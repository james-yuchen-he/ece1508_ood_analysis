"""Question-conditioned Q-Former for BLIP-2 OPT (the paper's VQA setup, Fig. 7).

The released OPT checkpoints ship the Q-Former without its text pathway: the
word/position embeddings and each layer's text feed-forward branch were
dropped after stage-1 pretraining. This module rebuilds that pathway and
initializes it from BLIP-2's stage-1 checkpoint (Salesforce/blip2-itm-vit-g),
whose text branch is exactly what stage-2 training left untouched. The 32
learned queries and the question tokens are concatenated and share the
Q-Former's self-attention (query_length=32 keeps cross-attention into the
image on the query positions), so the queries extract question-relevant
image features.

Run once to fetch the stage-1 text-side weights into qformer_text_init.pt:
    python clevr_blip2_qcond.py
"""

import copy
import os
import re

import torch
from transformers import AutoTokenizer, Blip2Config, Blip2ForConditionalGeneration
from transformers.models.blip_2.modeling_blip_2 import Blip2TextEmbeddings

MODEL_ID = "Salesforce/blip2-opt-2.7b"
STAGE1_ID = "Salesforce/blip2-itm-vit-g"
TEXT_INIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qformer_text_init.pt")

# Direct-child intermediate/output of an encoder layer = the text FFN branch
# (attention.output.* and *_query.* must not match).
TEXT_FFN_RE = re.compile(r"qformer\.encoder\.layer\.\d+\.(intermediate|output)\.")


def text_pathway_state(model):
    """All parameters of the grafted text pathway, by full name."""
    return {
        n: p for n, p in model.named_parameters()
        if n.startswith("qformer_text_embeddings.") or TEXT_FFN_RE.match(n)
    }


def freeze_text_pathway(model):
    """Freeze the grafted stage-1 text pathway (embeddings + text FFNs).

    The question then conditions the queries through the Q-Former's shared
    (trainable) self-attention while the text branch acts as a fixed question
    encoder — its optimizer state would not fit a 12GB GPU otherwise.
    """
    for p in text_pathway_state(model).values():
        p.requires_grad_(False)


def fetch_text_init():
    """Extract the Q-Former text pathway from the stage-1 checkpoint (one-time)."""
    from transformers import Blip2ForImageTextRetrieval

    itm = Blip2ForImageTextRetrieval.from_pretrained(STAGE1_ID, torch_dtype=torch.float32)
    sd = itm.state_dict()
    state = {
        "qformer_text_embeddings.word_embeddings.weight": sd["embeddings.word_embeddings.weight"],
        "qformer_text_embeddings.position_embeddings.weight": sd["embeddings.position_embeddings.weight"],
    }
    for k, v in sd.items():
        if TEXT_FFN_RE.match(k):
            state[k] = v
    torch.save(state, TEXT_INIT_PATH)
    print(f"saved {len(state)} tensors -> {TEXT_INIT_PATH}")


def build_qcond_model(dtype, state=None):
    """BLIP-2 OPT model whose Q-Former takes the question as text input.

    state: a tensor dict holding the text pathway (and optionally the rest of
    the finetuned bridge, e.g. a checkpoint from clevr_blip2_finetune.py).
    Defaults to the stage-1 init fetched by this file.
    Returns (model, qformer_tokenizer).
    """
    if state is None:
        assert os.path.exists(TEXT_INIT_PATH), (
            f"{TEXT_INIT_PATH} not found - run `python clevr_blip2_qcond.py` once to fetch it"
        )
        state = torch.load(TEXT_INIT_PATH, weights_only=True)

    config = Blip2Config.from_pretrained(MODEL_ID)
    config.qformer_config.use_qformer_text_input = True  # adds per-layer text FFN
    model = Blip2ForConditionalGeneration.from_pretrained(MODEL_ID, config=config, torch_dtype=dtype)

    emb_config = copy.deepcopy(config.qformer_config)
    emb_config.vocab_size = state["qformer_text_embeddings.word_embeddings.weight"].shape[0]
    model.qformer_text_embeddings = Blip2TextEmbeddings(emb_config).to(dtype)

    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    # Newly initialized modules (the text FFNs) materialize in fp32 regardless
    # of torch_dtype; unify so no-autocast inference gets one dtype throughout.
    model.to(dtype)

    tokenizer = AutoTokenizer.from_pretrained(STAGE1_ID)
    return model, tokenizer


def qcond_visual_prefix(model, pixel_values, qformer_input_ids, qformer_attention_mask):
    """The 32 projected query outputs, conditioned on image and question."""
    image_embeds = model.vision_model(pixel_values=pixel_values).last_hidden_state
    image_mask = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=image_embeds.device)

    query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_embeds = model.qformer_text_embeddings(
        input_ids=qformer_input_ids, query_embeds=query_tokens
    )
    query_mask = torch.ones(query_tokens.shape[:-1], dtype=torch.long, device=query_tokens.device)
    attention_mask = torch.cat([query_mask, qformer_attention_mask], dim=1)

    out = model.qformer(
        query_embeds=query_embeds,
        query_length=query_tokens.shape[1],
        attention_mask=attention_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_mask,
    )
    return model.language_projection(out.last_hidden_state[:, : query_tokens.shape[1]])


def _lm_inputs(model, prefix, input_ids, attention_mask):
    embeds = model.language_model.get_input_embeddings()(input_ids)
    inputs_embeds = torch.cat([prefix.to(embeds.dtype), embeds], dim=1)
    prefix_mask = torch.ones(
        prefix.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device
    )
    return inputs_embeds, torch.cat([prefix_mask, attention_mask], dim=1)


def qcond_loss(model, pixel_values, input_ids, attention_mask, labels,
               qformer_input_ids, qformer_attention_mask):
    prefix = qcond_visual_prefix(model, pixel_values, qformer_input_ids, qformer_attention_mask)
    inputs_embeds, full_mask = _lm_inputs(model, prefix, input_ids, attention_mask)
    prefix_labels = torch.full(prefix.shape[:2], -100, dtype=labels.dtype, device=labels.device)
    full_labels = torch.cat([prefix_labels, labels], dim=1)
    return model.language_model(
        inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=full_labels
    ).loss


@torch.no_grad()
def qcond_generate(model, pixel_values, input_ids, attention_mask,
                   qformer_input_ids, qformer_attention_mask, **generate_kwargs):
    prefix = qcond_visual_prefix(model, pixel_values, qformer_input_ids, qformer_attention_mask)
    inputs_embeds, full_mask = _lm_inputs(model, prefix, input_ids, attention_mask)
    return model.language_model.generate(
        inputs_embeds=inputs_embeds, attention_mask=full_mask, **generate_kwargs
    )


if __name__ == "__main__":
    fetch_text_init()
