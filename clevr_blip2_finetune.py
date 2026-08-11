"""Fine-tune BLIP-2 (OPT-2.7B) for visual question answering on CLEVR train.

Follows the BLIP-2 paper's VQA finetuning recipe (Sec 4.3, Table 8) with the
open-ended answer generation loss: autoregressive cross-entropy on the answer
tokens (+ EOS) only, conditioned on the image and the question. Recipe: AdamW
beta=(0.9, 0.999), weight decay 0.05, lr 1e-5 with linear warmup and cosine
decay, effective batch size = batch_size * grad_accum = 128 by default (the
paper trains 5 epochs; CLEVR train is large, so the default here is 1).

Deviations from the paper, by design:
- Only the Q-Former bridge trains (Q-Former, its 32 query tokens, and the
  projection into the LLM). The paper also finetunes the ViT; here it stays
  frozen along with the LLM. (Remove language_projection below to train
  strictly the Q-Former weights.)
- Image resolution stays at the checkpoint's 224 (paper uses 490).
- The prompt keeps "Short answer:" to match our eval.

As in the paper, the Q-Former is additionally conditioned on the question
(see clevr_blip2_qcond.py; run `python clevr_blip2_qcond.py` once first to
fetch the stage-1 text-pathway weights). The grafted text pathway itself
stays frozen as a fixed question encoder — training it too would not fit a
12GB GPU. Pass --no-qcond for the image-only Q-Former behavior.

The checkpoint stores only the trainable parameters. To use it at eval time:

    model = Blip2ForConditionalGeneration.from_pretrained(MODEL_ID, ...)
    model.load_state_dict(torch.load("qformer_clevr.pt"), strict=False)

Usage:
    python clevr_blip2_finetune.py [--epochs 1] [--batch-size 16]
        [--grad-accum 8] [--lr 1e-5] [--warmup-steps 1000] [--max-samples N]
        [--out qformer_clevr.pt]
"""

import argparse
import json
import math
import os
import random

# Reduce CUDA memory fragmentation; must be set before torch allocates.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from clevr_blip2_qcond import (
    build_qcond_model,
    freeze_text_pathway,
    qcond_loss,
    text_pathway_state,
)
from clevr_common import PROMPT

CLEVR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "CLEVR_v1.0")
MODEL_ID = "Salesforce/blip2-opt-2.7b"


class ClevrTrain(Dataset):
    def __init__(self, max_samples=None):
        with open(os.path.join(CLEVR_ROOT, "questions", "CLEVR_train_questions.json")) as f:
            questions = json.load(f)["questions"]
        if max_samples is not None:
            # Fixed-seed sample so a subset spans many images, reproducibly.
            questions = random.Random(0).sample(questions, max_samples)
        self.questions = questions

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, i):
        q = self.questions[i]
        image = Image.open(
            os.path.join(CLEVR_ROOT, "images", "train", q["image_filename"])
        ).convert("RGB")
        return {"image": image, "question": q["question"], "answer": q["answer"]}


def make_collate(processor, qformer_tokenizer=None):
    tokenizer = processor.tokenizer

    def collate(batch):
        images = [b["image"] for b in batch]
        texts = [
            PROMPT.format(b["question"]) + " " + b["answer"] + tokenizer.eos_token
            for b in batch
        ]
        enc = processor(images=images, text=texts, padding=True, return_tensors="pt")
        # Supervise only the answer + EOS: mask padding and everything before
        # the answer (prompt, and image placeholders if the processor adds
        # any). Assumes right padding.
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i, b in enumerate(batch):
            answer_len = len(
                tokenizer(" " + b["answer"] + tokenizer.eos_token, add_special_tokens=False).input_ids
            )
            seq_len = int(enc["attention_mask"][i].sum())
            labels[i, : seq_len - answer_len] = -100
        enc["labels"] = labels
        if qformer_tokenizer is not None:
            q_enc = qformer_tokenizer(
                [b["question"] for b in batch],
                padding=True, truncation=True, max_length=64, return_tensors="pt",
            )
            enc["qformer_input_ids"] = q_enc["input_ids"]
            enc["qformer_attention_mask"] = q_enc["attention_mask"]
        return enc

    return collate


def save_trainable(model, path):
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    if hasattr(model, "qformer_text_embeddings"):
        # Include the frozen text pathway so the checkpoint is self-contained.
        state.update({n: p.detach().cpu() for n, p in text_pathway_state(model).items()})
    torch.save(state, path)
    print(f"saved checkpoint -> {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="paper uses 5")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="micro-batch; 4 fits a 12GB GPU (8 with --no-qcond)")
    parser.add_argument("--grad-accum", type=int, default=32,
                        help="batch_size * grad_accum = effective batch (paper: 128)")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="optimizer steps of linear warmup (paper: 1000)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="train on a random subset instead of all 700k questions")
    parser.add_argument("--out", default="qformer_clevr_fine_tuned.pt")
    parser.add_argument("--save-every", type=int, default=500,
                        help="optimizer steps between checkpoints")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-qcond", action="store_true",
                        help="do not condition the Q-Former on the question")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "finetuning requires a CUDA GPU"
    # No GradScaler in this script, so fp16 autocast would risk gradient
    # underflow: require a bf16-capable (Ampere or newer) GPU instead.
    assert torch.cuda.is_bf16_supported(), "finetuning requires a bf16-capable GPU"
    device = "cuda"
    frozen_dtype = torch.bfloat16

    processor = Blip2Processor.from_pretrained(MODEL_ID)
    # Right padding for training so every sequence starts at position 0 and
    # labels line up (generation-time left padding is not needed here).
    processor.tokenizer.padding_side = "right"

    if args.no_qcond:
        model = Blip2ForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=frozen_dtype)
        qformer_tokenizer = None
    else:
        model, qformer_tokenizer = build_qcond_model(frozen_dtype)
    model.to(device)

    # Freeze everything, then re-enable the bridge in fp32 for stable
    # optimization; the frozen towers stay in half precision.
    for p in model.parameters():
        p.requires_grad_(False)
    for module in (model.qformer, model.language_projection):
        module.to(torch.float32)
        for p in module.parameters():
            p.requires_grad_(True)
    if not args.no_qcond:
        freeze_text_pathway(model)
    model.query_tokens.data = model.query_tokens.data.float()
    model.query_tokens.requires_grad_(True)

    model.eval()           # dropout off in the frozen towers
    model.qformer.train()  # dropout on where we train

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable) / 1e6:.1f}M")

    dataset = ClevrTrain(args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=make_collate(processor, qformer_tokenizer),
    )

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = min(args.warmup_steps, max(1, total_steps // 10))  # guard short runs
    print(f"{len(dataset)} samples, {total_steps} optimizer steps, {warmup} warmup")

    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.05
    )

    def lr_lambda(s):
        if s < warmup:
            return (s + 1) / warmup
        progress = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step = 0
    running = 0.0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=frozen_dtype):
                if args.no_qcond:
                    loss = model(**batch).loss / args.grad_accum
                else:
                    loss = qcond_loss(model, **batch) / args.grad_accum
            loss.backward()
            running += loss.item()

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                step += 1
                if step % 50 == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    print(
                        f"epoch {epoch} step {step}/{total_steps} "
                        f"loss {running / 50:.4f} lr {lr_now:.2e}",
                        flush=True,
                    )
                    running = 0.0
                if step % args.save_every == 0:
                    save_trainable(model, args.out)

    save_trainable(model, args.out)


if __name__ == "__main__":
    main()
