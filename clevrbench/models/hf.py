"""Adapters for HuggingFace image-text-to-text models.

`HFImageTextToText` carries everything that is the same across this family --
batched processing, left padding for decoder-only language models, greedy
decoding, stripping an echoed prompt back off the output. A specific model is
then usually just a class attribute or two, which is the point: swapping BLIP-2
for InstructBLIP should not mean reimplementing an evaluation loop.
"""

from __future__ import annotations

import torch

from .base import VLMAdapter

_DTYPES = {
    "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float32": torch.float32, "fp32": torch.float32, "float": torch.float32,
}


def _auto_model_class():
    """The generic image-text-to-text auto class, across transformers versions."""
    import transformers

    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        if hasattr(transformers, name):
            return getattr(transformers, name)
    raise SystemExit(
        "transformers exposes neither AutoModelForImageTextToText nor "
        "AutoModelForVision2Seq; pass an explicit adapter instead of hf-auto."
    )


class HFImageTextToText(VLMAdapter):
    """Batched greedy VQA over a HuggingFace vision-language model."""

    #: Concrete model class; None means resolve the auto class at load time.
    model_cls = None
    #: Concrete processor class; None means AutoProcessor.
    processor_cls = None
    default_model_id = None
    default_prompt = "Question: {question} Answer:"

    def __init__(self, model_id=None, prompt=None, dtype="float16", device="auto",
                 max_new_tokens=10, num_beams=1, cache_dir=None,
                 trust_remote_code=False):
        self.model_id = model_id or self.default_model_id
        if not self.model_id:
            raise SystemExit(
                f"{type(self).__name__} needs a checkpoint: "
                "--model-arg model_id=<hf-repo-or-path>"
            )
        self.prompt = prompt or self.default_prompt
        if "{question}" not in self.prompt:
            raise SystemExit("--model-arg prompt=... must contain '{question}'")
        if dtype not in _DTYPES and dtype != "auto":
            raise SystemExit(f"unknown dtype {dtype!r}; try {sorted(_DTYPES)} or 'auto'")

        self.dtype = dtype
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.num_beams = int(num_beams)
        self.cache_dir = cache_dir
        self.trust_remote_code = bool(trust_remote_code)
        self.name = self.model_id.rstrip("/").split("/")[-1]
        self.model = None
        self.processor = None

    # -- lifecycle --------------------------------------------------------
    def _torch_dtype(self):
        return _DTYPES.get(self.dtype)

    def _is_decoder_only(self):
        """Whether generation is decoder-only, so batches need left padding."""
        config = self.model.config
        # BLIP-2 and friends state this directly.
        explicit = getattr(config, "use_decoder_only_language_model", None)
        if explicit is not None:
            return bool(explicit)
        text_config = getattr(config, "text_config", config)
        return not getattr(text_config, "is_encoder_decoder", False)

    def load(self):
        from transformers import AutoProcessor

        processor_cls = self.processor_cls or AutoProcessor
        kwargs = {"cache_dir": self.cache_dir}
        if self.trust_remote_code:
            kwargs["trust_remote_code"] = True

        self.processor = processor_cls.from_pretrained(self.model_id, **kwargs)

        model_cls = self.model_cls or _auto_model_class()
        load_kwargs = dict(kwargs)
        if self.dtype != "auto":
            load_kwargs["dtype"] = self._torch_dtype()
        if self.device:
            load_kwargs["device_map"] = self.device
        self.model = model_cls.from_pretrained(self.model_id, **load_kwargs)
        self.model.eval()

        # A batched prompt for a decoder-only LM must be left-padded, or short
        # prompts generate from a run of pad tokens.
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and self._is_decoder_only():
            tokenizer.padding_side = "left"

    def unload(self):
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- inference --------------------------------------------------------
    def build_prompts(self, questions):
        """Question strings -> model input strings. Override for chat templates."""
        return [self.prompt.format(question=q) for q in questions]

    def extract_answer(self, decoded, prompt):
        """Strip an echoed prompt off the decoded output.

        Decoder-only models return the prompt followed by the continuation;
        encoder-decoder models return the continuation alone. Handling both
        here keeps the same adapter working across model families.
        """
        text = decoded.strip()
        stripped = prompt.strip()
        if text.startswith(stripped):
            return text[len(stripped):].strip()
        if stripped and stripped in text:
            return text.split(stripped, 1)[1].strip()
        if "Answer:" in text:
            return text.rsplit("Answer:", 1)[-1].strip()
        return text

    def answer_batch(self, images, questions):
        if self.model is None:
            raise RuntimeError("load() must be called before answer_batch()")

        prompts = self.build_prompts(questions)
        inputs = self.processor(
            images=list(images), text=prompts, return_tensors="pt", padding=True
        )
        # BatchFeature.to casts only floating-point tensors, so token ids stay
        # integral while pixel values move to the model's compute dtype.
        target_dtype = self._torch_dtype()
        inputs = (
            inputs.to(self.model.device, target_dtype)
            if target_dtype is not None
            else inputs.to(self.model.device)
        )

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
            )
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)
        return [self.extract_answer(d, p) for d, p in zip(decoded, prompts)]

    def describe(self):
        return {
            "name": self.name,
            "adapter": type(self).__name__,
            "model_id": self.model_id,
            "prompt": self.prompt,
            "dtype": self.dtype,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "decoding": "greedy" if self.num_beams == 1 else f"beam({self.num_beams})",
        }


class Blip2Adapter(HFImageTextToText):
    """BLIP-2 (OPT or Flan-T5 decoder)."""

    default_model_id = "Salesforce/blip2-opt-2.7b"

    def load(self):
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        self.model_cls = Blip2ForConditionalGeneration
        self.processor_cls = Blip2Processor
        super().load()


class InstructBlipAdapter(HFImageTextToText):
    """InstructBLIP, instruction-tuned; prefers an explicit short-answer cue."""

    default_model_id = "Salesforce/instructblip-vicuna-7b"
    default_prompt = "Question: {question} Short answer:"

    def load(self):
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

        self.model_cls = InstructBlipForConditionalGeneration
        self.processor_cls = InstructBlipProcessor
        super().load()


class HFAutoAdapter(HFImageTextToText):
    """Any image-text-to-text checkpoint, resolved through the auto classes."""
