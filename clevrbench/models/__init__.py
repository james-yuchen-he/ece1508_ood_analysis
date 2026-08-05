"""Model registry: names on the command line, adapter classes behind them.

Three ways to get a model into the harness, in increasing order of effort:

    --model blip2-opt-2.7b              a built-in name (below)
    --model hf-auto --model-arg model_id=some/hf-repo
                                        any HF image-text-to-text checkpoint
    --model my_models.py:MyVLM          your own VLMAdapter subclass

Built-ins are declared lazily as dotted paths, so `clevrbench models` lists
them without importing torch and the torch-free baselines run on a machine with
no GPU stack at all.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

from .base import VLMAdapter

# name -> (dotted class path, default kwargs, one-line description)
_BUILTIN = {
    "blip2-opt-2.7b": (
        "clevrbench.models.hf:Blip2Adapter",
        {"model_id": "Salesforce/blip2-opt-2.7b"},
        "BLIP-2 with an OPT-2.7B decoder (zero-shot VQA)",
    ),
    "blip2-opt-6.7b": (
        "clevrbench.models.hf:Blip2Adapter",
        {"model_id": "Salesforce/blip2-opt-6.7b"},
        "BLIP-2 with an OPT-6.7B decoder",
    ),
    "blip2-flan-t5-xl": (
        "clevrbench.models.hf:Blip2Adapter",
        {"model_id": "Salesforce/blip2-flan-t5-xl"},
        "BLIP-2 with a Flan-T5-XL decoder; follows short-answer prompts more closely",
    ),
    "instructblip-vicuna-7b": (
        "clevrbench.models.hf:InstructBlipAdapter",
        {"model_id": "Salesforce/instructblip-vicuna-7b"},
        "InstructBLIP, instruction-tuned on VQA data",
    ),
    "hf-auto": (
        "clevrbench.models.hf:HFAutoAdapter",
        {},
        "any HF image-text-to-text checkpoint; pass --model-arg model_id=...",
    ),
    "constant": (
        "clevrbench.models.baselines:ConstantAdapter",
        {},
        "baseline: always the same answer (--model-arg answer=yes)",
    ),
    "random": (
        "clevrbench.models.baselines:RandomAdapter",
        {},
        "baseline: uniform over CLEVR's 28 answers (--model-arg seed=0)",
    ),
}

_REGISTERED = {}


def register(name, factory, description=""):
    """Register an adapter factory under `name` at runtime."""
    _REGISTERED[name] = (factory, description)
    return factory


def available():
    """(name, description) for every model the CLI will accept by name."""
    entries = [(n, spec[2]) for n, spec in _BUILTIN.items()]
    entries += [(n, desc) for n, (_, desc) in _REGISTERED.items()]
    return sorted(entries)


def _import_from_path(spec):
    """Resolve 'pkg.module:Class' or '/path/to/file.py:Class' to the class."""
    module_part, _, attr = spec.rpartition(":")
    if not module_part or not attr:
        raise SystemExit(f"malformed model spec {spec!r}; expected 'module:Class'")

    if module_part.endswith(".py"):
        path = Path(module_part).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"no such adapter file: {path}")
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(module_part)
        except ImportError as exc:
            raise SystemExit(f"cannot import {module_part!r} for model {spec!r}: {exc}")

    try:
        return getattr(module, attr)
    except AttributeError:
        raise SystemExit(f"{module_part} has no attribute {attr!r}")


def _instantiate(factory, kwargs, spec):
    """Construct an adapter, turning a bad --model-arg into a usable message."""
    try:
        return factory(**kwargs)
    except TypeError as exc:
        import inspect

        try:
            accepted = [p for p in inspect.signature(factory).parameters if p != "self"]
            hint = f"\naccepted --model-arg keys: {', '.join(accepted) or '(none)'}"
        except (TypeError, ValueError):
            hint = ""
        raise SystemExit(f"cannot construct model {spec!r}: {exc}{hint}") from exc


def build(spec, **kwargs):
    """Instantiate a model from a registry name or a module:Class path."""
    if spec in _REGISTERED:
        factory, _ = _REGISTERED[spec]
        model = _instantiate(factory, kwargs, spec)
    elif spec in _BUILTIN:
        path, defaults, _ = _BUILTIN[spec]
        merged = {**defaults, **kwargs}
        model = _instantiate(_import_from_path(path), merged, spec)
        model.name = model.name if model.name != "unnamed" else spec
    elif ":" in spec:
        model = _instantiate(_import_from_path(spec), kwargs, spec)
    else:
        names = ", ".join(n for n, _ in available())
        raise SystemExit(
            f"unknown model {spec!r}.\nKnown names: {names}\n"
            "Or pass an adapter directly as 'module:Class' / 'file.py:Class'."
        )

    if not isinstance(model, VLMAdapter):
        raise SystemExit(f"{spec} is not a VLMAdapter subclass")
    return model


__all__ = ["VLMAdapter", "available", "build", "register"]
