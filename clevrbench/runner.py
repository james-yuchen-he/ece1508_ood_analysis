"""The evaluation loop.

Model-agnostic by construction: it knows how to batch questions, hand images to
an adapter, normalize what comes back, and attach the taxonomy labels each
prediction gets scored under. It does not know what the model is.

Each record keeps the raw model string alongside the normalized prediction, so
any scoring decision can be re-examined after the fact without re-running the
model -- which matters, because on a free-text model against a closed answer
vocabulary the normalizer is part of the measurement.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path

from PIL import Image

from . import answers, taxonomy


def _progress(iterable, total, enabled, desc):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc, unit="batch")
    except ImportError:
        return iterable


class ImageCache:
    """Small LRU of decoded images.

    Questions arrive sorted by filename and CLEVR asks ~10 questions per image,
    so a handful of slots removes almost all repeated PNG decoding.
    """

    def __init__(self, directory, maxsize=32):
        self.directory = Path(directory)
        self.maxsize = maxsize
        self._cache = OrderedDict()
        self.loads = 0

    def get(self, filename):
        if filename in self._cache:
            self._cache.move_to_end(filename)
            return self._cache[filename]
        path = self.directory / filename
        if not path.exists():
            raise SystemExit(f"image missing from the subset: {path}")
        image = Image.open(path).convert("RGB")
        self.loads += 1
        self._cache[filename] = image
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return image


def evaluate(model, questions, images_dir, batch_size=8, parse_mode="typed",
             progress=True):
    """Score `model` on `questions`; returns (records, runtime_stats)."""
    cache = ImageCache(images_dir)
    records = []
    batches = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]

    start = time.time()
    for batch in _progress(batches, len(batches), progress, "eval"):
        texts = [q["question"] for q in batch]
        if model.needs_images:
            images = [cache.get(q["image_filename"]) for q in batch]
        else:
            images = [None] * len(batch)

        try:
            raw_answers = model.answer_batch(images, texts)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise SystemExit(
                    f"CUDA out of memory at batch size {batch_size}. "
                    "Retry with a smaller --batch-size."
                ) from exc
            raise

        if len(raw_answers) != len(batch):
            raise SystemExit(
                f"{model.name} returned {len(raw_answers)} answers "
                f"for {len(batch)} questions; adapters must preserve order and length."
            )

        for question, raw in zip(batch, raw_answers):
            raw = "" if raw is None else str(raw)
            prediction, parse_path = answers.normalize(raw, question, mode=parse_mode)
            gold = question["answer"].strip().lower()
            features = taxonomy.features(question)
            records.append({
                "question_index": question.get("question_index"),
                "image_filename": question["image_filename"],
                "question": question["question"],
                "gold": gold,
                "raw": raw,
                "pred": prediction,
                "parse_path": parse_path,
                "correct": prediction == gold,
                **features,
            })

    elapsed = time.time() - start
    stats = {
        "seconds": round(elapsed, 2),
        "questions_per_second": round(len(records) / elapsed, 2) if elapsed else 0.0,
        "batch_size": batch_size,
        "images_decoded": cache.loads,
    }
    return records, stats
