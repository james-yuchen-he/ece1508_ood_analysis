"""Reference points that need no weights, no GPU, and no torch.

Two jobs. First, they make the harness testable end to end in a second, which
is how you tell a broken pipeline from a bad model. Second, they anchor the
scores: CLEVR's answer distribution is skewed enough that always saying "no"
scores about 0.21 overall and around 0.5 on the yes/no categories, so a model
reporting 0.3 overall has not obviously done anything. The metrics module also
computes a per-slice majority baseline directly from the gold answers; these
adapters are the runnable version of the same idea.
"""

from __future__ import annotations

from random import Random

from ..answers import CLEVR_ANSWERS
from .base import VLMAdapter


class ConstantAdapter(VLMAdapter):
    """Always answers the same thing."""

    needs_images = False

    def __init__(self, answer="yes"):
        self.answer = str(answer)
        self.name = f"constant[{self.answer}]"

    def answer_batch(self, images, questions):
        return [self.answer] * len(questions)

    def describe(self):
        return {"name": self.name, "adapter": "ConstantAdapter", "answer": self.answer}


class RandomAdapter(VLMAdapter):
    """Uniform over CLEVR's 28 answers, ignoring the image and the question."""

    needs_images = False

    def __init__(self, seed=0):
        self.seed = int(seed)
        self.name = "random"
        self._vocabulary = sorted(CLEVR_ANSWERS)
        self._rng = Random(self.seed)

    def load(self):
        self._rng = Random(self.seed)  # reset, so a run is reproducible

    def answer_batch(self, images, questions):
        return [self._rng.choice(self._vocabulary) for _ in questions]

    def describe(self):
        return {
            "name": self.name,
            "adapter": "RandomAdapter",
            "seed": self.seed,
            "vocabulary_size": len(self._vocabulary),
        }
