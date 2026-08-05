"""The contract every model plugs into.

Deliberately one required method. The harness hands an adapter a batch of
images and question strings and gets back whatever the model said, verbatim;
everything downstream -- normalizing to CLEVR's vocabulary, scoring, slicing by
compositional axis -- is the harness's job, not the model's. An adapter that
post-processes its own answers would be scoring itself under different rules
than the other models in the comparison.

A new model is therefore:

    class MyVLM(VLMAdapter):
        name = "my-vlm"

        def load(self):
            self.model = ...

        def answer_batch(self, images, questions):
            return [self.model.ask(i, q) for i, q in zip(images, questions)]

and either a `register("my-vlm", ...)` call or `--model path/to/file.py:MyVLM`
on the command line. See `clevrbench/models/hf.py` for a real one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class VLMAdapter(ABC):
    """A vision-language model the harness can score."""

    #: Short identifier recorded in results.json.
    name = "unnamed"

    #: Set False for adapters that ignore the image (the baselines), letting
    #: the runner skip PNG decoding entirely.
    needs_images = True

    def load(self):
        """Bring the model into memory. Called once, before any batch.

        Kept separate from __init__ so listing and configuring models stays
        cheap -- `clevrbench models` should not load 14 GB of weights.
        """

    @abstractmethod
    def answer_batch(self, images, questions):
        """Answer `questions` about `images`, positionally paired.

        Args:
            images: list of PIL.Image.Image, already RGB.
            questions: list of question strings, same length.

        Returns:
            list of raw model output strings, same length and order. Return
            what the model actually produced -- do not normalize or truncate.
        """

    def unload(self):
        """Release resources. Called once, after the last batch."""

    def describe(self):
        """Configuration to record in results.json for reproducibility."""
        return {"name": self.name, "adapter": type(self).__name__}

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *exc):
        self.unload()
        return False
