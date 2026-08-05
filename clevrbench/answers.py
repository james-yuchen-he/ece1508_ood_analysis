"""Mapping free-form model text onto CLEVR's closed answer vocabulary.

CLEVR only ever answers one of 28 things. Generative VLMs emit sentences, so
scoring needs a normalizer, and the normalizer is part of the measurement:
too strict and you report format failures as reasoning failures, too loose and
you credit the model for words that happen to appear in a rambling answer.

Three modes, so the choice is explicit and reportable:

    strict    only a terse, already-valid answer counts
    typed     (default) prefer answers of the kind the question asks for
    lenient   first vocabulary word anywhere in the output

`typed` exists because the question's program says exactly which answers are
admissible: a `query_material` question can only be answered rubber/metal. On
"the ball is red" for a colour question, `lenient` returns `sphere` (the first
vocabulary word) while `typed` returns `red`. Whichever mode is used, the raw
string is kept in predictions.jsonl so scoring stays auditable.
"""

from __future__ import annotations

import re

COUNTS = {str(i) for i in range(11)}
COLORS = {"gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"}
SHAPES = {"cube", "sphere", "cylinder"}
SIZES = {"large", "small"}
MATERIALS = {"rubber", "metal"}
BOOLEAN = {"yes", "no"}

CLEVR_ANSWERS = BOOLEAN | COUNTS | COLORS | SHAPES | SIZES | MATERIALS

PARSE_MODES = ("strict", "typed", "lenient")

# Which answers the question's final program node admits. Anything else is,
# by construction, not an answer to the question that was asked.
_EXPECTED_BY_FUNCTION = {
    "count": COUNTS,
    "exist": BOOLEAN,
    "equal_integer": BOOLEAN,
    "greater_than": BOOLEAN,
    "less_than": BOOLEAN,
    "query_color": COLORS,
    "query_shape": SHAPES,
    "query_size": SIZES,
    "query_material": MATERIALS,
    **{f"equal_{a}": BOOLEAN for a in ("color", "size", "material", "shape")},
}

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "none": "0",
}

# Free text the model emits for concepts CLEVR names differently.
SYNONYMS = {
    "grey": "gray",
    "ball": "sphere", "balls": "sphere", "round": "sphere", "spheres": "sphere",
    "block": "cube", "blocks": "cube", "box": "cube", "square": "cube",
    "cubes": "cube",
    "can": "cylinder", "cylindrical": "cylinder", "cylinders": "cylinder",
    "metallic": "metal", "shiny": "metal", "steel": "metal", "silver": "metal",
    "gold": "metal", "golden": "metal",
    "matte": "rubber", "rubbery": "rubber", "plastic": "rubber", "dull": "rubber",
    "big": "large", "bigger": "large", "biggest": "large", "larger": "large",
    "little": "small", "tiny": "small", "smaller": "small", "smallest": "small",
    "yep": "yes", "yeah": "yes", "true": "yes", "correct": "yes",
    "nope": "no", "false": "no",
}


def expected_answers(question):
    """The answer set admitted by this question's program, or None if unknown."""
    program = question.get("program") or []
    if not program:
        return None
    tail = program[-1]
    fn = tail.get("function") or tail.get("type") or ""
    return _EXPECTED_BY_FUNCTION.get(fn)


def _clean(text):
    """Lowercase first line, punctuation stripped, whitespace collapsed."""
    t = text.strip().lower().split("\n")[0]
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return " ".join(t.split())


def _resolve(token, numeric):
    """Map one token into the CLEVR vocabulary.

    `numeric` says whether digits are admissible here: "no" is a legitimate
    yes/no answer, so it is only read as the count 0 for counting questions.
    """
    token = SYNONYMS.get(token, token)
    if numeric:
        if token == "no":
            return "0"
        return NUMBER_WORDS.get(token, token)
    return token


def normalize(text, question=None, mode="typed"):
    """Normalize model output to a CLEVR answer.

    Returns (answer, parse_path) where answer is "" if nothing in the output
    resolves to a valid answer. parse_path records how the answer was reached
    -- "exact", "typed", "fallback", or "none" -- so a run can report how much
    of its score depended on lenient parsing.
    """
    if mode not in PARSE_MODES:
        raise ValueError(f"unknown parse mode {mode!r}; expected one of {PARSE_MODES}")

    expected = expected_answers(question) if question is not None else None
    numeric = expected is COUNTS
    cleaned = _clean(text)
    if not cleaned:
        return "", "none"

    # A terse, correctly formatted answer lands here.
    whole = _resolve(cleaned, numeric)
    if whole in CLEVR_ANSWERS:
        return whole, "exact"
    if mode == "strict":
        return "", "none"

    tokens = cleaned.split()

    # Prefer a token of the kind the question actually asks for.
    if mode == "typed" and expected:
        for tok in tokens:
            cand = _resolve(tok, numeric)
            if cand in expected:
                return cand, "typed"

    # Otherwise the first token resolving anywhere into the vocabulary.
    for tok in tokens:
        cand = _resolve(tok, numeric)
        if cand in CLEVR_ANSWERS:
            return cand, "fallback"
    return "", "none"
