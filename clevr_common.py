"""Shared CLEVR answer vocabulary and soft-match normalization.

CLEVR answers come from a closed vocabulary of 28 values: the numbers 0-10,
eight colors, three shapes, two sizes, two materials, and yes/no. Soft
matching maps a free-form generation onto that vocabulary: normalize the
text, canonicalize synonyms word by word, then take the first word that is
in the vocabulary expected for the question type.
"""

import re
import string

# Prompt shared by evaluation and finetuning so train and test match.
PROMPT = "Question: {} Short answer:"

# The question type is the last function of the CLEVR functional program.
# Result CSVs store question_type_id = index into this list.
QUESTION_TYPES = [
    "count",           # 0
    "exist",           # 1
    "query_color",     # 2
    "query_size",      # 3
    "query_material",  # 4
    "query_shape",     # 5
    "equal_color",     # 6
    "equal_size",      # 7
    "equal_material",  # 8
    "equal_shape",     # 9
    "equal_integer",   # 10
    "greater_than",    # 11
    "less_than",       # 12
]
QUESTION_TYPE_ID = {name: i for i, name in enumerate(QUESTION_TYPES)}

NUMBERS = [str(n) for n in range(11)]
COLORS = ["blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"]
SHAPES = ["cube", "cylinder", "sphere"]
SIZES = ["large", "small"]
MATERIALS = ["metal", "rubber"]
YES_NO = ["yes", "no"]

ANSWER_VOCAB = set(NUMBERS + COLORS + SHAPES + SIZES + MATERIALS + YES_NO)

# Word-level canonicalization into the CLEVR vocabulary: number words, plus
# the attribute synonyms CLEVR's own question generator uses.
CANONICAL = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "big": "large",
    "huge": "large",
    "tiny": "small",
    "little": "small",
    "metallic": "metal",
    "shiny": "metal",
    "matte": "rubber",
    "block": "cube",
    "blocks": "cube",
    "cubes": "cube",
    "ball": "sphere",
    "balls": "sphere",
    "spheres": "sphere",
    "cylinders": "cylinder",
    "grey": "gray",
}

# Answer vocabulary each question type is expected to draw from.
TYPE_VOCAB = {
    "count": set(NUMBERS),
    "query_color": set(COLORS),
    "query_size": set(SIZES),
    "query_material": set(MATERIALS),
    "query_shape": set(SHAPES),
}
for _t in ("exist", "equal_color", "equal_size", "equal_material", "equal_shape",
           "equal_integer", "greater_than", "less_than"):
    TYPE_VOCAB[_t] = set(YES_NO)

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.strip().lower().translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", text).strip()


def accuracy_report(corrects, totals) -> str:
    """Format per-question-type and overall accuracy as a table.

    corrects/totals are lists indexed by question_type_id.
    """
    lines = [f"{'id':>2}  {'question_type':<16} {'correct':>8} {'total':>8} {'accuracy':>8}"]
    for type_id, name in enumerate(QUESTION_TYPES):
        if totals[type_id] == 0:
            continue
        acc = corrects[type_id] / totals[type_id]
        lines.append(
            f"{type_id:>2}  {name:<16} {corrects[type_id]:>8} {totals[type_id]:>8} {acc:>8.4f}"
        )
    total = sum(totals)
    correct = sum(corrects)
    lines.append("")
    lines.append(f"{'overall':<20} {correct:>8} {total:>8} {correct / total:>8.4f}")
    return "\n".join(lines)


def extract_answer(raw: str, question_type: str = None) -> str:
    """Map a raw generation to the CLEVR answer it soft-matches.

    Returns the first canonicalized word found in the expected vocabulary for
    the question type (any CLEVR answer if the type is unknown). If nothing
    from the vocabulary appears, returns the whole normalized text, which
    will simply fail to match.
    """
    words = [CANONICAL.get(w, w) for w in normalize(raw).split()]
    vocab = TYPE_VOCAB.get(question_type, ANSWER_VOCAB)
    for word in words:
        if word in vocab:
            return word
    return " ".join(words)
