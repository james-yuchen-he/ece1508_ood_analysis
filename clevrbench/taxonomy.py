"""CLEVR question taxonomy — the axes accuracy gets sliced along.

Every CLEVR question ships with the functional program that generated it, and
that program is what makes CLEVR a *compositional* benchmark rather than a
recognition one. Two questions can both be "count" questions while one is a
3-step lookup and the other chains four spatial relations across a scene.
Aggregate accuracy hides that difference; these axes expose it:

    type          the five standard reporting categories
    depth         number of program nodes — how many operations compose
    hops          number of `relate` nodes — chained spatial reference
    same_attr     requires matching an attribute across objects
    family        one of CLEVR's 90 question templates

`features()` returns all of them for a question, and the metrics module
reports accuracy per level of each.
"""

from __future__ import annotations

# Final program node -> the five standard CLEVR reporting categories.
_TYPE_BY_FUNCTION = {
    "count": "count",
    "exist": "exist",
    "equal_integer": "compare_number",
    "greater_than": "compare_number",
    "less_than": "compare_number",
    **{f"query_{a}": "query_attribute" for a in ("color", "size", "material", "shape")},
    **{f"equal_{a}": "compare_attribute" for a in ("color", "size", "material", "shape")},
}

QUESTION_TYPES = (
    "count",
    "exist",
    "compare_number",
    "query_attribute",
    "compare_attribute",
)

# Depth buckets chosen from the val-split distribution (programs run 2..25
# nodes) so that no bucket is too thin to estimate accuracy in: roughly
# 14% / 38% / 22% / 26% of the split.
DEPTH_BUCKETS = ((2, 6), (7, 10), (11, 15), (16, 10**6))


def _nodes(question):
    """Program nodes, tolerating both CLEVR key spellings and missing programs."""
    return question.get("program") or []


def _function(node):
    # v1.0 uses "function"; some derived releases use "type".
    return node.get("function") or node.get("type") or ""


def program_functions(question):
    """Multiset of function names appearing in the question's program."""
    return [_function(n) for n in _nodes(question)]


def question_type(question):
    """Standard CLEVR category, read off the final node of the program."""
    nodes = _nodes(question)
    if not nodes:
        return "unknown"
    return _TYPE_BY_FUNCTION.get(_function(nodes[-1]), "unknown")


def program_depth(question):
    """Number of program nodes: how many operations the answer composes over."""
    return len(_nodes(question))


def depth_bucket(depth):
    """Human-readable bucket label for a program depth."""
    for lo, hi in DEPTH_BUCKETS:
        if lo <= depth <= hi:
            return f"{lo}-{hi}" if hi < 10**6 else f"{lo}+"
    return "unknown"


def relate_hops(question):
    """Count of `relate` nodes — how far spatial reference is chained.

    "the cube left of the sphere behind the cylinder" is 2 hops. This is the
    axis where compositional generalization tends to break down first.
    """
    return sum(1 for f in program_functions(question) if f == "relate")


def needs_same_attribute(question):
    """Whether the question requires matching an attribute across objects.

    The `same_color` / `same_shape` / ... family forces a comparison against
    every other object in the scene rather than a single lookup.
    """
    return any(f.startswith("same_") for f in program_functions(question))


def features(question):
    """All taxonomy axes for one question, as a flat dict of stratum labels.

    The keys here are exactly the slices `metrics` reports on, so adding an
    axis means adding it once, here.
    """
    depth = program_depth(question)
    hops = relate_hops(question)
    return {
        "type": question_type(question),
        "depth": depth,
        "depth_bucket": depth_bucket(depth),
        "hops": hops,
        "hops_bucket": ("1 hop" if hops == 1 else f"{hops} hops") if hops < 3 else "3+ hops",
        "same_attr": bool(needs_same_attribute(question)),
        "family": question.get("question_family_index"),
    }
