"""Aggregating predictions into the numbers a compositional-reasoning claim needs.

Three things this module insists on that a bare accuracy number leaves out:

1. **Error bars.** A 500-question subset estimates accuracy to about +-4 points.
   Reporting 0.31 vs 0.28 as a difference without an interval is not a result.
   Wilson intervals are used (they behave at the extremes, where accuracy on
   small strata tends to live).

2. **A baseline per slice.** CLEVR's yes/no categories are ~50% answerable by
   guessing; its open-vocabulary categories are not. An overall score mixes
   those together. Every slice therefore also reports the accuracy of always
   answering that slice's most common gold answer, and the delta against it.
   A model at 0.50 on `exist` and one at 0.50 on `count` are not comparable.

3. **Whether the model is actually answering.** `unparsed_rate` catches output
   that never lands in the vocabulary; the prediction histogram catches the
   opposite failure, a model that collapses onto one answer and rides the
   class prior.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from .taxonomy import QUESTION_TYPES

# Slice name -> the record field holding that slice's label.
AXES = (
    ("by_type", "type"),
    ("by_depth", "depth_bucket"),
    ("by_hops", "hops_bucket"),
    ("by_same_attribute", "same_attr"),
)

Z_95 = 1.959963984540054


def wilson_interval(successes, total, z=Z_95):
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because subsets are small and
    per-slice accuracies often sit near 0 or 1, where the normal interval
    runs outside [0, 1].
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (max(0.0, center - half), min(1.0, center + half))


def _sort_key(field, label):
    """Order slices for display: canonical for types, numeric for buckets."""
    if field == "type":
        order = {t: i for i, t in enumerate(QUESTION_TYPES)}
        return (order.get(label, len(order)), str(label))
    if field == "depth_bucket":
        return (int(str(label).split("-")[0].rstrip("+")), str(label))
    if field == "hops_bucket":
        return (int(str(label).split()[0].rstrip("+")), str(label))
    return (0, str(label))


def _slice_stats(records):
    """Accuracy, interval, majority baseline and parse health for one group."""
    n = len(records)
    if n == 0:
        return {"n": 0, "accuracy": 0.0}
    correct = sum(1 for r in records if r["correct"])
    gold_counts = Counter(r["gold"] for r in records)
    majority_answer, majority_n = gold_counts.most_common(1)[0]
    accuracy = correct / n
    majority = majority_n / n
    lo, hi = wilson_interval(correct, n)
    return {
        "n": n,
        "accuracy": accuracy,
        "ci95": [lo, hi],
        "n_correct": correct,
        # What always answering this slice's most common gold answer would score.
        "majority_baseline": majority,
        "majority_answer": majority_answer,
        "delta_vs_majority": accuracy - majority,
        "unparsed_rate": sum(1 for r in records if r["pred"] == "") / n,
    }


def aggregate(records):
    """Full metric set for a finished run."""
    n = len(records)
    if n == 0:
        return {"n_questions": 0, "overall": {"n": 0, "accuracy": 0.0}}

    overall = _slice_stats(records)
    overall["fallback_parse_rate"] = (
        sum(1 for r in records if r.get("parse_path") == "fallback") / n
    )

    result = {
        "n_questions": n,
        "overall": overall,
        "prediction_distribution": dict(
            Counter(r["pred"] or "<unparsed>" for r in records).most_common()
        ),
        "gold_distribution": dict(Counter(r["gold"] for r in records).most_common()),
    }

    for name, field in AXES:
        groups = defaultdict(list)
        for r in records:
            groups[r[field]].append(r)
        result[name] = {
            str(label): _slice_stats(rs)
            for label, rs in sorted(groups.items(), key=lambda kv: _sort_key(field, kv[0]))
        }

    # The overall majority baseline is a single answer for every question, which
    # a model that merely recognizes question types can beat without looking at
    # an image (answer "0" to counts, "no" to exists, ...). That stronger floor
    # is the honest reference for an overall score, so report it too.
    overall["majority_by_type_baseline"] = sum(
        s["majority_baseline"] * s["n"] for s in result["by_type"].values()
    ) / n
    overall["delta_vs_type_baseline"] = (
        overall["accuracy"] - overall["majority_by_type_baseline"]
    )

    # How concentrated the answers are. A model riding the class prior shows a
    # top-1 share far above the gold distribution's.
    preds = Counter(r["pred"] for r in records if r["pred"])
    if preds:
        top_pred, top_n = preds.most_common(1)[0]
        result["answer_collapse"] = {
            "top_prediction": top_pred,
            "top_prediction_share": top_n / n,
            "distinct_predictions": len(preds),
            "distinct_gold_answers": len(set(r["gold"] for r in records)),
        }
    return result
