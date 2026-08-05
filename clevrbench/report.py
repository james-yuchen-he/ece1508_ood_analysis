"""Console tables and the on-disk result format.

The console output leads with the comparison that matters -- accuracy against
the majority baseline for that same set of questions -- because on CLEVR the
raw number is easy to misread. A model can post ~0.50 on `exist` and look
halfway competent while doing nothing but guessing between two options.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import __version__

_AXIS_TITLES = {
    "by_type": "question type",
    "by_depth": "program depth (nodes)",
    "by_hops": "chained spatial relations",
    "by_same_attribute": "needs same-attribute match",
}


def _fmt_row(label, stats, width=28):
    lo, hi = stats.get("ci95", (0.0, 0.0))
    return (
        f"{label:<{width}}{stats['n']:>6}{stats['accuracy']:>9.3f}"
        f"   [{lo:.3f}, {hi:.3f}]{stats.get('majority_baseline', 0):>10.3f}"
        f"{stats.get('delta_vs_majority', 0):>+9.3f}"
    )


def _table(title, slices, width=28):
    header = (
        f"{title:<{width}}{'n':>6}{'acc':>9}{'95% CI':>18}"
        f"{'majority':>10}{'delta':>9}"
    )
    lines = ["", header, "-" * len(header)]
    for label, stats in slices.items():
        if stats.get("n"):
            lines.append(_fmt_row(label, stats, width))
    return lines


def format_report(results):
    """Render a finished run as text."""
    if "metrics" not in results:
        raise SystemExit(
            "this results.json predates clevrbench (no 'metrics' block) -- it was "
            "written by the standalone clevr_eval.py. Re-run it with "
            "`python -m clevrbench eval` to get the full breakdown."
        )
    metrics = results["metrics"]
    overall = metrics["overall"]
    model = results.get("model", {})
    subset = results.get("subset", {})
    runtime = results.get("runtime", {})

    title = f"{model.get('name', '?')} — CLEVR {subset.get('split', 'val')}"
    lo, hi = overall.get("ci95", (0.0, 0.0))
    lines = ["", "=" * 78, title, "=" * 78]
    lines.append(
        f"{metrics['n_questions']} questions"
        + (f" over {subset['n_images']} images" if subset.get("n_images") else "")
        + (f"  ·  {subset['path']}" if subset.get("path") else "")
    )
    lines.append("")
    lines.append(f"{'accuracy':<22}{overall['accuracy']:.4f}   [{lo:.4f}, {hi:.4f}] 95% CI")
    lines.append(
        f"{'majority baseline':<22}{overall['majority_baseline']:.4f}   "
        f"always answering {overall['majority_answer']!r}"
    )
    lines.append(f"{'delta vs baseline':<22}{overall['delta_vs_majority']:+.4f}")
    if "majority_by_type_baseline" in overall:
        lines.append(
            f"{'per-type baseline':<22}{overall['majority_by_type_baseline']:.4f}   "
            "best fixed answer per question type, no image needed"
        )
        lines.append(
            f"{'delta vs per-type':<22}{overall['delta_vs_type_baseline']:+.4f}"
            "   <- the number a compositional claim rests on"
        )
    lines.append(
        f"{'unparsed':<22}{overall['unparsed_rate']:.4f}   "
        "output outside CLEVR's 28-answer vocabulary"
    )
    if overall.get("fallback_parse_rate"):
        lines.append(
            f"{'loose parses':<22}{overall['fallback_parse_rate']:.4f}   "
            "answer recovered from an off-type word"
        )
    if runtime:
        lines.append(
            f"{'throughput':<22}{runtime.get('questions_per_second', 0):.2f} q/s   "
            f"({runtime.get('seconds', 0) / 60:.1f} min, batch {runtime.get('batch_size')})"
        )

    for axis, title in _AXIS_TITLES.items():
        if axis in metrics:
            lines += _table(title, metrics[axis])

    collapse = metrics.get("answer_collapse")
    if collapse:
        lines += [
            "",
            f"answer diversity: {collapse['distinct_predictions']} distinct predictions "
            f"vs {collapse['distinct_gold_answers']} distinct gold answers; "
            f"most frequent {collapse['top_prediction']!r} "
            f"({collapse['top_prediction_share']:.1%} of output)",
        ]
    return "\n".join(lines) + "\n"


def build_results(model, records, metrics, stats, subset_info, config):
    """Assemble the results.json payload."""
    return {
        "clevrbench_version": __version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model.describe(),
        "subset": subset_info,
        "config": config,
        "runtime": stats,
        "metrics": metrics,
    }


def write_run(out_dir, results, records):
    """Write results.json and predictions.jsonl; returns the directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    with open(out_dir / "predictions.jsonl", "w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return out_dir


def load_run(path):
    """Read back a results.json written by `write_run`."""
    path = Path(path)
    if path.is_dir():
        path = path / "results.json"
    if not path.exists():
        raise SystemExit(f"no results at {path}")
    with open(path) as fh:
        return json.load(fh)
