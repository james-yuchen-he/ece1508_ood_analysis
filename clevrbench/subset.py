"""Carving a small, self-contained, reproducible slice of CLEVR val.

The full val split is 149,991 questions over 15,000 images; a pass over it is
hours of GPU time for a number you can estimate to within a couple of points
from a fraction of it. A subset here is a directory:

    val_1000/
      questions.json    CLEVR-format questions, programs and answers intact
      images/           only the PNGs those questions reference
      subset.json       manifest: seed, strata, counts, provenance

which is portable (no 19 GB dataset needed to run it), reproducible (seed and
counts recorded), and re-usable across models, so two models are compared on
identical questions rather than on two different random draws.

Sampling is stratified by question type by default. A uniform random draw of
1,000 questions leaves `compare_number` with ~90 examples and a +-10 point
interval; stratifying fixes the per-slice counts instead of letting them land
where they may. `--balanced` goes further and equalizes them, which is what
you want when comparing *across* categories rather than reporting an overall
number comparable to published CLEVR results.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

from . import __version__, taxonomy
from .data import ClevrData

STRATA_CHOICES = ("type", "type+depth", "none")


def sample_checksum(questions):
    """Content hash identifying exactly which questions a subset contains.

    The subset directory is too big to commit, so what gets shared is the
    recipe (size, seed, strata) plus this digest. Regenerating with the same
    recipe and getting the same digest is what makes a result reproducible;
    the digest covers the questions and their order, which is all that affects
    a score. It deliberately ignores the manifest's timestamp and provenance
    fields, which differ per machine and change nothing.
    """
    payload = ",".join(str(q["question_index"]) for q in questions)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stratum_key(question, strata):
    """The label whose population this question counts toward."""
    if strata == "none":
        return "all"
    features = taxonomy.features(question)
    if strata == "type":
        return features["type"]
    if strata == "type+depth":
        return f"{features['type']}|{features['depth_bucket']}"
    raise ValueError(f"unknown strata {strata!r}; expected one of {STRATA_CHOICES}")


def _allocate(available, size, balanced):
    """Split `size` draws across strata, respecting how many each one has.

    Proportional allocation preserves the split's natural mix so the overall
    number stays comparable to a full-split evaluation; balanced allocation
    equalizes statistical power per stratum. Either way, strata that run out
    hand their remainder back to the others rather than shrinking the subset.
    """
    alloc = {k: 0 for k in available}
    remaining = min(size, sum(available.values()))
    # Sorted lists, never sets: set iteration order depends on PYTHONHASHSEED,
    # which would make a --balanced subset differ between machines even at the
    # same --seed. Ties are broken by stratum name for the same reason.
    active = sorted(k for k in available if available[k] > 0)

    while remaining > 0 and active:
        weights = {k: (1 if balanced else available[k]) for k in active}
        total = sum(weights.values())
        shares = {k: remaining * weights[k] / total for k in active}
        order = sorted(active, key=lambda k: (-shares[k], k))
        added = 0
        for key in order:
            take = min(int(shares[key]), available[key] - alloc[key], remaining - added)
            if take > 0:
                alloc[key] += take
                added += take
        if added == 0:  # every share rounded below 1 -- hand out singles
            for key in order:
                if added >= remaining or available[key] - alloc[key] <= 0:
                    continue
                alloc[key] += 1
                added += 1
        remaining -= added
        active = [k for k in active if available[k] - alloc[k] > 0]
    return alloc


def sample_questions(questions, size, seed=0, strata="type", balanced=False,
                     questions_per_image=None):
    """Draw a stratified sample; deterministic given (questions, size, seed)."""
    rng = Random(seed)

    if questions_per_image:
        # Fewer distinct images means far less to download and decode. It also
        # correlates the sample (same scene, several questions), so it is off
        # by default and worth stating when used.
        by_image = defaultdict(list)
        for q in questions:
            by_image[q["image_filename"]].append(q)
        kept = []
        for filename in sorted(by_image):
            group = by_image[filename][:]
            rng.shuffle(group)
            kept.extend(group[:questions_per_image])
        questions = kept

    pools = defaultdict(list)
    for q in questions:
        pools[stratum_key(q, strata)].append(q)

    available = {k: len(v) for k, v in pools.items()}
    allocation = _allocate(available, size, balanced)

    sampled = []
    for key in sorted(pools):
        take = allocation.get(key, 0)
        if take:
            sampled.extend(rng.sample(pools[key], take))

    # Image order, so the runner touches each PNG once.
    sampled.sort(key=lambda q: (q["image_filename"], q["question_index"]))
    return sampled


def _place_image(src, dest, link):
    if dest.exists():
        return
    if link:
        try:
            os.link(src, dest)
            return
        except OSError:
            pass  # different filesystem; fall back to a copy
    shutil.copyfile(src, dest)


def build_subset(out_dir, size=1000, split="val", seed=0, strata="type",
                 balanced=False, questions_per_image=None, data=None,
                 link=False, quiet=False):
    """Build a subset directory and return its path.

    Everything needed is fetched on demand: the split's question file, then
    only the images the sample references.
    """
    data = data or ClevrData(quiet=quiet)
    out_dir = Path(out_dir)

    questions = data.load_questions(split)
    total_available = len(questions)
    sampled = sample_questions(
        questions, size, seed=seed, strata=strata, balanced=balanced,
        questions_per_image=questions_per_image,
    )
    if not sampled:
        raise SystemExit("sampling produced no questions")

    filenames = sorted({q["image_filename"] for q in sampled})
    image_dir = data.ensure_images(split, filenames)

    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        _place_image(image_dir / filename, out_images / filename, link)

    with open(out_dir / "questions.json", "w") as fh:
        json.dump({"info": {"split": split, "source": "CLEVR_v1.0"},
                   "questions": sampled}, fh)

    features = [taxonomy.features(q) for q in sampled]
    manifest = {
        "clevrbench_version": __version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        # Identifies the sample itself; two machines running the same recipe
        # must agree on this or their numbers are not comparable.
        "checksum": sample_checksum(sampled),
        "split": split,
        "n_questions": len(sampled),
        "n_images": len(filenames),
        "n_available": total_available,
        "seed": seed,
        "strata": strata,
        "balanced": balanced,
        "questions_per_image": questions_per_image,
        "source": data.source_used or f"local dataset {data.root}",
        "counts": {
            "type": dict(Counter(f["type"] for f in features).most_common()),
            "depth_bucket": dict(sorted(Counter(f["depth_bucket"] for f in features).items())),
            "hops_bucket": dict(sorted(Counter(f["hops_bucket"] for f in features).items())),
            "answer": dict(Counter(q["answer"] for q in sampled).most_common()),
        },
    }
    with open(out_dir / "subset.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    if not quiet:
        print(f"subset: {len(sampled)} questions over {len(filenames)} images -> {out_dir}")
        for qtype, n in manifest["counts"]["type"].items():
            print(f"  {qtype:<20}{n:>6}")
        print(f"  checksum {manifest['checksum']}")
    return out_dir


def load_subset(path):
    """Read a subset directory into (questions, images_dir, manifest)."""
    path = Path(path)
    questions_file = path / "questions.json"
    if not questions_file.exists():
        raise SystemExit(f"{path} is not a subset directory (no questions.json)")
    with open(questions_file) as fh:
        questions = json.load(fh)["questions"]
    manifest_file = path / "subset.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
    return questions, path / "images", manifest


def verify_subset(path, expected=None):
    """Check a subset is intact and, optionally, that it matches a published digest.

    `expected` is the point of the exercise: a regenerated subset always agrees
    with its own manifest, so self-consistency proves nothing. Comparing against
    the digest the original author published is what confirms two people are
    scoring the same questions.
    """
    questions, images_dir, manifest = load_subset(path)
    actual = sample_checksum(questions)
    referenced = {q["image_filename"] for q in questions}
    missing = sorted(f for f in referenced if not (images_dir / f).exists())

    result = {
        "path": str(path),
        "n_questions": len(questions),
        "n_images": len(referenced),
        "checksum": actual,
        "manifest_checksum": manifest.get("checksum"),
        "missing_images": missing,
        "recipe": {k: manifest.get(k) for k in
                   ("split", "n_questions", "seed", "strata", "balanced",
                    "questions_per_image")},
    }
    result["manifest_match"] = (
        None if not manifest.get("checksum") else manifest["checksum"] == actual
    )
    result["expected_match"] = None if expected is None else expected == actual
    result["ok"] = (
        not missing
        and result["manifest_match"] is not False
        and result["expected_match"] is not False
    )
    return result


def default_subset_dir(root, split, size, seed, strata, balanced):
    """Stable directory name, so identical settings reuse an existing subset."""
    name = f"{split}_{size}_seed{seed}_{strata.replace('+', '-')}"
    if balanced:
        name += "_balanced"
    return Path(root) / name
