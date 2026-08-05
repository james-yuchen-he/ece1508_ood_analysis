"""Command line entry point.

    clevrbench models                       what can be evaluated
    clevrbench status                       what data is on disk
    clevrbench prepare                      build an eval subset
    clevrbench eval --model blip2-opt-2.7b  build it if needed, then score
    clevrbench report runs/<name>           reprint a finished run

`eval` is the out-of-the-box path: with no data present it fetches the CLEVR
questions file and the images its sample needs, builds the subset, and runs.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from . import __version__, models, report, runner, subset as subset_mod
from .answers import PARSE_MODES
from .data import ClevrData
from .metrics import aggregate


def _add_data_args(parser):
    parser.add_argument("--data-dir", default="data",
                        help="where CLEVR_v1.0/ lives or should be written")
    parser.add_argument("--no-download", action="store_true",
                        help="fail instead of fetching anything missing")
    parser.add_argument("--full-download", action="store_true",
                        help="download the whole 19 GB archive up front")


def _add_subset_args(parser):
    parser.add_argument("--split", default="val", choices=["val", "train"],
                        help="CLEVR test answers are withheld, so val by default")
    parser.add_argument("--size", type=int, default=1000,
                        help="questions in the subset (default: 1000)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strata", default="type", choices=subset_mod.STRATA_CHOICES,
                        help="sampling strata (default: type)")
    parser.add_argument("--balanced", action="store_true",
                        help="equal questions per stratum instead of proportional")
    parser.add_argument("--questions-per-image", type=int, default=None,
                        help="cap questions per image; fewer images to fetch/decode")
    parser.add_argument("--subset-root", default="data/subsets",
                        help="where built subsets are stored")
    parser.add_argument("--link", action="store_true",
                        help="hardlink subset images instead of copying them")


def _data_from(args):
    return ClevrData(
        data_dir=args.data_dir,
        allow_download=not args.no_download,
        force_full_download=args.full_download,
    )


def _parse_model_arg(text):
    if "=" not in text:
        raise SystemExit(f"--model-arg expects key=value, got {text!r}")
    key, _, raw = text.partition("=")
    lowered = raw.lower()
    if lowered in ("true", "false"):
        value = lowered == "true"
    else:
        try:
            value = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                value = raw
    return key.strip(), value


def _slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def _resolve_subset(args, data):
    """Return (path, questions, images_dir, manifest), building if needed."""
    if getattr(args, "subset", None):
        path = Path(args.subset)
        if not path.exists():
            raise SystemExit(f"no subset at {path}")
    else:
        path = subset_mod.default_subset_dir(
            args.subset_root, args.split, args.size, args.seed, args.strata, args.balanced
        )
        if not (path / "questions.json").exists():
            subset_mod.build_subset(
                path, size=args.size, split=args.split, seed=args.seed,
                strata=args.strata, balanced=args.balanced,
                questions_per_image=args.questions_per_image, data=data, link=args.link,
            )
        else:
            print(f"using existing subset {path}")
    questions, images_dir, manifest = subset_mod.load_subset(path)
    return path, questions, images_dir, manifest


# -- subcommands ----------------------------------------------------------
def cmd_models(args):
    print("models (use --model <name>, or module:Class / file.py:Class for your own)\n")
    for name, description in models.available():
        print(f"  {name:<26}{description}")
    return 0


def cmd_status(args):
    data = _data_from(args)
    info = data.status()
    print(f"data dir       {info['data_dir']}")
    print(f"dataset root   {info['root']}")
    print(f"archive zip    {'present' if info['archive_present'] else 'absent'}")
    for split, state in info["splits"].items():
        print(f"  {split:<6} questions {'yes' if state['questions'] else 'no ':<4} "
              f"images {state['images_present']}")
    root = Path(args.subset_root) if hasattr(args, "subset_root") else Path("data/subsets")
    if root.is_dir():
        built = sorted(p.name for p in root.iterdir() if (p / "questions.json").exists())
        print(f"subsets        {', '.join(built) if built else 'none'}")
    return 0


def cmd_prepare(args):
    data = _data_from(args)
    path = subset_mod.default_subset_dir(
        args.subset_root, args.split, args.size, args.seed, args.strata, args.balanced
    )
    if args.out:
        path = Path(args.out)
    subset_mod.build_subset(
        path, size=args.size, split=args.split, seed=args.seed, strata=args.strata,
        balanced=args.balanced, questions_per_image=args.questions_per_image,
        data=data, link=args.link,
    )
    return 0


def cmd_eval(args):
    data = _data_from(args)
    subset_path, questions, images_dir, manifest = _resolve_subset(args, data)

    if args.limit:
        questions = questions[: args.limit]

    model_kwargs = dict(_parse_model_arg(a) for a in args.model_arg or [])
    model = models.build(args.model, **model_kwargs)

    out_dir = Path(args.out) if args.out else Path("runs") / (
        f"{_slug(model.name)}_{_slug(subset_path.name)}"
    )

    print(f"loading {args.model}")
    with model:
        records, stats = runner.evaluate(
            model, questions, images_dir,
            batch_size=args.batch_size, parse_mode=args.parse,
            progress=not args.no_progress,
        )

    subset_info = {
        "path": str(subset_path),
        "split": manifest.get("split", args.split),
        "n_questions": len(questions),
        # Recount rather than trusting the manifest: --limit takes a prefix.
        "n_images": len({q["image_filename"] for q in questions}),
        # Identifies the questions actually scored, so a collaborator can
        # confirm two runs are comparable without shipping the data.
        "checksum": subset_mod.sample_checksum(questions),
        "subset_checksum": manifest.get("checksum"),
        "seed": manifest.get("seed"),
        "strata": manifest.get("strata"),
        "balanced": manifest.get("balanced"),
        "source": manifest.get("source"),
    }
    config = {
        "model_spec": args.model,
        "model_args": model_kwargs,
        "parse_mode": args.parse,
        "batch_size": args.batch_size,
        "limit": args.limit,
    }
    results = report.build_results(
        model, records, aggregate(records), stats, subset_info, config
    )
    report.write_run(out_dir, results, records)

    text = report.format_report(results)
    print(text)
    (out_dir / "report.txt").write_text(text)
    print(f"wrote {out_dir}/results.json, predictions.jsonl, report.txt")
    return 0


def cmd_verify(args):
    result = subset_mod.verify_subset(args.subset, expected=args.expect)
    recipe = result["recipe"]

    print(f"subset     {result['path']}")
    print(f"recipe     --split {recipe.get('split')} --size {recipe.get('n_questions')} "
          f"--seed {recipe.get('seed')} --strata {recipe.get('strata')}"
          f"{' --balanced' if recipe.get('balanced') else ''}"
          + (f" --questions-per-image {recipe['questions_per_image']}"
             if recipe.get("questions_per_image") else ""))
    print(f"contents   {result['n_questions']} questions over {result['n_images']} images")
    print(f"checksum   {result['checksum']}")

    if result["manifest_match"] is False:
        print("  MISMATCH against this subset's own manifest -- it has been edited "
              f"since it was built (manifest says {result['manifest_checksum']})")
    elif result["manifest_match"] is None:
        print("  (manifest predates checksums; rebuild to record one)")
    if result["expected_match"] is True:
        print("  MATCHES the expected checksum -- identical questions, in the same order")
    elif result["expected_match"] is False:
        print(f"  MISMATCH against --expect {args.expect}")
    if result["missing_images"]:
        print(f"  {len(result['missing_images'])} referenced images are missing, "
              f"first: {result['missing_images'][0]}")

    if not result["ok"]:
        return 1
    if args.expect is None:
        print("\nintact. To confirm it matches someone else's subset, compare "
              "checksums with --expect")
    return 0


def cmd_report(args):
    print(report.format_report(report.load_run(args.run)))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="clevrbench",
        description="Evaluate vision-language models on CLEVR compositional reasoning.",
    )
    parser.add_argument("--version", action="version", version=f"clevrbench {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_models = subparsers.add_parser("models", help="list available models")
    p_models.set_defaults(func=cmd_models)

    p_status = subparsers.add_parser("status", help="show what data is present")
    _add_data_args(p_status)
    p_status.add_argument("--subset-root", default="data/subsets")
    p_status.set_defaults(func=cmd_status)

    p_prepare = subparsers.add_parser("prepare", help="build an evaluation subset")
    _add_data_args(p_prepare)
    _add_subset_args(p_prepare)
    p_prepare.add_argument("--out", default=None, help="subset directory to write")
    p_prepare.set_defaults(func=cmd_prepare)

    p_eval = subparsers.add_parser("eval", help="evaluate a model")
    _add_data_args(p_eval)
    _add_subset_args(p_eval)
    p_eval.add_argument("--model", required=True,
                        help="registry name, or module:Class / file.py:Class")
    p_eval.add_argument("--model-arg", action="append", metavar="KEY=VALUE",
                        help="adapter keyword argument; repeatable")
    p_eval.add_argument("--subset", default=None,
                        help="use this subset directory instead of building one")
    p_eval.add_argument("--batch-size", type=int, default=8)
    p_eval.add_argument("--parse", default="typed", choices=PARSE_MODES,
                        help="answer normalization strictness (default: typed)")
    p_eval.add_argument("--limit", type=int, default=0,
                        help="score only the first N subset questions (smoke tests)")
    p_eval.add_argument("--out", default=None, help="run directory (default: runs/<model>_<subset>)")
    p_eval.add_argument("--no-progress", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    p_verify = subparsers.add_parser(
        "verify", help="check a subset is intact and matches a published checksum")
    p_verify.add_argument("subset", help="subset directory")
    p_verify.add_argument("--expect", default=None, metavar="SHA256",
                          help="checksum this subset must match (from the README "
                               "or a collaborator's results.json)")
    p_verify.set_defaults(func=cmd_verify)

    p_report = subparsers.add_parser("report", help="reprint a finished run")
    p_report.add_argument("run", help="run directory or results.json")
    p_report.set_defaults(func=cmd_report)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
