#!/usr/bin/env python
"""Zero-shot evaluation of BLIP-2 on CLEVR v1.0 VQA.

CLEVR's test split ships without answers, so evaluation uses `val` by default.
Val has 149,991 questions over 15,000 images; the default is a random subsample
(--limit) since a full pass takes many hours on a single consumer GPU.

Example:
    python clevr_eval.py --limit 2000
    python clevr_eval.py --limit 0 --batch-size 16 --out runs/full_val
"""
import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2ForConditionalGeneration, Blip2Processor

# ── CLEVR answer space ────────────────────────────────────────────────────
# The 28 answers CLEVR ever uses. Anything outside this set is a model
# artifact (verbosity, hallucination) and scores as wrong.
COUNTS = {str(i) for i in range(11)}
COLORS = {"gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"}
SHAPES = {"cube", "sphere", "cylinder"}
SIZES = {"large", "small"}
MATERIALS = {"rubber", "metal"}
CLEVR_ANSWERS = {"yes", "no"} | COUNTS | COLORS | SHAPES | SIZES | MATERIALS

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "none": "0", "no": "0",  # "no" only reached via count path
}

# Free-text the model emits for concepts CLEVR names differently.
SYNONYMS = {
    "grey": "gray",
    "ball": "sphere", "balls": "sphere", "round": "sphere",
    "block": "cube", "blocks": "cube", "box": "cube", "square": "cube",
    "can": "cylinder", "cylindrical": "cylinder",
    "metallic": "metal", "shiny": "metal", "steel": "metal", "silver": "metal",
    "matte": "rubber", "rubbery": "rubber", "plastic": "rubber",
    "big": "large", "bigger": "large", "biggest": "large",
    "little": "small", "tiny": "small", "smaller": "small", "smallest": "small",
    "yep": "yes", "yeah": "yes", "true": "yes",
    "nope": "no", "false": "no",
}

# Program function -> the five standard CLEVR reporting categories.
_TYPE_BY_FUNCTION = {
    "count": "count",
    "exist": "exist",
    "equal_integer": "compare_number",
    "greater_than": "compare_number",
    "less_than": "compare_number",
    **{f"query_{a}": "query_attribute" for a in ("color", "size", "material", "shape")},
    **{f"equal_{a}": "compare_attribute" for a in ("color", "size", "material", "shape")},
}


def question_type(question):
    """Standard CLEVR category from the final node of the functional program."""
    program = question.get("program") or []
    if not program:
        return "unknown"
    tail = program[-1]
    fn = tail.get("function") or tail.get("type")  # key name varies by release
    return _TYPE_BY_FUNCTION.get(fn, "unknown")


def normalize(text, is_count_question=False):
    """Map free-form model output onto the CLEVR answer vocabulary.

    Returns "" when nothing in the output resolves to a valid answer, which is
    tracked separately as the unparsed rate — it separates "the model reasoned
    wrongly" from "the model ignored the answer format".
    """
    t = text.strip().lower().split("\n")[0]
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = " ".join(t.split())
    if not t:
        return ""

    def resolve(tok):
        tok = SYNONYMS.get(tok, tok)
        # "no" is a legitimate yes/no answer, so only read it as 0 for counts.
        if tok == "no" and not is_count_question:
            return "no"
        return NUMBER_WORDS.get(tok, tok)

    # Whole string first: terse, correctly-formatted answers land here.
    whole = resolve(t)
    if whole in CLEVR_ANSWERS:
        return whole

    # Otherwise the first token that resolves into the vocabulary.
    for tok in t.split():
        cand = resolve(tok)
        if cand in CLEVR_ANSWERS:
            return cand
    return ""


def load_questions(clevr_root, split):
    path = Path(clevr_root) / "questions" / f"CLEVR_{split}_questions.json"
    if not path.exists():
        raise SystemExit(f"Questions file not found: {path}")
    with open(path) as f:
        questions = json.load(f)["questions"]
    if not questions or "answer" not in questions[0]:
        raise SystemExit(
            f"Split '{split}' has no ground-truth answers (CLEVR withholds test "
            f"answers). Use --split val or --split train."
        )
    return questions


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clevr-root", default="data/CLEVR_v1.0")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--model-id", default="Salesforce/blip2-opt-2.7b")
    ap.add_argument("--limit", type=int, default=2000,
                    help="Random question subsample; 0 = full split.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=10)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--prompt", default="Question: {question} Answer:",
                    help="Prompt template; {question} is substituted.")
    ap.add_argument("--out", default="runs/blip2_clevr",
                    help="Output directory for results.json + predictions.jsonl")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(args.clevr_root, args.split)
    total_available = len(questions)
    if args.limit and args.limit < len(questions):
        random.Random(args.seed).shuffle(questions)
        questions = questions[: args.limit]
    # Image locality: consecutive questions share an image, so the loader
    # only touches each PNG once.
    questions.sort(key=lambda q: q["image_filename"])
    print(f"Evaluating {len(questions)} / {total_available} {args.split} questions")

    processor = Blip2Processor.from_pretrained(args.model_id)
    # Decoder-only generation needs left padding for a batched prompt.
    processor.tokenizer.padding_side = "left"
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    image_dir = Path(args.clevr_root) / "images" / args.split
    cache = {}  # single-image cache; questions are sorted by filename

    def get_image(filename):
        if filename not in cache:
            cache.clear()
            cache[filename] = Image.open(image_dir / filename).convert("RGB")
        return cache[filename]

    records = []
    n_correct = 0
    n_unparsed = 0
    by_type = defaultdict(lambda: [0, 0])  # type -> [correct, total]
    start = time.time()

    for i in tqdm(range(0, len(questions), args.batch_size), desc="eval"):
        batch = questions[i : i + args.batch_size]
        images = [get_image(q["image_filename"]) for q in batch]
        prompts = [args.prompt.format(question=q["question"]) for q in batch]

        inputs = processor(
            images=images, text=prompts, return_tensors="pt", padding=True
        ).to(model.device, torch.float16)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                do_sample=False,
            )
        decoded = processor.batch_decode(out, skip_special_tokens=True)

        for q, raw in zip(batch, decoded):
            # Some versions echo the prompt; keep only what follows "Answer:".
            text = raw.split("Answer:")[-1].strip()
            qtype = question_type(q)
            pred = normalize(text, is_count_question=(qtype == "count"))
            gold = q["answer"].strip().lower()
            correct = pred == gold

            n_correct += correct
            n_unparsed += pred == ""
            by_type[qtype][0] += correct
            by_type[qtype][1] += 1
            records.append({
                "question_index": q["question_index"],
                "image_filename": q["image_filename"],
                "question": q["question"],
                "type": qtype,
                "gold": gold,
                "raw": text,
                "pred": pred,
                "correct": bool(correct),
            })

    elapsed = time.time() - start
    n = len(records)
    results = {
        "config": vars(args),
        "n_questions": n,
        "n_available": total_available,
        "accuracy": n_correct / n if n else 0.0,
        "unparsed_rate": n_unparsed / n if n else 0.0,
        "seconds": round(elapsed, 1),
        "questions_per_second": round(n / elapsed, 2) if elapsed else 0.0,
        "accuracy_by_type": {
            t: {"accuracy": c / tot, "n": tot} for t, (c, tot) in sorted(by_type.items())
        },
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "predictions.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\n{'='*52}\n{args.model_id} — CLEVR {args.split}\n{'='*52}")
    print(f"overall accuracy   {results['accuracy']:.4f}  ({n_correct}/{n})")
    print(f"unparsed outputs   {results['unparsed_rate']:.4f}  "
          f"(answer not in CLEVR vocabulary)")
    print(f"throughput         {results['questions_per_second']:.2f} q/s "
          f"({elapsed/60:.1f} min)\n")
    print(f"{'question type':<20}{'acc':>8}{'n':>8}")
    print("-" * 36)
    for t, v in results["accuracy_by_type"].items():
        print(f"{t:<20}{v['accuracy']:>8.4f}{v['n']:>8}")
    print(f"\nwrote {out_dir}/results.json and predictions.jsonl")


if __name__ == "__main__":
    main()
