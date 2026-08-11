"""Bare-minimum harness: evaluate BLIP-2 OPT-2.7B on the CLEVR 1.0 val split.

Each run writes into its own output folder (default eval_runs/<timestamp>):
    results.csv   one row per question: image index, question, ground truth,
                  raw and processed prediction, complexity, question type id
    accuracy.txt  soft-match accuracy per question type and overall
    config.json   the run configuration

Usage:
    python clevr_blip2_eval.py [--batch-size 64] [--max-samples N]
        [--qformer-checkpoint ckpt.pt] [--out-dir DIR]
"""

import argparse
import csv
import json
import os
import random
import time

import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from clevr_blip2_qcond import build_qcond_model, qcond_generate
from clevr_common import (
    PROMPT,
    QUESTION_TYPE_ID,
    QUESTION_TYPES,
    accuracy_report,
    extract_answer,
    normalize,
)

CLEVR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "CLEVR_v1.0")
MODEL_ID = "Salesforce/blip2-opt-2.7b"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", default=None,
                        help="output folder (default: eval_runs/<timestamp>)")
    parser.add_argument("--qformer-checkpoint", default=None,
                        help="trainable-weights checkpoint from clevr_blip2_finetune.py")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="evaluate on a fixed-seed random subset of the val set")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join("eval_runs", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"model": MODEL_ID, "prompt": PROMPT, "max_new_tokens": 10, **vars(args)}, f, indent=2)
    print(f"writing to {out_dir}/")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = Blip2Processor.from_pretrained(MODEL_ID)
    # OPT is decoder-only: left-pad so every sequence in a batch ends at the
    # generation position.
    processor.tokenizer.padding_side = "left"
    # This checkpoint's generation config stops only at "\n" (id 50118), but
    # finetuning (clevr_blip2_finetune.py) supervises the tokenizer's </s>;
    # accept either as end-of-answer.
    stop_ids = [processor.tokenizer.eos_token_id] + processor.tokenizer(
        "\n", add_special_tokens=False
    ).input_ids
    qformer_tokenizer = None
    if args.qformer_checkpoint:
        state = torch.load(args.qformer_checkpoint, weights_only=True)
        if any(k.startswith("qformer_text_embeddings.") for k in state):
            # Question-conditioned Q-Former checkpoint (clevr_blip2_qcond.py).
            model, qformer_tokenizer = build_qcond_model(dtype, state=state)
        else:
            model = Blip2ForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=dtype)
            missing, unexpected = model.load_state_dict(state, strict=False)
            assert not unexpected, f"unexpected keys in checkpoint: {unexpected[:5]}"
        print(f"loaded {len(state)} finetuned tensors from {args.qformer_checkpoint}"
              f" (question-conditioned: {qformer_tokenizer is not None})")
    else:
        model = Blip2ForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=dtype)
    model.to(device)
    model.eval()

    with open(os.path.join(CLEVR_ROOT, "questions", "CLEVR_val_questions.json")) as f:
        questions = json.load(f)["questions"]
    if args.max_samples is not None:
        # Fixed seed so repeated runs (e.g. zero-shot vs finetuned) score the
        # exact same subset.
        questions = random.Random(0).sample(questions, args.max_samples)

    totals = [0] * len(QUESTION_TYPES)
    corrects = [0] * len(QUESTION_TYPES)
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                "image_index",
                "question",
                "ground_truth",
                "raw_prediction",
                "processed_prediction",
                "question_complexity",
                "question_type_id",
            ]
        )

        for start in range(0, len(questions), args.batch_size):
            batch = questions[start : start + args.batch_size]
            image_paths = [
                os.path.join(CLEVR_ROOT, "images", "val", q["image_filename"]) for q in batch
            ]
            images = [Image.open(p).convert("RGB") for p in image_paths]
            prompts = [PROMPT.format(q["question"]) for q in batch]

            inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt").to(
                device, dtype
            )
            with torch.no_grad():
                if qformer_tokenizer is not None:
                    q_enc = qformer_tokenizer(
                        [q["question"] for q in batch],
                        padding=True, truncation=True, max_length=64, return_tensors="pt",
                    ).to(device)
                    generated = qcond_generate(
                        model, **inputs,
                        qformer_input_ids=q_enc["input_ids"],
                        qformer_attention_mask=q_enc["attention_mask"],
                        max_new_tokens=10, eos_token_id=stop_ids,
                    )
                else:
                    generated = model.generate(**inputs, max_new_tokens=10, eos_token_id=stop_ids)
            decoded = processor.batch_decode(generated, skip_special_tokens=True)
            # Some transformers versions echo the prompt in the output; keep
            # only the generated continuation.
            raw_preds = [
                d[len(p):] if d.startswith(p) else d for d, p in zip(decoded, prompts)
            ]
            # The model often terminates its answer with a newline token; keep
            # raw predictions single-line so each CSV record stays on one line.
            raw_preds = [" ".join(r.split()) for r in raw_preds]

            for q, raw in zip(batch, raw_preds):
                q_type = q["program"][-1]["function"]
                processed = extract_answer(raw, q_type)
                type_id = QUESTION_TYPE_ID[q_type]
                totals[type_id] += 1
                corrects[type_id] += processed == normalize(q["answer"])
                writer.writerow(
                    [
                        q["image_index"],
                        q["question"],
                        q["answer"],
                        raw,
                        processed,
                        len(q["program"]),
                        type_id,
                    ]
                )
            f.flush()
            print(f"{min(start + args.batch_size, len(questions))}/{len(questions)}", flush=True)

    report = accuracy_report(corrects, totals)
    with open(os.path.join(out_dir, "accuracy.txt"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
