"""Evaluate Qwen2.5-VL-7B-Instruct on the CLEVR 1.0 val split.

Same protocol and outputs as clevr_blip2_eval.py: each run writes into its
own folder (default eval_runs/<timestamp>) containing results.csv (same
columns), accuracy.txt (per-question-type soft-match table), and config.json.
--max-samples uses the same fixed seed as the BLIP-2 harness, so subset runs
score the exact same questions and are directly comparable.

The model is loaded 4-bit quantized (NF4, bf16 compute) so 7B fits a 12GB
GPU; the first run downloads ~16GB of weights.

Usage:
    python clevr_qwen_eval.py [--batch-size 8] [--max-samples N] [--out-dir DIR]
"""

import argparse
import csv
import json
import os
import random
import time

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from clevr_common import (
    QUESTION_TYPE_ID,
    QUESTION_TYPES,
    accuracy_report,
    extract_answer,
    normalize,
)

CLEVR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "CLEVR_v1.0")
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
INSTRUCTION = "{} Answer with a single word or number."


def chat_text(processor, question):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": INSTRUCTION.format(question)},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out-dir", default=None,
                        help="output folder (default: eval_runs/<timestamp>)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="evaluate on a fixed-seed random subset of the val set")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join("eval_runs", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(
            {"model": MODEL_ID, "instruction": INSTRUCTION, "max_new_tokens": 10,
             "quantization": "4bit-nf4", **vars(args)},
            f, indent=2,
        )
    print(f"writing to {out_dir}/")

    assert torch.cuda.is_available(), "evaluation requires a CUDA GPU"

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map="auto",
    )
    model.eval()

    with open(os.path.join(CLEVR_ROOT, "questions", "CLEVR_val_questions.json")) as f:
        questions = json.load(f)["questions"]
    if args.max_samples is not None:
        # Same seed as clevr_blip2_eval.py: identical subset across models.
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
            images = [
                Image.open(
                    os.path.join(CLEVR_ROOT, "images", "val", q["image_filename"])
                ).convert("RGB")
                for q in batch
            ]
            texts = [chat_text(processor, q["question"]) for q in batch]

            inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(
                model.device
            )
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            new_tokens = generated[:, inputs["input_ids"].shape[1]:]
            raw_preds = processor.batch_decode(new_tokens, skip_special_tokens=True)
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
