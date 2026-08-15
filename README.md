# BLIP-2 on CLEVR — zero-shot evaluation and Q-Former finetuning

Evaluates BLIP-2 (OPT-2.7B) on the CLEVR 1.0 validation split, finetunes
only its Q-Former bridge on CLEVR train (following the BLIP-2 paper's VQA
recipe), and compares against Qwen2.5-VL-7B-Instruct as a modern baseline.

## Setup

- Python env with: `torch`, `transformers`, `pillow`, `accelerate`,
  `bitsandbytes` (Qwen 4-bit only), `numpy`, `matplotlib`, `seaborn`
  (analysis only)
- GPU: 12GB+ VRAM; finetuning requires bf16 support (Ampere or newer)
- Dataset: [CLEVR v1.0](https://cs.stanford.edu/people/jcjohns/clevr/)
  extracted to `data/CLEVR_v1.0/` (with `images/` and `questions/` inside)
- Model weights download from Hugging Face on first use
  (~15GB BLIP-2, ~16GB Qwen)

## Files

| file | purpose |
|---|---|
| `clevr_common.py` | shared: question-type taxonomy, answer vocabulary, soft-match normalization, prompt, accuracy report |
| `clevr_blip2_eval.py` | evaluate BLIP-2 (zero-shot or a finetuned checkpoint) on CLEVR val |
| `clevr_blip2_finetune.py` | finetune the Q-Former bridge on CLEVR train |
| `clevr_qwen_eval.py` | evaluate Qwen2.5-VL-7B-Instruct (4-bit) on CLEVR val |
| `cosine_analysis.py` | redundancy analysis of the 32 Q-Former query vectors (needs `*_qformer.npy` tensors from `inspect_qformer.py --save-tensors` and a `config.py` defining `RESULTS_DIR` — not part of this repo yet) |

## Commands

### Evaluate BLIP-2 zero-shot

```bash
# full val split (149,991 questions; several hours)
python clevr_blip2_eval.py

# fixed-seed random subset (fast; same subset every run and across models)
python clevr_blip2_eval.py --max-samples 2000 --out-dir eval_runs/zeroshot_2k
```

### Finetune the Q-Former

```bash
# full train split (~700k questions, ~5.5k optimizer steps; about a day)
python clevr_blip2_finetune.py

# quicker: subset and/or more epochs
python clevr_blip2_finetune.py --max-samples 100000 --out qformer_100k.pt
python clevr_blip2_finetune.py --max-samples 2000 --epochs 20
```

Flags (defaults): `--epochs 1` · `--batch-size 8` (micro-batch; fits 12GB)
· `--grad-accum 16` (effective batch 8×16=128, as in the paper) · `--lr 1e-5`
· `--warmup-steps 1000` · `--max-samples N` (random fixed-seed subset)
· `--out qformer_clevr_fine_tuned.pt` · `--save-every 500` (optimizer steps
between checkpoints; also saves at the end).

Only the Q-Former, its 32 query tokens, and the LLM projection train
(~107M params); the ViT and OPT stay frozen. Checkpoints hold just those
tensors (~410MB).

### Evaluate a finetuned checkpoint

```bash
python clevr_blip2_eval.py --qformer-checkpoint qformer_clevr_fine_tuned.pt \
    --max-samples 2000 --out-dir eval_runs/finetuned_2k
```

### Evaluate Qwen2.5-VL-7B

```bash
python clevr_qwen_eval.py --max-samples 2000 --out-dir eval_runs/qwen_2k
```

## Outputs

Each evaluation run writes a self-contained folder (default
`eval_runs/<timestamp>/`):

- `results.csv` — one row per question:
  `image_index` (N in `CLEVR_val_00000N.png`), `question`, `ground_truth`,
  `raw_prediction` (whitespace-collapsed model output),
  `processed_prediction` (soft-match extraction), `question_complexity`
  (functional-program length), `question_type_id` (index into
  `QUESTION_TYPES` in `clevr_common.py`: 0=count, 1=exist, 2–5=query_*,
  6–9=equal_<attr>, 10–12=integer comparisons)
- `accuracy.txt` — soft-match accuracy per question type and overall
- `config.json` — the run's exact configuration (model, prompt, checkpoint,
  subset size)

Soft matching: normalize (lowercase, strip punctuation), canonicalize
synonyms and number words into CLEVR's 28-answer vocabulary
("none"→"0", "big"→"large", "shiny"→"metal", "ball"→"sphere", ...), then
take the first word in the vocabulary expected for the question type.

## Notes

- `--max-samples` always samples with seed 0, so subset runs of any model
  or checkpoint score the exact same questions and are directly comparable
  (95% CI at 2000 samples ≈ ±2.2pp).
- Generation stops at either `</s>` (what finetuning supervises) or `\n`
  (the stock checkpoint's terminator); both are passed as stop tokens at
  eval. Max 10 new tokens.
- Deviations from the BLIP-2 paper's VQA recipe (frozen ViT, 224px images,
  no question-conditioning of the Q-Former, "Short answer:" prompt) are
  documented in the `clevr_blip2_finetune.py` docstring.
