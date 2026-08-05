# clevrbench

A test harness for measuring **compositional reasoning** in vision-language
models on [CLEVR v1.0](https://cs.stanford.edu/people/jcjohns/clevr/), built so
that swapping in a new model is one class with one method, and so that a first
run needs no setup beyond installing the dependencies.

BLIP-2 is the reference model; no code outside `clevrbench/models/` imports it
or special-cases it — the rest of the harness only ever sees an adapter.

## Quickstart

```bash
conda activate dl                      # torch 2.12+cu132, transformers 5.10.2

python -m clevrbench eval --model blip2-opt-2.7b
```

That single command downloads whatever CLEVR data it needs, builds a
1,000-question stratified subset of `val`, runs the model, and writes
`runs/<model>_<subset>/` containing `results.json`, `predictions.jsonl`, and
`report.txt`.

```bash
python -m clevrbench models                     # what can be evaluated
python -m clevrbench status                     # what data is on disk
python -m clevrbench prepare --size 2000        # build a subset, don't run
python -m clevrbench eval --model constant --model-arg answer=no   # baseline
python -m clevrbench verify data/subsets/val_1000_seed0_type   # reproducibility check
python -m clevrbench report runs/blip2_val1k    # reprint a finished run
python tests/test_harness.py                    # 47 tests, no GPU or data needed
```

## Adding a model

The whole contract is one method. `answer_batch` receives a list of PIL images
and a list of question strings, and returns the model's raw output strings, in
order. Normalization, scoring, and slicing are the harness's job — an adapter
that cleaned up its own answers would be scoring itself under different rules
than the models it is compared against.

```python
# my_vlm.py
from clevrbench.models.base import VLMAdapter

class MyVLM(VLMAdapter):
    name = "my-vlm"

    def load(self):                       # called once, before the first batch
        self.model = load_my_model()

    def answer_batch(self, images, questions):
        return [self.model.ask(img, q) for img, q in zip(images, questions)]
```

```bash
python -m clevrbench eval --model my_vlm.py:MyVLM
```

Three ways in, in increasing order of effort:

| you have | how |
| --- | --- |
| any HF image-text-to-text checkpoint | `--model hf-auto --model-arg model_id=org/repo` |
| your own adapter class | `--model path/to/file.py:MyVLM` or `--model pkg.module:MyVLM` |
| something worth a short name | add a line to `_BUILTIN` in `clevrbench/models/__init__.py` |

`--model-arg key=value` is passed through to the adapter's constructor and is
repeatable, so `--model-arg prompt="Q: {question} A:" --model-arg num_beams=5`
works without touching code. For HuggingFace models, `HFImageTextToText` already
handles batching, left padding for decoder-only LMs, greedy decoding, and
stripping an echoed prompt — `Blip2Adapter` is six lines on top of it.

## Data: nothing to set up

CLEVR ships as a single 19 GB zip, which is a poor fit for evaluating 1,000
questions. Members are resolved through a chain, cheapest first:

1. already extracted under `data/CLEVR_v1.0/` — free
2. `data/CLEVR_v1.0.zip` on disk — local read
3. **the published archive over HTTP range requests** — fetch only the members needed
4. full download, then (2) — only if the server refuses ranges

Step 3 is what makes a cold start cheap. `zipfile` is handed a seekable
file-like object that reads over HTTP (`clevrbench/remote_zip.py`), so stdlib
handles ZIP64 and decompression while only the touched byte ranges cross the
network. Measured against the live archive:

| | cost |
| --- | --- |
| parse the central directory (100,015 entries) | 13.45 MB, 14 requests — once per fetch session |
| fetch one image (185 KB mean) | 1 request, 1.01× overhead |
| val questions file (152 MB on disk) | ~8 MB, stored deflated |
| **cold start for a 1,000-question subset** | **≈ 220 MB** instead of 19 GB |

(184 MB of images for 971 distinct scenes, ~8 MB of questions, and two central
directory reads — one to fetch the questions, one to fetch the images the
sample turns out to need.)

Anything fetched lands in `data/CLEVR_v1.0/` in the dataset's own layout, so it
is indistinguishable from a partial manual extraction and is reused next time.
`--no-download` refuses to fetch anything; `--full-download` forces the archive.

## Subsets

A subset is a self-contained directory, so an evaluation is portable and
repeatable without the full dataset present:

```
data/subsets/val_1000_seed0_type/
  questions.json    CLEVR-format questions, programs and answers intact
  images/           only the PNGs those questions reference
  subset.json       manifest: seed, strata, counts, provenance
```

Sampling is **stratified by question type** by default, so per-category counts
are fixed by construction rather than left to the draw: `compare_number` is 9%
of val, and a uniform sample of 1,000 hands it 90 ± 9 questions where
stratifying pins it at exactly 90 every time. Proportional allocation preserves
the split's natural mix, which keeps the overall number comparable to a
full-split evaluation; `--balanced` equalizes strata instead, which is what you
want when comparing *across* categories — at n ≈ 90 a category's interval is
already ±10 points. `--questions-per-image` caps questions per image to cut
download and decode cost, at the price of correlating the sample.

Reusing one subset across models is the point: two models get identical
questions rather than two different draws.

### Reproducing a subset exactly

`data/` is gitignored — neither CLEVR nor the built subsets are shared. What is
shared is the **recipe** and a **checksum**, which is enough to regenerate the
identical sample anywhere:

```bash
python -m clevrbench prepare --size 1000 --seed 0 --strata type

python -m clevrbench verify data/subsets/val_1000_seed0_type \
  --expect sha256:7e45f11a182ee34b560592225b4ecedf4e35eb223a01cd49ecd775156ece8a38
```

`verify` exits non-zero on a mismatch, so it drops straight into CI. The digest
covers the sampled questions and their order — the only things that affect a
score — and ignores per-machine fields like the manifest timestamp. Every run's
`results.json` records the checksum of the questions it actually scored, so two
collaborators can confirm they are comparing like with like without shipping any
data.

| subset | recipe | checksum |
| --- | --- | --- |
| default | `--size 1000 --seed 0 --strata type` | `sha256:7e45f11a182ee34b5605…8a38` |

Sampling is deterministic by construction: a seeded RNG, sorted iteration
everywhere, and no iteration over sets — set order depends on `PYTHONHASHSEED`,
which varies per process and silently changed `--balanced` subsets between
machines until it was fixed. Verified identical across four hash seeds and
across Python 3.8 and 3.12; `tests/test_harness.py` pins both.

Two caveats worth stating plainly. The checksum identifies the *data*, not the
*result*: floating-point non-determinism, GPU model, dtype, and
transformers/torch versions can still move a score by a fraction of a point on
identical questions. And it assumes CLEVR v1.0 proper — question indices come
from the official archive, so a mirror with reordered questions would not match.

## What gets measured

The five standard CLEVR categories, plus the axes that actually vary
compositional load — read off each question's functional program:

- **program depth** — how many operations compose (2–25 nodes)
- **chained spatial relations** — `relate` hops, where generalization tends to break first
- **same-attribute matching** — requires comparing against every other object

Each slice reports accuracy, a **95% Wilson interval**, and a **majority
baseline** computed from that slice's own gold answers. The baselines are not
decoration. CLEVR's yes/no categories are ~50% answerable by guessing and its
open-vocabulary categories are not, so an aggregate score mixes together things
that are not comparable. The headline block also reports the **per-type
baseline**: the best fixed answer per question type, which a model that merely
recognizes question types can reach without looking at an image. That is the
floor a compositional claim has to clear.

`unparsed_rate` tracks output that never lands in CLEVR's 28-answer vocabulary,
separating *reasoned incorrectly* from *ignored the answer format*. The
prediction histogram catches the opposite failure — a model collapsing onto one
answer and riding the class prior.

### Answer normalization

Generative models emit sentences; CLEVR accepts one of 28 words. The normalizer
is therefore part of the measurement, and `--parse` makes the choice explicit:

- `strict` — only a terse, already-valid answer counts
- `typed` (default) — prefer answers of the kind the question's program admits
- `lenient` — first vocabulary word anywhere in the output

`typed` exists because a `query_material` question can only be answered
rubber/metal. On *"the ball is red"* for a colour question, `lenient` returns
`sphere`; `typed` returns `red`. Raw strings are kept in `predictions.jsonl`, so
any scoring decision can be revisited without re-running the model.

The choice is not doing the work: on the subset below, BLIP-2 scores 0.283
under `typed` and 0.277 under `lenient`, well inside each other's intervals.
What the mode changes is the *diagnosis* — `typed` attributes 18.2% of answers
to off-type words where `lenient` hides them at 44.3%.

## Baseline results

`blip2-opt-2.7b`, zero-shot, greedy, 1,000-question stratified val subset
(seed 0), ~35 q/s on an RTX 3080 Ti (`runs/blip2_val1k`):

| | accuracy | 95% CI |
| --- | ---: | :---: |
| **BLIP-2 (blip2-opt-2.7b)** | **0.283** | [0.256, 0.312] |
| per-type baseline (no image) | 0.342 | — |
| always "no" | 0.287 | [0.260, 0.316] |
| uniform random over 28 answers | 0.045 | [0.034, 0.060] |

| question type | accuracy | majority | delta | n |
| --- | ---: | ---: | ---: | ---: |
| exist | 0.500 | 0.522 | −0.022 | 134 |
| compare_attribute | 0.439 | 0.517 | −0.078 | 180 |
| compare_number | 0.422 | 0.567 | −0.144 | 90 |
| count | 0.173 | 0.333 | −0.160 | 237 |
| query_attribute | 0.162 | 0.136 | +0.025 | 359 |

**BLIP-2 does not clear the no-vision floor.** It scores 0.283 against a
per-type baseline of 0.342, and it is at or below the majority baseline in four
of five categories. The three binary categories sit near chance while the
open-vocabulary ones fall far below it, so the aggregate is mostly coin-flips on
yes/no questions. Accuracy is also flat in program depth (0.259 at 2–6 nodes,
0.292 at 11–15, 0.394 at 16+) — a model composing operations would degrade with
depth; one answering from surface features and the class prior does not.

The dominant failure mode is answering the wrong attribute type: asked for an
object's material, the model replies *"It's a cube"*. 182 of 1,000 answers
(18.2%) contained no word of the type the question asked for, and **all 182 were
wrong**. Only 1.4% of output fell outside the vocabulary entirely, so this is a
reasoning/grounding failure, not a formatting one.

## Layout

```
clevrbench/
  taxonomy.py    what makes a question compositional (type, depth, hops)
  answers.py     free text -> CLEVR's 28-answer vocabulary
  data.py        locating/fetching CLEVR, in whatever state the machine is in
  remote_zip.py  seekable file object over HTTP range requests
  subset.py      stratified, reproducible, self-contained eval sets
  models/        the adapter interface, registry, BLIP-2, baselines
  runner.py      the evaluation loop (model-agnostic)
  metrics.py     accuracy per compositional axis, with intervals and baselines
  report.py      console tables + results.json
  cli.py         models / status / prepare / eval / verify / report
tests/           47 hermetic tests: no CLEVR, no network, no torch
```

Requires Pillow for the core; torch/transformers/accelerate only to run a real
model (`requirements.txt`). The baselines and the whole test suite run without
them.

`clevr_eval.py` is the original single-file version, kept for reference; its
results are in `runs/val2k`. It is superseded by this package — the harness
reproduces its numbers (0.283 [0.256, 0.312] here vs 0.301 on its own 2,000-question
draw, and per-category within sampling error).

## Next steps

- **Constrained decoding**: score all 28 vocabulary answers by likelihood and take
  the argmax, removing format failures and the normalizer entirely.
- `blip2-flan-t5-xl` and `instructblip-vicuna-7b` are registered but unrun; both
  follow short-answer instructions more reliably than the OPT variant.
- Slice by `question_family_index` (90 templates) to find which specific
  constructions fail, not just which categories.
