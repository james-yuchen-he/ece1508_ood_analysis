#!/usr/bin/env python
"""Tests for the harness itself, so a bad number is a model result and not a bug.

Hermetic: no CLEVR, no network, no torch, no GPU. Synthetic questions and
generated PNGs stand in for the dataset. Run directly or under pytest:

    python tests/test_harness.py
    pytest tests/test_harness.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clevrbench import answers, metrics, models, report, runner, subset, taxonomy
from clevrbench.data import ClevrData, image_member, questions_member
from clevrbench.models.base import VLMAdapter
from clevrbench.remote_zip import HTTPRangeFile


# -- fixtures -------------------------------------------------------------
def make_question(index=0, function="count", answer="2", program=None, image=None):
    """A CLEVR-shaped question with a plausible program."""
    if program is None:
        program = [{"function": "scene", "inputs": [], "value_inputs": []},
                   {"function": function, "inputs": [0], "value_inputs": []}]
    return {
        "question_index": index,
        "image_index": index,
        "image_filename": image or f"CLEVR_val_{index:06d}.png",
        "question_family_index": 0,
        "split": "val",
        "question": "How many things are there?",
        "answer": answer,
        "program": program,
    }


def chain(*functions):
    return [{"function": f, "inputs": [max(0, i - 1)], "value_inputs": []}
            for i, f in enumerate(functions)]


# -- answer normalization -------------------------------------------------
def test_normalize_exact_answers():
    q = make_question(function="query_color", answer="red")
    assert answers.normalize("red", q) == ("red", "exact")
    assert answers.normalize("  Red.\n", q) == ("red", "exact")


def test_normalize_synonyms_and_numbers():
    count_q = make_question(function="count", answer="3")
    assert answers.normalize("three", count_q)[0] == "3"
    shape_q = make_question(function="query_shape", answer="sphere")
    assert answers.normalize("ball", shape_q)[0] == "sphere"
    material_q = make_question(function="query_material", answer="metal")
    assert answers.normalize("metallic", material_q)[0] == "metal"


def test_no_means_zero_only_for_counts():
    """'no' is a yes/no answer everywhere except a counting question."""
    assert answers.normalize("no", make_question(function="exist", answer="no"))[0] == "no"
    assert answers.normalize("no", make_question(function="count", answer="0"))[0] == "0"


def test_typed_parsing_prefers_the_asked_attribute():
    """The failure the lenient rule has: first vocabulary word wins."""
    q = make_question(function="query_color", answer="red")
    assert answers.normalize("the ball is red", q, mode="typed")[0] == "red"
    assert answers.normalize("the ball is red", q, mode="lenient")[0] == "sphere"


def test_strict_mode_rejects_verbose_output():
    q = make_question(function="query_color", answer="red")
    assert answers.normalize("the ball is red", q, mode="strict") == ("", "none")
    assert answers.normalize("red", q, mode="strict")[0] == "red"


def test_out_of_vocabulary_output_is_unparsed():
    q = make_question(function="query_color", answer="red")
    assert answers.normalize("a photograph of objects", q)[0] == ""
    assert answers.normalize("", q) == ("", "none")


def test_parse_path_reports_how_the_answer_was_found():
    q = make_question(function="query_material", answer="metal")
    assert answers.normalize("metal", q)[1] == "exact"
    assert answers.normalize("it is a metal cube", q)[1] == "typed"
    # No material word anywhere, so scoring falls back to any vocabulary word.
    assert answers.normalize("it is a cube", q)[1] == "fallback"


def test_expected_answers_follow_the_program():
    assert answers.expected_answers(make_question(function="count")) == answers.COUNTS
    assert answers.expected_answers(make_question(function="exist")) == answers.BOOLEAN
    assert answers.expected_answers(make_question(function="query_size")) == answers.SIZES
    assert answers.expected_answers({"program": []}) is None


# -- taxonomy -------------------------------------------------------------
def test_question_type_from_final_program_node():
    cases = {
        "count": "count", "exist": "exist", "greater_than": "compare_number",
        "equal_integer": "compare_number", "query_color": "query_attribute",
        "equal_shape": "compare_attribute",
    }
    for function, expected in cases.items():
        assert taxonomy.question_type(make_question(function=function)) == expected
    assert taxonomy.question_type({"program": []}) == "unknown"


def test_type_key_fallback_for_alternate_releases():
    """Some CLEVR derivatives spell the node key 'type' instead of 'function'."""
    q = {"program": [{"type": "scene"}, {"type": "count"}]}
    assert taxonomy.question_type(q) == "count"


def test_depth_hops_and_same_attribute():
    q = make_question(program=chain("scene", "filter_color", "relate", "filter_shape",
                                    "relate", "count"))
    assert taxonomy.program_depth(q) == 6
    assert taxonomy.relate_hops(q) == 2
    assert not taxonomy.needs_same_attribute(q)
    same = make_question(program=chain("scene", "same_color", "count"))
    assert taxonomy.needs_same_attribute(same)


def test_depth_buckets_cover_the_range():
    assert taxonomy.depth_bucket(2) == "2-6"
    assert taxonomy.depth_bucket(6) == "2-6"
    assert taxonomy.depth_bucket(7) == "7-10"
    assert taxonomy.depth_bucket(25) == "16+"


def test_features_are_complete():
    features = taxonomy.features(make_question(function="count"))
    for key in ("type", "depth", "depth_bucket", "hops", "hops_bucket", "same_attr"):
        assert key in features, key
    assert features["hops_bucket"] == "0 hops"


# -- metrics --------------------------------------------------------------
def test_wilson_interval_brackets_the_estimate():
    lo, hi = metrics.wilson_interval(50, 100)
    assert lo < 0.5 < hi and 0 <= lo and hi <= 1
    # Degenerate cases stay inside [0, 1] where a normal interval would not.
    lo, hi = metrics.wilson_interval(0, 10)
    assert lo == 0.0 and 0 < hi < 1
    lo, hi = metrics.wilson_interval(10, 10)
    assert abs(hi - 1.0) < 1e-9 and 0 < lo < 1  # exactly 1.0 up to float error
    assert metrics.wilson_interval(0, 0) == (0.0, 0.0)
    # More data, tighter interval.
    narrow = metrics.wilson_interval(500, 1000)
    wide = metrics.wilson_interval(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def _records(specs):
    """specs: (type, gold, pred) -> scored records."""
    out = []
    for i, (qtype, gold, pred) in enumerate(specs):
        out.append({
            "question_index": i, "image_filename": "x.png", "question": "q",
            "gold": gold, "raw": pred, "pred": pred, "parse_path": "exact",
            "correct": gold == pred, "type": qtype, "depth": 5,
            "depth_bucket": "2-6", "hops": 0, "hops_bucket": "0 hops",
            "same_attr": False, "family": 0,
        })
    return out


def test_aggregate_scores_and_slices():
    result = metrics.aggregate(_records([
        ("count", "2", "2"), ("count", "3", "1"),
        ("exist", "yes", "yes"), ("exist", "no", "yes"),
    ]))
    assert result["n_questions"] == 4
    assert result["overall"]["accuracy"] == 0.5
    assert result["by_type"]["count"]["n"] == 2
    assert result["by_type"]["exist"]["accuracy"] == 0.5
    assert result["by_depth"]["2-6"]["n"] == 4


def test_majority_baseline_and_per_type_floor():
    # 'no' is 3 of 5 golds overall; within exist it is 3 of 3.
    result = metrics.aggregate(_records([
        ("exist", "no", "no"), ("exist", "no", "no"), ("exist", "no", "no"),
        ("count", "1", "1"), ("count", "2", "1"),
    ]))
    assert result["overall"]["majority_answer"] == "no"
    assert abs(result["overall"]["majority_baseline"] - 0.6) < 1e-9
    # Per-type floor: exist -> 'no' (3/3), count -> either (1/2) = 4/5.
    assert abs(result["overall"]["majority_by_type_baseline"] - 0.8) < 1e-9


def test_unparsed_and_collapse_are_tracked():
    result = metrics.aggregate(_records([
        ("exist", "yes", "yes"), ("exist", "no", ""), ("count", "1", "yes"),
    ]))
    assert abs(result["overall"]["unparsed_rate"] - 1 / 3) < 1e-9
    assert result["answer_collapse"]["top_prediction"] == "yes"
    assert result["prediction_distribution"]["<unparsed>"] == 1


def test_aggregate_handles_no_records():
    assert metrics.aggregate([])["n_questions"] == 0


# -- subset ---------------------------------------------------------------
def test_allocation_is_proportional_and_capped():
    available = {"a": 500, "b": 300, "c": 200}
    alloc = subset._allocate(available, 100, balanced=False)
    assert sum(alloc.values()) == 100
    assert alloc["a"] > alloc["b"] > alloc["c"]

    # A stratum that runs dry hands its share to the others.
    alloc = subset._allocate({"a": 5, "b": 1000}, 100, balanced=True)
    assert alloc["a"] == 5 and sum(alloc.values()) == 100

    # Never over-draws a stratum, and never invents questions.
    alloc = subset._allocate({"a": 3, "b": 4}, 100, balanced=False)
    assert alloc == {"a": 3, "b": 4}


def test_balanced_allocation_equalizes():
    alloc = subset._allocate({"a": 500, "b": 300, "c": 200}, 90, balanced=True)
    assert sum(alloc.values()) == 90
    assert max(alloc.values()) - min(alloc.values()) <= 1


def test_sampling_is_stratified_and_reproducible():
    pool = ([make_question(i, "count", "1") for i in range(300)]
            + [make_question(300 + i, "exist", "yes") for i in range(100)])
    first = subset.sample_questions(pool, 100, seed=0)
    again = subset.sample_questions(pool, 100, seed=0)
    other = subset.sample_questions(pool, 100, seed=1)

    indices = [q["question_index"] for q in first]
    assert indices == [q["question_index"] for q in again], "seed must be deterministic"
    assert indices != [q["question_index"] for q in other], "different seeds must differ"
    assert len(first) == 100

    types = [taxonomy.question_type(q) for q in first]
    # Proportional to the 300/100 pool, within rounding.
    assert abs(types.count("count") - 75) <= 1
    assert abs(types.count("exist") - 25) <= 1


def test_sampling_respects_questions_per_image():
    pool = [make_question(i, "count", "1", image=f"CLEVR_val_{i // 10:06d}.png")
            for i in range(100)]
    sampled = subset.sample_questions(pool, 100, seed=0, questions_per_image=2)
    assert len(sampled) == 20, "10 images x 2 questions"
    assert len({q["image_filename"] for q in sampled}) == 10


def test_subset_roundtrip_on_disk():
    """Build a subset from a fake dataset and read it back."""
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data = ClevrData(data_dir=tmp / "data", allow_download=False, quiet=True)

        questions = [make_question(i, "count", str(i % 3), image=f"CLEVR_val_{i:06d}.png")
                     for i in range(20)]
        qpath = data.path(questions_member("val"))
        qpath.parent.mkdir(parents=True, exist_ok=True)
        qpath.write_text(json.dumps({"questions": questions}))
        for q in questions:
            ipath = data.path(image_member("val", q["image_filename"]))
            ipath.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (10, 20, 30)).save(ipath)

        out = subset.build_subset(tmp / "sub", size=10, data=data, quiet=True)
        loaded, images_dir, manifest = subset.load_subset(out)

        assert len(loaded) == 10
        assert manifest["n_questions"] == 10 and manifest["seed"] == 0
        assert sum(1 for _ in images_dir.glob("*.png")) == manifest["n_images"]
        # Programs survive, or none of the compositional slicing works.
        assert all("program" in q and "answer" in q for q in loaded)


def test_allocation_is_independent_of_dict_order():
    """Regression: ties were broken by set iteration order, i.e. by PYTHONHASHSEED.

    With --balanced every stratum has an identical share, so whichever strata
    received the remainder depended on a hash seed that varies per process --
    two machines at the same --seed built different subsets.
    """
    forward = subset._allocate({"a": 100, "b": 100, "c": 100, "d": 100}, 10, balanced=True)
    reverse = subset._allocate({"d": 100, "c": 100, "b": 100, "a": 100}, 10, balanced=True)
    assert forward == reverse
    assert sum(forward.values()) == 10


def test_allocation_is_deterministic_across_hash_seeds():
    """The same check across real processes, where PYTHONHASHSEED actually varies."""
    import subprocess

    root = Path(__file__).resolve().parent.parent
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from clevrbench import subset\n"
        "print(subset._allocate({'a':100,'b':100,'c':100,'d':100}, 10, balanced=True))\n"
        % str(root)
    )
    outputs = set()
    for hash_seed in ("0", "1", "2", "3"):
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                              text=True, env={"PYTHONHASHSEED": hash_seed, "PATH": ""})
        assert proc.returncode == 0, proc.stderr
        outputs.add(proc.stdout.strip())
    assert len(outputs) == 1, f"allocation varied with PYTHONHASHSEED: {outputs}"


def test_sample_checksum_identifies_the_sample():
    a = [make_question(1), make_question(2), make_question(3)]
    assert subset.sample_checksum(a) == subset.sample_checksum(list(a))
    # Order matters: it changes which questions land in which batch.
    assert subset.sample_checksum(a) != subset.sample_checksum(list(reversed(a)))
    # Contents matter.
    assert subset.sample_checksum(a) != subset.sample_checksum(a[:2])
    assert subset.sample_checksum(a).startswith("sha256:")


def test_checksum_survives_a_rebuild_from_the_same_recipe():
    """The property a contributor relies on: same recipe, same questions."""
    pool = ([make_question(i, "count", "1") for i in range(300)]
            + [make_question(300 + i, "exist", "yes") for i in range(100)])
    first = subset.sample_questions(pool, 100, seed=0, strata="type")
    again = subset.sample_questions(pool, 100, seed=0, strata="type")
    assert subset.sample_checksum(first) == subset.sample_checksum(again)
    other = subset.sample_questions(pool, 100, seed=1, strata="type")
    assert subset.sample_checksum(first) != subset.sample_checksum(other)


def test_verify_subset_detects_tampering_and_mismatch():
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data = ClevrData(data_dir=tmp / "data", allow_download=False, quiet=True)
        questions = [make_question(i, "count", str(i % 3), image=f"CLEVR_val_{i:06d}.png")
                     for i in range(20)]
        qpath = data.path(questions_member("val"))
        qpath.parent.mkdir(parents=True, exist_ok=True)
        qpath.write_text(json.dumps({"questions": questions}))
        for q in questions:
            ipath = data.path(image_member("val", q["image_filename"]))
            ipath.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (10, 20, 30)).save(ipath)

        out = subset.build_subset(tmp / "sub", size=10, data=data, quiet=True)

        good = subset.verify_subset(out)
        assert good["ok"] and good["manifest_match"] is True
        assert good["expected_match"] is None
        assert subset.verify_subset(out, expected=good["checksum"])["expected_match"]
        assert subset.verify_subset(out, expected="sha256:nope")["ok"] is False

        # A missing image is a broken subset, not a different one.
        loaded, images_dir, _ = subset.load_subset(out)
        next(iter(images_dir.glob("*.png"))).unlink()
        assert subset.verify_subset(out)["missing_images"]

        # Editing questions.json is caught against the manifest.
        with open(out / "questions.json") as fh:
            payload = json.load(fh)
        payload["questions"] = payload["questions"][:5]
        (out / "questions.json").write_text(json.dumps(payload))
        assert subset.verify_subset(out)["manifest_match"] is False


def test_subset_dir_name_is_stable():
    a = subset.default_subset_dir("r", "val", 1000, 0, "type", False)
    b = subset.default_subset_dir("r", "val", 1000, 0, "type", False)
    c = subset.default_subset_dir("r", "val", 1000, 1, "type", False)
    assert a == b and a != c


# -- model registry -------------------------------------------------------
def test_builtin_models_are_listed():
    names = [name for name, _ in models.available()]
    assert "blip2-opt-2.7b" in names and "constant" in names


def test_build_baselines_and_pass_arguments():
    model = models.build("constant", answer="no")
    assert model.answer_batch([None], ["q?"]) == ["no"]
    rng_model = models.build("random", seed=3)
    rng_model.load()
    assert len(rng_model.answer_batch([None] * 5, ["q?"] * 5)) == 5


def test_random_baseline_is_reproducible():
    first, second = models.build("random", seed=7), models.build("random", seed=7)
    first.load()
    second.load()
    assert first.answer_batch([None] * 20, ["q"] * 20) == second.answer_batch(
        [None] * 20, ["q"] * 20)


def test_unknown_model_names_are_rejected():
    for spec in ("no-such-model", "clevrbench.models.baselines:NotAClass"):
        try:
            models.build(spec)
        except SystemExit:
            continue
        raise AssertionError(f"{spec} should have been rejected")


def test_external_adapter_loads_from_a_file():
    """The documented way to plug in a model without touching the package."""
    source = '''
from clevrbench.models.base import VLMAdapter

class MyVLM(VLMAdapter):
    name = "my-vlm"
    needs_images = False

    def __init__(self, reply="cube"):
        self.reply = reply

    def answer_batch(self, images, questions):
        return [self.reply] * len(questions)
'''
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "my_adapter.py"
        path.write_text(source)
        model = models.build(f"{path}:MyVLM", reply="sphere")
        assert model.name == "my-vlm"
        assert model.answer_batch([None, None], ["a", "b"]) == ["sphere", "sphere"]


def test_registered_models_are_buildable():
    class Stub(VLMAdapter):
        name = "stub"
        needs_images = False

        def answer_batch(self, images, questions):
            return ["yes"] * len(questions)

    models.register("stub-model", Stub, "test stub")
    assert "stub-model" in [n for n, _ in models.available()]
    assert models.build("stub-model").answer_batch([None], ["q"]) == ["yes"]


# -- runner ---------------------------------------------------------------
class ScriptedAdapter(VLMAdapter):
    """Replays a fixed list of outputs, to test scoring rather than a model."""

    name = "scripted"
    needs_images = False

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.batches_seen = 0

    def answer_batch(self, images, questions):
        self.batches_seen += 1
        taken, self.outputs = self.outputs[: len(questions)], self.outputs[len(questions):]
        return taken


def test_runner_scores_and_labels_records():
    questions = [
        make_question(0, "count", "2"),
        make_question(1, "exist", "yes"),
        make_question(2, "query_color", "red"),
    ]
    model = ScriptedAdapter(["2", "no", "the ball is red"])
    records, stats = runner.evaluate(model, questions, "/nonexistent", batch_size=2,
                                     progress=False)

    assert [r["correct"] for r in records] == [True, False, True]
    assert records[2]["pred"] == "red" and records[2]["raw"] == "the ball is red"
    assert records[0]["type"] == "count" and "depth_bucket" in records[0]
    assert stats["questions_per_second"] >= 0
    assert model.batches_seen == 2


def test_runner_rejects_a_misbehaving_adapter():
    """Length mismatch would silently misalign predictions against gold."""
    model = ScriptedAdapter(["yes"])
    try:
        runner.evaluate(model, [make_question(0), make_question(1)], "/nonexistent",
                        batch_size=2, progress=False)
    except SystemExit:
        return
    raise AssertionError("adapter returning the wrong number of answers must fail loudly")


def test_runner_decodes_images_once_per_file():
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        Image.new("RGB", (8, 8)).save(tmp / "CLEVR_val_000000.png")

        class NeedsImages(ScriptedAdapter):
            needs_images = True

        questions = [make_question(0, "count", "1", image="CLEVR_val_000000.png")
                     for _ in range(6)]
        model = NeedsImages(["1"] * 6)
        _, stats = runner.evaluate(model, questions, tmp, batch_size=2, progress=False)
        assert stats["images_decoded"] == 1, "the LRU should decode each PNG once"


def test_missing_image_fails_clearly():
    class NeedsImages(ScriptedAdapter):
        needs_images = True

    try:
        runner.evaluate(NeedsImages(["1"]), [make_question(0)], "/nonexistent",
                        batch_size=1, progress=False)
    except SystemExit as exc:
        assert "image missing" in str(exc)
        return
    raise AssertionError("a missing image must fail loudly")


# -- reporting ------------------------------------------------------------
def test_report_renders_and_roundtrips():
    records = _records([("count", "2", "2"), ("exist", "yes", "no")])
    model = models.build("constant", answer="2")
    results = report.build_results(
        model, records, metrics.aggregate(records),
        {"seconds": 1.0, "questions_per_second": 2.0, "batch_size": 8},
        {"path": "sub", "split": "val", "n_images": 2}, {"parse_mode": "typed"},
    )
    text = report.format_report(results)
    for expected in ("accuracy", "question type", "majority", "count"):
        assert expected in text

    with tempfile.TemporaryDirectory() as tmp:
        out = report.write_run(Path(tmp) / "run", results, records)
        assert (out / "results.json").exists()
        reloaded = report.load_run(out)
        assert reloaded["metrics"]["n_questions"] == 2
        lines = (out / "predictions.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2 and json.loads(lines[0])["gold"] == "2"


# -- HTTP range reader ----------------------------------------------------
class FakeRangeFile(HTTPRangeFile):
    """HTTPRangeFile over an in-memory buffer, so the logic is testable offline."""

    def __init__(self, payload, block_size=16):
        self.payload = payload
        self.url = "memory://"
        self.block_size = block_size
        self.timeout = 1
        self.max_retries = 1
        self._pos = 0
        self._cache_start = 0
        self._cache = b""
        self.bytes_fetched = 0
        self.requests_made = 0
        self.size = len(payload)

    def _fetch(self, start, end):
        end = min(end, self.size - 1)
        data = self.payload[start:end + 1]
        self.requests_made += 1
        self.bytes_fetched += len(data)
        return data


def test_range_reader_matches_the_underlying_bytes():
    payload = bytes(range(256)) * 8  # 2048 bytes
    reader = FakeRangeFile(payload, block_size=64)

    assert reader.read(10) == payload[:10]
    reader.seek(1000)
    assert reader.read(100) == payload[1000:1100]
    reader.seek(-20, 2)
    assert reader.read() == payload[-20:]
    reader.seek(0)
    assert reader.read() == payload, "a full sequential read must reproduce the file"
    assert reader.read(10) == b"", "reads past EOF return nothing"


def test_range_reader_serves_repeat_reads_from_cache():
    reader = FakeRangeFile(bytes(range(256)) * 8, block_size=512)
    reader.seek(0)
    reader.read(16)
    after_first = reader.requests_made
    for offset in range(0, 400, 16):  # all inside the cached block
        reader.seek(offset)
        reader.read(16)
    assert reader.requests_made == after_first, "cached bytes must not refetch"


def test_prefetch_fetches_an_exact_range_once():
    payload = bytes(range(256)) * 8
    reader = FakeRangeFile(payload, block_size=1024)
    reader.prefetch(100, 200)
    assert reader.requests_made == 1
    assert reader.bytes_fetched == 200, "prefetch must not over-read"

    reader.seek(100)
    assert reader.read(200) == payload[100:300]
    assert reader.requests_made == 1, "the prefetched range serves the read"


def test_prefetch_clamps_at_the_end_of_file():
    reader = FakeRangeFile(b"abcdef", block_size=4)
    reader.prefetch(4, 999)
    reader.seek(4)
    assert reader.read(999) == b"ef"


# -- data layer -----------------------------------------------------------
def test_member_paths_follow_the_archive_layout():
    assert questions_member("val") == "CLEVR_v1.0/questions/CLEVR_val_questions.json"
    assert image_member("val", "a.png") == "CLEVR_v1.0/images/val/a.png"


def test_missing_data_with_downloads_disabled_fails_clearly():
    with tempfile.TemporaryDirectory() as tmp:
        data = ClevrData(data_dir=Path(tmp) / "data", allow_download=False, quiet=True)
        assert not data.have(questions_member("val"))
        try:
            data.load_questions("val")
        except SystemExit as exc:
            assert "download" in str(exc).lower()
            return
        raise AssertionError("should refuse to invent data")


def test_test_split_answers_are_refused():
    """CLEVR withholds test answers; scoring against them is meaningless."""
    with tempfile.TemporaryDirectory() as tmp:
        data = ClevrData(data_dir=Path(tmp) / "data", allow_download=False, quiet=True)
        path = data.path(questions_member("test"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"questions": [{"question": "q", "program": []}]}))
        try:
            data.load_questions("test")
        except SystemExit as exc:
            assert "ground-truth" in str(exc)
            return
        raise AssertionError("a split without answers must be rejected")


# -- runner ---------------------------------------------------------------
def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  pass  {name}")
        except Exception:
            failures.append((name, traceback.format_exc()))
            print(f"  FAIL  {name}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    for name, tb in failures:
        print(f"\n--- {name} ---\n{tb}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
