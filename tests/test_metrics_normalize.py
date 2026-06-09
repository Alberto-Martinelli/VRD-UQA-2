"""Step-1 normalize checks. Run: uv run python -m tests.test_metrics_normalize"""
import json
import tempfile
from pathlib import Path
from config.paths import REPO_ROOT
import importlib.util


def _load(rel_path, name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


STEP1 = _load("VQA_analysis/metrics/1_normalize_unanswerable_responses.py", "step1")


def test_canonicalize_rule_based():
    ident = lambda a: a  # identity classifier; must not be called for these
    assert STEP1.canonicalize_answer("Not available", ident) == "unable to determine"
    assert STEP1.canonicalize_answer("", ident) == "unable to determine"
    assert STEP1.canonicalize_answer("$100", ident) == "$100"  # numeric passthrough


def test_canonicalize_uses_classifier_for_freetext():
    calls = []
    def stub(a):
        calls.append(a)
        return "stubbed"
    assert STEP1.canonicalize_answer("Paris", stub) == "stubbed"
    assert calls == ["Paris"]


def test_label_vqa_answers_both_sides():
    item = {"verification_result": {"vqa_results": [{
        "answer_corrupted": [{"pages": ["p1"], "answer": "Not available"}],
        "answer_clean": [{"pages": ["p1"], "answer": "$42"}],
    }]}}
    src = {"corrupted_questions": [item]}
    tmp = Path(tempfile.mkdtemp())
    inp, out = tmp / "predictions.json", tmp / "normalized.json"
    inp.write_text(json.dumps(src))
    STEP1.label_vqa_answers(str(inp), str(out), classify_fn=lambda a: a)
    r = json.load(open(out))["corrupted_questions"][0]["verification_result"]["vqa_results"][0]
    assert r["answer_corrupted"][0]["answer_converted"] == "unable to determine"
    assert r["answer_clean"][0]["answer_converted"] == "$42"


if __name__ == "__main__":
    test_canonicalize_rule_based()
    test_canonicalize_uses_classifier_for_freetext()
    test_label_vqa_answers_both_sides()
    print("OK: metrics step 1 normalize")
