"""Step-3 compute checks. Run: uv run python -m tests.test_metrics_compute"""
import json
from pathlib import Path
from config.paths import REPO_ROOT
import importlib.util


def _load(rel_path, name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


STEP3 = _load("VQA_analysis/metrics/3_compute_metrics.py", "step3")
FIX = REPO_ROOT / "tests" / "fixtures" / "normalized_min.json"


def _results():
    with open(FIX) as f:
        return json.load(f)["corrupted_questions"]


def test_qur_reads_corrupted_side():
    az = STEP3.VQAAnalyzer(_results(), None, "BDocs", side="corrupted")
    qur = az.QUR()
    assert qur[0] == 1.0  # both items fully refused on corrupted side


def test_frr_reads_clean_side():
    az = STEP3.VQAAnalyzer(_results(), None, "BDocs", side="clean")
    frr = az.FRR()
    # item0 answered (Rome) -> not refused; item1 refused -> FRR = 1/2
    assert abs(frr[0] - 0.5) < 1e-9


def test_get_answers_side_selection():
    az = STEP3.VQAAnalyzer(_results(), None, "BDocs", side="corrupted")
    assert az._get_answers(az.valid_results[0])[0]["answer_converted"] == "unable to determine"
    az2 = STEP3.VQAAnalyzer(_results(), None, "BDocs", side="clean")
    assert az2._get_answers(az2.valid_results[0])[0]["answer_converted"] == "Rome"


def test_get_answers_tolerates_nonlist():
    # A legacy/error record may store a flat string under answer_<side>; metrics
    # must coerce it to [] and not iterate over its characters.
    res = {"is_corrupted": True, "complexity": 1, "verification_result":
           {"vqa_results": [{"answer_corrupted": "Unable to determine: error"}]}}
    az = STEP3.VQAAnalyzer([res], None, "BDocs", side="corrupted")
    assert az._get_answers(az.valid_results[0]) == []
    assert az.QUR()[0] == 0.0  # does not crash; counts as a non-refusal


if __name__ == "__main__":
    test_qur_reads_corrupted_side()
    test_frr_reads_clean_side()
    test_get_answers_side_selection()
    test_get_answers_tolerates_nonlist()
    print("OK: metrics step 3 compute")
