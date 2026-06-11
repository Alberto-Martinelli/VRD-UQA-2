"""Mock-mode evaluator checks. Run: uv run python -m tests.test_evaluator_mock"""
import json
import os
import subprocess
import tempfile
from pathlib import Path
from config.paths import REPO_ROOT

SAMPLE = REPO_ROOT / "VQA_analysis" / "evaluation_files" / "BDocs_sample15.json"


def _write_mock_config(tmp):
    base = json.load(open(REPO_ROOT / "VQA_analysis" / "config_mock.json"))
    base["dataset"] = "BDocs"
    base["input_file"] = str(SAMPLE)
    base["ocr_enabled"] = False
    base["sampling_percentage"] = 100
    base["seed"] = 42
    base["few_shot"] = {"enabled": False}  # no images on disk in test env
    cfg = Path(tmp) / "mock_cfg.json"
    cfg.write_text(json.dumps(base))
    return cfg


def _run(cfg, run_id, questions, run_root):
    env = dict(os.environ)
    env["VQA_RUN_ID"] = run_id
    env["VQA_EVAL_RUNS_DIR"] = str(run_root)
    subprocess.check_call(
        ["uv", "run", "python", "VQA_analysis/evaluators/qwen2.5_evaluator.py",
         "--config_path", str(cfg), "--finetuned", "--questions", questions],
        cwd=str(REPO_ROOT), env=env,
    )


def test_dual_answer_predictions_and_manifest():
    tmp = tempfile.mkdtemp()
    run_root = Path(tmp) / "runs"
    cfg = _write_mock_config(tmp)
    run_id = "eval_val_15_20260101_000000"
    _run(cfg, run_id, "both", run_root)

    leaf = run_root / run_id / "BDocs" / "finetuned_noocr"
    preds = json.load(open(leaf / "predictions.json"))
    item0 = preds["corrupted_questions"][0]["verification_result"]["vqa_results"][0]
    assert "answer_corrupted" in item0 and "answer_clean" in item0
    assert item0["question_corrupted"] != item0["question_clean"]

    man = json.load(open(leaf / "manifest.json"))
    assert man["dataset"] == "BDocs"
    assert man["config"] == "finetuned_noocr"
    assert man["questions"] == "both"
    assert man["seed"] == 42
    assert man["model_name"] == "Qwen_2.5_7B_finetuned"
    assert man["adapter"]  # finetuned -> non-null
    assert "git_commit" in man and "created_at" in man


def test_questions_corrupted_only_omits_clean():
    tmp = tempfile.mkdtemp()
    run_root = Path(tmp) / "runs"
    cfg = _write_mock_config(tmp)
    run_id = "eval_val_15_20260101_000001"
    _run(cfg, run_id, "corrupted", run_root)
    leaf = run_root / run_id / "BDocs" / "finetuned_noocr"
    item0 = json.load(open(leaf / "predictions.json"))["corrupted_questions"][0]["verification_result"]["vqa_results"][0]
    assert "answer_corrupted" in item0 and "answer_clean" not in item0


def test_resume_skips_completed_leaf():
    tmp = tempfile.mkdtemp()
    run_root = Path(tmp) / "runs"
    cfg = _write_mock_config(tmp)
    run_id = "eval_val_15_20260101_000002"
    _run(cfg, run_id, "both", run_root)
    preds = run_root / run_id / "BDocs" / "finetuned_noocr" / "predictions.json"

    # Tamper with the file; a resumed (same run_id) run must NOT overwrite it.
    with open(preds) as f:
        d = json.load(f)
    d["_resume_marker"] = True
    with open(preds, "w") as f:
        json.dump(d, f)

    _run(cfg, run_id, "both", run_root)  # second pass -> should skip
    with open(preds) as f:
        assert json.load(f).get("_resume_marker") is True  # untouched => skipped


if __name__ == "__main__":
    test_dual_answer_predictions_and_manifest()
    test_questions_corrupted_only_omits_clean()
    test_resume_skips_completed_leaf()
    print("OK: evaluator mock dual-answer")
