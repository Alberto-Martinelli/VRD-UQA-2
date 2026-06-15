"""Mock-mode evaluator checks for all model entrypoints.
Run: uv run python -m tests.test_evaluator_mock"""
import json
import os
import subprocess
import tempfile
from pathlib import Path
from config.paths import REPO_ROOT

SAMPLE = REPO_ROOT / "VQA_analysis" / "evaluation_files" / "BDocs_sample15.json"

# (cli_model_key, leaf_prefix, finetuned). Qwen has an adapter entry (finetuned);
# the others run zero-shot. cli_model_key is the run_eval.py --model value.
MODELS = [
    ("qwen2.5",  "",         True),
    ("phi4",     "phi4",     False),
    ("internvl", "internvl", False),
    ("gemma4",   "gemma",    False),
]


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


def _run(model, cfg, run_id, questions, run_root, finetuned, input_file=SAMPLE):
    env = dict(os.environ)
    env["VQA_RUN_ID"] = run_id
    env["VQA_EVAL_RUNS_DIR"] = str(run_root)
    cmd = ["uv", "run", "python", "VQA_analysis/evaluators/run_eval.py",
           "--model", model, "--dataset", "BDocs", "--split", "val_15",
           "--config", str(cfg), "--input-file", str(input_file),
           "--questions", questions]
    if finetuned:
        cmd.append("--finetuned")
    subprocess.check_call(cmd, cwd=str(REPO_ROOT), env=env)


def _leaf_name(prefix, finetuned):
    slug = "finetuned_noocr" if finetuned else "zeroshot_noocr"
    return f"{prefix}__{slug}" if prefix else slug


def test_all_models_dual_answer_and_namespaced_leaf():
    for i, (key, prefix, finetuned) in enumerate(MODELS):
        tmp = tempfile.mkdtemp()
        run_root = Path(tmp) / "runs"
        cfg = _write_mock_config(tmp)
        run_id = f"eval_val_15_2026010100000{i}"
        _run(key, cfg, run_id, "both", run_root, finetuned)

        leaf = run_root / run_id / "BDocs" / _leaf_name(prefix, finetuned)
        preds = json.load(open(leaf / "predictions.json"))
        item0 = preds["corrupted_questions"][0]["verification_result"]["vqa_results"][0]
        assert "answer_corrupted" in item0 and "answer_clean" in item0, key
        assert item0["question_corrupted"] != item0["question_clean"], key
        assert item0["model_type"] in ("qwen", "phi4", "internvl", "gemma"), key

        man = json.load(open(leaf / "manifest.json"))
        assert man["dataset"] == "BDocs", key
        assert man["questions"] == "both", key
        assert man["seed"] == 42, key


def test_corrupted_only_omits_clean_qwen():
    tmp = tempfile.mkdtemp()
    run_root = Path(tmp) / "runs"
    cfg = _write_mock_config(tmp)
    run_id = "eval_val_15_20260101_000010"
    _run(MODELS[0][0], cfg, run_id, "corrupted", run_root, True)
    leaf = run_root / run_id / "BDocs" / "finetuned_noocr"
    item0 = json.load(open(leaf / "predictions.json"))["corrupted_questions"][0]["verification_result"]["vqa_results"][0]
    assert "answer_corrupted" in item0 and "answer_clean" not in item0


def test_resume_skips_completed_leaf_qwen():
    tmp = tempfile.mkdtemp()
    run_root = Path(tmp) / "runs"
    cfg = _write_mock_config(tmp)
    run_id = "eval_val_15_20260101_000011"
    _run(MODELS[0][0], cfg, run_id, "both", run_root, True)
    preds = run_root / run_id / "BDocs" / "finetuned_noocr" / "predictions.json"
    with open(preds) as f:
        d = json.load(f)
    d["_resume_marker"] = True
    with open(preds, "w") as f:
        json.dump(d, f)
    _run(MODELS[0][0], cfg, run_id, "both", run_root, True)  # second pass -> should skip
    with open(preds) as f:
        assert json.load(f).get("_resume_marker") is True


if __name__ == "__main__":
    test_all_models_dual_answer_and_namespaced_leaf()
    test_corrupted_only_omits_clean_qwen()
    test_resume_skips_completed_leaf_qwen()
    print("OK: evaluator mock dual-answer (all 4 models)")
