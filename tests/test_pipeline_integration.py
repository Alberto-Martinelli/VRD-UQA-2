"""End-to-end mock pipeline integration. Run: uv run python -m tests.test_pipeline_integration

Runs evaluator (mock) -> normalize (--no-model) -> enrich -> compute on the
BDocs sample, entirely inside tempdirs (VQA_EVAL_RUNS_DIR), with dummy page
images so the image-dependent metrics in step 3 run for real.
"""
import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path
from PIL import Image
from config.paths import REPO_ROOT

SAMPLE = REPO_ROOT / "VQA_analysis" / "evaluation_files" / "BDocs_sample15.json"


def _page_basenames(sample_data):
    names = set()
    for item in sample_data.get("corrupted_questions", []):
        for page_id in item.get("layout_analysis", {}).get("pages", {}):
            names.add(os.path.basename(page_id))
    return names


def _make_dummy_images(images_dir, basenames):
    images_dir.mkdir(parents=True, exist_ok=True)
    for name in basenames:
        Image.new("RGB", (200, 200), color=(255, 255, 255)).save(images_dir / name)


def test_full_mock_run_produces_clean_tree():
    tmp = Path(tempfile.mkdtemp())
    run_root = tmp / "runs"
    images_dir = tmp / "images"
    run_id = "eval_val_15_20260101_010101"

    sample_data = json.loads(SAMPLE.read_text())
    _make_dummy_images(images_dir, _page_basenames(sample_data))

    # Repoint the sample's images at our dummy dir, write a local copy.
    sample_data["base_image_dir"] = str(images_dir)
    sample_path = tmp / "sample.json"
    sample_path.write_text(json.dumps(sample_data))

    base = json.loads((REPO_ROOT / "VQA_analysis" / "config_mock.json").read_text())
    base.update({"dataset": "BDocs", "input_file": str(sample_path), "ocr_enabled": False,
                 "sampling_percentage": 100, "seed": 42, "split": "val",
                 "few_shot": {"enabled": False}})
    cfg = tmp / "cfg.json"
    cfg.write_text(json.dumps(base))

    env = dict(os.environ)
    env.update({"VQA_RUN_ID": run_id, "VQA_EVAL_RUNS_DIR": str(run_root)})

    def run(*cmd):
        subprocess.check_call(["uv", "run", "python", *cmd], cwd=str(REPO_ROOT), env=env)

    run("VQA_analysis/evaluators/qwen2.5_evaluator.py", "--config_path", str(cfg), "--finetuned", "--questions", "both")
    run("VQA_analysis/metrics/1_normalize_unanswerable_responses.py", "--run-id", run_id, "--no-model")
    run("VQA_analysis/metrics/2_enrich_metadata.py", "--run-id", run_id)
    run("VQA_analysis/metrics/3_compute_metrics.py", "--run-id", run_id)

    leaf = run_root / run_id / "BDocs" / "finetuned_noocr"
    assert (leaf / "predictions.json").is_file()
    assert (leaf / "manifest.json").is_file()
    assert (leaf / "_cache" / "normalized.json").is_file()
    assert not (leaf / "_cache" / "enriched.json").exists()  # no third heavy file
    assert (leaf / "metrics" / "QUR.csv").is_file()
    assert (leaf / "metrics" / "FRR.csv").is_file()

    with open(leaf / "_cache" / "normalized.json") as f:
        norm = json.load(f)
    vr = norm["corrupted_questions"][0]["verification_result"]["vqa_results"][0]
    assert "answer_converted" in vr["answer_corrupted"][0]
    assert "answer_converted" in vr["answer_clean"][0]
    assert norm.get("_enriched") is True

    with open(run_root / run_id / "summary.csv") as f:
        summary = list(csv.DictReader(f))
    metrics_seen = {r["metric"] for r in summary}
    assert {"QUR", "UR", "FRR"} <= metrics_seen
    assert all(set(r.keys()) >= {"dataset", "config", "label", "model", "metric", "complexity", "value"} for r in summary)


if __name__ == "__main__":
    test_full_mock_run_produces_clean_tree()
    print("OK: end-to-end mock pipeline")
