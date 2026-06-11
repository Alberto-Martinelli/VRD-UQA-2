"""Run: uv run python -m tests.test_aggregate_summaries"""
import csv
import os
import tempfile
from pathlib import Path


def _write_summary(run_dir, rows):
    run_dir.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "config", "label", "model", "metric", "complexity", "value"]
    with open(run_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_aggregates_multiple_runs():
    tmp = tempfile.mkdtemp()
    runs = Path(tmp) / "runs"
    os.environ["VQA_EVAL_RUNS_DIR"] = str(runs)
    _write_summary(runs / "eval_val_100_llama_BDocs",
                   [{"dataset": "BDocs", "config": "zeroshot_ocr", "label": "Zero-Shot",
                     "model": "Llama3.2-11B", "metric": "QUR", "complexity": "overall", "value": "0.5"}])
    _write_summary(runs / "eval_val_100_phi4_BDocs",
                   [{"dataset": "BDocs", "config": "zeroshot_ocr", "label": "Zero-Shot",
                     "model": "Phi4-multimodal", "metric": "QUR", "complexity": "overall", "value": "0.6"}])

    from importlib.machinery import SourceFileLoader
    from config.paths import REPO_ROOT
    mod = SourceFileLoader(
        "agg", str(REPO_ROOT / "VQA_analysis" / "metrics" / "4_aggregate_summaries.py")
    ).load_module()
    out = mod.aggregate(tag="test")

    with open(out) as f:
        rows = list(csv.DictReader(f))
    models = sorted(r["model"] for r in rows)
    assert models == ["Llama3.2-11B", "Phi4-multimodal"], models
    assert {r["run_id"] for r in rows} == {"eval_val_100_llama_BDocs", "eval_val_100_phi4_BDocs"}


if __name__ == "__main__":
    test_aggregates_multiple_runs()
    print("OK: aggregate summaries")
