"""Validates config/run_layout.py. Run: uv run python -m tests.test_run_layout"""
import datetime
import json
import tempfile
from pathlib import Path
from config import run_layout as rl


def test_make_run_id_format():
    when = datetime.datetime(2026, 6, 8, 22, 22, 13)
    assert rl.make_run_id("val", 300, when) == "eval_val_300_20260608_222213"


def test_derive_mode():
    assert rl.derive_mode(finetuned=True, few_shot_enabled=True) == "finetuned"
    assert rl.derive_mode(finetuned=False, few_shot_enabled=True) == "fewshot"
    assert rl.derive_mode(finetuned=False, few_shot_enabled=False) == "zeroshot"


def test_build_slug():
    assert rl.build_slug("finetuned", ocr_enabled=True, window_size=1) == "finetuned_ocr"
    assert rl.build_slug("zeroshot", ocr_enabled=False, window_size=1) == "zeroshot_noocr"
    assert rl.build_slug("fewshot", ocr_enabled=True, window_size=2) == "fewshot_ocr_w2"


def test_human_label():
    m = {"config": "finetuned_ocr", "ocr_enabled": True, "window_size": 1}
    assert rl.human_label(m) == "Fine-Tuned (LoRA) · OCR"
    m2 = {"config": "zeroshot_noocr", "ocr_enabled": False, "window_size": 1}
    assert rl.human_label(m2) == "Zero-Shot · no-OCR"
    # Unknown / missing mode falls back to "?".
    assert rl.human_label({}) == "? · no-OCR"


def test_paths_and_manifest_roundtrip():
    import os
    tmp = Path(tempfile.mkdtemp())
    # Point EVAL_RUNS_DIR at a temp location for this test only.
    rl.EVAL_RUNS_DIR = tmp
    leaf = rl.leaf_dir("eval_val_5_20260101_000000", "BDocs", "finetuned_ocr")
    assert leaf == tmp / "eval_val_5_20260101_000000" / "BDocs" / "finetuned_ocr"
    mpath = leaf / "manifest.json"
    rl.write_manifest(mpath, {"dataset": "BDocs", "config": "finetuned_ocr"})
    assert mpath.is_file()
    assert rl.read_manifest(mpath)["dataset"] == "BDocs"


def test_git_helpers_return_types():
    assert isinstance(rl.git_commit(), str) and rl.git_commit()
    assert isinstance(rl.git_dirty(), bool)


def test_append_summary_rows_writes_header_once():
    tmp = Path(tempfile.mkdtemp())
    rl.EVAL_RUNS_DIR = tmp
    run_id = "eval_val_5_20260101_000000"
    rl.append_summary_rows(run_id, [{"dataset": "BDocs", "config": "finetuned_ocr",
        "label": "Fine-Tuned (LoRA) · OCR", "model": "Qwen_2.5_7B_finetuned",
        "metric": "QUR", "complexity": "overall", "value": 0.77}])
    rl.append_summary_rows(run_id, [{"dataset": "BDocs", "config": "finetuned_ocr",
        "label": "Fine-Tuned (LoRA) · OCR", "model": "Qwen_2.5_7B_finetuned",
        "metric": "FRR", "complexity": "overall", "value": 0.10}])
    text = (tmp / run_id / "summary.csv").read_text()
    assert text.count("dataset,config,label,model,metric,complexity,value") == 1
    assert "QUR" in text and "FRR" in text


def test_update_latest_symlink():
    tmp = Path(tempfile.mkdtemp())
    rl.EVAL_RUNS_DIR = tmp
    (tmp / "eval_a").mkdir(parents=True)
    rl.update_latest_symlink("eval_a")
    link = tmp / "latest"
    assert link.is_symlink()
    assert (link.resolve()) == (tmp / "eval_a").resolve()
    # Re-pointing replaces the old target without error.
    (tmp / "eval_b").mkdir()
    rl.update_latest_symlink("eval_b")
    assert (link.resolve()) == (tmp / "eval_b").resolve()


if __name__ == "__main__":
    test_make_run_id_format()
    test_derive_mode()
    test_build_slug()
    test_human_label()
    test_paths_and_manifest_roundtrip()
    test_git_helpers_return_types()
    test_append_summary_rows_writes_header_once()
    test_update_latest_symlink()
    print("OK: config/run_layout.py (ids/slugs/labels)")
