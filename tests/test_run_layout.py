"""Validates config/run_layout.py. Run: uv run python -m tests.test_run_layout"""
import contextlib
import datetime
import tempfile
from pathlib import Path
from config import run_layout as rl


@contextlib.contextmanager
def _tmp_runs_dir():
    """Point rl.EVAL_RUNS_DIR at a fresh tempdir, restoring it afterwards so
    tests stay hermetic regardless of order (e.g. under a future pytest)."""
    orig = rl.EVAL_RUNS_DIR
    tmp = Path(tempfile.mkdtemp())
    rl.EVAL_RUNS_DIR = tmp
    try:
        yield tmp
    finally:
        rl.EVAL_RUNS_DIR = orig


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
    # few_shot param encodes k / shot_type / selection into the slug
    fs_rand = {"enabled": True, "n_shots": 2, "shot_type": "mixed", "selection": "random"}
    assert rl.build_slug("fewshot", ocr_enabled=True, window_size=1, few_shot=fs_rand) == "fewshot_ocr_k2_mixed_random"
    fs_spec = {"enabled": True, "n_shots": 4, "shot_type": "mixed", "selection": "specific"}
    assert rl.build_slug("fewshot", ocr_enabled=True, window_size=1, few_shot=fs_spec) == "fewshot_ocr_k4_mixed_specific"
    # Non-fewshot modes ignore the few_shot arg
    assert rl.build_slug("zeroshot", ocr_enabled=True, window_size=1, few_shot=fs_rand) == "zeroshot_ocr"
    # Disabled few_shot block produces no suffix
    assert rl.build_slug("fewshot", ocr_enabled=True, window_size=1, few_shot={"enabled": False}) == "fewshot_ocr"


def test_human_label():
    m = {"config": "finetuned_ocr", "ocr_enabled": True, "window_size": 1}
    assert rl.human_label(m) == "Fine-Tuned (LoRA) · OCR"
    m2 = {"config": "zeroshot_noocr", "ocr_enabled": False, "window_size": 1}
    assert rl.human_label(m2) == "Zero-Shot · no-OCR"
    # Unknown / missing mode falls back to "?".
    assert rl.human_label({}) == "? · no-OCR"
    # Few-shot manifests include k and selection in the label
    m3 = {
        "config": "fewshot_ocr_k2_mixed_specific",
        "ocr_enabled": True, "window_size": 1,
        "few_shot": {"enabled": True, "n_shots": 2, "shot_type": "mixed", "selection": "specific"},
    }
    assert rl.human_label(m3) == "Few-Shot · OCR · k2-specific"
    m4 = {
        "config": "fewshot_ocr_k4_mixed_random",
        "ocr_enabled": True, "window_size": 1,
        "few_shot": {"enabled": True, "n_shots": 4, "shot_type": "mixed", "selection": "random"},
    }
    assert rl.human_label(m4) == "Few-Shot · OCR · k4-random"


def test_paths_and_manifest_roundtrip():
    with _tmp_runs_dir() as tmp:
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
    with _tmp_runs_dir() as tmp:
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
    with _tmp_runs_dir() as tmp:
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
    print("OK: config/run_layout.py")
