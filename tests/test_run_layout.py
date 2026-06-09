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


if __name__ == "__main__":
    test_make_run_id_format()
    test_derive_mode()
    test_build_slug()
    test_human_label()
    print("OK: config/run_layout.py (ids/slugs/labels)")
