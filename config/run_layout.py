"""Single source of truth for the evaluation-run artifact layout.

Imported by the evaluator, every metrics step, and the plotting notebook so the
folder structure and manifests are defined in exactly one place. Mirrors the
packaging intent in pyproject.toml: `config` is installed editable, so this
imports cleanly from SLURM by-path scripts, `-m`, and notebooks.
"""
from __future__ import annotations

import csv
import datetime
import json
import subprocess
from pathlib import Path

from config.paths import REPO_ROOT

EVAL_RUNS_DIR = REPO_ROOT / "artifacts" / "evaluation_runs"

MODE_LABELS = {
    "zeroshot": "Zero-Shot",
    "fewshot": "Few-Shot",
    "finetuned": "Fine-Tuned (LoRA)",
}

SUMMARY_COLUMNS = ["dataset", "config", "label", "model", "metric", "complexity", "value"]


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id(split: str, n: int, when: datetime.datetime | None = None) -> str:
    # UTC default keeps the run_id timestamp consistent with utc_now_iso()
    # (used for created_at) regardless of the node's local timezone.
    when = when or datetime.datetime.now(datetime.timezone.utc)
    return f"eval_{split}_{n}_{when:%Y%m%d_%H%M%S}"


def derive_mode(finetuned: bool, few_shot_enabled: bool) -> str:
    if finetuned:
        return "finetuned"
    if few_shot_enabled:
        return "fewshot"
    return "zeroshot"


def build_slug(mode: str, ocr_enabled: bool, window_size: int = 1) -> str:
    slug = f"{mode}_ocr" if ocr_enabled else f"{mode}_noocr"
    if window_size and window_size != 1:
        slug += f"_w{window_size}"
    return slug


def human_label(manifest: dict) -> str:
    mode = (manifest.get("config", "") or "").split("_")[0]
    label = MODE_LABELS.get(mode, mode or "?")
    label += " · OCR" if manifest.get("ocr_enabled") else " · no-OCR"
    w = manifest.get("window_size", 1)
    if w and w != 1:
        label += f" · w{w}"
    return label
