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
import os
import subprocess
from pathlib import Path

from config.paths import REPO_ROOT

# Default mirrors the repo layout; VQA_EVAL_RUNS_DIR wins at runtime (same
# env-override pattern as config/paths.py SCRATCH_FLASH) so the evaluator and
# every metrics step can be redirected to a scratch/temp tree in lockstep.
EVAL_RUNS_DIR = Path(
    os.getenv("VQA_EVAL_RUNS_DIR", str(REPO_ROOT / "artifacts" / "evaluation_runs"))
)

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


def build_slug(mode: str, ocr_enabled: bool, window_size: int = 1,
               few_shot: dict | None = None) -> str:
    slug = f"{mode}_ocr" if ocr_enabled else f"{mode}_noocr"
    if window_size and window_size != 1:
        slug += f"_w{window_size}"
    if mode == "fewshot" and few_shot and few_shot.get("enabled"):
        k = few_shot.get("n_shots", 0)
        shot_type = few_shot.get("shot_type", "mixed")
        selection = few_shot.get("selection", "random")
        slug += f"_k{k}_{shot_type}_{selection}"
    return slug


def human_label(manifest: dict) -> str:
    mode = (manifest.get("config", "") or "").split("_")[0]
    label = MODE_LABELS.get(mode, mode or "?")
    label += " · OCR" if manifest.get("ocr_enabled") else " · no-OCR"
    w = manifest.get("window_size", 1)
    if w and w != 1:
        label += f" · w{w}"
    if mode == "fewshot":
        fs = manifest.get("few_shot") or {}
        if fs.get("enabled"):
            k = fs.get("n_shots", 0)
            sel = fs.get("selection", "random")
            label += f" · k{k}-{sel}"
    return label


def run_dir(run_id: str) -> Path:
    return EVAL_RUNS_DIR / run_id


def leaf_dir(run_id: str, dataset: str, slug: str, model_prefix: str = "") -> Path:
    # model_prefix lets >1 model share a run without colliding on one leaf.
    # "" preserves the historical bare path (Qwen), so existing artifacts and
    # in-flight k-sweep resume are undisturbed.
    name = f"{model_prefix}__{slug}" if model_prefix else slug
    return run_dir(run_id) / dataset / name


def write_manifest(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def read_manifest(path: Path | str) -> dict:
    with open(path) as f:
        return json.load(f)


def _git(args: list[str]) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(REPO_ROOT), text=True, stderr=subprocess.DEVNULL
    ).strip()


def git_commit() -> str:
    try:
        return _git(["rev-parse", "--short", "HEAD"])
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(_git(["status", "--porcelain"]))
    except Exception:
        return False


def append_summary_rows(run_id: str, rows: list[dict]) -> None:
    # Single writer per run (one step-3 metrics process), so the header-once
    # check needs no cross-process lock.
    path = run_dir(run_id) / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if not exists:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in SUMMARY_COLUMNS})


def update_latest_symlink(run_id: str) -> None:
    EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    link = EVAL_RUNS_DIR / "latest"
    # missing_ok swallows only ENOENT (already gone); a real failure
    # (e.g. permission/stale handle on BeeGFS) propagates with its true cause
    # instead of surfacing as a misleading FileExistsError below.
    link.unlink(missing_ok=True)
    link.symlink_to(run_id)  # relative target inside EVAL_RUNS_DIR
