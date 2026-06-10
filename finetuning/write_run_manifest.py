"""Write a run_manifest.json describing a fine-tuning experiment.

Standalone on purpose: this runs inside LLaMA-Factory's own venv (where the
repo's `config` package is NOT installed), so it depends only on the stdlib plus
PyYAML (a LLaMA-Factory dependency). It scrapes the training YAML config plus the
artifacts LLaMA-Factory leaves in output_dir (adapter_config.json, all_results.json,
trainer_state.json) and writes run_manifest.json into that output_dir, so the
existing copy-back carries it alongside the adapter.

Best-effort by contract: a fine-tuning run can take many hours, so manifest
generation must NEVER fail the job. Every read is guarded and any top-level
error is downgraded to a warning with exit code 0.

Usage:
    python finetuning/write_run_manifest.py --config <train.yaml> \
        [--output-dir <dir>] [--git-commit <sha>] [--git-dirty true|false]
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path


def _safe(fn, default=None):
    """Run fn(), returning default on any exception (best-effort scraping)."""
    try:
        return fn()
    except Exception:
        return default


def _load_yaml(path):
    import yaml  # provided by LLaMA-Factory's env
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _utc_now_iso():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(args, cwd):
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL
    ).strip()


def build_manifest(config_path, output_dir, git_commit=None, git_dirty=None):
    cfg = _safe(lambda: _load_yaml(config_path), {}) or {}
    output_dir = Path(output_dir or cfg.get("output_dir") or ".")

    adapter = _safe(lambda: _load_json(output_dir / "adapter_config.json"), {}) or {}
    results = _safe(lambda: _load_json(output_dir / "all_results.json"), {}) or {}
    trainer_state = _safe(lambda: _load_json(output_dir / "trainer_state.json"), {}) or {}

    # git: prefer values passed by the orchestrator (which can read the real repo
    # .git, not the rsync'd work-copy that excludes it); fall back to local git.
    if git_commit is None:
        git_commit = _safe(lambda: _git(["rev-parse", "--short", "HEAD"], str(Path(config_path).parent)), "unknown")
    if git_dirty is None:
        git_dirty = _safe(
            lambda: bool(_git(["status", "--porcelain"], str(Path(config_path).parent))), None
        )

    return {
        "experiment": output_dir.name,
        "created_at": _utc_now_iso(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "base_model": cfg.get("model_name_or_path") or adapter.get("base_model_name_or_path"),
        "method": {
            "finetuning_type": cfg.get("finetuning_type"),
            "stage": cfg.get("stage"),
            "peft_type": adapter.get("peft_type"),
            "lora_rank": adapter.get("r", cfg.get("lora_rank")),
            "lora_alpha": adapter.get("lora_alpha", cfg.get("lora_alpha")),
            "lora_dropout": adapter.get("lora_dropout", cfg.get("lora_dropout")),
            "lora_target": cfg.get("lora_target"),
            "num_target_modules": len(adapter.get("target_modules", []) or []),
        },
        "dataset": {
            "train": cfg.get("dataset"),
            "eval": cfg.get("eval_dataset"),
            "template": cfg.get("template"),
            "cutoff_len": cfg.get("cutoff_len"),
            "dataset_dir": cfg.get("dataset_dir"),
        },
        "hyperparams": {
            "learning_rate": cfg.get("learning_rate"),
            "num_train_epochs": cfg.get("num_train_epochs"),
            "per_device_train_batch_size": cfg.get("per_device_train_batch_size"),
            "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps"),
            "lr_scheduler_type": cfg.get("lr_scheduler_type"),
            "warmup_steps": cfg.get("warmup_steps"),
            "bf16": cfg.get("bf16"),
            "seed": cfg.get("seed", 42),  # LLaMA-Factory default when unset
        },
        "results": {
            "train_loss": results.get("train_loss"),
            "eval_loss": results.get("eval_loss"),
            "epoch": results.get("epoch"),
            "train_runtime_s": results.get("train_runtime"),
            "train_samples_per_second": results.get("train_samples_per_second"),
            "total_flos": results.get("total_flos"),
            "global_step": trainer_state.get("global_step"),
            "best_metric": trainer_state.get("best_metric"),
        },
        "config_path": str(config_path),
        "output_dir": str(output_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Write run_manifest.json for a fine-tuning run.")
    parser.add_argument("--config", required=True, help="Path to the LLaMA-Factory training YAML.")
    parser.add_argument("--output-dir", default=None, help="Training output_dir (default: read from the YAML).")
    parser.add_argument("--git-commit", default=None, help="Short SHA of the VRD-UQA repo (orchestrator-supplied).")
    parser.add_argument("--git-dirty", default=None, choices=["true", "false"], help="Repo dirty flag.")
    args = parser.parse_args()

    git_dirty = {"true": True, "false": False}.get(args.git_dirty) if args.git_dirty else None
    manifest = build_manifest(args.config, args.output_dir, args.git_commit, git_dirty)

    output_dir = Path(manifest["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "run_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote fine-tuning run manifest -> {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # never fail the training job over a manifest
        print(f"WARNING: run_manifest generation failed (non-fatal): {e}", file=sys.stderr)
        sys.exit(0)
