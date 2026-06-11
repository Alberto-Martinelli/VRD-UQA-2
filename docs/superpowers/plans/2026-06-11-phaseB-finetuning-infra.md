# Phase B — LoRA Fine-tuning Infrastructure for the 3 New Models — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LLaMA-Factory LoRA SFT configs + a parallel-safe SLURM launcher so Llama-3.2-Vision, InternVL3.5-8B, and Phi-4-multimodal can each be fine-tuned (3 sbatch jobs in parallel, no collision), reusing the existing Qwen dataset and hyperparameters, and wire the resulting adapters into the evaluators.

**Architecture:** Each model gets a YAML that is a verbatim copy of `finetuning/qwen25vl_lora_sft.yaml` except `model_name_or_path` / `template` / `output_dir`. A generalized `run_finetune_vlm.sh <model_key>` clones LLaMA-Factory into a **per-job** scratch dir (model + `$SLURM_JOB_ID`) with its own venv and HF cache, trains, writes a run manifest, and copies the adapter back to `artifacts/finetuning/<model_key>_lora_sft`. The evaluators consume adapters via the `*_finetuned` config entries already referenced by each subclass's `FINETUNED_MODEL_KEY` (Phase A).

**Tech Stack:** LLaMA-Factory (`llamafactory-cli train`), LoRA/PEFT, SLURM/bash, the model-agnostic dataset `vrd_uqa_train` / `vrd_uqa_val`.

**Spec:** [docs/superpowers/specs/2026-06-11-multimodel-evaluators-and-finetuning-design.md](../specs/2026-06-11-multimodel-evaluators-and-finetuning-design.md) (Phase B).

**Depends on:** Phase A (the subclasses define `FINETUNED_MODEL_KEY`; the base `_load_model` paths consume `adapter_path` / merged `model_name`).

---

## File Structure

| File | Responsibility |
|---|---|
| `finetuning/llama32vl_lora_sft.yaml` / `internvl35_lora_sft.yaml` / `phi4mm_lora_sft.yaml` | **New.** Full-run LoRA configs (verbatim Qwen params; model/template/output_dir differ). |
| `finetuning/llama32vl_lora_smoke.yaml` / `internvl35_lora_smoke.yaml` / `phi4mm_lora_smoke.yaml` | **New.** Fast smoke configs (mirror `qwen25vl_lora_smoke.yaml`). |
| `scripts/slurm/run_finetune_vlm.sh` | **New.** Generalized, parallel-safe finetune launcher (`<model_key>`). |
| `finetuning/merge_lora.py` | **New.** Merge a LoRA adapter into base weights (fallback for Phi-4 / InternVL eval). |
| `finetuning/phi4mm_official/README.md` | **New.** Phi-4-mm official-sample fallback scaffold + decision note. |
| `VQA_analysis/config_*.json` | **Modify.** Add `llama3.2_finetuned` / `phi4_finetuned` / `internvl3_5_finetuned` entries. |
| `tests/test_finetune_configs.py` | **New.** Validate each new YAML (dataset registered, verbatim hyperparams). |
| `finetuning/readme.txt` / `sbatch_commands.txt` | **Modify.** Parallel finetune launch docs. |

The existing `scripts/slurm/run_finetune_qwen25vl.sh` is **left untouched** (Qwen keeps its working path); `run_finetune_vlm.sh qwen25vl` is also available for uniform parallel runs.

The dataset registration `finetuning/dataset_info.json` (`vrd_uqa_train` / `vrd_uqa_val`) is **reused unchanged** — the training data is model-agnostic.

---

## Task 1: Full-run LoRA YAMLs for the 3 models

**Files:**
- Create: `finetuning/llama32vl_lora_sft.yaml`, `finetuning/internvl35_lora_sft.yaml`, `finetuning/phi4mm_lora_sft.yaml`

> Every hyperparameter is identical to `finetuning/qwen25vl_lora_sft.yaml`. Only `model_name_or_path`, `template`, and `output_dir` change. `dataset_dir` is a default that the launcher overrides per job (Task 3).

- [ ] **Step 1: Write `finetuning/llama32vl_lora_sft.yaml`**

```yaml
# LLaMA-Factory training config: Llama-3.2-11B-Vision-Instruct LoRA SFT (full run)
# Hyperparameters are identical to qwen25vl_lora_sft.yaml; only model/template/output_dir differ.
# dataset_dir is overridden per job by run_finetune_vlm.sh.
### model
model_name_or_path: meta-llama/Llama-3.2-11B-Vision-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 32

### dataset
dataset: vrd_uqa_train
eval_dataset: vrd_uqa_val
dataset_dir: /mnt/beegfs/amartinelli/finetune_llama32vl/VRD-UQA/finetuning
template: mllama
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: /mnt/beegfs/amartinelli/finetune_out/llama32vl_lora_sft
logging_steps: 10
eval_strategy: steps
eval_steps: 100
save_strategy: steps
save_steps: 100
save_total_limit: 3
plot_loss: true
overwrite_output_dir: true
report_to: none
load_best_model_at_end: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_steps: 50
bf16: true
ddp_timeout: 180000000
```

- [ ] **Step 2: Write `finetuning/internvl35_lora_sft.yaml`** (identical except the 3 keys)

```yaml
# LLaMA-Factory training config: InternVL3.5-8B LoRA SFT (full run)
# Hyperparameters identical to qwen25vl_lora_sft.yaml; only model/template/output_dir differ.
# dataset_dir is overridden per job by run_finetune_vlm.sh.
### model
model_name_or_path: OpenGVLab/InternVL3_5-8B
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 32

### dataset
dataset: vrd_uqa_train
eval_dataset: vrd_uqa_val
dataset_dir: /mnt/beegfs/amartinelli/finetune_internvl35/VRD-UQA/finetuning
template: intern_vl
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: /mnt/beegfs/amartinelli/finetune_out/internvl35_lora_sft
logging_steps: 10
eval_strategy: steps
eval_steps: 100
save_strategy: steps
save_steps: 100
save_total_limit: 3
plot_loss: true
overwrite_output_dir: true
report_to: none
load_best_model_at_end: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_steps: 50
bf16: true
ddp_timeout: 180000000
```

- [ ] **Step 3: Write `finetuning/phi4mm_lora_sft.yaml`** (identical except the 3 keys)

```yaml
# LLaMA-Factory training config: Phi-4-multimodal-instruct LoRA SFT (full run)
# Hyperparameters identical to qwen25vl_lora_sft.yaml; only model/template/output_dir differ.
# dataset_dir is overridden per job by run_finetune_vlm.sh.
# NOTE: the `phi4` template + Phi-4-mm vision finetuning support must be confirmed against
# the fresh LLaMA-Factory clone (Task 3 Step 4). If LF cannot train its vision path, use the
# Microsoft official-sample fallback (Task 7) instead of this YAML.
### model
model_name_or_path: microsoft/Phi-4-multimodal-instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 32

### dataset
dataset: vrd_uqa_train
eval_dataset: vrd_uqa_val
dataset_dir: /mnt/beegfs/amartinelli/finetune_phi4mm/VRD-UQA/finetuning
template: phi4
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: /mnt/beegfs/amartinelli/finetune_out/phi4mm_lora_sft
logging_steps: 10
eval_strategy: steps
eval_steps: 100
save_strategy: steps
save_steps: 100
save_total_limit: 3
plot_loss: true
overwrite_output_dir: true
report_to: none
load_best_model_at_end: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_steps: 50
bf16: true
ddp_timeout: 180000000
```

- [ ] **Step 4: Commit**

```bash
git add finetuning/llama32vl_lora_sft.yaml finetuning/internvl35_lora_sft.yaml finetuning/phi4mm_lora_sft.yaml
git commit -m "feat(finetune): full-run LoRA YAMLs for llama32vl / internvl35 / phi4mm"
```

---

## Task 2: Smoke YAMLs (fast config validation)

**Files:**
- Create: `finetuning/llama32vl_lora_smoke.yaml`, `finetuning/internvl35_lora_smoke.yaml`, `finetuning/phi4mm_lora_smoke.yaml`

> Mirror `qwen25vl_lora_smoke.yaml`: same model as the full run, `dataset: vrd_uqa_smoke`, `max_samples: 50`, tiny logging/save cadence — just enough to confirm the config + template load and one training loop runs.

- [ ] **Step 1: Write `finetuning/llama32vl_lora_smoke.yaml`**

```yaml
# LLaMA-Factory smoke test: Llama-3.2-11B-Vision-Instruct LoRA (fast config check)
### model
model_name_or_path: meta-llama/Llama-3.2-11B-Vision-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset: vrd_uqa_smoke
dataset_dir: /mnt/beegfs/amartinelli/finetune_llama32vl/VRD-UQA/finetuning
template: mllama
cutoff_len: 4096
max_samples: 50
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: /mnt/beegfs/amartinelli/finetune_out/llama32vl_lora_smoke
logging_steps: 5
save_steps: 200
plot_loss: true
overwrite_output_dir: true
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
ddp_timeout: 180000000
```

- [ ] **Step 2: Write `finetuning/internvl35_lora_smoke.yaml`** (same as Step 1 with model `OpenGVLab/InternVL3_5-8B`, `template: intern_vl`, `dataset_dir: …/finetune_internvl35/…`, `output_dir: …/internvl35_lora_smoke`)

```yaml
# LLaMA-Factory smoke test: InternVL3.5-8B LoRA (fast config check)
### model
model_name_or_path: OpenGVLab/InternVL3_5-8B
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset: vrd_uqa_smoke
dataset_dir: /mnt/beegfs/amartinelli/finetune_internvl35/VRD-UQA/finetuning
template: intern_vl
cutoff_len: 4096
max_samples: 50
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: /mnt/beegfs/amartinelli/finetune_out/internvl35_lora_smoke
logging_steps: 5
save_steps: 200
plot_loss: true
overwrite_output_dir: true
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
ddp_timeout: 180000000
```

- [ ] **Step 3: Write `finetuning/phi4mm_lora_smoke.yaml`** (model `microsoft/Phi-4-multimodal-instruct`, `template: phi4`, `dataset_dir: …/finetune_phi4mm/…`, `output_dir: …/phi4mm_lora_smoke`)

```yaml
# LLaMA-Factory smoke test: Phi-4-multimodal-instruct LoRA (fast config check)
### model
model_name_or_path: microsoft/Phi-4-multimodal-instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset: vrd_uqa_smoke
dataset_dir: /mnt/beegfs/amartinelli/finetune_phi4mm/VRD-UQA/finetuning
template: phi4
cutoff_len: 4096
max_samples: 50
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: /mnt/beegfs/amartinelli/finetune_out/phi4mm_lora_smoke
logging_steps: 5
save_steps: 200
plot_loss: true
overwrite_output_dir: true
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
ddp_timeout: 180000000
```

- [ ] **Step 4: Commit**

```bash
git add finetuning/llama32vl_lora_smoke.yaml finetuning/internvl35_lora_smoke.yaml finetuning/phi4mm_lora_smoke.yaml
git commit -m "feat(finetune): smoke LoRA YAMLs for the 3 new models"
```

---

## Task 3: Generalized, parallel-safe `run_finetune_vlm.sh`

**Files:**
- Create: `scripts/slurm/run_finetune_vlm.sh`

> Based on `run_finetune_qwen25vl.sh`, with every shared writable resource made per-job-unique so 3 jobs run concurrently without collision (spec B.3.1): per-job `WORK_DIR` (model + `$SLURM_JOB_ID`), per-job LF clone + venv, per-job `HF_HOME`, `dataset_dir` overridden to the per-job repo copy, and the slurm-log move scoped to `$SLURM_JOB_ID`.

- [ ] **Step 1: Write `scripts/slurm/run_finetune_vlm.sh`**

```bash
#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-16:00:00
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-finetune-%j.out

# Parallel-safe LoRA fine-tuning for one model per job.
# Usage:
#   sbatch run_finetune_vlm.sh <model_key> [smoke]
#   <model_key> = qwen25vl | llama32vl | internvl35 | phi4mm
# Run 3 in parallel (no collision):
#   for M in llama32vl internvl35 phi4mm; do sbatch --job-name=ft-$M run_finetune_vlm.sh $M; done

set -euo pipefail
module purge
module load miniconda3/3.13.25
module load nvhpc/25.1
source "$HOME/VRD-UQA/scripts/env.sh"
export UV_LINK_MODE=copy

MODEL_KEY="${1:?usage: run_finetune_vlm.sh <model_key> [smoke]}"
VARIANT="${2:-sft}"   # sft (full) | smoke
case "$MODEL_KEY" in
  qwen25vl|llama32vl|internvl35|phi4mm) : ;;
  *) echo "ERROR: unknown model_key '$MODEL_KEY'"; exit 2 ;;
esac
CONFIG_NAME="${MODEL_KEY}_lora_${VARIANT}.yaml"

# ---- Per-job scratch (model + job id) => no cross-job collision ----
WORK_DIR="$SCRATCH_FLASH/finetune_${MODEL_KEY}_${SLURM_JOB_ID}"
LF_DIR="$WORK_DIR/LLaMA-Factory"
VENV_DIR="$WORK_DIR/.venv"
export HF_HOME="$WORK_DIR/hf_home"     # isolates the tokenized/dataset cache per job
mkdir -p "$WORK_DIR"
trap 'cd "$HOME"; rm -rf "$WORK_DIR"' EXIT

# ---- Sync repo (for finetuning configs + dataset registration) ----
rsync -aq --exclude='.git' --exclude='.venv' --exclude='data' \
      --exclude='corruption-scripts/results' \
      "$HOME/VRD-UQA/" "$WORK_DIR/VRD-UQA/"

# ---- Clone LLaMA-Factory (per job) ----
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "$LF_DIR"

# ---- Python env via uv (per job) ----
cd "$LF_DIR"
uv venv --python 3.11 "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
uv pip install --upgrade pip
uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121
uv pip install -e ".[torch,metrics]"
uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121
# Model-specific vision deps
case "$MODEL_KEY" in
  qwen25vl) uv pip install qwen-vl-utils ;;
  *)        uv pip install qwen-vl-utils ;;  # harmless; LF imports it lazily
esac

# ---- Train (override dataset_dir to THIS job's repo copy) ----
CONFIG="$WORK_DIR/VRD-UQA/finetuning/$CONFIG_NAME"
DATASET_DIR="$WORK_DIR/VRD-UQA/finetuning"
echo "Using config: $CONFIG (dataset_dir=$DATASET_DIR)"
cat "$CONFIG"
llamafactory-cli train "$CONFIG" dataset_dir="$DATASET_DIR"

# ---- Run manifest (read output_dir straight from the YAML) ----
OUTPUT_DIR="$(sed -nE 's/^output_dir:[[:space:]]*//p' "$CONFIG" | tr -d "\"'")"
GIT_COMMIT="$(cd "$HOME/VRD-UQA" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ -n "$(cd "$HOME/VRD-UQA" && git status --porcelain 2>/dev/null)" ]; then GIT_DIRTY=true; else GIT_DIRTY=false; fi
python "$WORK_DIR/VRD-UQA/finetuning/write_run_manifest.py" \
    --config "$CONFIG" --output-dir "$OUTPUT_DIR" \
    --git-commit "$GIT_COMMIT" --git-dirty "$GIT_DIRTY" || true

# ---- Copy adapter back to a per-model dest (distinct across the 3 models) ----
DEST="$HOME/VRD-UQA/artifacts/finetuning/${MODEL_KEY}_lora_${VARIANT}"
if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
    mkdir -p "$DEST"
    rsync -av --exclude='checkpoint-*' "$OUTPUT_DIR/" "$DEST/"
    echo "Copied trained adapter -> $DEST"
else
    echo "WARNING: output_dir '$OUTPUT_DIR' not found; nothing copied back." >&2
fi

# ---- Per-job slurm log move (scoped to THIS job id, never a sibling's) ----
mv "$HOME"/slurm-finetune-${SLURM_JOB_ID}.out "$HOME/VRD-UQA/" 2>/dev/null || true
```

- [ ] **Step 2: Syntax-check**

Run: `cd /home/amartinelli/VRD-UQA && bash -n scripts/slurm/run_finetune_vlm.sh && echo "syntax ok"`
Expected: `syntax ok`

- [ ] **Step 3: Commit**

```bash
git add scripts/slurm/run_finetune_vlm.sh
git commit -m "feat(finetune): parallel-safe generalized LoRA finetune launcher"
```

- [ ] **Step 4 (HPC, manual): Smoke each model, verify template support**

```bash
for M in llama32vl internvl35 phi4mm; do
  sbatch --job-name=ftsmoke-$M scripts/slurm/run_finetune_vlm.sh $M smoke
done
```
Expected: each produces `artifacts/finetuning/<M>_lora_smoke/adapter_config.json` + `run_manifest.json`.
If `phi4mm` errors on the `phi4` template / vision path, **switch Phi-4 to the official-sample fallback (Task 7)**.

---

## Task 4: Validate the new YAMLs (config test)

**Files:**
- Create: `tests/test_finetune_configs.py`

> A flat `key: value` YAML parser (no PyYAML dependency) checks each new full YAML: dataset is registered in `dataset_info.json`, and the comparison hyperparameters match the Qwen reference verbatim.

- [ ] **Step 1: Write the failing test**

```python
"""Validate the new LoRA YAMLs against dataset_info + the Qwen reference.
Run: uv run python -m tests.test_finetune_configs"""
import json
from config.paths import REPO_ROOT

FT = REPO_ROOT / "finetuning"
NEW_FULL = ["llama32vl_lora_sft.yaml", "internvl35_lora_sft.yaml", "phi4mm_lora_sft.yaml"]
EXPECTED = {  # (model_name_or_path, template, output_dir basename)
    "llama32vl_lora_sft.yaml": ("meta-llama/Llama-3.2-11B-Vision-Instruct", "mllama", "llama32vl_lora_sft"),
    "internvl35_lora_sft.yaml": ("OpenGVLab/InternVL3_5-8B", "intern_vl", "internvl35_lora_sft"),
    "phi4mm_lora_sft.yaml": ("microsoft/Phi-4-multimodal-instruct", "phi4", "phi4mm_lora_sft"),
}
# Verbatim-from-Qwen comparison hyperparameters
SHARED = {
    "stage": "sft", "finetuning_type": "lora", "lora_target": "all",
    "lora_rank": "16", "lora_alpha": "32", "cutoff_len": "4096",
    "dataset": "vrd_uqa_train", "eval_dataset": "vrd_uqa_val",
    "per_device_train_batch_size": "1", "gradient_accumulation_steps": "8",
    "learning_rate": "2.0e-5", "num_train_epochs": "1.0",
    "lr_scheduler_type": "cosine", "warmup_steps": "50", "bf16": "true",
    "save_total_limit": "3", "load_best_model_at_end": "true",
}


def _parse_flat_yaml(path):
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, val = line.partition(":")
        out[key.strip()] = val.strip()
    return out


def test_new_yamls_valid_and_match_qwen_params():
    registered = set(json.loads((FT / "dataset_info.json").read_text()).keys())
    for fname in NEW_FULL:
        cfg = _parse_flat_yaml(FT / fname)
        model, template, out_base = EXPECTED[fname]
        assert cfg["model_name_or_path"] == model, fname
        assert cfg["template"] == template, fname
        assert cfg["output_dir"].rstrip("/").split("/")[-1] == out_base, fname
        assert cfg["dataset"] in registered and cfg["eval_dataset"] in registered, fname
        for k, v in SHARED.items():
            assert cfg.get(k) == v, f"{fname}: {k}={cfg.get(k)!r} expected {v!r}"


if __name__ == "__main__":
    test_new_yamls_valid_and_match_qwen_params()
    print("OK: finetune configs")
```

- [ ] **Step 2: Run it (passes once Task 1 YAMLs exist)**

Run: `cd /home/amartinelli/VRD-UQA && uv run python -m tests.test_finetune_configs`
Expected: `OK: finetune configs`. (If run before Task 1, it FAILS with `FileNotFoundError` — confirming it actually checks the files.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_finetune_configs.py
git commit -m "test(finetune): validate new LoRA YAMLs vs dataset_info + Qwen params"
```

---

## Task 5: Wire `*_finetuned` entries into the eval configs

**Files:**
- Modify: `VQA_analysis/config_zeroshot.json`, `config_fewshot.json`, `config_mock.json`

> Each subclass's `FINETUNED_MODEL_KEY` (Phase A) points here. Default to PEFT-attach (`adapter_path` + base `model_name`). For Phi-4 / InternVL, if PEFT-attach fails at eval, switch that entry to a merged checkpoint (Task 6): set `model_name` to the merged dir and remove `adapter_path`.

- [ ] **Step 1: Add three entries to each config's `open_source_models`**

```json
        "llama3.2_finetuned": {
            "model_name": "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "adapter_path": "/home/amartinelli/VRD-UQA/artifacts/finetuning/llama32vl_lora_sft",
            "batch_size": 1,
            "max_tokens": 1024,
            "use_flash_attention": true,
            "name": "Llama3.2-11B_finetuned"
        },
        "phi4_finetuned": {
            "model_name": "microsoft/Phi-4-multimodal-instruct",
            "adapter_path": "/home/amartinelli/VRD-UQA/artifacts/finetuning/phi4mm_lora_sft",
            "batch_size": 1,
            "max_tokens": 1024,
            "use_flash_attention": true,
            "name": "Phi4-multimodal_finetuned"
        },
        "internvl3_5_finetuned": {
            "model_name": "OpenGVLab/InternVL3_5-8B",
            "adapter_path": "/home/amartinelli/VRD-UQA/artifacts/finetuning/internvl35_lora_sft",
            "batch_size": 1,
            "max_tokens": 1024,
            "max_tiles": 12,
            "input_size": 448,
            "use_flash_attention": true,
            "name": "InternVL3.5-8B_finetuned"
        }
```

- [ ] **Step 2: Verify all three configs still parse and expose the keys**

Run: `cd /home/amartinelli/VRD-UQA && uv run python -c "import json; [print(p, all(k in json.load(open(p))['open_source_models'] for k in ['llama3.2_finetuned','phi4_finetuned','internvl3_5_finetuned'])) for p in ['VQA_analysis/config_zeroshot.json','VQA_analysis/config_fewshot.json','VQA_analysis/config_mock.json']]"`
Expected: each line ends with `True`.

- [ ] **Step 3: Commit**

```bash
git add VQA_analysis/config_zeroshot.json VQA_analysis/config_fewshot.json VQA_analysis/config_mock.json
git commit -m "feat(config): finetuned eval entries for llama/phi4/internvl adapters"
```

---

## Task 6: LoRA-merge helper (eval fallback for custom-arch models)

**Files:**
- Create: `finetuning/merge_lora.py`

> For Phi-4 / InternVL, attaching a PEFT adapter onto the `trust_remote_code` class can be fragile. This helper exports a **merged** checkpoint so eval loads a plain model (set the `*_finetuned` entry's `model_name` to the merged dir, drop `adapter_path`).

- [ ] **Step 1: Write `finetuning/merge_lora.py`**

```python
"""Merge a LoRA adapter into its base model and save a standalone checkpoint.

Eval fallback for models whose custom (trust_remote_code) classes don't cleanly
accept a PEFT adapter at load time (Phi-4-multimodal, InternVL3.5). After merging,
point the model_<...>_finetuned config entry at --out and remove its adapter_path.

Usage:
  python finetuning/merge_lora.py --base <hf_id_or_path> --adapter <adapter_dir> --out <dir>
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from peft import PeftModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print(f"Loading base {args.base} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu")
    print(f"Attaching adapter {args.adapter} ...")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("Merging ...")
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)

    # Save tokenizer/processor alongside so the merged dir loads standalone.
    for loader in (AutoProcessor, AutoTokenizer):
        try:
            loader.from_pretrained(args.base, trust_remote_code=True).save_pretrained(args.out)
            break
        except Exception as e:
            print(f"({loader.__name__} skipped: {e})")
    print(f"Merged checkpoint written to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it imports / shows help**

Run: `cd /home/amartinelli/VRD-UQA && uv run python finetuning/merge_lora.py --help`
Expected: argparse usage text listing `--base --adapter --out`.

- [ ] **Step 3: Commit**

```bash
git add finetuning/merge_lora.py
git commit -m "feat(finetune): LoRA merge helper for custom-arch eval fallback"
```

---

## Task 7: Phi-4-multimodal official-sample fallback scaffold

**Files:**
- Create: `finetuning/phi4mm_official/README.md`

> Only used if Task 3 Step 4 shows LLaMA-Factory can't train Phi-4-mm's vision path. Documents the decision and the isolated path so the adapter is still consumable at eval (Task 5 / Task 6).

- [ ] **Step 1: Write `finetuning/phi4mm_official/README.md`**

```markdown
# Phi-4-multimodal fine-tuning — official-sample fallback

Use this path ONLY if `run_finetune_vlm.sh phi4mm smoke` (LLaMA-Factory, `phi4`
template) fails to train Phi-4-multimodal's vision path. Phi-4-mm ships internal
speech/vision LoRA adapters, which LLaMA-Factory may not handle.

## Procedure
1. Clone Microsoft's Phi-4-multimodal finetuning sample (from the model card /
   `microsoft/Phi-4-multimodal-instruct` repo) into this directory.
2. Convert `artifacts/finetuning/dataset/train.json` (+ `val.json`) — the same
   model-agnostic VRD-UQA SFT data registered as `vrd_uqa_train`/`vrd_uqa_val` —
   into the sample's expected format (image path + prompt + target).
3. Train a LoRA on the language path (keep the vision adapter frozen), batch_size 1,
   matching the LoRA rank/alpha/lr from `phi4mm_lora_sft.yaml` (16 / 32 / 2e-5) as
   closely as the sample allows.
4. Export the result so eval can consume it:
   - If it produces a PEFT adapter → copy to `artifacts/finetuning/phi4mm_lora_sft`
     and keep the eval `phi4_finetuned` entry as-is (adapter_path).
   - Else merge with `finetuning/merge_lora.py` → set `phi4_finetuned.model_name`
     to the merged dir and remove its `adapter_path`.

## Decision log
- [ ] LLaMA-Factory `phi4` template result: __pass / fail__ (fill after Task 3 Step 4)
- [ ] Fallback used: __yes / no__
```

- [ ] **Step 2: Commit**

```bash
git add finetuning/phi4mm_official/README.md
git commit -m "docs(finetune): Phi-4-mm official-sample fallback scaffold"
```

---

## Task 8: Parallel finetune launch docs

**Files:**
- Modify: `finetuning/readme.txt`, `sbatch_commands.txt`

- [ ] **Step 1: Append to `finetuning/readme.txt`**

```text

--- Multi-model LoRA fine-tuning (parallel, one model per job) ---
Run all three in parallel (no collision; each uses its own scratch dir + venv + HF cache):
    for M in llama32vl internvl35 phi4mm; do
        sbatch --job-name=ft-$M scripts/slurm/run_finetune_vlm.sh $M
    done
Smoke first (fast config check): append 'smoke' ->  run_finetune_vlm.sh <M> smoke
Adapters land in: artifacts/finetuning/<M>_lora_sft/   (consumed by the *_finetuned eval entries)
Qwen still uses its original script: sbatch scripts/slurm/run_finetune_qwen25vl.sh
```

- [ ] **Step 2: Append to `sbatch_commands.txt`**

```text

# --- LoRA fine-tuning the 3 new models in parallel ---
for M in llama32vl internvl35 phi4mm; do
  sbatch --job-name=ft-$M scripts/slurm/run_finetune_vlm.sh $M
done
# Then evaluate fine-tuned models (Phase A launcher + FINETUNE flag):
#   FINETUNE=--finetuned CONFIG=VQA_analysis/config_zeroshot.json \
#     sbatch scripts/slurm/run_vqa_analysis.sh llama BDocs val_100
```

- [ ] **Step 3: Commit**

```bash
git add finetuning/readme.txt sbatch_commands.txt
git commit -m "docs(finetune): parallel multi-model launch instructions"
```

---

## Self-Review Notes (author)

- **Spec coverage:** B.1 approach → all tasks; B.2 verbatim YAMLs → T1 (+T4 enforces verbatim params); B.3 generalized script → T3; B.3.1 parallel safety (per-job WORK_DIR/clone/venv/HF_HOME, scoped log move, dataset_dir override) → T3; B.4 Phi-4 fallback → T3 Step 4 + T7; B.5 adapter consumption (attach default, merge fallback) → T5 + T6; B.6 smoke YAMLs + config test → T2 + T4.
- **Cross-plan consistency:** the `*_finetuned` config keys (T5) match the `FINETUNED_MODEL_KEY` values set in Phase A subclasses (`llama3.2_finetuned`, `phi4_finetuned`, `internvl3_5_finetuned`); adapter dest dirs `artifacts/finetuning/<model_key>_lora_sft` match the `adapter_path` values.
- **Non-disruptive:** `run_finetune_qwen25vl.sh` and `dataset_info.json` are untouched; the Qwen finetuning path keeps working.
- **Manual/HPC steps:** real training + the `phi4` template verification are HPC jobs (T3 Step 4, T4 note), not CI — the CI-runnable checks are the YAML/config tests (T4, T5) and `bash -n` (T3).
