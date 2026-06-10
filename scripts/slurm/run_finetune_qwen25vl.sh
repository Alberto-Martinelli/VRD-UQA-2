#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-16:00:00
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-finetune-%j.out

set -euo pipefail

module purge
module load miniconda3/3.13.25
module load nvhpc/25.1

source "$HOME/VRD-UQA/scripts/env.sh"
export UV_LINK_MODE=copy

# ---- Workspace ----
WORK_DIR="$SCRATCH_FLASH/finetune_qwen25vl"
LF_DIR="$WORK_DIR/LLaMA-Factory"
VENV_DIR="$WORK_DIR/.venv"
mkdir -p "$WORK_DIR"

# ---- Sync repo (for finetuning configs & train.json) ----
rsync -aq \
  --exclude='.git' --exclude='.venv' \
  --exclude='data' --exclude='data_standard_pipeline' \
  --exclude='corruption-scripts/results' \
  "$HOME/VRD-UQA/" "$WORK_DIR/VRD-UQA/"

# ---- Clone LLaMA-Factory (only if missing) ----
if [ ! -d "$LF_DIR" ]; then
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "$LF_DIR"
fi

# ---- Python env via uv ----
cd "$LF_DIR"
if [ ! -d "$VENV_DIR" ]; then
    uv venv --python 3.11 "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

uv pip install --upgrade pip
# Pin torch/torchvision to versions compiled for CUDA 12.1 (compatible with driver 12.0.80 via forward-compat).
# --reinstall forces replacement of any previously cached wheels (e.g. cu124/cu130).
uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121
uv pip install -e ".[torch,metrics]"
# LLaMA-Factory's editable install may have pulled torch/torchaudio/torchvision from PyPI (cu13);
# pin them back to the cu121 builds in case they got upgraded.
uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121
# Vision deps for Qwen2.5-VL
uv pip install qwen-vl-utils

# ---- Train ----
CONFIG="$WORK_DIR/VRD-UQA/finetuning/qwen25vl_lora_sft.yaml"
echo "Using config: $CONFIG"
cat "$CONFIG"

llamafactory-cli train "$CONFIG"

# Read output_dir straight from the config so the two never drift.
OUTPUT_DIR="$(sed -nE 's/^output_dir:[[:space:]]*//p' "$CONFIG" | tr -d "\"'")"

# ---- Write run_manifest.json into the training output dir (best-effort) ----
# Capture git from the real repo ($HOME has .git; the rsync'd work-copy doesn't).
# The manifest writer never fails the job; '|| true' is belt-and-suspenders.
GIT_COMMIT="$(cd "$HOME/VRD-UQA" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ -n "$(cd "$HOME/VRD-UQA" && git status --porcelain 2>/dev/null)" ]; then GIT_DIRTY=true; else GIT_DIRTY=false; fi
python "$WORK_DIR/VRD-UQA/finetuning/write_run_manifest.py" \
    --config "$CONFIG" --output-dir "$OUTPUT_DIR" \
    --git-commit "$GIT_COMMIT" --git-dirty "$GIT_DIRTY" || true

# ---- Copy trained adapter + training artifacts back to the repo ----
# (run_manifest.json lives in OUTPUT_DIR and is carried back by the rsync below.)
DEST="$HOME/VRD-UQA/artifacts/finetuning/qwen25vl_lora_sft"
if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
    mkdir -p "$DEST"
    # Bring back the final adapter, tokenizer, metrics and loss plot — but skip
    # the intermediate checkpoint-* dirs (large, reproducible from the final state).
    rsync -av --exclude='checkpoint-*' "$OUTPUT_DIR/" "$DEST/"
    echo "Copied trained adapter -> $DEST"
else
    echo "WARNING: output_dir '$OUTPUT_DIR' not found; nothing copied back." >&2
fi

# ---- Move slurm log back ----
mv "$HOME"/slurm-finetune-*.out "$HOME/VRD-UQA/" 2>/dev/null || true
