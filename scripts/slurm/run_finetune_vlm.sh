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
#   <model_key> = qwen25vl | internvl35 | phi4mm
# Run them in parallel (no collision):
#   for M in internvl35 phi4mm; do sbatch --job-name=ft-$M run_finetune_vlm.sh $M; done

set -euo pipefail
module purge
module load miniconda3/3.13.25
module load nvhpc/25.1
source "$HOME/VRD-UQA/scripts/env.sh"
# HF auth for gated / rate-limited model downloads (e.g. Llama-3.2 is gated).
if [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN="$(cat "$HOME/.hf_token")"
fi
export UV_LINK_MODE=copy

MODEL_KEY="${1:?usage: run_finetune_vlm.sh <model_key> [smoke]}"
VARIANT="${2:-sft}"   # sft (full) | smoke
case "$MODEL_KEY" in
  qwen25vl|internvl35|phi4mm) : ;;
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
      --exclude='artifacts/evaluation_runs' --exclude='artifacts/evaluation_archive' \
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
