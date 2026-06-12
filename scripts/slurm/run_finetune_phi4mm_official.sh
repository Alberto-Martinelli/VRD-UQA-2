#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-16:00:00
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-finetune-phi4mm-%j.out

# Phi-4-multimodal vision fine-tuning via Microsoft's official path (B.4 fallback) —
# LLaMA-Factory's `phi4` template is text-only and rejects images, so this trains the
# model's built-in vision LoRA with finetuning/phi4mm_official/finetune_phi4mm.py.
#
# Usage:
#   sbatch run_finetune_phi4mm_official.sh [smoke]
#
# Phi-4-mm's remote code is written for an OLDER transformers; this builds a dedicated
# per-job venv with the versions the model needs (pinned below) — isolated from the
# main repo venv (which is on transformers 4.57.6 for Qwen/InternVL).
# Output is a FULL fine-tuned model dir (not a PEFT adapter); eval loads it via model_name.

set -euo pipefail
module purge
module load miniconda3/3.13.25
module load nvhpc/25.1
source "$HOME/VRD-UQA/scripts/env.sh"
if [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN="$(cat "$HOME/.hf_token")"
fi
export UV_LINK_MODE=copy

VARIANT="${1:-sft}"   # sft (full) | smoke
WORK_DIR="$SCRATCH_FLASH/finetune_phi4mm_official_${SLURM_JOB_ID}"
VENV_DIR="$WORK_DIR/.venv"
export HF_HOME="$WORK_DIR/hf_home"
mkdir -p "$WORK_DIR"
trap 'cd "$HOME"; rm -rf "$WORK_DIR"' EXIT

# ---- Sync repo (script + dataset); skip the big eval-artifact trees ----
rsync -aq --exclude='.git' --exclude='.venv' --exclude='data' \
      --exclude='corruption-scripts/results' \
      --exclude='artifacts/evaluation_runs' --exclude='artifacts/evaluation_archive' \
      "$HOME/VRD-UQA/" "$WORK_DIR/VRD-UQA/"

# ---- Per-job venv with the versions Phi-4-mm's remote code expects ----
uv venv --python 3.11 "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
uv pip install --upgrade pip
uv pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1+cu121 torchvision==0.19.1+cu121
uv pip install \
    transformers==4.47.0 peft==0.13.2 accelerate==1.3.0 scipy==1.15.1 backoff==2.2.1 \
    Pillow soundfile sentencepiece protobuf

# ---- Train ----
DATA_JSON="$WORK_DIR/VRD-UQA/artifacts/finetuning/dataset/train.json"
SCRATCH_OUT="$WORK_DIR/out"
SCRIPT="$WORK_DIR/VRD-UQA/finetuning/phi4mm_official/finetune_phi4mm.py"

EXTRA=""
if [ "$VARIANT" = "smoke" ]; then
    EXTRA="--max_samples 50"
fi

echo "Phi-4-mm vision fine-tune | variant=$VARIANT | data=$DATA_JSON"
accelerate launch --num_processes 1 "$SCRIPT" \
    --model_name_or_path microsoft/Phi-4-multimodal-instruct \
    --data_json "$DATA_JSON" \
    --output_dir "$SCRATCH_OUT" \
    --dynamic_hd 16 \
    --num_train_epochs 1 \
    --learning_rate 4.0e-5 \
    --batch_size 8 --batch_size_per_gpu 1 \
    $EXTRA

# ---- Copy the full fine-tuned model back (large ~10-20GB) ----
DEST="$HOME/VRD-UQA/artifacts/finetuning/phi4mm_vision_${VARIANT}"
if [ -d "$SCRATCH_OUT" ]; then
    mkdir -p "$DEST"
    rsync -a "$SCRATCH_OUT/" "$DEST/"
    echo "Copied fine-tuned Phi-4-mm model -> $DEST"
    echo "NOTE: set the phi4_finetuned eval entry to model_name=$DEST (remove adapter_path);"
    echo "      eval of this model needs the same pinned transformers (4.47.0)."
else
    echo "WARNING: $SCRATCH_OUT not found; nothing copied back." >&2
fi

# ---- Per-job slurm log move (scoped to THIS job id) ----
mv "$HOME"/slurm-finetune-phi4mm-${SLURM_JOB_ID}.out "$HOME/VRD-UQA/" 2>/dev/null || true
