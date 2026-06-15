#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-16:00:00
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-finetune-gemma4-%j.out

# Gemma 4 12B (it) LoRA fine-tuning via HF Trainer + PEFT (standalone script) —
# LLaMA-Factory has no Gemma 4 vision template, so this trains a PEFT LoRA adapter with
# finetuning/gemma4/finetune_gemma4.py.
#
# Usage:
#   sbatch run_finetune_gemma4.sh [smoke]
#
# Gemma 4 needs a NEWER transformers than the rest of the repo; this builds a dedicated
# per-job venv (pinned below) isolated from the main venv (4.57.6) and the Phi-4 path
# (4.47.0). Output is a PEFT LoRA adapter (~500MB); eval loads base + adapter.
# Gemma 4 is GATED -> needs HF_TOKEN (verify the HF account has Gemma 4 access).

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
WORK_DIR="$SCRATCH_FLASH/finetune_gemma4_${SLURM_JOB_ID}"
VENV_DIR="$WORK_DIR/.venv"
export HF_HOME="$WORK_DIR/hf_home"
mkdir -p "$WORK_DIR"
trap 'cd "$HOME"; rm -rf "$WORK_DIR"' EXIT

# ---- Sync repo (script + dataset); skip the big eval-artifact trees ----
rsync -aq --exclude='.git' --exclude='.venv' --exclude='data' \
      --exclude='corruption-scripts/results' \
      --exclude='artifacts/evaluation_runs' --exclude='artifacts/evaluation_archive' \
      "$HOME/VRD-UQA/" "$WORK_DIR/VRD-UQA/"

# ---- Per-job venv with the versions Gemma 4 needs ----
uv venv --python 3.11 "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
uv pip install --upgrade pip
uv pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1+cu121 torchvision==0.19.1+cu121
uv pip install \
    "transformers>=5.5.2" "peft>=0.19.0" accelerate bitsandbytes \
    Pillow sentencepiece protobuf
# flash-attn is optional (perf only); training falls back to sdpa+bf16 if it is absent.
uv pip install flash-attn --no-build-isolation || echo "flash-attn unavailable; using sdpa"

# ---- Train ----
DATA_JSON="$WORK_DIR/VRD-UQA/artifacts/finetuning/dataset/train.json"
SCRATCH_OUT="$WORK_DIR/out"
SCRIPT="$WORK_DIR/VRD-UQA/finetuning/gemma4/finetune_gemma4.py"

FLASH_FLAG="--use_flash_attention"
python -c "import flash_attn" 2>/dev/null || FLASH_FLAG="--no_flash_attention"

EXTRA=""
if [ "$VARIANT" = "smoke" ]; then
    EXTRA="--max_samples 50"
fi

echo "Gemma 4 LoRA fine-tune | variant=$VARIANT | flash=$FLASH_FLAG | data=$DATA_JSON"
accelerate launch --num_processes 1 "$SCRIPT" \
    --model_name_or_path google/gemma-4-12B-it \
    --data_json "$DATA_JSON" \
    --output_dir "$SCRATCH_OUT" \
    --visual_token_budget 560 \
    --num_train_epochs 1 \
    --learning_rate 2.0e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    $FLASH_FLAG \
    $EXTRA

# ---- Copy the LoRA adapter back (~500MB) ----
DEST="$HOME/VRD-UQA/artifacts/finetuning/gemma4_12b_lora_${VARIANT}"
if [ -d "$SCRATCH_OUT" ]; then
    mkdir -p "$DEST"
    rsync -a "$SCRATCH_OUT/" "$DEST/"
    echo "Copied Gemma 4 LoRA adapter -> $DEST"
    echo "NOTE: add a gemma4_finetuned eval entry (model_name=google/gemma-4-12B-it,"
    echo "      adapter_path=$DEST) once a Gemma 4 evaluator exists; eval needs the same"
    echo "      pinned transformers (>=5.5.2)."
else
    echo "WARNING: $SCRATCH_OUT not found; nothing copied back." >&2
fi

# ---- Per-job slurm log move (scoped to THIS job id) ----
mv "$HOME"/slurm-finetune-gemma4-${SLURM_JOB_ID}.out "$HOME/VRD-UQA/" 2>/dev/null || true
