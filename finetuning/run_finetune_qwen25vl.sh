#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-finetune-%j.out

set -euo pipefail

module purge
module load miniconda3/3.13.25
module load nvhpc/25.1

export SCRATCH_FLASH="/mnt/beegfs/amartinelli"
export HF_HOME="$SCRATCH_FLASH/.cache/huggingface"
export UV_LINK_MODE=copy

# ---- Workspace ----
WORK_DIR="$SCRATCH_FLASH/finetune_qwen25vl"
LF_DIR="$WORK_DIR/LLaMA-Factory"
VENV_DIR="$WORK_DIR/.venv"
mkdir -p "$WORK_DIR"

# ---- Sync repo (for finetuning configs & train.json) ----
rsync -av \
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

# ---- Move slurm log back ----
mv $HOME/slurm-finetune-*.out $HOME/VRD-UQA
