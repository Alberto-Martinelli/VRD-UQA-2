#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --partition=cpu_sapphire
#SBATCH --output=slurm-test_dataset-%x-%j.out

# Usage: sbatch --job-name=DUDE     download_dataset.sh DUDE
#        sbatch --job-name=MPDocVQA download_dataset.sh MPDocVQA
#        sbatch --job-name=SlideVQA download_dataset.sh SlideVQA
#        sbatch --job-name=BDocs    download_dataset.sh BDocs

set -euo pipefail

DATASET="${1:?Usage: sbatch download_dataset.sh <DATASET>}"

module purge
module load miniconda3/3.13.25
module load nvhpc/25.1

source "$HOME/VRD-UQA/scripts/env.sh"
mkdir -p "$HF_HOME"

if [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN="$(cat "$HOME/.hf_token")"
fi

WORK_DIR="$SCRATCH_FLASH/test_${DATASET}_dataset"
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

echo "Syncing repository to scratch..."
rsync -aq --exclude='.git' --exclude='.venv' --exclude='corruption-scripts/results' \
    --exclude='/data/' \
    "$HOME/VRD-UQA/" "$WORK_DIR/"

cd "$WORK_DIR"

uv --version
export UV_LINK_MODE=copy
export UV_CACHE_DIR=/tmp/uv_cache_$SLURM_JOB_ID
uv venv --python python3
uv sync -qq

echo "Sampling ${DATASET}..."
uv run python -m datasets_api.sample_dataset "$DATASET"

mv "$HOME"/slurm* "$HOME/VRD-UQA/" 2>/dev/null || true
rm -rf "$WORK_DIR"
echo "Done."
