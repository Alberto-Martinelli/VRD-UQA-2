#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-00:15:00
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-HPC-Test-%j.out

set -euo pipefail

PASS=0
FAIL=0

check() {
    local label="$1"
    shift
    if "$@" &>/dev/null; then
        echo "[PASS] $label"
        ((PASS++))
    else
        echo "[FAIL] $label"
        ((FAIL++))
    fi
}

echo "========================================"
echo " HPC Sanity Check — $(date)"
echo "========================================"

# --- Modules ---
echo ""
echo "--- Modules ---"
module purge
module load miniconda3/3.13.25
module load nvhpc/25.1
check "miniconda3 loaded"  python3 --version
check "nvhpc loaded"       nvcc --version

# --- GPU ---
echo ""
echo "--- GPU ---"
check "nvidia-smi available"   nvidia-smi
check "at least 1 GPU visible" bash -c '[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -ge 1 ]'

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
echo "      GPU: $gpu_name  ($gpu_mem)"

check "torch sees CUDA" bash -c 'python3 -c "import torch; assert torch.cuda.is_available(), \"no cuda\""'
cuda_dev=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "N/A")
echo "      torch device: $cuda_dev"

# --- Scratch storage ---
echo ""
echo "--- Scratch Flash ---"
export SCRATCH_FLASH="/mnt/beegfs-compat/amartinelli"
check "scratch_flash dir exists" test -d "$SCRATCH_FLASH"

tmpfile="$SCRATCH_FLASH/.hpc_test_$$"
check "scratch_flash writable"   bash -c "echo ok > $tmpfile"
check "scratch_flash readable"   bash -c "[ \"\$(cat $tmpfile)\" = ok ]"
rm -f "$tmpfile"

scratch_free=$(df -h "$SCRATCH_FLASH" 2>/dev/null | awk 'NR==2{print $4}')
echo "      Scratch Flash free: ${scratch_free:-unknown}"

# --- uv / Python env ---
echo ""
echo "--- uv / Python ---"
check "uv available" uv --version

export UV_LINK_MODE=copy
export UV_CACHE_DIR=/tmp/uv_cache_test_$$
tmpenv=$(mktemp -d)
check "uv venv create"  bash -c "uv venv --python python3 $tmpenv/testvenv &>/dev/null"
check "uv pip install"  bash -c "VIRTUAL_ENV=$tmpenv/testvenv uv pip install numpy &>/dev/null"
check "numpy importable" bash -c "VIRTUAL_ENV=$tmpenv/testvenv python3 -c 'import numpy' &>/dev/null"
rm -rf "$tmpenv" "$UV_CACHE_DIR"

# --- HuggingFace cache dir ---
echo ""
echo "--- HuggingFace ---"
export HF_HOME="$SCRATCH_FLASH/.cache/huggingface"
mkdir -p "$HF_HOME"
check "HF_HOME writable" test -w "$HF_HOME"
echo "      HF_HOME: $HF_HOME"

# --- Project rsync ---
echo ""
echo "--- Project sync ---"
tmpdir=$(mktemp -d "$SCRATCH_FLASH/.hpc_test_rsync_XXXXXX")
check "rsync project" bash -c "rsync -a --exclude='.git' --exclude='.venv' \
    --exclude='data/DUDE' --exclude='data/BDocs' --exclude='data/SlideVQA' \
    $HOME/VRD-UQA/ $tmpdir/ &>/dev/null"
check "config file present" test -f "$tmpdir/corruption-scripts/config.hpc.json"
rm -rf "$tmpdir"

# --- Summary ---
echo ""
echo "========================================"
echo " Results: $PASS passed, $FAIL failed"
echo "========================================"

mv "$HOME"/slurm-HPC-Test-*.out "$HOME/VRD-UQA/" 2>/dev/null || true

[ "$FAIL" -eq 0 ]
