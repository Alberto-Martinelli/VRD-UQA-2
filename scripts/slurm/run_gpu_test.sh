#!/usr/bin/env bash
#SBATCH -N 1 # one compute node for the job
#SBATCH --ntasks=1 # one process
#SBATCH --cpus-per-task=4 # 4 cores per process
#SBATCH --mem=32G # RAM memory
#SBATCH --time=0-00:30:00 # max wall time (D-HH:MM:SS)
#SBATCH --partition=gpu_a40 # partition name
#SBATCH --gres=gpu:1 # 1 GPU
#SBATCH --output=slurm-DUDE-%j.out # output file name

module purge

# Inspect available modules first with: module avail
module load miniconda3/3.13.25
module load nvhpc/25.1

source "$HOME/VRD-UQA/scripts/env.sh"

rm -rf $SCRATCH_FLASH/first_sample_test/
mkdir -p $SCRATCH_FLASH/first_sample_test/
cp -r $HOME/VRD-UQA $SCRATCH_FLASH/first_sample_test
cd $SCRATCH_FLASH/first_sample_test/VRD-UQA/
# rm -rf .venv/

# Overwrite the local config with the HPC-specific one
cp corruption-scripts/config.hpc.json corruption-scripts/config.json
# Remove old augmented file if it exists
rm -rf corruption-scripts/results/

uv --version
uv sync
uv run corruption-scripts/corruption/main.py --config corruption-scripts/config.json
# uv venv --python 3.12
# source .venv/bin/activate
# uv pip install torch --index https://download.pytorch.org/whl/cu124
# uv run gpu_test.py

mv $HOME/slurm* $HOME/VRD-UQA/