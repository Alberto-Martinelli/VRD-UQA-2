#!/usr/bin/env bash
#SBATCH -N 1 # one compute node for the job
#SBATCH --ntasks=1 # one process
#SBATCH --cpus-per-task=4 # 4 cores per process
#SBATCH --mem=64G # RAM memory
#SBATCH --time=0-23:59:00 # max wall time (D-HH:MM:SS)
#SBATCH --partition=gpu_v100_ext # partition name
#SBATCH --gres=gpu:1 # 1 GPU
#SBATCH --output=slurm-Verification-%j.out # output file name

module purge

# Inspect available modules first with: module avail
module load miniconda3/3.13.25
module load nvhpc/25.1

export SCRATCH="/mnt/beegfs-compat/amartinelli"
# export SCRATCH_FLASH="$SCRATCH"

export HF_HOME="$SCRATCH/.cache/huggingface"

rm -rf $SCRATCH/verification
mkdir -p $SCRATCH/verification/
rsync -av --exclude='data/DUDE' --exclude='data/BDocs' --exclude='data/SlideVQA' --exclude='.git' --exclude='.venv' $HOME/VRD-UQA/ $SCRATCH/verification/
cd $SCRATCH/verification/

uv --version
export UV_LINK_MODE=copy
export UV_CACHE_DIR=/tmp/uv_cache_$SLURM_JOB_ID
uv venv --python python3
uv sync

# ------ RUN THE ANSWERABILITY VERIFIER ------ 
uv run python corruption-scripts/verification/answerability_verifier.py --config corruption-scripts/config.hpc.json

# ------ RUN THE JUST FALSE ------
# uv run python corruption-scripts/verification/just_false.py --input ./MPDocVQA_unanswerable_corrupted_questions_verified.json --output ./MPDocVQA_unanswerable_corrupted_questions_just_false.json

mv $HOME/slurm* $HOME/VRD-UQA/
cp ./MPDocVQA_unanswerable_corrupted_questions_verified.json $HOME/VRD-UQA