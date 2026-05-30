#!/usr/bin/env bash
#SBATCH -N 1 # one compute node for the job
#SBATCH --ntasks=1 # one process
#SBATCH --cpus-per-task=4 # 4 cores per process
#SBATCH --mem=32G # RAM memory
#SBATCH --time=0-23:59:00 # max wall time (D-HH:MM:SS)
#SBATCH --partition=gpu_a40 # partition name
#SBATCH --gres=gpu:1 # 1 GPU
#SBATCH --output=slurm-VQA_analysis-%j.out # output file name

START_TIME=$SECONDS
echo "Job started at: $(date)"

module purge

# Inspect available modules first with: module avail
module load miniconda3/3.13.25
module load nvhpc/25.1

export SCRATCH_FLASH="/mnt/beegfs/amartinelli"

export HF_HOME="$SCRATCH_FLASH/.cache/huggingface"

rm -rf $SCRATCH_FLASH/VQA_analysis
# mkdir -p $SCRATCH_FLASH/VQA_analysis/
rsync -av --exclude='data' --exclude='.git' --exclude='.venv' --exclude='corruption-scripts/results' $HOME/VRD-UQA/ $SCRATCH_FLASH/VQA_analysis/
cd $SCRATCH_FLASH/VQA_analysis/
# rm -rf .venv/

uv --version
export UV_LINK_MODE=copy
uv sync

CONFIG="VQA_analysis/config_lora.json"

# ------ RUN THE QWEN EVALUATOR ------
uv run python VQA_analysis/new_evaluators/qwen_evaluator.py --config_path $CONFIG

# cp VQA_analysis/models/results/MPDocVQA/LLM/results_w2_UNABLE/original/Qwen_vqa_analysis_results.json $HOME/VRD-UQA/

# ------ RUN THE UNABLE CONVERTER ------
uv run python VQA_analysis/pipeline/1_normalize_unanswerable_responses.py

# ------ RUN THE ADDING INFORMATIONS ------
uv run python VQA_analysis/pipeline/2_enrich_metadata.py

# ------ RUN THE RESULT ANALYSIS ------
uv run python VQA_analysis/pipeline/3_compute_metrics.py --config $CONFIG

mv $HOME/slurm* $HOME/VRD-UQA/

cp -r $SCRATCH_FLASH/VQA_analysis/VQA_analysis/models/results $HOME/VRD-UQA


ELAPSED=$(( SECONDS - START_TIME ))
printf 'Job finished at: %s\nTotal execution time: %02d:%02d:%02d (%ds)\n' \
    "$(date)" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$ELAPSED"
