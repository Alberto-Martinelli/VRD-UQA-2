#!/usr/bin/env bash
#SBATCH -N 1 # one compute node for the job
#SBATCH --ntasks=1 # one process
#SBATCH --cpus-per-task=4 # 4 cores per process
#SBATCH --mem=32G # RAM memory
#SBATCH --time=0-4:00:00 # max wall time (D-HH:MM:SS)
#SBATCH --partition=gpu_a40 # partition name
#SBATCH --gres=gpu:1 # 1 GPU
#SBATCH --output=slurm-VQA_analysis-%j.out # output file name

START_TIME=$SECONDS
echo "Job started at: $(date)"

module purge

# Inspect available modules first with: module avail
module load miniconda3/3.13.25
module load nvhpc/25.1

source "$HOME/VRD-UQA/scripts/env.sh"

WORK_DIR=$SCRATCH_FLASH/VQA_analysis_${SLURM_JOB_ID}

rm -rf $WORK_DIR
# mkdir -p $WORK_DIR
rsync -aq --exclude='data' --exclude='.git' --exclude='.venv' --exclude='corruption-scripts/results' --exclude='finetuning' $HOME/VRD-UQA/ $WORK_DIR/
cd $WORK_DIR
# rm -rf .venv/

uv --version
export UV_LINK_MODE=copy
uv sync -qq

DATASET=$1
ZS_CONFIG="VQA_analysis/config_zeroshot.json"
FS_CONFIG="VQA_analysis/config_fewshot.json"

# Patch dataset and input_file into the scratch copies (originals untouched)
uv run python -c "
import json, sys
dataset = sys.argv[1]
input_file = f'/home/amartinelli/VRD-UQA/VQA_analysis/evaluation_files/complete/{dataset}_unanswerable_corrupted_questions_just_false.json'
for path in ['VQA_analysis/config_zeroshot.json', 'VQA_analysis/config_fewshot.json']:
    cfg = json.load(open(path))
    cfg['dataset'] = dataset
    cfg['input_file'] = input_file
    json.dump(cfg, open(path, 'w'), indent=4)
" $DATASET

# ------ PASS 1: ZERO-SHOT ------
# printf "\n\n=== QWEN2.5 — ZERO-SHOT ==="
# uv run python VQA_analysis/new_evaluators/qwen2.5_evaluator.py --config_path $ZS_CONFIG

# ------ PASS 2: FEW-SHOT ------
# printf "\n\n=== QWEN2.5 — FEW-SHOT FINETUNED ==="
# uv run python VQA_analysis/new_evaluators/qwen2.5_evaluator.py --config_path $FS_CONFIG --finetuned

# ------ PASS 2b: FEW-SHOT FINETUNED — ANSWERABLE ------
printf "\n\n=== QWEN2.5 — FEW-SHOT FINETUNED — ANSWERABLE ==="
uv run python VQA_analysis/new_evaluators/qwen2.5_evaluator.py --config_path $FS_CONFIG --finetuned --answerable

# ------ PASS 3: FINE-TUNING ------

# Normalize + enrich once (auto-skips already-processed files)
uv run python VQA_analysis/pipeline/1_normalize_unanswerable_responses.py
uv run python VQA_analysis/pipeline/2_enrich_metadata.py

# Metrics per condition
uv run python VQA_analysis/pipeline/3_compute_metrics.py --config $FS_CONFIG

mv $HOME/slurm* $HOME/VRD-UQA/

cp -r $WORK_DIR/VQA_analysis/models/results $HOME/VRD-UQA/results_${SLURM_JOB_ID}


ELAPSED=$(( SECONDS - START_TIME ))
printf 'Job finished at: %s\nTotal execution time: %02d:%02d:%02d (%ds)\n' \
    "$(date)" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$ELAPSED"
