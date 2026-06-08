#!/usr/bin/env bash
#SBATCH -N 1 # one compute node for the job
#SBATCH --ntasks=1 # one process
#SBATCH --cpus-per-task=4 # 4 cores per process
#SBATCH --mem=32G # RAM memory
#SBATCH --time=0-12:00:00 # max wall time (D-HH:MM:SS) — generous for 4 datasets x val_300; override with `sbatch --time=...`
#SBATCH --partition=gpu_a40 # partition name
#SBATCH --gres=gpu:1 # 1 GPU
#SBATCH --output=slurm-VQA_analysis-%j.out # output file name

# Usage (one job evaluates every requested dataset sequentially):
#   sbatch run_vqa_analysis.sh                      # all 4 datasets, full eval (val_300)
#   sbatch run_vqa_analysis.sh val_5                # all 4 datasets, smoke test (val_5)
#   sbatch run_vqa_analysis.sh val_300 DUDE BDocs   # specific datasets, full eval
#
#   Arg 1   = split folder suffix (val_300 | val_5 | ...). Default: val_300.
#   Args 2+ = dataset names.       Default: BDocs DUDE MPDocVQA SlideVQA.

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

# ---- Args: split + dataset list ----
SPLIT="${1:-val_300}"          # which corrupted set to evaluate: val_300 (full) or val_5 (smoke test)
shift || true
DATASETS=("$@")                # optional explicit list; empty -> all four
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=(BDocs DUDE MPDocVQA SlideVQA)
fi
echo "Split: $SPLIT | Datasets: ${DATASETS[*]}"

ZS_CONFIG="VQA_analysis/config_zeroshot.json"
FS_CONFIG="VQA_analysis/config_fewshot.json"

# ---- Evaluate each dataset ----
# One evaluator process per dataset (model is reloaded each time; for val_300 the
# per-dataset eval time dwarfs the load, and val_5 is a quick smoke test anyway).
for D in "${DATASETS[@]}"; do
    printf "\n\n########## DATASET: %s  (split=%s) ##########\n" "$D" "$SPLIT"

    # Patch dataset + input_file into the scratch config copies (originals in $HOME untouched).
    # data/ is not rsynced to the work-dir, so input_file points at the persistent copy.
    uv run python -c "
import json, sys
dataset, split = sys.argv[1], sys.argv[2]
input_file = f'/home/amartinelli/VRD-UQA/data/{dataset}/{dataset}_{split}/{dataset}_unanswerable_corrupted_questions_just_false.json'
for path in ['VQA_analysis/config_zeroshot.json', 'VQA_analysis/config_fewshot.json']:
    cfg = json.load(open(path))
    cfg['dataset'] = dataset
    cfg['input_file'] = input_file
    json.dump(cfg, open(path, 'w'), indent=4)
" "$D" "$SPLIT"

    # Active pass: few-shot, fine-tuned, answerable.
    # To change condition, edit the flags below: drop --finetuned for the base model,
    # drop --answerable for the unanswerable set, or use $ZS_CONFIG for zero-shot.
    printf "\n=== QWEN2.5 — FEW-SHOT FINETUNED — ANSWERABLE — %s ===\n" "$D"
    uv run python VQA_analysis/evaluators/qwen2.5_evaluator.py --config_path $FS_CONFIG --finetuned --answerable
done

# ---- Metrics ----
# Normalize + enrich once over the whole artifacts/evaluation tree (auto-skips processed files),
# then compute metrics per dataset.
uv run python VQA_analysis/metrics/1_normalize_unanswerable_responses.py
uv run python VQA_analysis/metrics/2_enrich_metadata.py
for D in "${DATASETS[@]}"; do
    uv run python VQA_analysis/metrics/3_compute_metrics.py --dataset "$D"
done

mv $HOME/slurm* $HOME/VRD-UQA/

# Copy results back under a human-readable, sortable name (split + timestamp)
RUN_NAME="eval_${SPLIT}_$(date +%Y%m%d_%H%M%S)"
DEST="$HOME/VRD-UQA/artifacts/evaluation_runs/$RUN_NAME"
mkdir -p "$(dirname "$DEST")"
cp -r "$WORK_DIR/artifacts/evaluation" "$DEST"
echo "Results copied to $DEST"


ELAPSED=$(( SECONDS - START_TIME ))
printf 'Job finished at: %s\nTotal execution time: %02d:%02d:%02d (%ds)\n' \
    "$(date)" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$ELAPSED"
