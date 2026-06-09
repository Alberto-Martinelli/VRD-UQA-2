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

# Reclaim the per-job scratch dir on any exit (success, error, timeout, scancel)
# so SCRATCH_FLASH doesn't accumulate a full repo copy per run. Results are
# copied back to $HOME before the script exits, so this only drops the scratch
# working copy. cd out first so removing the CWD is safe.
trap 'cd "$HOME"; rm -rf "$WORK_DIR"' EXIT

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

# One run_id for the whole job; the evaluator + metrics steps all write under it.
N="${SPLIT##*_}"                       # e.g. val_300 -> 300
SPLIT_NAME="${SPLIT%_*}"               # e.g. val_300 -> val
export VQA_RUN_ID="eval_${SPLIT_NAME}_${N}_$(date +%Y%m%d_%H%M%S)"
export VQA_CONFIG_PATH="$FS_CONFIG"
echo "Run id: $VQA_RUN_ID"

# ---- Evaluate each dataset (QUR+FRR in one pass via --questions both) ----
for D in "${DATASETS[@]}"; do
    printf "\n\n########## DATASET: %s  (split=%s) ##########\n" "$D" "$SPLIT"
    uv run python -c "
import json, sys
dataset, split = sys.argv[1], sys.argv[2]
input_file = f'/home/amartinelli/VRD-UQA/data/{dataset}/{dataset}_{split}/{dataset}_unanswerable_corrupted_questions_just_false.json'
for path in ['VQA_analysis/config_zeroshot.json', 'VQA_analysis/config_fewshot.json']:
    cfg = json.load(open(path))
    cfg['dataset'] = dataset
    cfg['split'] = split.split('_')[0]
    cfg['input_file'] = input_file
    json.dump(cfg, open(path, 'w'), indent=4)
" "$D" "$SPLIT"

    printf "\n=== QWEN2.5 — FEW-SHOT FINETUNED — QUR+FRR — %s ===\n" "$D"
    uv run python VQA_analysis/evaluators/qwen2.5_evaluator.py --config_path "$FS_CONFIG" --finetuned --questions both
done

# ---- Metrics (all operate on $VQA_RUN_ID) ----
uv run python VQA_analysis/metrics/1_normalize_unanswerable_responses.py --run-id "$VQA_RUN_ID"
uv run python VQA_analysis/metrics/2_enrich_metadata.py            --run-id "$VQA_RUN_ID"
uv run python VQA_analysis/metrics/3_compute_metrics.py            --run-id "$VQA_RUN_ID"

# ---- Run manifest + latest symlink ----
uv run python -c "
import json
from config import run_layout as rl
run_id = '$VQA_RUN_ID'
datasets = '${DATASETS[*]}'.split()
run = rl.run_dir(run_id)
configs = sorted({leaf.name for ds in datasets for leaf in (run / ds).glob('*') if leaf.is_dir()}) if run.exists() else []
rl.write_manifest(run / 'run_manifest.json', {
    'run_id': run_id, 'created_at': rl.utc_now_iso(),
    'git_commit': rl.git_commit(), 'git_dirty': rl.git_dirty(),
    'split': '${SPLIT_NAME}', 'n': int('${N}'), 'seed': 42,
    'datasets': datasets, 'configs': configs,
})
rl.update_latest_symlink(run_id)
print('Wrote run_manifest + latest ->', run_id)
"

mv $HOME/slurm* $HOME/VRD-UQA/ 2>/dev/null || true

# Copy the clean run tree back to \$HOME (1:1 — no reshaping).
SRC="$WORK_DIR/artifacts/evaluation_runs/$VQA_RUN_ID"
DEST="$HOME/VRD-UQA/artifacts/evaluation_runs/$VQA_RUN_ID"
if [ -d "$SRC" ]; then
    mkdir -p "$DEST"
    cp -r "$SRC/." "$DEST/"
    ln -sfn "$VQA_RUN_ID" "$HOME/VRD-UQA/artifacts/evaluation_runs/latest"
    echo "Results copied to $DEST"
else
    echo "WARNING: no run dir produced at $SRC"
fi


ELAPSED=$(( SECONDS - START_TIME ))
printf 'Job finished at: %s\nTotal execution time: %02d:%02d:%02d (%ds)\n' \
    "$(date)" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$ELAPSED"
