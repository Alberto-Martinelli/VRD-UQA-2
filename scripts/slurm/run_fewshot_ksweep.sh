#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-16:00:00
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-fewshot_ksweep-%j.out

# Few-shot experiment: Qwen2.5-7B base model, SlideVQA + BDocs.
# Conditions: k=2 random (baseline) + k=2,4,6 specific (corruption-matched selection).
# Evaluates only corrupted questions (--questions corrupted).
# Demos are drawn from the held-out train_750 pool (no val-set leakage).
# All leaves share one VQA_RUN_ID so the metrics pipeline produces a single summary.csv.
#
# Usage:
#   sbatch run_fewshot_ksweep.sh                        # val_300, both datasets
#   sbatch run_fewshot_ksweep.sh val_5                  # smoke test (val_5), both datasets
#   sbatch run_fewshot_ksweep.sh val_300 SlideVQA       # single dataset
#   VQA_RUN_ID=<id> sbatch run_fewshot_ksweep.sh ...    # resume a timed-out run
#
#   Arg 1 = split suffix (val_300 | val_5 | ...). Default: val_300.
#   Args 2+ = dataset names.  Default: SlideVQA BDocs.

START_TIME=$SECONDS
echo "Job started at: $(date)"

module purge
module load miniconda3/3.13.25
module load nvhpc/25.1

source "$HOME/VRD-UQA/scripts/env.sh"

SPLIT="${1:-val_300}"
shift || true
DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=(SlideVQA BDocs)
fi
N="${SPLIT##*_}"
SPLIT_NAME="${SPLIT%_*}"
export VQA_RUN_ID="${VQA_RUN_ID:-eval_${SPLIT_NAME}_${N}_$(date +%Y%m%d_%H%M%S)}"
echo "Split: $SPLIT | Datasets: ${DATASETS[*]} | Run id: $VQA_RUN_ID"

WORK_DIR=$SCRATCH_FLASH/VQA_ksweep_${SLURM_JOB_ID}
SRC_RUN="$WORK_DIR/artifacts/evaluation_runs/$VQA_RUN_ID"
DEST_RUN="$HOME/VRD-UQA/artifacts/evaluation_runs/$VQA_RUN_ID"

sync_back() {
    [ -d "$SRC_RUN" ] || return 0
    mkdir -p "$DEST_RUN"
    rsync -aq "$SRC_RUN/" "$DEST_RUN/" || true
    ln -sfn "$VQA_RUN_ID" "$HOME/VRD-UQA/artifacts/evaluation_runs/latest" 2>/dev/null || true
}

trap 'sync_back; cd "$HOME"; rm -rf "$WORK_DIR"' EXIT

rm -rf "$WORK_DIR"
rsync -aq --exclude='data' --exclude='.git' --exclude='.venv' \
    --exclude='corruption-scripts/results' --exclude='finetuning' \
    "$HOME/VRD-UQA/" "$WORK_DIR/"
cd "$WORK_DIR"

uv --version
export UV_LINK_MODE=copy
uv sync -qq

if [ -d "$DEST_RUN" ]; then
    echo "RESUME: prior results found at $DEST_RUN — restoring into scratch; finished leaves will be skipped."
    mkdir -p "$SRC_RUN"
    rsync -aq "$DEST_RUN/" "$SRC_RUN/"
fi

FS_CONFIG="VQA_analysis/config_fewshot.json"

# Patch a config JSON for a given dataset/split/few-shot parameters.
patch_config() {
    local CONFIG_PATH="$1"
    local DATASET="$2"
    local SPLIT_VAL="$3"
    local FS_ENABLED="$4"   # "true" | "false"
    local N_SHOTS="$5"
    local SELECTION="$6"    # "random" | "specific"
    local POOL_FILE="$7"

    uv run python -c "
import json, sys
config_path, dataset, split, fs_enabled, n_shots, selection, pool_file = sys.argv[1:]
fs_enabled = fs_enabled == 'true'
n_shots = int(n_shots)

input_file = f'/home/amartinelli/VRD-UQA/data/{dataset}/{dataset}_{split}/{dataset}_unanswerable_corrupted_questions_just_false.json'
cfg = json.load(open(config_path))
cfg['dataset'] = dataset
cfg['split'] = split.split('_')[0]
cfg['input_file'] = input_file
cfg['few_shot']['enabled'] = fs_enabled
cfg['few_shot']['n_shots'] = n_shots
cfg['few_shot']['selection'] = selection
cfg['few_shot']['pool_file'] = pool_file if fs_enabled else ''
json.dump(cfg, open(config_path, 'w'), indent=4)
" "$CONFIG_PATH" "$DATASET" "$SPLIT_VAL" "$FS_ENABLED" "$N_SHOTS" "$SELECTION" "$POOL_FILE"
}

for D in "${DATASETS[@]}"; do
    POOL_FILE="/home/amartinelli/VRD-UQA/data/$D/${D}_train_750/${D}_unanswerable_corrupted_questions_just_false.json"

    printf "\n\n########## DATASET: %s  (split=%s) ##########\n" "$D" "$SPLIT"

    # ---- k=2 random (baseline) ----
    printf "\n=== QWEN2.5 BASE — FEW-SHOT k=2 random — %s ===\n" "$D"
    patch_config "$FS_CONFIG" "$D" "$SPLIT" "true" 2 "random" "$POOL_FILE"
    export VQA_CONFIG_PATH="$FS_CONFIG"
    uv run python VQA_analysis/evaluators/qwen2.5_evaluator.py \
        --config_path "$FS_CONFIG" --questions corrupted
    sync_back

    # ---- k=2,4,6 specific (corruption-matched) ----
    for K in 2 4 6; do
        printf "\n=== QWEN2.5 BASE — FEW-SHOT k=%d specific — %s ===\n" "$K" "$D"
        patch_config "$FS_CONFIG" "$D" "$SPLIT" "true" "$K" "specific" "$POOL_FILE"
        export VQA_CONFIG_PATH="$FS_CONFIG"
        uv run python VQA_analysis/evaluators/qwen2.5_evaluator.py \
            --config_path "$FS_CONFIG" --questions corrupted
        sync_back
    done
done

# ---- Metrics pipeline ----
uv run python VQA_analysis/metrics/1_normalize_unanswerable_responses.py --run-id "$VQA_RUN_ID"
uv run python VQA_analysis/metrics/2_enrich_metadata.py                  --run-id "$VQA_RUN_ID"
uv run python VQA_analysis/metrics/3_compute_metrics.py                  --run-id "$VQA_RUN_ID"

# ---- Run manifest ----
uv run python -c "
import json
from config import run_layout as rl
run_id = '$VQA_RUN_ID'
datasets = '${DATASETS[*]}'.split()
run = rl.run_dir(run_id)
configs = sorted({leaf.name for ds in datasets for leaf in (run / ds).glob('*') if leaf.is_dir()}) if run.exists() else []
rl.write_manifest(run / 'run_manifest.json', {
    'run_id': run_id,
    'created_at': rl.utc_now_iso(),
    'git_commit': rl.git_commit(),
    'git_dirty': rl.git_dirty(),
    'split': '${SPLIT_NAME}',
    'n': int('${N}'),
    'seed': 42,
    'datasets': datasets,
    'configs': configs,
    'experiment': 'fewshot_ksweep',
    'k_values': [2, 4, 6],
    'selections': ['random', 'specific'],
    'conditions': 'k=2 random (baseline); k=2,4,6 specific',
    'model': 'Qwen2.5-VL-7B-Instruct',
    'pool': 'train_750',
})
print('Wrote run_manifest ->', run_id)
"

mv $HOME/slurm* $HOME/VRD-UQA/ 2>/dev/null || true

sync_back
echo "Results synced to $DEST_RUN"

ELAPSED=$(( SECONDS - START_TIME ))
printf 'Job finished at: %s\nTotal execution time: %02d:%02d:%02d (%ds)\n' \
    "$(date)" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$ELAPSED"
