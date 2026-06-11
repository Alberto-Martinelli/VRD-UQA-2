#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-VQA_analysis-%j.out

# One model x one dataset x one split per job, so jobs run in parallel safely.
# Usage:
#   sbatch run_vqa_analysis.sh <model> <dataset> <split>
#   e.g. sbatch run_vqa_analysis.sh llama BDocs val_100
#
#   <model>   = qwen2.5 | llama | phi4 | internvl
#   <dataset> = BDocs | DUDE | MPDocVQA | SlideVQA
#   <split>   = val_300 | val_100 | val_5 | ...
#
# Each job derives a deterministic, unique run_id eval_<split>_<n>_<model>_<dataset>,
# so parallel jobs never share a run dir (no sync/metrics collision). Resubmitting
# the same combo RESUMES. Set RUN_TAG=foo to force a distinct fresh run.

set -uo pipefail
START_TIME=$SECONDS
echo "Job started at: $(date)"

MODEL="${1:?usage: run_vqa_analysis.sh <model> <dataset> <split>}"
DATASET="${2:?usage: run_vqa_analysis.sh <model> <dataset> <split>}"
SPLIT="${3:-val_300}"
N="${SPLIT##*_}"
SPLIT_NAME="${SPLIT%_*}"
RUN_TAG="${RUN_TAG:-}"

# Map model key -> evaluator entrypoint. --finetuned only when an adapter exists
# for that model (Phase B wires the *_finetuned config entries + FINETUNE flag).
case "$MODEL" in
  qwen2.5)  ENTRY="VQA_analysis/evaluators/qwen2.5_evaluator.py" ;;
  llama)    ENTRY="VQA_analysis/evaluators/llama_evaluator.py" ;;
  phi4)     ENTRY="VQA_analysis/evaluators/phi4_evaluator.py" ;;
  internvl) ENTRY="VQA_analysis/evaluators/internvl_evaluator.py" ;;
  *) echo "ERROR: unknown model '$MODEL'"; exit 2 ;;
esac

# Eval condition is config-driven. Default: few-shot config, no --finetuned for the
# new models (no adapter yet). Override CONFIG / FINETUNE via env when needed.
CONFIG="${CONFIG:-VQA_analysis/config_fewshot.json}"
FINETUNE="${FINETUNE:-}"   # set FINETUNE=--finetuned to use a *_finetuned entry

module purge
module load miniconda3/3.13.25
module load nvhpc/25.1
source "$HOME/VRD-UQA/scripts/env.sh"

export VQA_RUN_ID="${VQA_RUN_ID:-eval_${SPLIT_NAME}_${N}_${MODEL}_${DATASET}${RUN_TAG:+_$RUN_TAG}}"
echo "Model: $MODEL | Dataset: $DATASET | Split: $SPLIT | Run id: $VQA_RUN_ID"

WORK_DIR=$SCRATCH_FLASH/VQA_analysis_${SLURM_JOB_ID}
SRC_RUN="$WORK_DIR/artifacts/evaluation_runs/$VQA_RUN_ID"
DEST_RUN="$HOME/VRD-UQA/artifacts/evaluation_runs/$VQA_RUN_ID"

sync_back() {
    [ -d "$SRC_RUN" ] || return 0
    mkdir -p "$DEST_RUN"
    rsync -aq "$SRC_RUN/" "$DEST_RUN/" || true
    # 'latest' is a global pointer; updating it from parallel jobs is a harmless
    # last-writer-wins race. Opt out with NO_LATEST=1 to avoid churn.
    [ -n "${NO_LATEST:-}" ] || ln -sfn "$VQA_RUN_ID" "$HOME/VRD-UQA/artifacts/evaluation_runs/latest" 2>/dev/null || true
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

# Resume: restore an already-computed run for this id from $HOME into scratch.
if [ -d "$DEST_RUN" ]; then
    echo "RESUME: prior results at $DEST_RUN — restoring into scratch; finished leaf will be skipped."
    mkdir -p "$SRC_RUN"
    rsync -aq "$DEST_RUN/" "$SRC_RUN/"
fi

export VQA_CONFIG_PATH="$CONFIG"

# Point the chosen config at this dataset/split's corrupted set.
uv run python -c "
import json, sys
dataset, split, cfg_path = sys.argv[1], sys.argv[2], sys.argv[3]
input_file = f'/home/amartinelli/VRD-UQA/data/{dataset}/{dataset}_{split}/{dataset}_unanswerable_corrupted_questions_just_false.json'
cfg = json.load(open(cfg_path))
cfg['dataset'] = dataset
cfg['split'] = split.split('_')[0]
cfg['input_file'] = input_file
json.dump(cfg, open(cfg_path, 'w'), indent=4)
" "$DATASET" "$SPLIT" "$CONFIG"

printf '\n=== %s — %s — QUR+FRR — %s ===\n' "$MODEL" "$CONFIG" "$DATASET"
uv run python "$ENTRY" --config_path "$CONFIG" $FINETUNE --questions both
sync_back

# Metrics — scoped to THIS run_id only (private tree => no cross-job collision).
uv run python VQA_analysis/metrics/1_normalize_unanswerable_responses.py --run-id "$VQA_RUN_ID"
uv run python VQA_analysis/metrics/2_enrich_metadata.py            --run-id "$VQA_RUN_ID"
uv run python VQA_analysis/metrics/3_compute_metrics.py            --run-id "$VQA_RUN_ID"

# Per-job slurm log move — scoped to THIS job id (never grab sibling jobs' logs).
mv "$HOME"/slurm-VQA_analysis-${SLURM_JOB_ID}.out "$HOME/VRD-UQA/" 2>/dev/null || true

sync_back
echo "Results synced to $DEST_RUN"
ELAPSED=$(( SECONDS - START_TIME ))
printf 'Job finished at: %s\nTotal execution time: %02d:%02d:%02d (%ds)\n' \
    "$(date)" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$ELAPSED"
