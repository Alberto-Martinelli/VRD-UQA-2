#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-09:00:00
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-VQA_analysis-%j.out
#
# One model x one dataset x one split per job (parallel-safe, resumable).
#   sbatch run_vqa_analysis.sh <model> <dataset> <split>
#   <model> = qwen2.5 | phi4 | internvl | gemma4
# Env knobs: CONFIG=<path>  FINETUNE=--finetuned  RUN_TAG=foo  NO_LATEST=1
set -uo pipefail
START_TIME=$SECONDS
echo "Job started at: $(date)"

# Absolute path: under sbatch this script runs from a spool copy, so $0 is unreliable.
source "$HOME/VRD-UQA/scripts/slurm/_prelude.sh"

MODEL="${1:?usage: run_vqa_analysis.sh <model> <dataset> <split>}"
DATASET="${2:?usage: run_vqa_analysis.sh <model> <dataset> <split>}"
SPLIT="${3:-val_300}"
CONFIG="${CONFIG:-VQA_analysis/config_fewshot.json}"
FINETUNE="${FINETUNE:-}"

export VQA_RUN_ID="${VQA_RUN_ID:-$(make_run_id "$MODEL" "$DATASET" "$SPLIT")}"
export VQA_CONFIG_PATH="$CONFIG"
echo "Model: $MODEL | Dataset: $DATASET | Split: $SPLIT | Run id: $VQA_RUN_ID"

setup_scratch_workdir "VQA_analysis_${SLURM_JOB_ID}"
restore_prior_run "$VQA_RUN_ID"
activate_eval_venv "$MODEL"

printf '\n=== %s — %s — %s ===\n' "$MODEL" "$CONFIG" "$DATASET"
"${EVAL_CMD[@]}" VQA_analysis/evaluators/run_eval.py \
    --model "$MODEL" --dataset "$DATASET" --split "$SPLIT" \
    --config "$CONFIG" $FINETUNE --questions both
sync_run_back "$VQA_RUN_ID"

# Metrics — model-agnostic, main uv venv, scoped to THIS run id.
uv run python VQA_analysis/metrics/1_normalize_unanswerable_responses.py --run-id "$VQA_RUN_ID"
uv run python VQA_analysis/metrics/2_enrich_metadata.py            --run-id "$VQA_RUN_ID"
uv run python VQA_analysis/metrics/3_compute_metrics.py            --run-id "$VQA_RUN_ID"

mv "$HOME"/slurm-VQA_analysis-${SLURM_JOB_ID}.out "$HOME/VRD-UQA/" 2>/dev/null || true
echo "Results synced for $VQA_RUN_ID"
ELAPSED=$(( SECONDS - START_TIME ))
printf 'Job finished at: %s\nTotal execution time: %02d:%02d:%02d (%ds)\n' \
    "$(date)" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$ELAPSED"
