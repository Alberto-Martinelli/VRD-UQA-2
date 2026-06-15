#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-09:00:00
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-VQA_analysis-%j.out

# One model x one dataset x one split per job, so jobs run in parallel safely.
# Usage:
#   sbatch run_vqa_analysis.sh <model> <dataset> <split>
#   e.g. sbatch run_vqa_analysis.sh phi4 BDocs val_100
#
#   <model>   = qwen2.5 | phi4 | internvl | gemma4
#   <dataset> = BDocs | DUDE | MPDocVQA | SlideVQA
#   <split>   = val_300 | val_100 | val_5 | ...
#
# qwen2.5/internvl run the evaluator in the main repo venv (transformers 4.57.6). phi4 and
# gemma4 need a different transformers, so the evaluator runs in a per-job PINNED venv built
# on scratch (phi4 ==4.47.0, gemma4 >=5.5.2); the metrics steps still use the main venv.
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
  phi4)     ENTRY="VQA_analysis/evaluators/phi4_evaluator.py" ;;
  internvl) ENTRY="VQA_analysis/evaluators/internvl_evaluator.py" ;;
  gemma4)   ENTRY="VQA_analysis/evaluators/gemma4_evaluator.py" ;;
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
# HF auth for gated / rate-limited model downloads (e.g. Llama-3.2 is gated).
if [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN="$(cat "$HOME/.hf_token")"
fi

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

# --- Evaluator interpreter selection -------------------------------------------------
# qwen2.5/internvl run in the main (uv) venv. phi4 and gemma4 need a different transformers
# than the repo venv (phi4: ==4.47.0 for its remote code; gemma4: >=5.5.2 for
# AutoModelForMultimodalLM), so they get a per-job PINNED venv on scratch, built once per
# job. The metrics steps below ALWAYS use the main venv (model-agnostic, run on 4.57.6).
EVAL_CMD=(uv run python)
case "$MODEL" in
  phi4|gemma4)
    EVAL_VENV="$WORK_DIR/.venv_eval"
    EVAL_VPY="$EVAL_VENV/bin/python"
    echo "Building pinned eval venv for $MODEL at $EVAL_VENV ..."
    uv venv --python 3.11 "$EVAL_VENV"
    uv pip install --python "$EVAL_VPY" --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1+cu121 torchvision==0.19.1+cu121
    if [ "$MODEL" = "phi4" ]; then
        uv pip install --python "$EVAL_VPY" \
            transformers==4.47.0 peft==0.13.2 accelerate==1.3.0 scipy==1.15.1 backoff==2.2.1 \
            Pillow soundfile sentencepiece protobuf tqdm numpy
    else  # gemma4
        uv pip install --python "$EVAL_VPY" \
            "transformers>=5.5.2" "peft>=0.19.0" accelerate bitsandbytes \
            Pillow sentencepiece protobuf tqdm numpy
    fi
    # flash-attn is optional (perf only); best-effort. If it does not build, set
    # use_flash_attention=false for this model in the config or the evaluator load will fail.
    uv pip install --python "$EVAL_VPY" flash-attn --no-build-isolation \
        || echo "flash-attn unavailable in eval venv; set use_flash_attention=false for $MODEL if load fails"
    # config (pure stdlib) + base_evaluator must import without the editable project install:
    # PYTHONPATH=repo-root exposes the `config` package; base_evaluator resolves via script dir.
    EVAL_CMD=(env "PYTHONPATH=$WORK_DIR" "$EVAL_VPY")
    ;;
esac

printf '\n=== %s — %s — QUR+FRR — %s ===\n' "$MODEL" "$CONFIG" "$DATASET"
"${EVAL_CMD[@]}" "$ENTRY" --config_path "$CONFIG" $FINETUNE --questions both
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
