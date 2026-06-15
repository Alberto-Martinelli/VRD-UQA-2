#!/usr/bin/env bash
# Shared SLURM prelude for the evaluation path. `source` this, then call the
# helpers. Bash stays in its lane (scheduler/env/exec); application logic lives
# in run_eval.py + the metrics steps.
#
# On source: module loads (skipped off-HPC so this is source-able locally for
# testing), env.sh, HF token. Helper functions are defined below.

if command -v module >/dev/null 2>&1; then
    module purge
    module load miniconda3/3.13.25
    module load nvhpc/25.1
fi
source "$HOME/VRD-UQA/scripts/env.sh"
if [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN="$(cat "$HOME/.hf_token")"
fi

# Deterministic, parallel-safe run id. Same string the eval path has always
# used: eval_<split_name>_<n>_<model>_<dataset>[_<RUN_TAG>].
make_run_id() {
    local model="$1" dataset="$2" split="$3"
    local n="${split##*_}" split_name="${split%_*}"
    printf 'eval_%s_%s_%s_%s%s' "$split_name" "$n" "$model" "$dataset" "${RUN_TAG:+_$RUN_TAG}"
}

# rsync the scratch run dir back to $HOME and refresh the 'latest' pointer.
# Safe on early exit / when nothing has been produced yet.
sync_run_back() {
    local run_id="${1:-${VQA_RUN_ID:-}}"
    [ -n "$run_id" ] || return 0
    local src="$WORK_DIR/artifacts/evaluation_runs/$run_id"
    local dest="$HOME/VRD-UQA/artifacts/evaluation_runs/$run_id"
    [ -d "$src" ] || return 0
    mkdir -p "$dest"
    rsync -aq "$src/" "$dest/" || true
    [ -n "${NO_LATEST:-}" ] || ln -sfn "$run_id" "$HOME/VRD-UQA/artifacts/evaluation_runs/latest" 2>/dev/null || true
}

# Per-job scratch workspace: sync repo in, install deps, arm the EXIT trap
# (sync results back, then tear down). Sets the global WORK_DIR.
setup_scratch_workdir() {
    WORK_DIR="$SCRATCH_FLASH/$1"
    trap 'sync_run_back; cd "$HOME"; rm -rf "$WORK_DIR"' EXIT
    rm -rf "$WORK_DIR"
    rsync -aq --exclude='data' --exclude='.git' --exclude='.venv' \
          --exclude='corruption-scripts/results' --exclude='finetuning' \
          "$HOME/VRD-UQA/" "$WORK_DIR/"
    cd "$WORK_DIR"
    uv --version
    export UV_LINK_MODE=copy
    uv sync -qq
}

# Restore a prior run's results from $HOME into scratch so a resubmit RESUMES
# (the evaluator skips finished leaves).
restore_prior_run() {
    local run_id="$1"
    local dest="$HOME/VRD-UQA/artifacts/evaluation_runs/$run_id"
    local src="$WORK_DIR/artifacts/evaluation_runs/$run_id"
    if [ -d "$dest" ]; then
        echo "RESUME: prior results at $dest — restoring into scratch; finished leaf will be skipped."
        mkdir -p "$src"
        rsync -aq "$dest/" "$src/"
    fi
}

# Select the interpreter for the eval step into the EVAL_CMD array.
# qwen2.5/internvl use the main uv venv. phi4/gemma4 need a different
# transformers, so they get a per-job PINNED venv on scratch. (This only MOVES
# the existing inline build into a function; caching is a later pass.)
activate_eval_venv() {
    local model="$1"
    EVAL_CMD=(uv run python)
    case "$model" in
      phi4|gemma4)
        local venv="$WORK_DIR/.venv_eval"
        echo "Building pinned eval venv for $model at $venv ..."
        uv venv --python 3.11 "$venv"
        uv pip install --python "$venv/bin/python" --index-url https://download.pytorch.org/whl/cu121 \
            torch==2.4.1+cu121 torchvision==0.19.1+cu121
        if [ "$model" = "phi4" ]; then
            uv pip install --python "$venv/bin/python" \
                transformers==4.47.0 peft==0.13.2 accelerate==1.3.0 scipy==1.15.1 backoff==2.2.1 \
                Pillow soundfile sentencepiece protobuf tqdm numpy
        else
            uv pip install --python "$venv/bin/python" \
                "transformers>=5.5.2" "peft>=0.19.0" accelerate bitsandbytes \
                Pillow sentencepiece protobuf tqdm numpy
        fi
        uv pip install --python "$venv/bin/python" flash-attn --no-build-isolation \
            || echo "flash-attn unavailable in eval venv; set use_flash_attention=false for $model if load fails"
        EVAL_CMD=(env "PYTHONPATH=$WORK_DIR" "$venv/bin/python")
        ;;
    esac
}
