# Thin SLURM Eval Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the evaluation SLURM path a thin wrapper — extract shared boilerplate into `scripts/slurm/_prelude.sh` and move config resolution + overrides + model→class routing out of `run_vqa_analysis.sh` into `VQA_analysis/evaluators/run_eval.py`.

**Architecture:** Bash keeps only scheduler + environment + exec (prelude helper functions). A by-path Python entrypoint `run_eval.py` holds a model→class factory (importlib, loads only the requested model) and applies in-memory config overrides (killing the `python -c` mutation). `BaseVQAEvaluator.__init__` is extended to accept a config dict or path. **No behavior change** — same runs, run-ids, run-dir layout, outputs.

**Tech Stack:** Python 3.12 (`argparse`, `importlib.util`), bash, `uv`. Tests are plain-assert scripts run via `uv run python -m tests.<name>` (repo has no pytest).

**Reference spec:** `docs/superpowers/specs/2026-06-15-thin-slurm-eval-wrapper-design.md`

**Branch:** `feat/thin-slurm-eval-wrapper` (spec already committed here).

**Convention:** This repo keeps uncommitted analysis files in the tree — **always `git add` explicit paths, never `git add -A`/`.`**

---

## File Structure

- **Modify** `VQA_analysis/evaluators/base_evaluator.py:21-23` — `__init__` accepts config dict *or* path.
- **Create** `VQA_analysis/evaluators/run_eval.py` — unified by-path entrypoint: factory + in-memory overrides.
- **Create** `scripts/slurm/_prelude.sh` — shared SLURM helpers (module/env/HF + scratch/run-id/venv functions).
- **Modify** `scripts/slurm/run_vqa_analysis.sh` — rewritten thin (~35 lines).
- **Create** `tests/test_base_evaluator_config.py` — unit test for the dict-or-path `__init__`.
- **Modify** `tests/test_evaluator_mock.py` — re-point to drive `run_eval.py` for all 4 models.

---

## Task 1: `BaseVQAEvaluator.__init__` accepts a dict or a path

**Files:**
- Create: `tests/test_base_evaluator_config.py`
- Modify: `VQA_analysis/evaluators/base_evaluator.py:21-23`

- [ ] **Step 1: Write the failing test**

Create `tests/test_base_evaluator_config.py`:

```python
"""Unit test: BaseVQAEvaluator.__init__ accepts a config dict OR a path.
Run: uv run python -m tests.test_base_evaluator_config"""
import importlib.util
import json
import tempfile
from pathlib import Path
from config.paths import REPO_ROOT


def _load_base():
    path = REPO_ROOT / "VQA_analysis" / "evaluators" / "base_evaluator.py"
    spec = importlib.util.spec_from_file_location("base_evaluator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dummy_cls(base):
    class Dummy(base.BaseVQAEvaluator):
        MODEL_KEY = "m"
    return Dummy


def _cfg():
    return {"open_source_models": {"m": {"max_tokens": 7}}, "seed": 13, "mock": True}


def test_init_accepts_dict():
    base = _load_base()
    e = _dummy_cls(base)(_cfg(), finetuned=False)
    assert e.config["seed"] == 13
    assert e.seed == 13
    assert e.max_tokens == 7


def test_init_still_accepts_path():
    base = _load_base()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cfg.json"
        p.write_text(json.dumps(_cfg()))
        e = _dummy_cls(base)(str(p), finetuned=False)
        assert e.seed == 13


if __name__ == "__main__":
    test_init_accepts_dict()
    test_init_still_accepts_path()
    print("OK: base_evaluator dict-or-path __init__")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/amartinelli/VRD-UQA && uv run python -m tests.test_base_evaluator_config`
Expected: FAIL — `test_init_accepts_dict` raises `TypeError` (current `__init__` does `open(config)` on a dict).

- [ ] **Step 3: Implement the change**

In `VQA_analysis/evaluators/base_evaluator.py`, replace lines 21-23:
```python
    def __init__(self, config_path, finetuned, questions="both"):
        with open(config_path) as f:
            self.config = json.load(f)
```
with:
```python
    def __init__(self, config, finetuned, questions="both"):
        # config may be a path (loaded here) or an already-built dict (passed by run_eval.py).
        if isinstance(config, (str, os.PathLike)):
            with open(config) as f:
                self.config = json.load(f)
        else:
            self.config = config
```
(`os` and `json` are already imported at the top of the file.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/amartinelli/VRD-UQA && uv run python -m tests.test_base_evaluator_config`
Expected: `OK: base_evaluator dict-or-path __init__`, exit 0.

- [ ] **Step 5: Confirm the per-evaluator entrypoints still construct correctly (path path unchanged)**

Run: `cd /home/amartinelli/VRD-UQA && uv run python -m tests.test_evaluator_mock`
Expected: PASS (still uses per-file entrypoints + path at this point) — `OK: evaluator mock dual-answer (all 4 models)`.

- [ ] **Step 6: Commit**

```bash
git add VQA_analysis/evaluators/base_evaluator.py tests/test_base_evaluator_config.py
git commit -m "feat(eval): BaseVQAEvaluator.__init__ accepts a config dict or path

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `run_eval.py` factory + in-memory overrides; re-point the mock test

**Files:**
- Create: `VQA_analysis/evaluators/run_eval.py`
- Modify: `tests/test_evaluator_mock.py`

- [ ] **Step 1: Re-point the mock test to drive `run_eval.py` (failing test first)**

In `tests/test_evaluator_mock.py`, replace the `MODELS` list (the 4-tuple block under the module docstring/imports) with this 3-tuple form:
```python
# (cli_model_key, leaf_prefix, finetuned). Qwen has an adapter entry (finetuned);
# the others run zero-shot. cli_model_key is the run_eval.py --model value.
MODELS = [
    ("qwen2.5",  "",         True),
    ("phi4",     "phi4",     False),
    ("internvl", "internvl", False),
    ("gemma4",   "gemma",    False),
]
```

Replace the `_run` function with:
```python
def _run(model, cfg, run_id, questions, run_root, finetuned, input_file=SAMPLE):
    env = dict(os.environ)
    env["VQA_RUN_ID"] = run_id
    env["VQA_EVAL_RUNS_DIR"] = str(run_root)
    cmd = ["uv", "run", "python", "VQA_analysis/evaluators/run_eval.py",
           "--model", model, "--dataset", "BDocs", "--split", "val_15",
           "--config", str(cfg), "--input-file", str(input_file),
           "--questions", questions]
    if finetuned:
        cmd.append("--finetuned")
    subprocess.check_call(cmd, cwd=str(REPO_ROOT), env=env)
```

In `test_all_models_dual_answer_and_namespaced_leaf`, change the loop unpacking and the `_run` call. Replace:
```python
    for i, (key, entry, prefix, finetuned) in enumerate(MODELS):
```
with:
```python
    for i, (key, prefix, finetuned) in enumerate(MODELS):
```
and replace:
```python
        _run(entry, cfg, run_id, "both", run_root, finetuned)
```
with:
```python
        _run(key, cfg, run_id, "both", run_root, finetuned)
```

In `test_corrupted_only_omits_clean_qwen` and `test_resume_skips_completed_leaf_qwen`, replace every `MODELS[0][1]` with `MODELS[0][0]` (the qwen CLI key is now at index 0). There are three such call sites total (one in `test_corrupted_only_omits_clean_qwen`, two in `test_resume_skips_completed_leaf_qwen`).

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/amartinelli/VRD-UQA && uv run python -m tests.test_evaluator_mock`
Expected: FAIL — `run_eval.py` does not exist yet (`python: can't open file '.../run_eval.py'` → `CalledProcessError`).

- [ ] **Step 3: Create `run_eval.py`**

Create `VQA_analysis/evaluators/run_eval.py`:
```python
"""Unified by-path eval launcher: model -> evaluator class factory + in-memory
config overrides. Replaces the bash `case` routing and the `python -c` config
mutation in run_vqa_analysis.sh.

Run by-path so it works under the phi4/gemma4 pinned venv (no editable install):
    python VQA_analysis/evaluators/run_eval.py --model qwen2.5 --dataset BDocs --split val_300
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path

from config.paths import REPO_ROOT

HERE = Path(__file__).resolve().parent

# CLI model key -> (evaluator module filename beside this file, class name).
EVALUATORS = {
    "qwen2.5":  ("qwen2.5_evaluator.py",  "QwenVQAEvaluator"),
    "phi4":     ("phi4_evaluator.py",     "Phi4VQAEvaluator"),
    "internvl": ("internvl_evaluator.py", "InternVLVQAEvaluator"),
    "gemma4":   ("gemma4_evaluator.py",   "Gemma4VQAEvaluator"),
}


def _load_evaluator_class(model):
    """Import ONLY the requested evaluator module. importlib handles the dot in
    'qwen2.5_evaluator.py', and loading just one module keeps the pinned venv
    from importing other models' incompatible deps."""
    module_file, class_name = EVALUATORS[model]
    mod_name = module_file[:-3].replace(".", "_")  # qwen2.5_evaluator -> qwen2_5_evaluator
    spec = importlib.util.spec_from_file_location(mod_name, HERE / module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _resolve_input_file(dataset, split):
    return str(
        REPO_ROOT / "data" / dataset / f"{dataset}_{split}"
        / f"{dataset}_unanswerable_corrupted_questions_just_false.json"
    )


def main():
    parser = argparse.ArgumentParser(description="Unified VQA evaluator launcher.")
    parser.add_argument("--model", required=True, choices=sorted(EVALUATORS))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="val_300")
    parser.add_argument("--config", default="VQA_analysis/config_fewshot.json")
    parser.add_argument("--input-file", default=None,
                        help="Explicit eval input; overrides the data/<dataset>/<split> derivation.")
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--questions", choices=["both", "corrupted", "clean"], default="both")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    config["dataset"] = args.dataset
    config["split"] = args.split.split("_")[0]
    config["input_file"] = args.input_file or _resolve_input_file(args.dataset, args.split)

    # Provenance for the run manifest (base_evaluator reads VQA_CONFIG_PATH).
    os.environ.setdefault("VQA_CONFIG_PATH", args.config)

    evaluator_cls = _load_evaluator_class(args.model)
    evaluator = evaluator_cls(config, args.finetuned, questions=args.questions)
    print(f"Running {args.model} | dataset={args.dataset} split={args.split} "
          f"finetuned={args.finetuned} questions={args.questions} seed={evaluator.seed}")
    evaluator.evaluate()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/amartinelli/VRD-UQA && uv run python -m tests.test_evaluator_mock`
Expected: `OK: evaluator mock dual-answer (all 4 models)`, exit 0. (All 4 models now route through `run_eval.py` and construct via the config dict.)

- [ ] **Step 5: Confirm `python -c` is no longer needed — sanity check the CLI directly**

Run:
```bash
cd /home/amartinelli/VRD-UQA && uv run python VQA_analysis/evaluators/run_eval.py --help
```
Expected: argparse help listing `--model {gemma4,internvl,phi4,qwen2.5} --dataset --split --config --input-file --finetuned --questions`.

- [ ] **Step 6: Commit**

```bash
git add VQA_analysis/evaluators/run_eval.py tests/test_evaluator_mock.py
git commit -m "feat(eval): unified run_eval.py factory + in-memory config overrides

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `_prelude.sh` + thin `run_vqa_analysis.sh`

**Files:**
- Create: `scripts/slurm/_prelude.sh`
- Modify: `scripts/slurm/run_vqa_analysis.sh` (full rewrite)

- [ ] **Step 1: Create `scripts/slurm/_prelude.sh`**

```bash
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
        echo "RESUME: restoring $dest into scratch"
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
```

- [ ] **Step 2: Rewrite `scripts/slurm/run_vqa_analysis.sh`**

Replace the **entire file** with:
```bash
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
```

- [ ] **Step 3: Syntax-check both scripts**

Run:
```bash
cd /home/amartinelli/VRD-UQA && bash -n scripts/slurm/_prelude.sh && bash -n scripts/slurm/run_vqa_analysis.sh && echo "OK bash -n"
```
Expected: `OK bash -n`.

- [ ] **Step 4: Functionally test the prelude helpers locally (module loads auto-skip off-HPC)**

Run:
```bash
cd /home/amartinelli/VRD-UQA && bash -c '
set -uo pipefail
source scripts/slurm/_prelude.sh
r="$(make_run_id qwen2.5 BDocs val_300)"
echo "run_id=$r"
[ "$r" = "eval_val_300_qwen2.5_BDocs" ] || { echo FAIL; exit 1; }
r2="$(RUN_TAG=foo make_run_id phi4 DUDE val_100)"
[ "$r2" = "eval_val_100_phi4_DUDE_foo" ] || { echo FAIL2; exit 1; }
echo OK
'
```
Expected: `run_id=eval_val_300_qwen2.5_BDocs` then `OK`. (Confirms `env.sh` sources and `make_run_id` matches the legacy run-id string.)

- [ ] **Step 5: Confirm the bash hell is gone (structural checks)**

Run:
```bash
cd /home/amartinelli/VRD-UQA
echo "lines: $(wc -l < scripts/slurm/run_vqa_analysis.sh)"
echo "python -c count: $(grep -c 'python -c' scripts/slurm/run_vqa_analysis.sh)"
echo "model-routing case count: $(grep -c 'case "\$MODEL"' scripts/slurm/run_vqa_analysis.sh)"
```
Expected: `lines:` ≤ 40; `python -c count: 0`; `model-routing case count: 0`. (The remaining `case` is the venv selection, which lives in `_prelude.sh` and is legitimate.)

- [ ] **Step 6: Commit**

```bash
git add scripts/slurm/_prelude.sh scripts/slurm/run_vqa_analysis.sh
git commit -m "refactor(slurm): thin run_vqa_analysis.sh via _prelude.sh + run_eval.py

Removes the python -c config mutation and the model->script case routing;
boilerplate (module/env/HF, scratch setup, run-id, venv select, sync-back)
moves to scripts/slurm/_prelude.sh helpers. No behavior change.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full CI-safe test suite**

Run each and confirm it prints its `OK` line / exits 0:
```bash
cd /home/amartinelli/VRD-UQA
for t in test_paths test_base_evaluator_config test_evaluator_mock \
         test_metrics_normalize test_metrics_enrich test_metrics_compute \
         test_pipeline_integration test_aggregate_summaries test_run_layout \
         test_finetune_configs test_finetune_manifest test_gemma4_finetune; do
  echo "=== $t ==="; uv run python -m tests.$t || echo "FAILED: $t"
done
```
Expected: every test passes; no `FAILED:` lines.

- [ ] **Step 2: Confirm the design's verification gates**

Run:
```bash
cd /home/amartinelli/VRD-UQA
grep -n "python -c" scripts/slurm/run_vqa_analysis.sh || echo "no python -c (good)"
bash -n scripts/slurm/_prelude.sh && bash -n scripts/slurm/run_vqa_analysis.sh && echo "bash -n clean"
uv run python VQA_analysis/evaluators/run_eval.py --help >/dev/null && echo "run_eval CLI ok"
```
Expected: `no python -c (good)`, `bash -n clean`, `run_eval CLI ok`.

- [ ] **Step 3: Confirm clean tree state and branch history**

Run: `cd /home/amartinelli/VRD-UQA && git status -s && git log --oneline feat/thin-slurm-eval-wrapper -6`
Expected: only intended files changed (no stray analysis files swept in); log shows the spec commit + Task 1-3 commits.

- [ ] **Step 4 (handoff):** Implementation complete and CI-verified. The HPC smoke run (`sbatch scripts/slurm/run_vqa_analysis.sh qwen2.5 BDocs val_5`) is a **user follow-up** (needs the A40). Then use `superpowers:finishing-a-development-branch` to integrate `feat/thin-slurm-eval-wrapper`.
```
