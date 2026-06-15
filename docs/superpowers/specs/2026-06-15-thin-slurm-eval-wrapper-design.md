# Thin SLURM Eval Wrapper — Design

**Date:** 2026-06-15
**Status:** Approved
**Scope:** Pass 1 of the "kill bash hell" effort — make the **evaluation** SLURM path a thin wrapper by extracting shared boilerplate into a prelude and moving application logic (config resolution, model routing, config overrides) out of bash into a small Python entrypoint.

## Problem

`scripts/slurm/run_vqa_analysis.sh` is 159 lines and conflates four jobs, only one of which needs bash:

1. SLURM contract (`#SBATCH`) — irreducibly bash.
2. Environment activation (`module load`, `source env.sh`, venv activate) — shell-only.
3. `exec` the program — bash.
4. **Application logic** — `python -c` config mutation, model→script `case` routing, run-id derivation, scratch rsync/resume/sync-back, venv build. This is ~80% of the lines and is unreadable + untestable in bash.

Boilerplate is duplicated across the SLURM scripts (`module load` + `source env.sh` in 10/10, `rsync` in 8/10, `uv venv` in 8/10).

Two specific smells this pass fixes:
- **In-place JSON mutation:** `run_vqa_analysis.sh` patches a scratch copy of the config via inline `python -c "json.load … json.dump"`.
- **Split responsibility:** a bash `case` maps `qwen2.5 → qwen2.5_evaluator.py`, putting model routing in bash.

## Goals

- `run_vqa_analysis.sh` drops from 159 lines to ~30 readable lines: scheduler + environment + a single Python `exec` + the (already clean) metrics lines.
- Application logic (config load + overrides, model→class routing, the eval run) lives in testable Python.
- **No behavior change.** Same eval runs, same run-ids, same run-dir layout, same outputs. Verified in mock mode + `bash -n`; HPC validation is a follow-up only if a smoke run surfaces a bug.

## Non-goals (deferred to pass 2+)

- `run_fewshot_ksweep.sh` (the other 170-line script) and the finetune scripts adopting the prelude.
- **Environment caching** (pain point 2). This pass only *moves* the per-job venv build into a named prelude function; it does not yet cache/reuse it. No behavior change.
- Removing the now-redundant per-evaluator `main()` functions (kept for back-compat).
- Hydra or any new dependency. Native `argparse` only.

## Hard constraints (must not break)

- **Pinned-venv isolation:** phi4/gemma4 run in a per-job venv with **no editable project install** — only `PYTHONPATH=repo-root` exposes the `config` package, and `base_evaluator` resolves via the script directory. The new entrypoint must work under this.
- **Import only the requested model.** The pinned venv must never import the other models' (incompatible) deps. The factory loads only the one evaluator module requested.
- **`qwen2.5_evaluator.py` has a dot in its name** and cannot be a normal `import`; it is run by-path today. The factory must load it via `importlib.util.spec_from_file_location`.
- **Parallel-safe, resumable run-ids** (one model×dataset×split per job) — preserved exactly; run-id/leaf logic already lives in `config/run_layout.py`.

## Chosen approach

Thin bash wrapper + shared `_prelude.sh` + a by-path Python entrypoint `run_eval.py` that holds the factory and in-memory config overrides. Decisions locked during brainstorming:
- **CLI scope:** eval step only; the 3 metrics steps stay as explicit `python metrics/N_*.py --run-id` lines (already clean, on the main venv).
- **CLI home:** `VQA_analysis/evaluators/run_eval.py`, run by-path (reuses the existing by-path + script-dir import convention; works under the pinned no-editable-install venv; no new package, no `pyproject` change).
- **Override mechanism:** in-memory dict (no file written). `BaseVQAEvaluator.__init__` accepts a dict *or* a path.

## Design

### Component boundary

| Concern | Lands in | Rationale |
|---|---|---|
| `#SBATCH`, `module load`, `source env.sh`, HF token | `scripts/slurm/_prelude.sh` | scheduler + shell-only |
| scratch workdir, rsync repo→scratch, `uv sync`, teardown trap | `_prelude.sh` (`setup_scratch_workdir`) | genuine HPC infra |
| pinned venv build/activate for phi4/gemma4 | `_prelude.sh` (`activate_eval_venv`) | interpreter selection is irreducibly bash |
| run-dir resume + sync-back | `run_vqa_analysis.sh` (calling prelude helpers) | eval-specific (finetune differs) |
| config load + overrides, model→class routing, eval run | `VQA_analysis/evaluators/run_eval.py` | application logic → testable Python |
| normalize → enrich → compute | three explicit lines in `run_vqa_analysis.sh` | already clean one-liners on the main venv |

### `scripts/slurm/_prelude.sh` (new)

Sourced by the eval script. On source: `module purge`/load, `source env.sh`, load `$HOME/.hf_token` → `HF_TOKEN`. Defines helper functions (kept bash because they are HPC/shell concerns):

- `setup_scratch_workdir <name>` — sets `WORK_DIR=$SCRATCH_FLASH/<name>`, `rm -rf` + rsync repo→scratch (same excludes as today), `cd "$WORK_DIR"`, `uv sync -qq`, and installs the `trap … rm -rf "$WORK_DIR"` teardown.
- `make_run_id <model> <dataset> <split>` — emits the deterministic run-id (same string the script builds today: `eval_<split_name>_<n>_<model>_<dataset>` with optional `RUN_TAG`). Single source so bash (sync paths) and Python (writing) agree; Python side already derives the same via `run_layout`/`VQA_RUN_ID`.
- `restore_prior_run <run_id>` — if a prior `$HOME` run dir exists, rsync it into scratch (resume).
- `activate_eval_venv <model>` — for phi4/gemma4: build + select the pinned venv exactly as today (per-job, `.venv_eval`, version-pinned installs, best-effort flash-attn), exporting the interpreter/`PYTHONPATH` the script then uses; for qwen2.5/internvl: no-op (main `uv` venv). Behavior identical to the current inline blocks, just named.
- `sync_run_back <run_id>` — rsync the scratch run dir back to `$HOME`, update the `latest` symlink (opt-out `NO_LATEST`), same as today's `sync_back`.

Pass 1 only requires the eval script to adopt these; other scripts adopt later.

### `VQA_analysis/evaluators/run_eval.py` (new)

A by-path entrypoint. Registry + importlib loader:

```python
EVALUATORS = {  # CLI key -> (module file beside this one, class name)
    "qwen2.5":  ("qwen2.5_evaluator.py", "QwenVQAEvaluator"),
    "phi4":     ("phi4_evaluator.py",    "Phi4VQAEvaluator"),
    "internvl": ("internvl_evaluator.py","InternVLVQAEvaluator"),
    "gemma4":   ("gemma4_evaluator.py",  "Gemma4VQAEvaluator"),
}
```

(Class names verified against the modules: `QwenVQAEvaluator`, `Phi4VQAEvaluator`, `InternVLVQAEvaluator`, `Gemma4VQAEvaluator` — all subclass `BaseVQAEvaluator`.)

CLI: `--model {qwen2.5,phi4,internvl,gemma4}` (required), `--dataset` (required), `--split` (default `val_300`), `--config` (default `VQA_analysis/config_fewshot.json`), `--finetuned` (flag), `--questions {both,corrupted,clean}` (default `both`).

Flow:
1. Resolve the requested module path relative to this file; `importlib.util.spec_from_file_location` to load **only** that one module; get the class. (Dot-name safe; pinned-venv safe — no other model imported.)
2. Load the config JSON into a dict; apply in-memory overrides:
   - `cfg["dataset"] = dataset`
   - `cfg["split"] = split.split("_")[0]`
   - `cfg["input_file"] = data/{dataset}/{dataset}_{split}/{dataset}_unanswerable_corrupted_questions_just_false.json` (absolute, `REPO_ROOT`-relative via `config.paths`)
3. Construct the evaluator with the dict + `finetuned` + `questions`; call `.evaluate()`.

Run-id and run-dir come from the existing env (`VQA_RUN_ID`) + `config/run_layout.py`, unchanged. No config file is written.

### `VQA_analysis/evaluators/base_evaluator.py` (one change, back-compat)

`__init__(self, config, finetuned, questions="both")`: if `config` is a `str`/`Path`, load JSON from it (today's behavior); if it is a `dict`, use it directly. Everything else unchanged. The per-evaluator `main()` functions keep passing a path, so they and any by-path usage still work. The manifest's `config_path` provenance field continues to read `VQA_CONFIG_PATH` (set by the script to the base config name).

### `scripts/slurm/run_vqa_analysis.sh` (rewritten thin, ~30 lines)

```bash
#!/usr/bin/env bash
#SBATCH ...headers (unchanged)...
set -uo pipefail
source "$(dirname "$0")/_prelude.sh"
MODEL=$1 DATASET=$2 SPLIT=${3:-val_300}
setup_scratch_workdir "VQA_analysis_${SLURM_JOB_ID}"
export VQA_RUN_ID="${VQA_RUN_ID:-$(make_run_id "$MODEL" "$DATASET" "$SPLIT")}"
restore_prior_run "$VQA_RUN_ID"
activate_eval_venv "$MODEL"
python VQA_analysis/evaluators/run_eval.py --model "$MODEL" --dataset "$DATASET" --split "$SPLIT" \
       --config "${CONFIG:-VQA_analysis/config_fewshot.json}" ${FINETUNE:+--finetuned} --questions both
sync_run_back "$VQA_RUN_ID"
uv run python VQA_analysis/metrics/1_normalize_unanswerable_responses.py --run-id "$VQA_RUN_ID"
uv run python VQA_analysis/metrics/2_enrich_metadata.py            --run-id "$VQA_RUN_ID"
uv run python VQA_analysis/metrics/3_compute_metrics.py            --run-id "$VQA_RUN_ID"
```

The `python` for the eval line is whatever `activate_eval_venv` selected (main `uv` venv for qwen/internvl, pinned interpreter for phi4/gemma4). The metrics lines always use the main `uv` venv.

## Testing

- **Re-point `tests/test_evaluator_mock.py`** to drive `run_eval.py --model <key> --dataset BDocs --split val_15` (mock mode) for all 4 models, instead of invoking the per-file entrypoints. Keeps the predictions/leaf-dir assertions; now also proves the factory + in-memory overrides. Run: `uv run python -m tests.test_evaluator_mock`.
- **`bash -n`** on `scripts/slurm/_prelude.sh` and `scripts/slurm/run_vqa_analysis.sh`.
- The other CI tests (`test_paths`, metrics tests, etc.) must remain green.

HPC smoke (`sbatch run_vqa_analysis.sh qwen2.5 BDocs val_5`) is a user follow-up; not required for this pass to be considered done.

## Verification

- `run_vqa_analysis.sh` contains no `python -c`, no `case "$MODEL"` script-routing, and is ≤ ~35 lines.
- `grep -n "python -c" scripts/slurm/run_vqa_analysis.sh` → none.
- `uv run python -m tests.test_evaluator_mock` passes for all 4 models via `run_eval.py`.
- `bash -n` clean on both scripts.
- Run-dir layout for a mock run matches the pre-refactor layout (same `<run_id>/<dataset>/<leaf>` paths).
