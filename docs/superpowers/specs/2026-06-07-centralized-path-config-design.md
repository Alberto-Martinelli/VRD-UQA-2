# Centralized Path Configuration — Design

**Date:** 2026-06-07
**Status:** Approved
**Scope:** Eliminate duplicated, hardcoded environment-specific paths across the repo by giving them a single source of truth shared between the SLURM bash scripts and the Python code.

## Problem

Environment-specific paths are defined independently in many files. Changing where data lives requires editing 5+ files, and the copies have already drifted:

- `SCRATCH_FLASH = /mnt/beegfs/amartinelli/` — defined in `datasets_api/BDocs_dataset.py`, `datasets_api/SlideVQA_dataset.py`, and 5 bash scripts under `scripts/slurm/` plus `corruption-scripts/verification/run_verification.sh`.
- `PROJECT_DIR` / `REPO_ROOT = /home/amartinelli/VRD-UQA/` — `datasets_api/datasets_utils.py` and `finetuning/build_dataset.py`.
- `SOURCE_DIR = /home/amartinelli/MPDocVQA/MPDocVQA_complete/qas` — `datasets_api/MPDocVQA_dataset.py` (external MPDocVQA raw QAs).
- `DUDE_IMAGE_DIR`, `MPDOCVQA_IMAGE_DIR` — spell out `/mnt/beegfs/amartinelli/<X>_images` in full, while BDocs/SlideVQA already derive it as `{SCRATCH_FLASH}/<X>_images`.
- `HF_HOME = {SCRATCH_FLASH}/.cache/huggingface` — repeated in every bash script.

### Drift / latent bugs found
- `test_hpc.sh` uses `/mnt/beegfs-compat/amartinelli` — a different mount from everything else. **Decision: drift; normalize to `/mnt/beegfs/amartinelli`.**
- Trailing-slash inconsistency (`.../amartinelli/` vs `.../amartinelli`).
- `PROJECT_DIR` / `REPO_ROOT` are hardcoded to the `$HOME` copy, but the SLURM scripts `rsync` the repo to `$SCRATCH_FLASH/<workdir>` and `cd` there at runtime — so the hardcoded value is **wrong on the compute node**. `save_sample()` writes back to the home copy instead of the running work-dir copy.

## Goals

- One file owns the environment roots for bash; one file owns them (plus all derivation) for Python; the two agree by construction.
- Eliminate all independently hardcoded copies of the five variables. After the change, `grep -rn` for them returns only the two new source-of-truth files.
- Fix the repo-root relocation bug by self-locating the repo root in Python.
- No new dependencies; no Python subprocess calls from bash.

## Non-goals (YAGNI)

- Not touching `corruption-scripts/config*.json` or `VQA_analysis/config_*.json` — a separate concern (different schema, different fix).
- No `python-dotenv` or any added dependency.
- No `.env.local` / secrets split — single-user research repo, values already committed.
- No unrelated refactoring of the dataset modules beyond the path constants.

## Chosen approach

**Shared `scripts/env.sh` + `config/paths.py`, bridged by environment-variable inheritance.** Bash owns the root *values* via exports in `env.sh`; Python derives the *structure* in `paths.py`, reading the same roots from `os.environ` with matching defaults. SLURM scripts `source scripts/env.sh` before `uv run python ...`, so the exported vars flow into `os.getenv`.

Rejected alternatives:
- *Python-only `paths.py`, bash calls `python -c`* — adds a subprocess call and startup cost to every bash script.
- *`.env` + python-dotenv* — adds a dependency, and `.env` can only hold flat roots (derived paths still live in `paths.py`).

## Design

### New file: `scripts/env.sh`

Canonical roots for bash. Sourced by every SLURM script. The `${VAR:-default}` form lets any machine override by exporting before sourcing.

```bash
# Single source of truth for environment roots (bash side).
# Mirror of config/paths.py defaults — keep the two in sync.
# Override any value by exporting it before sourcing this file.
export SCRATCH_FLASH="${SCRATCH_FLASH:-/mnt/beegfs/amartinelli}"
export MPDOCVQA_SOURCE_QAS="${MPDOCVQA_SOURCE_QAS:-/home/amartinelli/MPDocVQA/MPDocVQA_complete/qas}"
export HF_HOME="$SCRATCH_FLASH/.cache/huggingface"
export VRD_UQA_HOME="${VRD_UQA_HOME:-$HOME/VRD-UQA}"   # persistent repo, used as rsync source
```

### New file: `config/paths.py`

Canonical roots + all derivation for Python. `REPO_ROOT` is self-locating (never hardcoded), which fixes the relocation bug. Roots come from `os.getenv` with defaults that mirror `env.sh` (env vars win at runtime).

```python
from pathlib import Path
import os

# Self-locating: correct regardless of whether the repo runs from $HOME or a
# relocated SLURM work-dir. Fixes save_sample() writing to the wrong copy.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Roots mirror scripts/env.sh defaults; env vars take precedence at runtime.
SCRATCH_FLASH = Path(os.getenv("SCRATCH_FLASH", "/mnt/beegfs/amartinelli"))
MPDOCVQA_SOURCE_QAS = Path(
    os.getenv("MPDOCVQA_SOURCE_QAS", "/home/amartinelli/MPDocVQA/MPDocVQA_complete/qas")
)

def image_dir(dataset_name: str) -> str:
    """e.g. image_dir("DUDE") -> /mnt/beegfs/amartinelli/DUDE_images"""
    return str(SCRATCH_FLASH / f"{dataset_name}_images")
```

**Open decision resolved:** default fallback (shown above) over fail-loud, so Python runs on a dev machine without sourcing `env.sh`. Cost: the default literal lives in both files, kept in sync by a comment.

**Importability:** the only Python consumers are `datasets_api/` and `finetuning/`, both of which already run with repo-root on `sys.path` (they use `from datasets_api.x import ...`). `REPO_ROOT` self-locates via `__file__`, so `paths.py` works regardless of entry point. The corruption-scripts get their paths from `config.json`, not these constants, so they are untouched.

### Migration — call sites unchanged, only definitions move

Existing getters (`get_DUDE_image_dir()`, etc.) are kept so no consumer breaks; only their bodies are gutted to delegate.

| File | Before | After |
|------|--------|-------|
| `datasets_api/DUDE_dataset.py` | `DUDE_IMAGE_DIR = '/mnt/.../DUDE_images'` | `from config import paths`; `return paths.image_dir("DUDE")` |
| `datasets_api/MPDocVQA_dataset.py` | `SOURCE_DIR=...`, `MPDOCVQA_IMAGE_DIR=...` | `paths.MPDOCVQA_SOURCE_QAS`, `paths.image_dir("MPDocVQA")` |
| `datasets_api/BDocs_dataset.py` | `SCRATCH_FLASH=...`, `IMAGES_PATH=...` | `paths.image_dir("BDocs")` |
| `datasets_api/SlideVQA_dataset.py` | `SCRATCH_FLASH=...`, `IMAGES_PATH=...` | `paths.image_dir("SlideVQA")` |
| `datasets_api/datasets_utils.py` | `PROJECT_DIR = '/home/...'` | `paths.REPO_ROOT` |
| `finetuning/build_dataset.py` | `REPO_ROOT = Path('/home/...')` | `from config.paths import REPO_ROOT` |
| `scripts/slurm/*.sh` (5 files) | `export SCRATCH_FLASH=...` + `export HF_HOME=...` | `source` the shared `scripts/env.sh` (relative path adjusted per script location) |
| `corruption-scripts/verification/run_verification.sh` | same two exports | `source` shared `scripts/env.sh` |
| `test_hpc.sh` | `beegfs-compat` value | `source scripts/env.sh` (normalized to `/mnt/beegfs`) |

### Normalization rules
- No trailing slashes anywhere; all joins via `pathlib` / `os.path.join`.
- `/mnt/beegfs-compat/` → `/mnt/beegfs/`.

## Verification

- `grep -rn` for `SCRATCH_FLASH|PROJECT_DIR|SOURCE_DIR|MPDOCVQA_IMAGE_DIR|DUDE_IMAGE_DIR` (excluding `.venv`, `__pycache__`) returns **only** `scripts/env.sh` and `config/paths.py`.
- `python -c "from config import paths; print(paths.image_dir('DUDE'), paths.REPO_ROOT)"` prints the correct paths.
- `bash -c 'source scripts/env.sh && echo $SCRATCH_FLASH $HF_HOME'` prints normalized values.
- Each migrated dataset module still imports cleanly, and its `get_*_image_dir()` returns the same string as before the change.
