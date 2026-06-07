# Centralized Path Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every environment-specific path in the repo a single source of truth — `scripts/env.sh` for bash, `config/paths.py` for Python — so changing a machine path means editing one place.

**Architecture:** Bash scripts `source scripts/env.sh` (exports root values as env vars). `config/paths.py` reads the same roots via `os.getenv` with matching defaults and derives every dataset/image/cache path. The two agree by environment-variable inheritance: SLURM scripts source `env.sh` before `uv run python ...`, so the exports flow into `os.getenv`. No new dependencies, no subprocess calls.

**Tech Stack:** Python 3.12 (`pathlib`, `os.getenv`), bash, `uv` for running. No test framework exists in the repo, so tests are plain-`assert` scripts run via `uv run python -m tests.<name>` (the repo's existing `-m` idiom, e.g. `python -m datasets_api.X`).

**Reference spec:** `docs/superpowers/specs/2026-06-07-centralized-path-config-design.md`

**Branch:** `centralize-paths` (already created; the spec commit is its first commit).

---

## File Structure

**New files:**
- `config/__init__.py` — empty; makes `config` an importable package (mirrors `datasets_api/__init__.py`, `finetuning/__init__.py`).
- `config/paths.py` — Python source of truth: roots (`REPO_ROOT`, `SCRATCH_FLASH`, `MPDOCVQA_SOURCE_QAS`) + `image_dir()` derivation.
- `scripts/env.sh` — bash source of truth: `export`s of `SCRATCH_FLASH`, `MPDOCVQA_SOURCE_QAS`, `HF_HOME`, `VRD_UQA_HOME`.
- `tests/__init__.py` — empty; makes `tests` a package for `-m` running.
- `tests/test_paths.py` — assertion script validating `config/paths.py`.

**Modified files (definitions removed, call sites unchanged):**
- `datasets_api/MPDocVQA_dataset.py`, `datasets_api/DUDE_dataset.py`, `datasets_api/BDocs_dataset.py`, `datasets_api/SlideVQA_dataset.py`, `datasets_api/datasets_utils.py`
- `finetuning/build_dataset.py`
- `scripts/slurm/run_pipeline.sh`, `scripts/slurm/download_dataset.sh`, `scripts/slurm/run_finetune_qwen25vl.sh`, `scripts/slurm/run_vqa_analysis.sh`, `scripts/slurm/run_vqa_analysis_finetuned.sh`, `scripts/slurm/run_gpu_test.sh`
- `corruption-scripts/verification/run_verification.sh`
- `test_hpc.sh`

---

## Task 1: Create the `config` package and `config/paths.py` (TDD)

**Files:**
- Create: `config/__init__.py`
- Create: `config/paths.py`
- Create: `tests/__init__.py`
- Create: `tests/test_paths.py`

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` as an empty file. Then create `tests/test_paths.py`:

```python
"""Validates config/paths.py. Run from repo root: uv run python -m tests.test_paths"""
import os
import sys
import subprocess
from config import paths


def _run_with_env(env_overrides):
    """Run a clean subprocess that prints image_dir('DUDE') under the given env."""
    env = {k: v for k, v in os.environ.items() if k != "SCRATCH_FLASH"}
    env["PYTHONPATH"] = str(paths.REPO_ROOT)
    env.update(env_overrides)
    out = subprocess.check_output(
        [sys.executable, "-c", "from config import paths; print(paths.image_dir('DUDE'))"],
        env=env, cwd=str(paths.REPO_ROOT), text=True,
    )
    return out.strip()


def test_repo_root_self_locates():
    assert (paths.REPO_ROOT / "pyproject.toml").is_file(), paths.REPO_ROOT


def test_image_dir_derives_from_scratch_flash():
    assert paths.image_dir("DUDE") == str(paths.SCRATCH_FLASH / "DUDE_images")
    assert paths.image_dir("MPDocVQA") == str(paths.SCRATCH_FLASH / "MPDocVQA_images")


def test_default_value_when_env_unset():
    assert _run_with_env({}) == "/mnt/beegfs/amartinelli/DUDE_images"


def test_env_override_respected():
    assert _run_with_env({"SCRATCH_FLASH": "/tmp/scratch_test"}) == "/tmp/scratch_test/DUDE_images"


if __name__ == "__main__":
    test_repo_root_self_locates()
    test_image_dir_derives_from_scratch_flash()
    test_default_value_when_env_unset()
    test_env_override_respected()
    print("OK: config/paths.py")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/amartinelli/VRD-UQA && uv run python -m tests.test_paths`
Expected: FAIL with `ModuleNotFoundError: No module named 'config'`.

- [ ] **Step 3: Create the `config` package**

Create `config/__init__.py` as an empty file. Then create `config/paths.py`:

```python
"""Single source of truth for environment-specific paths (Python side).

Mirrors scripts/env.sh: the default literals below MUST match env.sh.
Environment variables (exported by env.sh) take precedence at runtime.
"""
from pathlib import Path
import os

# Self-locating: correct whether the repo runs from $HOME or a relocated SLURM
# work-dir. config/paths.py -> parent is config/, parent.parent is the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Roots — defaults mirror scripts/env.sh; env vars win at runtime.
SCRATCH_FLASH = Path(os.getenv("SCRATCH_FLASH", "/mnt/beegfs/amartinelli"))
MPDOCVQA_SOURCE_QAS = Path(
    os.getenv("MPDOCVQA_SOURCE_QAS", "/home/amartinelli/MPDocVQA/MPDocVQA_complete/qas")
)


def image_dir(dataset_name: str) -> str:
    """Absolute image dir for a dataset, e.g. image_dir("DUDE") ->
    /mnt/beegfs/amartinelli/DUDE_images (no trailing slash)."""
    return str(SCRATCH_FLASH / f"{dataset_name}_images")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/amartinelli/VRD-UQA && uv run python -m tests.test_paths`
Expected: `OK: config/paths.py` and exit code 0.

- [ ] **Step 5: Commit**

```bash
git add config/__init__.py config/paths.py tests/__init__.py tests/test_paths.py
git commit -m "Add config/paths.py single source of truth for Python paths

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Create `scripts/env.sh` (bash source of truth)

**Files:**
- Create: `scripts/env.sh`

- [ ] **Step 1: Create the file**

Create `scripts/env.sh`:

```bash
#!/usr/bin/env bash
# Single source of truth for environment roots (bash side).
# Mirror of config/paths.py defaults — keep the two in sync.
# Override any value by exporting it BEFORE sourcing this file.
export SCRATCH_FLASH="${SCRATCH_FLASH:-/mnt/beegfs/amartinelli}"
export MPDOCVQA_SOURCE_QAS="${MPDOCVQA_SOURCE_QAS:-/home/amartinelli/MPDocVQA/MPDocVQA_complete/qas}"
export HF_HOME="$SCRATCH_FLASH/.cache/huggingface"
export VRD_UQA_HOME="${VRD_UQA_HOME:-$HOME/VRD-UQA}"   # persistent repo, used as rsync source
```

- [ ] **Step 2: Verify it sources and exports correctly**

Run:
```bash
bash -c 'source /home/amartinelli/VRD-UQA/scripts/env.sh && echo "$SCRATCH_FLASH | $HF_HOME | $MPDOCVQA_SOURCE_QAS | $VRD_UQA_HOME"'
```
Expected: `/mnt/beegfs/amartinelli | /mnt/beegfs/amartinelli/.cache/huggingface | /home/amartinelli/MPDocVQA/MPDocVQA_complete/qas | /home/amartinelli/VRD-UQA`

- [ ] **Step 3: Verify the override path works**

Run:
```bash
bash -c 'export SCRATCH_FLASH=/tmp/x; source /home/amartinelli/VRD-UQA/scripts/env.sh && echo "$SCRATCH_FLASH | $HF_HOME"'
```
Expected: `/tmp/x | /tmp/x/.cache/huggingface`

- [ ] **Step 4: Commit**

```bash
git add scripts/env.sh
git commit -m "Add scripts/env.sh single source of truth for bash paths

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Migrate `datasets_api/MPDocVQA_dataset.py`

This module is the richest: it has both `SOURCE_DIR` and `MPDOCVQA_IMAGE_DIR`.

**Files:**
- Modify: `datasets_api/MPDocVQA_dataset.py:1-14`

- [ ] **Step 1: Replace the imports and constants**

Find lines 1-11:
```python
import os
import json
import random
from datasets_api.datasets_utils import save_sample
import pandas as pd

SOURCE_DIR = "/home/amartinelli/MPDocVQA/MPDocVQA_complete/qas"
MPDOCVQA_IMAGE_DIR = '/mnt/beegfs/amartinelli/MPDocVQA_images'

def get_MPDocVQA_image_dir():
    return MPDOCVQA_IMAGE_DIR
```

Replace with:
```python
import os
import json
import random
from datasets_api.datasets_utils import save_sample
from config import paths
import pandas as pd


def get_MPDocVQA_image_dir():
    return paths.image_dir("MPDocVQA")
```

- [ ] **Step 2: Update `get_MPDocVQA_split` to use the centralized source dir**

Find line 14 (inside `get_MPDocVQA_split`):
```python
    source = os.path.join(SOURCE_DIR, f"{split_type}.json")
```

Replace with:
```python
    source = os.path.join(str(paths.MPDOCVQA_SOURCE_QAS), f"{split_type}.json")
```

- [ ] **Step 3: Verify the getter returns the same value as before**

Run:
```bash
cd /home/amartinelli/VRD-UQA && uv run python -c "from datasets_api.MPDocVQA_dataset import get_MPDocVQA_image_dir; v=get_MPDocVQA_image_dir(); print(v); assert v == '/mnt/beegfs/amartinelli/MPDocVQA_images', v; print('OK')"
```
Expected: prints `/mnt/beegfs/amartinelli/MPDocVQA_images` then `OK`.

- [ ] **Step 4: Commit**

```bash
git add datasets_api/MPDocVQA_dataset.py
git commit -m "Centralize MPDocVQA paths via config/paths.py

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Migrate `datasets_api/DUDE_dataset.py`

**Files:**
- Modify: `datasets_api/DUDE_dataset.py:1-10`

- [ ] **Step 1: Replace the imports and constant**

Find lines 1-10:
```python
from datasets import load_dataset
from datasets_api.datasets_utils import save_sample
import pandas as pd
import os
import logging

DUDE_IMAGE_DIR = '/mnt/beegfs/amartinelli/DUDE_images'

def get_DUDE_image_dir():
    return DUDE_IMAGE_DIR
```

Replace with:
```python
from datasets import load_dataset
from datasets_api.datasets_utils import save_sample
from config import paths
import pandas as pd
import os
import logging


def get_DUDE_image_dir():
    return paths.image_dir("DUDE")
```

- [ ] **Step 2: Verify the getter returns the same value as before**

Run:
```bash
cd /home/amartinelli/VRD-UQA && uv run python -c "from datasets_api.DUDE_dataset import get_DUDE_image_dir; v=get_DUDE_image_dir(); print(v); assert v == '/mnt/beegfs/amartinelli/DUDE_images', v; print('OK')"
```
Expected: prints `/mnt/beegfs/amartinelli/DUDE_images` then `OK`.

- [ ] **Step 3: Commit**

```bash
git add datasets_api/DUDE_dataset.py
git commit -m "Centralize DUDE paths via config/paths.py

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Migrate `datasets_api/BDocs_dataset.py` and `datasets_api/SlideVQA_dataset.py`

Both build their image dir as `os.path.join(SCRATCH_FLASH, IMAGES_PATH)` with a trailing slash. After migration they return the normalized (no trailing slash) value from `paths.image_dir(...)`. This only affects *future* samplings; `os.path.join(image_dir, filename)` is unaffected by the dropped trailing slash, and existing data files already contain baked-in absolute paths.

**Files:**
- Modify: `datasets_api/BDocs_dataset.py:1-13`
- Modify: `datasets_api/SlideVQA_dataset.py:1-11`

- [ ] **Step 1: Migrate BDocs**

In `datasets_api/BDocs_dataset.py`, find lines 1-13:
```python
import os
from datasets import load_dataset
from tqdm import tqdm
import json
from langdetect import detect, LangDetectException
from datasets_api.datasets_utils import save_sample
import pandas as pd

SCRATCH_FLASH = '/mnt/beegfs/amartinelli/'
IMAGES_PATH = 'BDocs_images/'

def get_BDocs_image_dir():
    return os.path.join(SCRATCH_FLASH, IMAGES_PATH)
```

Replace with:
```python
import os
from datasets import load_dataset
from tqdm import tqdm
import json
from langdetect import detect, LangDetectException
from datasets_api.datasets_utils import save_sample
from config import paths
import pandas as pd


def get_BDocs_image_dir():
    return paths.image_dir("BDocs")
```

- [ ] **Step 2: Migrate SlideVQA**

In `datasets_api/SlideVQA_dataset.py`, find lines 1-11:
```python
from datasets import load_dataset
import os
from tqdm import tqdm
from datasets_api.datasets_utils import save_sample
import pandas as pd

SCRATCH_FLASH = '/mnt/beegfs/amartinelli/'
IMAGES_PATH = 'SlideVQA_images/'

def get_SlideVQA_image_dir():
    return os.path.join(SCRATCH_FLASH, IMAGES_PATH)
```

Replace with:
```python
from datasets import load_dataset
import os
from tqdm import tqdm
from datasets_api.datasets_utils import save_sample
from config import paths
import pandas as pd


def get_SlideVQA_image_dir():
    return paths.image_dir("SlideVQA")
```

- [ ] **Step 3: Verify both getters**

Run:
```bash
cd /home/amartinelli/VRD-UQA && uv run python -c "
from datasets_api.BDocs_dataset import get_BDocs_image_dir
from datasets_api.SlideVQA_dataset import get_SlideVQA_image_dir
b, s = get_BDocs_image_dir(), get_SlideVQA_image_dir()
print(b); print(s)
assert b == '/mnt/beegfs/amartinelli/BDocs_images', b
assert s == '/mnt/beegfs/amartinelli/SlideVQA_images', s
print('OK')
"
```
Expected: prints the two dirs (no trailing slash) then `OK`.

- [ ] **Step 4: Commit**

```bash
git add datasets_api/BDocs_dataset.py datasets_api/SlideVQA_dataset.py
git commit -m "Centralize BDocs and SlideVQA paths via config/paths.py

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Migrate `datasets_api/datasets_utils.py` (PROJECT_DIR → REPO_ROOT)

**Files:**
- Modify: `datasets_api/datasets_utils.py:1-8`

- [ ] **Step 1: Replace the constant and its use**

Find lines 1-8:
```python
from pathlib import Path
import os
import json

PROJECT_DIR = "/home/amartinelli/VRD-UQA/"

def save_sample(dataset_name: str, split_type: str, num_questions: int, output_data):
    out_dir = Path(PROJECT_DIR) / f"{dataset_name}_{split_type}_{num_questions}" / "qas"
```

Replace with:
```python
from pathlib import Path
import os
import json
from config.paths import REPO_ROOT

def save_sample(dataset_name: str, split_type: str, num_questions: int, output_data):
    # REPO_ROOT self-locates to the running copy of the repo. On a relocated SLURM
    # work-dir this now writes into the work-dir copy (the previously-hardcoded
    # $HOME path wrote to the wrong copy on the compute node). See design doc.
    out_dir = REPO_ROOT / f"{dataset_name}_{split_type}_{num_questions}" / "qas"
```

- [ ] **Step 2: Verify the module imports and `save_sample` resolves the right base**

Run:
```bash
cd /home/amartinelli/VRD-UQA && uv run python -c "
from datasets_api import datasets_utils
from config.paths import REPO_ROOT
print(REPO_ROOT)
assert (REPO_ROOT / 'pyproject.toml').is_file(), REPO_ROOT
print('OK')
"
```
Expected: prints the repo root path then `OK`.

- [ ] **Step 3: Commit**

```bash
git add datasets_api/datasets_utils.py
git commit -m "Centralize repo root in datasets_utils via config/paths.REPO_ROOT

Self-locating REPO_ROOT fixes save_sample writing to the wrong copy on
relocated SLURM work-dirs.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Migrate `finetuning/build_dataset.py` (REPO_ROOT)

**Files:**
- Modify: `finetuning/build_dataset.py:1-13`

- [ ] **Step 1: Replace the hardcoded REPO_ROOT with the import**

Find lines 1-13:
```python
from __future__ import annotations

import json
import random
from pathlib import Path
from tqdm import tqdm

from datasets_api.BDocs_dataset import get_BDocs_split, sample_BDocs_different_from, standardize_BDocs_for_corruption_pipeline
from datasets_api.DUDE_dataset import get_DUDE_split, sample_DUDE_different_from, standardize_DUDE_for_corruption_pipeline
from datasets_api.MPDocVQA_dataset import get_MPDocVQA_split, sample_MPDocVQA_different_from, standardize_MPDocVQA_for_corruption_pipeline
from datasets_api.SlideVQA_dataset import get_SlideVQA_split, sample_SlideVQA_different_from, standardize_SlideVQA_for_corruption_pipeline

REPO_ROOT = Path("/home/amartinelli/VRD-UQA")
```

Replace with:
```python
from __future__ import annotations

import json
import random
from pathlib import Path
from tqdm import tqdm

from datasets_api.BDocs_dataset import get_BDocs_split, sample_BDocs_different_from, standardize_BDocs_for_corruption_pipeline
from datasets_api.DUDE_dataset import get_DUDE_split, sample_DUDE_different_from, standardize_DUDE_for_corruption_pipeline
from datasets_api.MPDocVQA_dataset import get_MPDocVQA_split, sample_MPDocVQA_different_from, standardize_MPDocVQA_for_corruption_pipeline
from datasets_api.SlideVQA_dataset import get_SlideVQA_split, sample_SlideVQA_different_from, standardize_SlideVQA_for_corruption_pipeline
from config.paths import REPO_ROOT
```

(`Path` is still imported and used elsewhere in the file — leave the import line in place. All existing `REPO_ROOT / ...` usages are unchanged.)

- [ ] **Step 2: Verify the module imports and REPO_ROOT resolves**

Run:
```bash
cd /home/amartinelli/VRD-UQA && uv run python -c "
import finetuning.build_dataset as b
print(b.REPO_ROOT)
assert (b.REPO_ROOT / 'pyproject.toml').is_file(), b.REPO_ROOT
print('OK')
"
```
Expected: prints the repo root path then `OK`.

- [ ] **Step 3: Commit**

```bash
git add finetuning/build_dataset.py
git commit -m "Centralize repo root in build_dataset via config/paths.REPO_ROOT

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Migrate the bash scripts to source `scripts/env.sh`

All SLURM scripts source from the persistent `$HOME/VRD-UQA/scripts/env.sh` because they source *before* `cd`-ing into the relocated work-dir, and they already assume the repo lives at `$HOME/VRD-UQA` (their rsync lines reference it).

**Files:**
- Modify: `scripts/slurm/run_pipeline.sh:36-37`
- Modify: `scripts/slurm/download_dataset.sh:23-24`
- Modify: `scripts/slurm/run_finetune_qwen25vl.sh:17-18`
- Modify: `scripts/slurm/run_vqa_analysis.sh:20-22`
- Modify: `scripts/slurm/run_vqa_analysis_finetuned.sh:20-22`
- Modify: `scripts/slurm/run_gpu_test.sh:15-16`
- Modify: `corruption-scripts/verification/run_verification.sh:20-22`
- Modify: `test_hpc.sh:58,85`

- [ ] **Step 1: `run_pipeline.sh`** — replace lines 36-37:
```bash
export SCRATCH_FLASH="/mnt/beegfs/amartinelli"
export HF_HOME="$SCRATCH_FLASH/.cache/huggingface"
```
with:
```bash
source "$HOME/VRD-UQA/scripts/env.sh"
```

- [ ] **Step 2: `download_dataset.sh`** — replace lines 23-24:
```bash
export SCRATCH_FLASH="/mnt/beegfs/amartinelli"
export HF_HOME="$SCRATCH_FLASH/.cache/huggingface"
```
with:
```bash
source "$HOME/VRD-UQA/scripts/env.sh"
```
(Leave the following `mkdir -p "$HF_HOME"` and the `$HOME/.hf_token` block intact.)

- [ ] **Step 3: `run_finetune_qwen25vl.sh`** — replace lines 17-18:
```bash
export SCRATCH_FLASH="/mnt/beegfs/amartinelli"
export HF_HOME="$SCRATCH_FLASH/.cache/huggingface"
```
with:
```bash
source "$HOME/VRD-UQA/scripts/env.sh"
```
(Leave the following `export UV_LINK_MODE=copy` intact.)

- [ ] **Step 4: `run_vqa_analysis.sh`** — replace lines 20-22 (two exports separated by a blank line):
```bash
export SCRATCH_FLASH="/mnt/beegfs/amartinelli"

export HF_HOME="$SCRATCH_FLASH/.cache/huggingface"
```
with:
```bash
source "$HOME/VRD-UQA/scripts/env.sh"
```

- [ ] **Step 5: `run_vqa_analysis_finetuned.sh`** — replace lines 20-22 (identical to Step 4):
```bash
export SCRATCH_FLASH="/mnt/beegfs/amartinelli"

export HF_HOME="$SCRATCH_FLASH/.cache/huggingface"
```
with:
```bash
source "$HOME/VRD-UQA/scripts/env.sh"
```

- [ ] **Step 6: `run_gpu_test.sh`** — this script *uses* `$SCRATCH_FLASH` (lines 17-20) but never defines it. Add the source line right after the module loads. Find lines 14-17:
```bash
module load miniconda3/3.13.25
module load nvhpc/25.1

rm -rf $SCRATCH_FLASH/first_sample_test/
```
Replace with:
```bash
module load miniconda3/3.13.25
module load nvhpc/25.1

source "$HOME/VRD-UQA/scripts/env.sh"

rm -rf $SCRATCH_FLASH/first_sample_test/
```

- [ ] **Step 7: `run_verification.sh`** — replace lines 20-22:
```bash
export SCRATCH_FLASH="/mnt/beegfs/amartinelli"

export HF_HOME="$SCRATCH_FLASH/.cache/huggingface"
```
with:
```bash
source "$HOME/VRD-UQA/scripts/env.sh"
```

- [ ] **Step 8: `test_hpc.sh`** — this script sets `SCRATCH_FLASH` and `HF_HOME` inline mid-script. Source `env.sh` once near the top and remove the two inline exports (this also normalizes `beegfs-compat` → `beegfs`).

  First, find lines 36-37:
```bash
module load miniconda3/3.13.25
module load nvhpc/25.1
```
  Replace with:
```bash
module load miniconda3/3.13.25
module load nvhpc/25.1

source "$HOME/VRD-UQA/scripts/env.sh"
```

  Next, find line 58:
```bash
export SCRATCH_FLASH="/mnt/beegfs-compat/amartinelli"
```
  Delete it (the `SCRATCH_FLASH` value now comes from `env.sh`). Keep the surrounding `check "scratch_flash dir exists" ...` lines.

  Next, find line 85:
```bash
export HF_HOME="$SCRATCH_FLASH/.cache/huggingface"
```
  Delete it (the `HF_HOME` value now comes from `env.sh`). Keep the following `mkdir -p "$HF_HOME"` line.

- [ ] **Step 9: Verify no script still defines the roots, and `beegfs-compat` is gone**

Run:
```bash
cd /home/amartinelli/VRD-UQA && grep -rn "export SCRATCH_FLASH=" scripts/ corruption-scripts/ test_hpc.sh ; echo "---" ; grep -rn "beegfs-compat" . --include=*.sh --include=*.py
```
Expected: the first `grep` prints **nothing** (no script defines it anymore); the `---` separator prints; the second `grep` prints **nothing**. (Both greps exiting non-zero / empty is the success condition.)

- [ ] **Step 10: Verify a representative script still resolves the vars when sourced**

Run:
```bash
cd /home/amartinelli/VRD-UQA && bash -c 'source ./scripts/env.sh; echo "$SCRATCH_FLASH | $HF_HOME"'
```
Expected: `/mnt/beegfs/amartinelli | /mnt/beegfs/amartinelli/.cache/huggingface`

- [ ] **Step 11: Commit**

```bash
git add scripts/slurm/run_pipeline.sh scripts/slurm/download_dataset.sh \
        scripts/slurm/run_finetune_qwen25vl.sh scripts/slurm/run_vqa_analysis.sh \
        scripts/slurm/run_vqa_analysis_finetuned.sh scripts/slurm/run_gpu_test.sh \
        corruption-scripts/verification/run_verification.sh test_hpc.sh
git commit -m "Source scripts/env.sh in all bash scripts

Removes duplicated SCRATCH_FLASH/HF_HOME exports, normalizes the
beegfs-compat drift in test_hpc.sh, and fixes the undefined SCRATCH_FLASH
in run_gpu_test.sh.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Confirm only the two source-of-truth files define the roots**

Run:
```bash
cd /home/amartinelli/VRD-UQA && grep -rnE '(export SCRATCH_FLASH=|^SCRATCH_FLASH *=|^PROJECT_DIR *=|^SOURCE_DIR *=|^MPDOCVQA_IMAGE_DIR *=|^DUDE_IMAGE_DIR *=|^IMAGES_PATH *=)' --include=*.py --include=*.sh . | grep -vE '\.venv/|__pycache__'
```
Expected: exactly two lines — one in `scripts/env.sh` (`export SCRATCH_FLASH=...`) and one in `config/paths.py` (`SCRATCH_FLASH = Path(...)`). No matches in `datasets_api/`, `finetuning/`, or any other script.

- [ ] **Step 2: Run the paths test suite**

Run: `cd /home/amartinelli/VRD-UQA && uv run python -m tests.test_paths`
Expected: `OK: config/paths.py`, exit code 0.

- [ ] **Step 3: Smoke-import every migrated Python module**

Run:
```bash
cd /home/amartinelli/VRD-UQA && uv run python -c "
from datasets_api.MPDocVQA_dataset import get_MPDocVQA_image_dir
from datasets_api.DUDE_dataset import get_DUDE_image_dir
from datasets_api.BDocs_dataset import get_BDocs_image_dir
from datasets_api.SlideVQA_dataset import get_SlideVQA_image_dir
from datasets_api import datasets_utils
import finetuning.build_dataset
print(get_MPDocVQA_image_dir())
print(get_DUDE_image_dir())
print(get_BDocs_image_dir())
print(get_SlideVQA_image_dir())
print('OK: all modules import and getters resolve')
"
```
Expected: the four image dirs (all under `/mnt/beegfs/amartinelli`, no trailing slashes) then `OK: all modules import and getters resolve`.

- [ ] **Step 4: Confirm the working tree is clean and the branch history is coherent**

Run: `cd /home/amartinelli/VRD-UQA && git status && git log --oneline centralize-paths -10`
Expected: clean working tree; the log shows the spec commit plus the Task 1-8 commits.

- [ ] **Step 5 (handoff):** Implementation complete. Use `superpowers:finishing-a-development-branch` to decide how to integrate `centralize-paths` (merge to `main`, open a PR, etc.).
