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
