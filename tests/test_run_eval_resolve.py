"""Unit test for run_eval._resolve_input_file data-root resolution.

Guards the regression where the SLURM job derived the corrupted-questions
input under the scratch WORK_DIR (data/ rsync-excluded) instead of the
persistent repo. _resolve_input_file must honor VRD_UQA_HOME (the persistent
root exported by scripts/env.sh), falling back to REPO_ROOT locally.

Run: uv run python -m tests.test_run_eval_resolve
"""
import importlib.util
import os
from pathlib import Path
from config.paths import REPO_ROOT


def _load_run_eval():
    path = REPO_ROOT / "VQA_analysis" / "evaluators" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("run_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_root_is_repo_root():
    m = _load_run_eval()
    os.environ.pop("VRD_UQA_HOME", None)
    got = m._resolve_input_file("BDocs", "val_300")
    expected = str(
        REPO_ROOT / "data" / "BDocs" / "BDocs_val_300"
        / "BDocs_unanswerable_corrupted_questions_just_false.json"
    )
    assert got == expected, got


def test_vrd_uqa_home_overrides_root():
    """The SLURM case: persistent root differs from the (relocated) REPO_ROOT."""
    m = _load_run_eval()
    os.environ["VRD_UQA_HOME"] = "/persist/repo"
    try:
        got = m._resolve_input_file("DUDE", "val_5")
        assert got == (
            "/persist/repo/data/DUDE/DUDE_val_5/"
            "DUDE_unanswerable_corrupted_questions_just_false.json"
        ), got
    finally:
        os.environ.pop("VRD_UQA_HOME", None)


if __name__ == "__main__":
    test_default_root_is_repo_root()
    test_vrd_uqa_home_overrides_root()
    print("OK: run_eval._resolve_input_file data-root resolution")
