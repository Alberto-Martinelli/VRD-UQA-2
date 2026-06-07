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
