"""Fine-tuning run-manifest checks. Run: uv run python -m tests.test_finetune_manifest"""
import json
import tempfile
from pathlib import Path
from config.paths import REPO_ROOT
import importlib.util


def _load(rel_path, name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


WRM = _load("finetuning/write_run_manifest.py", "write_run_manifest")


def _fixture(tmp):
    """A minimal training output_dir + YAML mirroring a real LLaMA-Factory run."""
    out = Path(tmp) / "qwen25vl_lora_sft_demo"
    out.mkdir(parents=True)
    (out / "adapter_config.json").write_text(json.dumps({
        "r": 16, "lora_alpha": 32, "lora_dropout": 0.0, "peft_type": "LORA",
        "base_model_name_or_path": "Qwen/Qwen2.5-VL-7B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj"],
    }))
    (out / "all_results.json").write_text(json.dumps({
        "epoch": 3.0, "eval_loss": 0.0333, "train_loss": 0.0665,
        "train_runtime": 35681.18, "train_samples_per_second": 0.498,
        "total_flos": 7.23e17,
    }))
    (out / "trainer_state.json").write_text(json.dumps({"global_step": 222, "best_metric": 0.0333}))
    cfg = Path(tmp) / "train.yaml"
    cfg.write_text(
        "model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct\n"
        "finetuning_type: lora\nstage: sft\nlora_target: all\n"
        "lora_rank: 16\nlora_alpha: 32\n"
        "dataset: vrd_uqa_train\neval_dataset: vrd_uqa_val\n"
        "template: qwen2_vl\ncutoff_len: 4096\n"
        "learning_rate: 2.0e-5\nnum_train_epochs: 3.0\n"
        "per_device_train_batch_size: 1\ngradient_accumulation_steps: 8\n"
        "lr_scheduler_type: cosine\nwarmup_steps: 50\nbf16: true\n"
        f"output_dir: {out}\n"
    )
    return cfg, out


def test_build_manifest_scrapes_all_sources():
    tmp = tempfile.mkdtemp()
    cfg, out = _fixture(tmp)
    m = WRM.build_manifest(str(cfg), None, git_commit="abc1234", git_dirty=False)

    assert m["experiment"] == "qwen25vl_lora_sft_demo"
    assert m["git_commit"] == "abc1234" and m["git_dirty"] is False
    assert m["base_model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    # LoRA details prefer adapter_config.json (ground truth) over the YAML
    assert m["method"]["lora_rank"] == 16 and m["method"]["lora_alpha"] == 32
    assert m["method"]["peft_type"] == "LORA"
    assert m["method"]["num_target_modules"] == 3
    assert m["dataset"]["train"] == "vrd_uqa_train" and m["dataset"]["eval"] == "vrd_uqa_val"
    assert abs(m["hyperparams"]["learning_rate"] - 2.0e-5) < 1e-12
    assert m["hyperparams"]["num_train_epochs"] == 3.0
    assert m["hyperparams"]["seed"] == 42  # default when YAML omits it
    assert abs(m["results"]["train_loss"] - 0.0665) < 1e-9
    assert abs(m["results"]["eval_loss"] - 0.0333) < 1e-9
    assert m["results"]["global_step"] == 222
    assert m["output_dir"] == str(out)


def test_main_writes_manifest_and_output_dir_from_yaml():
    tmp = tempfile.mkdtemp()
    cfg, out = _fixture(tmp)
    import sys
    argv = sys.argv
    sys.argv = ["write_run_manifest.py", "--config", str(cfg),
                "--git-commit", "deadbee", "--git-dirty", "true"]
    try:
        WRM.main()
    finally:
        sys.argv = argv
    written = json.loads((out / "run_manifest.json").read_text())
    assert written["git_commit"] == "deadbee" and written["git_dirty"] is True
    assert written["method"]["lora_rank"] == 16


def test_missing_artifacts_are_tolerated():
    # No adapter_config/all_results present: manifest still builds with None-ish fields.
    tmp = Path(tempfile.mkdtemp())
    out = tmp / "empty_run"
    out.mkdir()
    cfg = tmp / "train.yaml"
    cfg.write_text(f"model_name_or_path: foo/bar\nfinetuning_type: lora\noutput_dir: {out}\n")
    m = WRM.build_manifest(str(cfg), None, git_commit="x", git_dirty=False)
    assert m["base_model"] == "foo/bar"
    assert m["method"]["lora_rank"] is None       # no adapter_config, no YAML lora_rank
    assert m["method"]["num_target_modules"] == 0
    assert m["results"]["train_loss"] is None     # no all_results.json


if __name__ == "__main__":
    test_build_manifest_scrapes_all_sources()
    test_main_writes_manifest_and_output_dir_from_yaml()
    test_missing_artifacts_are_tolerated()
    print("OK: finetuning run manifest")
