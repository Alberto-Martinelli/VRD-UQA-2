"""Validate the new LoRA YAMLs against dataset_info + the Qwen reference.
Run: uv run python -m tests.test_finetune_configs"""
import json
from config.paths import REPO_ROOT

FT = REPO_ROOT / "finetuning"
NEW_FULL = ["internvl35_lora_sft.yaml"]  # Phi-4-mm uses the MS official path, not LLaMA-Factory
EXPECTED = {  # (model_name_or_path, template, output_dir basename)
    "internvl35_lora_sft.yaml": ("OpenGVLab/InternVL3_5-8B-HF", "intern_vl", "internvl35_lora_sft"),
}
# Verbatim-from-Qwen comparison hyperparameters
SHARED = {
    "stage": "sft", "finetuning_type": "lora", "lora_target": "all",
    "lora_rank": "16", "lora_alpha": "32", "cutoff_len": "4096",
    "dataset": "vrd_uqa_train", "eval_dataset": "vrd_uqa_val",
    "per_device_train_batch_size": "1", "gradient_accumulation_steps": "8",
    "learning_rate": "2.0e-5", "num_train_epochs": "1.0",
    "lr_scheduler_type": "cosine", "warmup_steps": "50", "bf16": "true",
    "save_total_limit": "3", "load_best_model_at_end": "true",
}


def _parse_flat_yaml(path):
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, val = line.partition(":")
        out[key.strip()] = val.strip()
    return out


def test_new_yamls_valid_and_match_qwen_params():
    registered = set(json.loads((FT / "dataset_info.json").read_text()).keys())
    for fname in NEW_FULL:
        cfg = _parse_flat_yaml(FT / fname)
        model, template, out_base = EXPECTED[fname]
        assert cfg["model_name_or_path"] == model, fname
        assert cfg["template"] == template, fname
        assert cfg["output_dir"].rstrip("/").split("/")[-1] == out_base, fname
        assert cfg["dataset"] in registered and cfg["eval_dataset"] in registered, fname
        for k, v in SHARED.items():
            assert cfg.get(k) == v, f"{fname}: {k}={cfg.get(k)!r} expected {v!r}"


if __name__ == "__main__":
    test_new_yamls_valid_and_match_qwen_params()
    print("OK: finetune configs")
