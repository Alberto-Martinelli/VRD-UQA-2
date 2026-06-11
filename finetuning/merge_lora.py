"""Merge a LoRA adapter into its base model and save a standalone checkpoint.

Eval fallback for models whose custom (trust_remote_code) classes don't cleanly
accept a PEFT adapter at load time (Phi-4-multimodal, InternVL3.5). After merging,
point the model_<...>_finetuned config entry at --out and remove its adapter_path.

Usage:
  python finetuning/merge_lora.py --base <hf_id_or_path> --adapter <adapter_dir> --out <dir>
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from peft import PeftModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print(f"Loading base {args.base} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu")
    print(f"Attaching adapter {args.adapter} ...")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("Merging ...")
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)

    # Save tokenizer/processor alongside so the merged dir loads standalone.
    for loader in (AutoProcessor, AutoTokenizer):
        try:
            loader.from_pretrained(args.base, trust_remote_code=True).save_pretrained(args.out)
            break
        except Exception as e:
            print(f"({loader.__name__} skipped: {e})")
    print(f"Merged checkpoint written to {args.out}")


if __name__ == "__main__":
    main()
