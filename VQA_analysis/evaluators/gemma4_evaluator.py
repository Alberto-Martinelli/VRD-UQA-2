import argparse
import torch
from PIL import Image
from transformers import AutoProcessor

from base_evaluator import BaseVQAEvaluator


class Gemma4VQAEvaluator(BaseVQAEvaluator):
    """Gemma 4 12B (google/gemma-4-12B-it) via the HF-native integration:
    AutoProcessor + AutoModelForMultimodalLM + chat template + generate. Gemma 4 is a
    dense, encoder-free unified multimodal model with a configurable per-image visual
    token budget (no dynamic tiling). Same base for train (TRL+PEFT) and eval, so a LoRA
    adapter attaches cleanly. Mirrors the InternVL evaluator's message-building pattern.

    NOTE: AutoModelForMultimodalLM (transformers>=5.5.2) is imported lazily in _load_model
    so this module imports under the main repo venv (4.57.6) for the mock test; real eval
    runs in the pinned >=5.5.2 env, same situation as Phi-4 (4.47.0)."""

    MODEL_KEY = "gemma4"
    FINETUNED_MODEL_KEY = "gemma4_finetuned"
    MODEL_TYPE = "gemma"
    MODEL_LEAF_PREFIX = "gemma"

    def _load_model(self):
        print("Initializing Gemma 4 12B model...")
        print("Model configuration:", self.model_config)
        from transformers import AutoModelForMultimodalLM  # needs transformers>=5.5.2

        model_name = self.model_config["model_name"]
        budget = self.model_config.get("visual_token_budget", 560)
        self.processor = AutoProcessor.from_pretrained(model_name, visual_token_budget=budget)
        model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        if self.model_config.get("use_flash_attention", False):
            model_kwargs["attn_implementation"] = "flash_attention_2"
        self.model = AutoModelForMultimodalLM.from_pretrained(model_name, **model_kwargs)

        adapter_path = self.model_config.get("adapter_path")
        if adapter_path:
            import os
            resolved = os.path.abspath(os.path.expandvars(os.path.expanduser(adapter_path)))
            print(f"Loading PEFT/LoRA adapter from: {resolved}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, resolved)

        self.model = self.model.eval()
        self.max_tokens = self.model_config.get("max_tokens", 1024)
        print("Gemma 4 12B model initialized successfully")

    def _to_native(self, neutral_turns, window_image_paths, question_prompt):
        """Convert neutral few-shot turns + the current window into Gemma 4 chat messages
        with embedded PIL images (the processor projects them under the visual token budget)."""
        messages = []
        for t in neutral_turns or []:
            if t["role"] == "user":
                content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in t["image_paths"]]
                content.append({"type": "text", "text": t["text"]})
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": t["text"]}]})
        content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in window_image_paths]
        content.append({"type": "text", "text": question_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def _generate(self, window_image_paths, question_prompt, few_shot_turns):
        messages = self._to_native(few_shot_turns, window_image_paths, question_prompt)
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
        trimmed = generated[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--questions", choices=["both", "corrupted", "clean"], default="both")
    args = parser.parse_args()
    evaluator = Gemma4VQAEvaluator(args.config_path, args.finetuned, questions=args.questions)
    print(f"Starting Gemma 4 12B evaluator (questions={args.questions}, seed={evaluator.seed})")
    evaluator.evaluate()


if __name__ == "__main__":
    main()
