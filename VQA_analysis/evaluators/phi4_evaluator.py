import argparse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from base_evaluator import BaseVQAEvaluator

USER, ASSISTANT, END = "<|user|>", "<|assistant|>", "<|end|>"


class Phi4VQAEvaluator(BaseVQAEvaluator):
    MODEL_KEY = "phi4"
    FINETUNED_MODEL_KEY = "phi4_finetuned"  # entry added post-fine-tuning (Phase B)
    MODEL_TYPE = "phi4"
    MODEL_LEAF_PREFIX = "phi4"

    def _load_model(self):
        print("Initializing Phi-4-multimodal model...")
        print("Model configuration:", self.model_config)
        model_name = self.model_config["model_name"]
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto", "trust_remote_code": True}
        # The upstream checkpoint's config.json bakes in _attn_implementation="flash_attention_2",
        # so an explicit override is required even to opt OUT of it (omitting the kwarg keeps FA2).
        model_kwargs["_attn_implementation"] = (
            "flash_attention_2" if self.model_config.get("use_flash_attention", False) else "sdpa"
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        adapter_path = self.model_config.get("adapter_path")
        if adapter_path:
            import os
            resolved = os.path.abspath(os.path.expandvars(os.path.expanduser(adapter_path)))
            print(f"Loading PEFT/LoRA adapter from: {resolved}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, resolved)

        self.model = self.model.eval()
        self.max_tokens = self.model_config.get("max_tokens", 1024)
        try:
            self.gen_config = GenerationConfig.from_pretrained(model_name)
        except Exception:
            self.gen_config = None
        print("Phi-4-multimodal model initialized successfully")

    def _build_prompt_and_images(self, neutral_turns, window_image_paths, question_prompt):
        """Phi-4 placeholders are numbered globally across the whole prompt."""
        images, prompt, idx = [], "", 1
        for t in neutral_turns or []:
            if t["role"] == "user":
                tags = ""
                for p in t["image_paths"]:
                    tags += f"<|image_{idx}|>"
                    images.append(Image.open(p).convert("RGB"))
                    idx += 1
                prompt += f"{USER}{tags}{t['text']}{END}"
            else:
                prompt += f"{ASSISTANT}{t['text']}{END}"
        tags = ""
        for p in window_image_paths:
            tags += f"<|image_{idx}|>"
            images.append(Image.open(p).convert("RGB"))
            idx += 1
        prompt += f"{USER}{tags}{question_prompt}{END}{ASSISTANT}"
        return prompt, images

    def _generate(self, window_image_paths, question_prompt, few_shot_turns):
        prompt, images = self._build_prompt_and_images(few_shot_turns, window_image_paths, question_prompt)
        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.model.device)
        gen_kwargs = {"max_new_tokens": self.max_tokens}
        if self.gen_config is not None:
            gen_kwargs["generation_config"] = self.gen_config
        out = self.model.generate(**inputs, **gen_kwargs)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--questions", choices=["both", "corrupted", "clean"], default="both")
    args = parser.parse_args()
    evaluator = Phi4VQAEvaluator(args.config_path, args.finetuned, questions=args.questions)
    print(f"Starting Phi-4-multimodal evaluator (questions={args.questions}, seed={evaluator.seed})")
    evaluator.evaluate()


if __name__ == "__main__":
    main()
