import argparse
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

from base_evaluator import BaseVQAEvaluator


class LlamaVQAEvaluator(BaseVQAEvaluator):
    MODEL_KEY = "llama3.2"
    FINETUNED_MODEL_KEY = "llama3.2_finetuned"  # entry added post-fine-tuning (Phase B)
    MODEL_TYPE = "llama"
    MODEL_LEAF_PREFIX = "llama"

    def _load_model(self):
        print("Initializing Llama-3.2-Vision model...")
        print("Model configuration:", self.model_config)
        model_name = self.model_config["model_name"]
        self.processor = AutoProcessor.from_pretrained(model_name)
        model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        if self.model_config.get("use_flash_attention", False):
            model_kwargs["attn_implementation"] = "flash_attention_2"
        self.model = MllamaForConditionalGeneration.from_pretrained(model_name, **model_kwargs)

        adapter_path = self.model_config.get("adapter_path")
        if adapter_path:
            import os
            resolved = os.path.abspath(os.path.expandvars(os.path.expanduser(adapter_path)))
            print(f"Loading PEFT/LoRA adapter from: {resolved}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, resolved)

        self.model = self.model.eval()
        self.max_tokens = self.model_config.get("max_tokens", 1024)
        print("Llama-3.2-Vision model initialized successfully")

    def _to_native(self, neutral_turns, window_image_paths, question_prompt):
        """Returns (messages, pil_images) for the processor. Images in conversation
        order; each {"type":"image"} placeholder consumes the next PIL image."""
        messages, images = [], []
        for t in neutral_turns or []:
            if t["role"] == "user":
                content = []
                for p in t["image_paths"]:
                    content.append({"type": "image"})
                    images.append(Image.open(p).convert("RGB"))
                content.append({"type": "text", "text": t["text"]})
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": t["text"]}]})
        content = []
        for p in window_image_paths:
            content.append({"type": "image"})
            images.append(Image.open(p).convert("RGB"))
        content.append({"type": "text", "text": question_prompt})
        messages.append({"role": "user", "content": content})
        return messages, images

    def _generate(self, window_image_paths, question_prompt, few_shot_turns):
        if len(window_image_paths) > 1:
            print(f"WARNING: Llama-3.2-Vision got {len(window_image_paths)} images in one "
                  f"prompt; it is designed for one. Pin batch_size=1.")
        messages, images = self._to_native(few_shot_turns, window_image_paths, question_prompt)
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        trimmed = out[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--questions", choices=["both", "corrupted", "clean"], default="both")
    args = parser.parse_args()
    evaluator = LlamaVQAEvaluator(args.config_path, args.finetuned, questions=args.questions)
    print(f"Starting Llama-3.2-Vision evaluator (questions={args.questions}, seed={evaluator.seed})")
    evaluator.evaluate()


if __name__ == "__main__":
    main()
