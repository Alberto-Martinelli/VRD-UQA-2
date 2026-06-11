import argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from base_evaluator import BaseVQAEvaluator


class QwenVQAEvaluator(BaseVQAEvaluator):
    MODEL_KEY = "qwen2.5"
    FINETUNED_MODEL_KEY = "qwen2.5_finetuned"
    MODEL_TYPE = "qwen"
    MODEL_LEAF_PREFIX = ""  # bare path for back-compat

    def _load_model(self):
        print("Initializing Qwen model...")
        print("Model configuration:", self.model_config)
        model_name = self.model_config["model_name"]

        processor_kwargs = {}
        if "min_pixels" in self.model_config and "max_pixels" in self.model_config:
            processor_kwargs.update({
                "min_pixels": self.model_config["min_pixels"],
                "max_pixels": self.model_config["max_pixels"],
            })
        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)

        model_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
        if self.model_config.get("use_flash_attention", False):
            model_kwargs.update({"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"})
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)

        adapter_path = self.model_config.get("adapter_path")
        if adapter_path:
            import os
            resolved = os.path.abspath(os.path.expandvars(os.path.expanduser(adapter_path)))
            print(f"Loading PEFT/LoRA adapter from: {resolved}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, resolved)

        self.model = self.model.eval()
        self.max_tokens = self.model_config.get("max_tokens", 1024)
        print("Qwen model initialized successfully")

    def _to_native_turns(self, neutral_turns):
        native = []
        for t in neutral_turns or []:
            if t["role"] == "user":
                native.append({"role": "user", "content": [
                    *[{"type": "image", "image": f"file://{p}"} for p in t["image_paths"]],
                    {"type": "text", "text": t["text"]},
                ]})
            else:
                native.append({"role": "assistant", "content": [{"type": "text", "text": t["text"]}]})
        return native

    def _generate(self, window_image_paths, question_prompt, few_shot_turns):
        messages = self._to_native_turns(few_shot_turns)
        messages.append({"role": "user", "content": [
            *[{"type": "image", "image": f"file://{p}"} for p in window_image_paths],
            {"type": "text", "text": question_prompt},
        ]})
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--questions", choices=["both", "corrupted", "clean"], default="both",
                        help="Which question side(s): both (default), corrupted (QUR), clean (FRR).")
    args = parser.parse_args()
    evaluator = QwenVQAEvaluator(args.config_path, args.finetuned, questions=args.questions)
    print(f"Starting QWEN 2.5 evaluator (questions={args.questions}, seed={evaluator.seed})")
    evaluator.evaluate()


if __name__ == "__main__":
    main()
