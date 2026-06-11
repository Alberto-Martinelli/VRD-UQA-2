import argparse
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from base_evaluator import BaseVQAEvaluator

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_ratio(ar, ratios, w, h, input_size):
    best, best_diff = (1, 1), float("inf")
    area = w * h
    for r in ratios:
        target = r[0] / r[1]
        diff = abs(ar - target)
        if diff < best_diff or (diff == best_diff and area > 0.5 * input_size * input_size * r[0] * r[1]):
            best_diff, best = diff, r
    return best


def _dynamic_preprocess(image, min_num=1, max_num=12, input_size=448, use_thumbnail=True):
    w, h = image.size
    ar = w / h
    ratios = sorted({(i, j) for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if min_num <= i * j <= max_num}, key=lambda x: x[0] * x[1])
    rw, rh = _find_closest_ratio(ar, ratios, w, h, input_size)
    tw, th = input_size * rw, input_size * rh
    blocks = rw * rh
    resized = image.resize((tw, th))
    tiles = []
    for i in range(blocks):
        box = ((i % (tw // input_size)) * input_size,
               (i // (tw // input_size)) * input_size,
               ((i % (tw // input_size)) + 1) * input_size,
               ((i // (tw // input_size)) + 1) * input_size)
        tiles.append(resized.crop(box))
    if use_thumbnail and blocks != 1:
        tiles.append(image.resize((input_size, input_size)))
    return tiles


def _load_image_tiles(path, input_size=448, max_num=12):
    image = Image.open(path).convert("RGB")
    transform = _build_transform(input_size)
    tiles = _dynamic_preprocess(image, max_num=max_num, input_size=input_size, use_thumbnail=True)
    pixel_values = torch.stack([transform(t) for t in tiles])
    return pixel_values  # (num_tiles, 3, H, W)


class InternVLVQAEvaluator(BaseVQAEvaluator):
    MODEL_KEY = "internvl3_5"
    FINETUNED_MODEL_KEY = "internvl3_5_finetuned"  # entry added post-fine-tuning (Phase B)
    MODEL_TYPE = "internvl"
    MODEL_LEAF_PREFIX = "internvl"

    def _load_model(self):
        print("Initializing InternVL3.5-8B model...")
        print("Model configuration:", self.model_config)
        model_name = self.model_config["model_name"]
        self.max_tiles = self.model_config.get("max_tiles", 12)
        self.input_size = self.model_config.get("input_size", 448)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False)
        model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto",
                        "trust_remote_code": True, "low_cpu_mem_usage": True}
        if self.model_config.get("use_flash_attention", False):
            model_kwargs["use_flash_attn"] = True
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)

        adapter_path = self.model_config.get("adapter_path")
        if adapter_path:
            import os
            resolved = os.path.abspath(os.path.expandvars(os.path.expanduser(adapter_path)))
            print(f"Loading PEFT/LoRA adapter from: {resolved}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, resolved)

        self.model = self.model.eval()
        self.max_tokens = self.model_config.get("max_tokens", 1024)
        print("InternVL3.5-8B model initialized successfully")

    def _prep_images(self, image_paths):
        """Returns (pixel_values, num_patches_list, image_tag_block)."""
        pv_list, num_patches, tags = [], [], ""
        for i, p in enumerate(image_paths):
            pv = _load_image_tiles(p, input_size=self.input_size, max_num=self.max_tiles)
            pv_list.append(pv)
            num_patches.append(pv.shape[0])
            tags += f"Image-{i + 1}: <image>\n"
        pixel_values = torch.cat(pv_list).to(torch.bfloat16).to(self.model.device)
        return pixel_values, num_patches, tags

    def _generate(self, window_image_paths, question_prompt, few_shot_turns):
        gen_config = {"max_new_tokens": self.max_tokens, "do_sample": False}
        history = []
        for i in range(0, len(few_shot_turns or []), 2):
            u = few_shot_turns[i]
            a = few_shot_turns[i + 1]
            history.append((u["text"], a["text"]))
        pixel_values, num_patches, tags = self._prep_images(window_image_paths)
        question = f"{tags}{question_prompt}" if window_image_paths else question_prompt
        response = self.model.chat(
            self.tokenizer, pixel_values, question, gen_config,
            num_patches_list=num_patches, history=history or None, return_history=False,
        )
        return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--questions", choices=["both", "corrupted", "clean"], default="both")
    args = parser.parse_args()
    evaluator = InternVLVQAEvaluator(args.config_path, args.finetuned, questions=args.questions)
    print(f"Starting InternVL3.5-8B evaluator (questions={args.questions}, seed={evaluator.seed})")
    evaluator.evaluate()


if __name__ == "__main__":
    main()
