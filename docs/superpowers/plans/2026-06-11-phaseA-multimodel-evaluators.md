# Phase A — Multi-model Evaluators + Parallel-safe Orchestration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Llama-3.2-Vision, Phi-4-multimodal, and InternVL3.5-8B evaluators behind a shared `BaseVQAEvaluator` (with Qwen migrated onto it), make evaluation runs parallel-safe via per-job run IDs, and add a cross-run aggregator.

**Architecture:** Extract all model-agnostic logic from the existing `QwenVQAEvaluator` into `BaseVQAEvaluator`; each model becomes a subclass implementing only `_load_model` + `_generate`. Run-artifact leaves gain a model prefix (Qwen stays bare for back-compat). The SLURM launcher takes `model dataset split` and derives a deterministic per-`(model,dataset,N)` run_id so parallel jobs never share a directory. A read-only aggregator concatenates per-run `summary.csv` files.

**Tech Stack:** Python 3.11, transformers 4.57.6, torch, qwen-vl-utils, pytest-style standalone tests (run via `uv run python -m tests.<module>`), SLURM/bash.

**Spec:** [docs/superpowers/specs/2026-06-11-multimodel-evaluators-and-finetuning-design.md](../specs/2026-06-11-multimodel-evaluators-and-finetuning-design.md) (Phase A + caveats).

---

## File Structure

| File | Responsibility |
|---|---|
| `VQA_analysis/evaluators/base_evaluator.py` | **New.** `BaseVQAEvaluator`: all model-agnostic logic + abstract `_load_model`/`_generate` + neutral few-shot turns. |
| `VQA_analysis/evaluators/qwen2.5_evaluator.py` | **Refactor.** `QwenVQAEvaluator(BaseVQAEvaluator)` — only Qwen `_load_model`/`_generate` + `main()`. CLI unchanged. |
| `VQA_analysis/evaluators/llama_evaluator.py` | **New.** `LlamaVQAEvaluator` (Mllama). |
| `VQA_analysis/evaluators/phi4_evaluator.py` | **New.** `Phi4VQAEvaluator` (Phi-4-multimodal). |
| `VQA_analysis/evaluators/internvl_evaluator.py` | **New.** `InternVLVQAEvaluator` (InternVL3.5, `model.chat`). |
| `config/run_layout.py` | **Modify.** `leaf_dir()` gains `model_prefix`. |
| `VQA_analysis/config_zeroshot.json` / `config_fewshot.json` / `config_mock.json` | **Modify.** Confirm `llama3.2`/`phi4`; add `internvl3_5`. |
| `VQA_analysis/metrics/4_aggregate_summaries.py` | **New.** Read-only cross-run aggregator. |
| `scripts/slurm/run_vqa_analysis.sh` | **Refactor.** `run_vqa_analysis.sh <model> <dataset> <split>`, per-job run_id, glob/symlink fixes. |
| `tests/test_evaluator_mock.py` | **Modify.** Parametrize over all 4 entrypoints + namespaced-leaf assertions. |
| `tests/test_aggregate_summaries.py` | **New.** Aggregator fixture test. |

**Mock-mode is the test backbone:** `mock: true` in config short-circuits in the base before any model load, so all four evaluators run in CI with no GPU and no downloads. Real-model `val_5` smoke runs are manual on the HPC (commands in Task 12).

---

## Task 1: Create `BaseVQAEvaluator` with the model-agnostic logic

**Files:**
- Create: `VQA_analysis/evaluators/base_evaluator.py`

The base holds everything the current `qwen2.5_evaluator.py` does **except** model load and per-window inference. Most methods move **verbatim**; three change shape (`generate_answer`, `_build_few_shot_turns`, `_resolve_leaf`) and `__init__` gains a mock guard.

- [ ] **Step 1: Write `base_evaluator.py`**

Header + class attributes + `__init__` (note the `_load_model` mock guard) + abstract hooks:

```python
import gc
import json
import os
import random
import datetime
import traceback
import torch
from tqdm.auto import tqdm

from config import run_layout as rl


class BaseVQAEvaluator:
    """Model-agnostic VQA evaluation. Subclasses implement _load_model + _generate."""

    MODEL_KEY: str = ""            # config key under open_source_models
    FINETUNED_MODEL_KEY = None     # set only where a fine-tuned variant exists
    MODEL_TYPE: str = "base"       # written to vqa_result["model_type"]
    MODEL_LEAF_PREFIX: str = ""     # "" for Qwen (back-compat); model key otherwise

    def __init__(self, config_path, finetuned, questions="both"):
        with open(config_path) as f:
            self.config = json.load(f)

        self.finetuned = finetuned
        self.questions = questions
        self.seed = self.config.get("seed", 42)
        self._set_seed()

        if self.finetuned:
            if not self.FINETUNED_MODEL_KEY:
                raise ValueError(
                    f"--finetuned requested but {type(self).__name__} has no "
                    f"fine-tuned variant configured (FINETUNED_MODEL_KEY is None)."
                )
            self.model_config = self.config["open_source_models"][self.FINETUNED_MODEL_KEY]
        else:
            self.model_config = self.config["open_source_models"][self.MODEL_KEY]

        self.sampling_percentage = self.config.get("sampling_percentage", 100)
        self.unable_to_respond_aware = self.config.get("unable_to_respond_aware", True)

        # Extract base_image_dir from the input JSON if present, else images_base_path
        input_file = self.config.get("input_file")
        self.images_base_path = self.config.get("images_base_path")
        if input_file and os.path.exists(input_file):
            try:
                with open(input_file) as f_in:
                    in_data = json.load(f_in)
                    if "base_image_dir" in in_data:
                        self.images_base_path = in_data["base_image_dir"]
                        print(f"Extracted images_base_path from input file: {self.images_base_path}")
            except Exception as e:
                print(f"Warning: could not parse base_image_dir from input file {input_file}: {e}")

        if self.config.get("mock", False):
            print("Mock mode enabled — skipping model initialization")
            self.max_tokens = self.model_config.get("max_tokens", 1024)
        else:
            self._load_model()

    # ---- abstract hooks (subclass implements) -------------------------------
    def _load_model(self):
        """Populate self.model, self.processor or self.tokenizer, self.max_tokens."""
        raise NotImplementedError

    def _generate(self, window_image_paths, question_prompt, few_shot_turns) -> str:
        """Run inference for one window of pages; return the decoded answer string.
        few_shot_turns is the neutral list from _build_few_shot_turns (may be None)."""
        raise NotImplementedError
```

- [ ] **Step 2: Add the verbatim-moved methods**

Copy these methods **unchanged** from the current `VQA_analysis/evaluators/qwen2.5_evaluator.py` into the class (drop the unused `from difflib import SequenceMatcher`):

- `_set_seed` (current lines 55–67)
- `_create_prompt` (69–95)
- `get_sorted_ocr_text` (156–166)
- `get_ocr_text` (168–185)
- `_cleanup_model` (146–154)
- `_save_results` (334–384)
- `_get_fs_pool` (386–400)
- `_fs_specific_score` (402–410)
- `_select_few_shot_examples` (412–466)
- `QUESTION_SIDES` class dict (538–542)
- `_process_single_question` (544–603) — but change the literal `"model_type": "qwen"` (line 571) to `"model_type": self.MODEL_TYPE`
- `evaluate` (605–668)

- [ ] **Step 3: Add the reshaped `generate_answer` (delegates to `_generate`)**

```python
    def generate_answer(self, question, image_paths, ocr_text=None, few_shot_turns=None):
        """Sliding-window image context + optional few-shot; per-window inference
        is delegated to the model-specific self._generate()."""
        if self.config.get("mock", False):
            window_size = self.model_config.get("batch_size", 1)
            windows = [image_paths[i:i + window_size] for i in range(0, len(image_paths), window_size)]
            return {
                "answer": [{"pages": w, "answer": "Mock answer"} for w in windows],
                "query": question,
                "image_paths": image_paths,
                "analysis_type": f"window_size_{window_size}_mock",
            }

        try:
            missing = [p for p in image_paths if not os.path.exists(p)]
            if missing:
                print(f"WARNING: {len(missing)} image path(s) not found on disk:")
                for p in missing:
                    print(f"  [MISSING] {p}")

            window_size = self.model_config.get("batch_size", 1)
            stride = self.model_config.get("stride", window_size // 2) if window_size > 1 else 1
            total_images = len(image_paths)

            windows = []
            start_idx = 0
            while start_idx < total_images:
                end_idx = min(start_idx + window_size, total_images)
                if end_idx == total_images and (end_idx - start_idx) < window_size and total_images >= window_size:
                    window = image_paths[-window_size:]
                else:
                    window = image_paths[start_idx:end_idx]
                if window not in windows:
                    windows.append(window)
                if end_idx == total_images:
                    break
                start_idx += stride

            all_responses = []
            for window in windows:
                batch_ocr = None
                if ocr_text:
                    ocr_lines = []
                    for path in window:
                        page_num = image_paths.index(path) + 1
                        page_ocr = ocr_text.get(path, "")
                        if page_ocr:
                            ocr_lines.append(f"Page {page_num}:\n{page_ocr}")
                    batch_ocr = "\n\n".join(ocr_lines) if ocr_lines else None

                question_prompt = self._create_prompt(question, batch_ocr)
                response = self._generate(window, question_prompt, few_shot_turns)
                all_responses.append({"pages": window, "answer": response})

            return {
                "answer": all_responses,
                "query": question,
                "image_paths": image_paths,
                "analysis_type": f"window_size_{window_size}",
            }
        except Exception as e:
            print(f"Error in generate_answer: {str(e)}")
            print(f"Full error: {traceback.format_exc()}")
            return {
                "answer": "Unable to determine: error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
```

- [ ] **Step 4: Add the neutral `_build_few_shot_turns`**

Same selection/anchoring/OCR logic as Qwen's current version (lines 468–536), but emit a **model-neutral** turn list instead of Qwen message dicts:

```python
    def _build_few_shot_turns(self, shots):
        """Neutral conversational turns for few-shot. Each subclass's _generate
        converts these to its native message format.
        Shape: {"role":"user","image_paths":[...],"text":...} / {"role":"assistant","text":...}"""
        window_size = self.model_config.get("batch_size", 1)
        turns = []
        for shot in shots:
            item = shot["item"]
            is_ans = shot["type"] == "answerable"

            all_pages = list(item["layout_analysis"]["pages"].keys())
            anchor = 0
            if item.get("original_answer_locations"):
                answer_page = os.path.basename(item["original_answer_locations"][0]["page_id"])
                all_basenames = [os.path.basename(p) for p in all_pages]
                if answer_page in all_basenames:
                    anchor = all_basenames.index(answer_page)

            start = max(0, min(anchor - window_size // 2, len(all_pages) - window_size))
            window_pages = all_pages[start:start + window_size]
            image_paths = [
                os.path.join(self.images_base_path, os.path.basename(p_id))
                for p_id in window_pages
            ]

            ocr_text = None
            if self.config.get("ocr_enabled", False):
                window_pages_dict = {p: item["layout_analysis"]["pages"][p] for p in window_pages}
                ocr_dict = self.get_ocr_text(window_pages_dict)
                ocr_lines = []
                for i, path in enumerate(image_paths):
                    page_ocr = ocr_dict.get(path, "")
                    if page_ocr:
                        ocr_lines.append(f"Page {i + 1}:\n{page_ocr}")
                ocr_text = "\n\n".join(ocr_lines) if ocr_lines else None

            if is_ans:
                q_text = item["original_question"]
                ans_text = item["original_answer_locations"][0]["answer"]
            else:
                q_text = item["corrupted_question"]
                ans_text = "Unable to determine"

            prompt = self._create_prompt(q_text, ocr_text)
            turns.append({"role": "user", "image_paths": image_paths, "text": prompt})
            turns.append({"role": "assistant", "text": ans_text})
        return turns
```

- [ ] **Step 5: Add `_resolve_leaf` threading the model prefix**

Same as Qwen's current `_resolve_leaf` (lines 319–332) but pass the prefix into `leaf_dir`:

```python
    def _resolve_leaf(self, n_items):
        few_shot_cfg = self.config.get("few_shot", {})
        few_shot_enabled = few_shot_cfg.get("enabled", False)
        mode = rl.derive_mode(self.finetuned, few_shot_enabled)
        slug = rl.build_slug(mode, bool(self.config.get("ocr_enabled", False)),
                             self.model_config.get("batch_size", 1),
                             few_shot=few_shot_cfg if few_shot_enabled else None)
        run_id = os.environ.get("VQA_RUN_ID") or rl.make_run_id(
            self.config.get("split", "val"), n_items
        )
        return run_id, slug, rl.leaf_dir(run_id, self.config["dataset"], slug, self.MODEL_LEAF_PREFIX)
```

- [ ] **Step 6: Verify the module imports cleanly**

Run: `cd /home/amartinelli/VRD-UQA && uv run python -c "from VQA_analysis.evaluators.base_evaluator import BaseVQAEvaluator; print('ok')"`
Expected: prints `ok` (no import errors). *(This import works here because cwd is the repo root; the runtime subclasses use same-dir `import base_evaluator`.)*

- [ ] **Step 7: Commit**

```bash
git add VQA_analysis/evaluators/base_evaluator.py
git commit -m "feat(eval): extract BaseVQAEvaluator with model-agnostic logic"
```

---

## Task 2: Add `leaf_dir` model prefix in `run_layout.py`

**Files:**
- Modify: `config/run_layout.py:87-88`

- [ ] **Step 1: Edit `leaf_dir`**

Replace:

```python
def leaf_dir(run_id: str, dataset: str, slug: str) -> Path:
    return run_dir(run_id) / dataset / slug
```

with:

```python
def leaf_dir(run_id: str, dataset: str, slug: str, model_prefix: str = "") -> Path:
    # model_prefix lets >1 model share a run without colliding on one leaf.
    # "" preserves the historical bare path (Qwen), so existing artifacts and
    # in-flight k-sweep resume are undisturbed.
    name = f"{model_prefix}__{slug}" if model_prefix else slug
    return run_dir(run_id) / dataset / name
```

- [ ] **Step 2: Verify back-compat (default arg keeps old path)**

Run: `cd /home/amartinelli/VRD-UQA && uv run python -c "from config.run_layout import leaf_dir; print(leaf_dir('r','BDocs','zeroshot_ocr').name, '|', leaf_dir('r','BDocs','zeroshot_ocr','llama').name)"`
Expected: `zeroshot_ocr | llama__zeroshot_ocr`

- [ ] **Step 3: Commit**

```bash
git add config/run_layout.py
git commit -m "feat(layout): optional model_prefix in leaf_dir for multi-model runs"
```

---

## Task 3: Migrate Qwen onto the base class

**Files:**
- Modify: `VQA_analysis/evaluators/qwen2.5_evaluator.py` (full rewrite to a thin subclass)
- Test: `tests/test_evaluator_mock.py` (existing — must still pass unchanged)

- [ ] **Step 1: Rewrite `qwen2.5_evaluator.py` as a subclass**

```python
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
```

- [ ] **Step 2: Run the existing mock test to verify parity**

Run: `cd /home/amartinelli/VRD-UQA && uv run python -m tests.test_evaluator_mock`
Expected: prints `OK: evaluator mock dual-answer` (all three existing assertions pass — leaf path `BDocs/finetuned_noocr` unchanged because `MODEL_LEAF_PREFIX=""`).

- [ ] **Step 3: Commit**

```bash
git add VQA_analysis/evaluators/qwen2.5_evaluator.py
git commit -m "refactor(eval): Qwen evaluator as BaseVQAEvaluator subclass"
```

---

## Task 4: Llama-3.2-Vision evaluator

**Files:**
- Create: `VQA_analysis/evaluators/llama_evaluator.py`

> Mllama is effectively single-image per prompt → `batch_size` is pinned to 1 in config (Task 7). Multi-image few-shot is best-effort; a runtime warning is emitted.

- [ ] **Step 1: Write `llama_evaluator.py`**

```python
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
```

- [ ] **Step 2: Verify it imports**

Run: `cd /home/amartinelli/VRD-UQA && uv run python -c "import importlib.util as u; print(u.spec_from_file_location('m','VQA_analysis/evaluators/llama_evaluator.py') is not None)"`
Expected: `True` (full mock-run is exercised by Task 8).

- [ ] **Step 3: Commit**

```bash
git add VQA_analysis/evaluators/llama_evaluator.py
git commit -m "feat(eval): Llama-3.2-Vision evaluator"
```

---

## Task 5: Phi-4-multimodal evaluator

**Files:**
- Create: `VQA_analysis/evaluators/phi4_evaluator.py`

> Phi-4-multimodal uses a manual prompt format with numbered `<|image_i|>` placeholders and `trust_remote_code`. **Verify at smoke time** (Task 12): the exact special tokens (`<|user|>`/`<|end|>`/`<|assistant|>`) and whether `generation_config` needs `num_logits_to_keep`. The structure below follows Microsoft's documented usage.

- [ ] **Step 1: Write `phi4_evaluator.py`**

```python
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
        if self.model_config.get("use_flash_attention", False):
            model_kwargs["_attn_implementation"] = "flash_attention_2"
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
```

- [ ] **Step 2: Commit**

```bash
git add VQA_analysis/evaluators/phi4_evaluator.py
git commit -m "feat(eval): Phi-4-multimodal evaluator"
```

---

## Task 6: InternVL3.5-8B evaluator

**Files:**
- Create: `VQA_analysis/evaluators/internvl_evaluator.py`

> Reuses the judge's proven load path (`AutoModel` + `trust_remote_code`, `model.chat`). The tiling helpers below are InternVL's standard preprocessing. **Verify at smoke time** (Task 12): `model.chat` history-with-images behavior for multi-shot.

- [ ] **Step 1: Write `internvl_evaluator.py`**

```python
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
```

> Note: few-shot images in `history` are not attached via `model.chat`'s text-only history; for multi-shot with images, smoke-test and, if needed, fold demonstrations into a single multi-image turn. Few-shot is not required for the zero-shot baseline that ships first.

- [ ] **Step 2: Commit**

```bash
git add VQA_analysis/evaluators/internvl_evaluator.py
git commit -m "feat(eval): InternVL3.5-8B evaluator (model.chat + dynamic tiling)"
```

---

## Task 7: Config entries for the new models

**Files:**
- Modify: `VQA_analysis/config_zeroshot.json`, `config_fewshot.json`, `config_mock.json`

- [ ] **Step 1: In all three configs, pin Llama `batch_size` and add `internvl3_5`**

In each file's `open_source_models`, ensure the `llama3.2` entry has `"batch_size": 1` and add this sibling entry:

```json
        "internvl3_5": {
            "model_name": "OpenGVLab/InternVL3_5-8B",
            "batch_size": 1,
            "max_tokens": 1024,
            "max_tiles": 12,
            "input_size": 448,
            "use_flash_attention": true,
            "name": "InternVL3.5-8B"
        }
```

(`config_mock.json` may keep `use_flash_attention` as it is elsewhere in that file; the value is irrelevant in mock mode.)

- [ ] **Step 2: Verify all three parse**

Run: `cd /home/amartinelli/VRD-UQA && uv run python -c "import json; [print(p, 'internvl3_5' in json.load(open(p))['open_source_models'], json.load(open(p))['open_source_models']['llama3.2']['batch_size']) for p in ['VQA_analysis/config_zeroshot.json','VQA_analysis/config_fewshot.json','VQA_analysis/config_mock.json']]"`
Expected: each line ends with `True 1`.

- [ ] **Step 3: Commit**

```bash
git add VQA_analysis/config_zeroshot.json VQA_analysis/config_fewshot.json VQA_analysis/config_mock.json
git commit -m "feat(config): add internvl3_5 entry; pin llama3.2 batch_size=1"
```

---

## Task 8: Parametrize mock tests over all 4 evaluators

**Files:**
- Modify: `tests/test_evaluator_mock.py`

- [ ] **Step 1: Replace the test module with a parametrized version**

```python
"""Mock-mode evaluator checks for all model entrypoints.
Run: uv run python -m tests.test_evaluator_mock"""
import json
import os
import subprocess
import tempfile
from pathlib import Path
from config.paths import REPO_ROOT

SAMPLE = REPO_ROOT / "VQA_analysis" / "evaluation_files" / "BDocs_sample15.json"

# (model_key, entrypoint, leaf_prefix). Qwen is finetuned (adapter entry exists);
# the new models have no adapter yet, so they run zero-shot (no --finetuned).
MODELS = [
    ("qwen2.5", "VQA_analysis/evaluators/qwen2.5_evaluator.py", "", True),
    ("llama3.2", "VQA_analysis/evaluators/llama_evaluator.py", "llama", False),
    ("phi4", "VQA_analysis/evaluators/phi4_evaluator.py", "phi4", False),
    ("internvl3_5", "VQA_analysis/evaluators/internvl_evaluator.py", "internvl", False),
]


def _write_mock_config(tmp):
    base = json.load(open(REPO_ROOT / "VQA_analysis" / "config_mock.json"))
    base["dataset"] = "BDocs"
    base["input_file"] = str(SAMPLE)
    base["ocr_enabled"] = False
    base["sampling_percentage"] = 100
    base["seed"] = 42
    base["few_shot"] = {"enabled": False}  # no images on disk in test env
    cfg = Path(tmp) / "mock_cfg.json"
    cfg.write_text(json.dumps(base))
    return cfg


def _run(entrypoint, cfg, run_id, questions, run_root, finetuned):
    env = dict(os.environ)
    env["VQA_RUN_ID"] = run_id
    env["VQA_EVAL_RUNS_DIR"] = str(run_root)
    cmd = ["uv", "run", "python", entrypoint, "--config_path", str(cfg), "--questions", questions]
    if finetuned:
        cmd.append("--finetuned")
    subprocess.check_call(cmd, cwd=str(REPO_ROOT), env=env)


def _leaf_name(prefix, finetuned):
    slug = "finetuned_noocr" if finetuned else "zeroshot_noocr"
    return f"{prefix}__{slug}" if prefix else slug


def test_all_models_dual_answer_and_namespaced_leaf():
    for i, (key, entry, prefix, finetuned) in enumerate(MODELS):
        tmp = tempfile.mkdtemp()
        run_root = Path(tmp) / "runs"
        cfg = _write_mock_config(tmp)
        run_id = f"eval_val_15_2026010100000{i}"
        _run(entry, cfg, run_id, "both", run_root, finetuned)

        leaf = run_root / run_id / "BDocs" / _leaf_name(prefix, finetuned)
        preds = json.load(open(leaf / "predictions.json"))
        item0 = preds["corrupted_questions"][0]["verification_result"]["vqa_results"][0]
        assert "answer_corrupted" in item0 and "answer_clean" in item0, key
        assert item0["question_corrupted"] != item0["question_clean"], key
        assert item0["model_type"] in ("qwen", "llama", "phi4", "internvl"), key

        man = json.load(open(leaf / "manifest.json"))
        assert man["dataset"] == "BDocs", key
        assert man["questions"] == "both", key
        assert man["seed"] == 42, key


def test_corrupted_only_omits_clean_qwen():
    tmp = tempfile.mkdtemp()
    run_root = Path(tmp) / "runs"
    cfg = _write_mock_config(tmp)
    run_id = "eval_val_15_20260101_000010"
    _run(MODELS[0][1], cfg, run_id, "corrupted", run_root, True)
    leaf = run_root / run_id / "BDocs" / "finetuned_noocr"
    item0 = json.load(open(leaf / "predictions.json"))["corrupted_questions"][0]["verification_result"]["vqa_results"][0]
    assert "answer_corrupted" in item0 and "answer_clean" not in item0


def test_resume_skips_completed_leaf_qwen():
    tmp = tempfile.mkdtemp()
    run_root = Path(tmp) / "runs"
    cfg = _write_mock_config(tmp)
    run_id = "eval_val_15_20260101_000011"
    _run(MODELS[0][1], cfg, run_id, "both", run_root, True)
    preds = run_root / run_id / "BDocs" / "finetuned_noocr" / "predictions.json"
    with open(preds) as f:
        d = json.load(f)
    d["_resume_marker"] = True
    with open(preds, "w") as f:
        json.dump(d, f)
    _run(MODELS[0][1], cfg, run_id, "both", run_root, True)  # second pass -> should skip
    with open(preds) as f:
        assert json.load(f).get("_resume_marker") is True


if __name__ == "__main__":
    test_all_models_dual_answer_and_namespaced_leaf()
    test_corrupted_only_omits_clean_qwen()
    test_resume_skips_completed_leaf_qwen()
    print("OK: evaluator mock dual-answer (all 4 models)")
```

- [ ] **Step 2: Run the test**

Run: `cd /home/amartinelli/VRD-UQA && uv run python -m tests.test_evaluator_mock`
Expected: `OK: evaluator mock dual-answer (all 4 models)`

- [ ] **Step 3: Commit**

```bash
git add tests/test_evaluator_mock.py
git commit -m "test(eval): parametrize mock test over all 4 evaluators + namespaced leaf"
```

---

## Task 9: Cross-run aggregator

**Files:**
- Create: `VQA_analysis/metrics/4_aggregate_summaries.py`
- Test: `tests/test_aggregate_summaries.py`

- [ ] **Step 1: Write the failing test**

```python
"""Run: uv run python -m tests.test_aggregate_summaries"""
import csv
import os
import tempfile
from pathlib import Path


def _write_summary(run_dir, rows):
    run_dir.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "config", "label", "model", "metric", "complexity", "value"]
    with open(run_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_aggregates_multiple_runs():
    tmp = tempfile.mkdtemp()
    runs = Path(tmp) / "runs"
    os.environ["VQA_EVAL_RUNS_DIR"] = str(runs)
    _write_summary(runs / "eval_val_100_llama_BDocs",
                   [{"dataset": "BDocs", "config": "zeroshot_ocr", "label": "Zero-Shot",
                     "model": "Llama3.2-11B", "metric": "QUR", "complexity": "overall", "value": "0.5"}])
    _write_summary(runs / "eval_val_100_phi4_BDocs",
                   [{"dataset": "BDocs", "config": "zeroshot_ocr", "label": "Zero-Shot",
                     "model": "Phi4-multimodal", "metric": "QUR", "complexity": "overall", "value": "0.6"}])

    from importlib.machinery import SourceFileLoader
    from config.paths import REPO_ROOT
    mod = SourceFileLoader(
        "agg", str(REPO_ROOT / "VQA_analysis" / "metrics" / "4_aggregate_summaries.py")
    ).load_module()
    out = mod.aggregate(tag="test")

    with open(out) as f:
        rows = list(csv.DictReader(f))
    models = sorted(r["model"] for r in rows)
    assert models == ["Llama3.2-11B", "Phi4-multimodal"], models
    assert {r["run_id"] for r in rows} == {"eval_val_100_llama_BDocs", "eval_val_100_phi4_BDocs"}


if __name__ == "__main__":
    test_aggregates_multiple_runs()
    print("OK: aggregate summaries")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /home/amartinelli/VRD-UQA && uv run python -m tests.test_aggregate_summaries`
Expected: FAIL — `FileNotFoundError`/`No such file` for `4_aggregate_summaries.py`.

- [ ] **Step 3: Write `4_aggregate_summaries.py`**

```python
"""Aggregate per-run summary.csv files into one comparison table.

Each parallel evaluation job writes a private run dir with its own summary.csv
(see run_vqa_analysis.sh). This read-only step concatenates them so models /
datasets evaluated in separate jobs can be compared in one table.

Run:
  uv run python VQA_analysis/metrics/4_aggregate_summaries.py [--tag NAME]
      [--run-glob 'eval_*'] [--datasets BDocs DUDE] [--run-ids r1 r2 ...]

Writes EVAL_RUNS_DIR/comparison_<tag>.csv. Honors VQA_EVAL_RUNS_DIR.
"""
import argparse
import csv
from config.run_layout import EVAL_RUNS_DIR, SUMMARY_COLUMNS

OUT_COLUMNS = ["run_id"] + SUMMARY_COLUMNS


def aggregate(run_glob="eval_*", tag="all", datasets=None, run_ids=None):
    if run_ids:
        run_dirs = [EVAL_RUNS_DIR / r for r in run_ids]
    else:
        run_dirs = sorted(p for p in EVAL_RUNS_DIR.glob(run_glob) if p.is_dir())

    rows = []
    used = []
    for rd in run_dirs:
        sc = rd / "summary.csv"
        if not sc.exists():
            continue
        used.append(rd.name)
        with open(sc, newline="") as f:
            for row in csv.DictReader(f):
                if datasets and row.get("dataset") not in datasets:
                    continue
                row["run_id"] = rd.name
                rows.append(row)

    out = EVAL_RUNS_DIR / f"comparison_{tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in OUT_COLUMNS})
    print(f"Wrote {out} ({len(rows)} rows from {len(used)} runs: {used})")
    return out


def main():
    p = argparse.ArgumentParser(description="Aggregate per-run summary.csv into a comparison table.")
    p.add_argument("--tag", default="all")
    p.add_argument("--run-glob", default="eval_*")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--run-ids", nargs="*", default=None)
    args = p.parse_args()
    aggregate(run_glob=args.run_glob, tag=args.tag, datasets=args.datasets, run_ids=args.run_ids)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/amartinelli/VRD-UQA && uv run python -m tests.test_aggregate_summaries`
Expected: `OK: aggregate summaries`

- [ ] **Step 5: Commit**

```bash
git add VQA_analysis/metrics/4_aggregate_summaries.py tests/test_aggregate_summaries.py
git commit -m "feat(metrics): cross-run summary aggregator + test"
```

---

## Task 10: Refactor `run_vqa_analysis.sh` to the parameterized, parallel-safe launcher

**Files:**
- Modify: `scripts/slurm/run_vqa_analysis.sh`

- [ ] **Step 1: Replace the script**

```bash
#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-VQA_analysis-%j.out

# One model x one dataset x one split per job, so jobs run in parallel safely.
# Usage:
#   sbatch run_vqa_analysis.sh <model> <dataset> <split>
#   e.g. sbatch run_vqa_analysis.sh llama BDocs val_100
#
#   <model>   = qwen2.5 | llama | phi4 | internvl
#   <dataset> = BDocs | DUDE | MPDocVQA | SlideVQA
#   <split>   = val_300 | val_100 | val_5 | ...
#
# Each job derives a deterministic, unique run_id eval_<split>_<n>_<model>_<dataset>,
# so parallel jobs never share a run dir (no sync/metrics collision). Resubmitting
# the same combo RESUMES. Set RUN_TAG=foo to force a distinct fresh run.

set -uo pipefail
START_TIME=$SECONDS
echo "Job started at: $(date)"

MODEL="${1:?usage: run_vqa_analysis.sh <model> <dataset> <split>}"
DATASET="${2:?usage: run_vqa_analysis.sh <model> <dataset> <split>}"
SPLIT="${3:-val_300}"
N="${SPLIT##*_}"
SPLIT_NAME="${SPLIT%_*}"
RUN_TAG="${RUN_TAG:-}"

# Map model key -> evaluator entrypoint. --finetuned only when an adapter exists
# for that model (Phase B wires the *_finetuned config entries + FINETUNE flag).
case "$MODEL" in
  qwen2.5)  ENTRY="VQA_analysis/evaluators/qwen2.5_evaluator.py" ;;
  llama)    ENTRY="VQA_analysis/evaluators/llama_evaluator.py" ;;
  phi4)     ENTRY="VQA_analysis/evaluators/phi4_evaluator.py" ;;
  internvl) ENTRY="VQA_analysis/evaluators/internvl_evaluator.py" ;;
  *) echo "ERROR: unknown model '$MODEL'"; exit 2 ;;
esac

# Eval condition is config-driven. Default: few-shot config, no --finetuned for the
# new models (no adapter yet). Override CONFIG / FINETUNE via env when needed.
CONFIG="${CONFIG:-VQA_analysis/config_fewshot.json}"
FINETUNE="${FINETUNE:-}"   # set FINETUNE=--finetuned to use a *_finetuned entry

module purge
module load miniconda3/3.13.25
module load nvhpc/25.1
source "$HOME/VRD-UQA/scripts/env.sh"

export VQA_RUN_ID="${VQA_RUN_ID:-eval_${SPLIT_NAME}_${N}_${MODEL}_${DATASET}${RUN_TAG:+_$RUN_TAG}}"
echo "Model: $MODEL | Dataset: $DATASET | Split: $SPLIT | Run id: $VQA_RUN_ID"

WORK_DIR=$SCRATCH_FLASH/VQA_analysis_${SLURM_JOB_ID}
SRC_RUN="$WORK_DIR/artifacts/evaluation_runs/$VQA_RUN_ID"
DEST_RUN="$HOME/VRD-UQA/artifacts/evaluation_runs/$VQA_RUN_ID"

sync_back() {
    [ -d "$SRC_RUN" ] || return 0
    mkdir -p "$DEST_RUN"
    rsync -aq "$SRC_RUN/" "$DEST_RUN/" || true
    # 'latest' is a global pointer; updating it from parallel jobs is a harmless
    # last-writer-wins race. Opt out with NO_LATEST=1 to avoid churn.
    [ -n "${NO_LATEST:-}" ] || ln -sfn "$VQA_RUN_ID" "$HOME/VRD-UQA/artifacts/evaluation_runs/latest" 2>/dev/null || true
}
trap 'sync_back; cd "$HOME"; rm -rf "$WORK_DIR"' EXIT

rm -rf "$WORK_DIR"
rsync -aq --exclude='data' --exclude='.git' --exclude='.venv' \
      --exclude='corruption-scripts/results' --exclude='finetuning' \
      "$HOME/VRD-UQA/" "$WORK_DIR/"
cd "$WORK_DIR"
uv --version
export UV_LINK_MODE=copy
uv sync -qq

# Resume: restore an already-computed run for this id from $HOME into scratch.
if [ -d "$DEST_RUN" ]; then
    echo "RESUME: prior results at $DEST_RUN — restoring into scratch; finished leaf will be skipped."
    mkdir -p "$SRC_RUN"
    rsync -aq "$DEST_RUN/" "$SRC_RUN/"
fi

export VQA_CONFIG_PATH="$CONFIG"

# Point the chosen config at this dataset/split's corrupted set.
uv run python -c "
import json, sys
dataset, split, cfg_path = sys.argv[1], sys.argv[2], sys.argv[3]
input_file = f'/home/amartinelli/VRD-UQA/data/{dataset}/{dataset}_{split}/{dataset}_unanswerable_corrupted_questions_just_false.json'
cfg = json.load(open(cfg_path))
cfg['dataset'] = dataset
cfg['split'] = split.split('_')[0]
cfg['input_file'] = input_file
json.dump(cfg, open(cfg_path, 'w'), indent=4)
" "$DATASET" "$SPLIT" "$CONFIG"

printf '\n=== %s — %s — QUR+FRR — %s ===\n' "$MODEL" "$CONFIG" "$DATASET"
uv run python "$ENTRY" --config_path "$CONFIG" $FINETUNE --questions both
sync_back

# Metrics — scoped to THIS run_id only (private tree => no cross-job collision).
uv run python VQA_analysis/metrics/1_normalize_unanswerable_responses.py --run-id "$VQA_RUN_ID"
uv run python VQA_analysis/metrics/2_enrich_metadata.py            --run-id "$VQA_RUN_ID"
uv run python VQA_analysis/metrics/3_compute_metrics.py            --run-id "$VQA_RUN_ID"

# Per-job slurm log move — scoped to THIS job id (never grab sibling jobs' logs).
mv "$HOME"/slurm-VQA_analysis-${SLURM_JOB_ID}.out "$HOME/VRD-UQA/" 2>/dev/null || true

sync_back
echo "Results synced to $DEST_RUN"
ELAPSED=$(( SECONDS - START_TIME ))
printf 'Job finished at: %s\nTotal execution time: %02d:%02d:%02d (%ds)\n' \
    "$(date)" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$ELAPSED"
```

- [ ] **Step 2: Shellcheck-lint the script (syntax only; no SLURM here)**

Run: `cd /home/amartinelli/VRD-UQA && bash -n scripts/slurm/run_vqa_analysis.sh && echo "syntax ok"`
Expected: `syntax ok`

- [ ] **Step 3: Commit**

```bash
git add scripts/slurm/run_vqa_analysis.sh
git commit -m "feat(slurm): per-(model,dataset,split) parallel-safe eval launcher"
```

---

## Task 11: Update `sbatch_commands.txt` with the parallel launch pattern

**Files:**
- Modify: `sbatch_commands.txt`

- [ ] **Step 1: Append a documented parallel-launch block**

Add to the end of `sbatch_commands.txt`:

```text

# --- Multi-model eval (one model x dataset x split per job; runs in parallel) ---
# sbatch run_vqa_analysis.sh <model> <dataset> <split>   model = qwen2.5|llama|phi4|internvl
for M in qwen2.5 llama phi4 internvl; do
  sbatch --job-name=vqa-$M scripts/slurm/run_vqa_analysis.sh $M BDocs val_100
done
# After the jobs finish, build a comparison table across the private runs:
uv run python VQA_analysis/metrics/4_aggregate_summaries.py --tag bdocs_val100 --datasets BDocs
```

- [ ] **Step 2: Commit**

```bash
git add sbatch_commands.txt
git commit -m "docs: parallel multi-model eval launch + aggregator commands"
```

---

## Task 12: Manual real-model smoke runs (HPC — documented, not CI)

> These require GPU + model downloads, so they are run by you on the HPC, not in the test suite. Record the commands; they validate the `_load_model`/`_generate` paths and the verify-at-smoke notes in Tasks 5–6.

- [ ] **Step 1: Smoke each new model on `val_5`, one dataset**

```bash
sbatch --job-name=smoke-llama    scripts/slurm/run_vqa_analysis.sh llama    BDocs val_5
sbatch --job-name=smoke-phi4     scripts/slurm/run_vqa_analysis.sh phi4     BDocs val_5
sbatch --job-name=smoke-internvl scripts/slurm/run_vqa_analysis.sh internvl BDocs val_5
```

- [ ] **Step 2: Confirm each produced predictions + metrics**

For each run_id `eval_val_5_<model>_BDocs`, check:
`ls artifacts/evaluation_runs/eval_val_5_<model>_BDocs/BDocs/*/predictions.json` and a non-empty `summary.csv`.
If Phi-4 or InternVL errors on the prompt/`model.chat` API, apply the verify-at-smoke notes in Tasks 5–6 (special tokens / `generation_config`; `model.chat` history signature) and re-run.

---

## Self-Review Notes (author)

- **Spec coverage:** A.1 base class → T1/T3; A.2 per-model → T3–T6; A.3 neutral few-shot → T1; A.4 leaf prefix → T2; A.5 launcher → T10/T11; A.5.1 parallel safety → T10; A.6 tests → T8; A.7 aggregator → T9. Caveats (Llama single-image, Phi-4/InternVL verify-at-smoke) → T4/T5/T6 + T12.
- **Finetuned config entries** (`*_finetuned`) and `FINETUNED_MODEL_KEY` adapter wiring are referenced here but **created in Phase B** (Plan 2); until then the new models run zero/few-shot only, which is what the mock test exercises (`--finetuned` only for Qwen).
