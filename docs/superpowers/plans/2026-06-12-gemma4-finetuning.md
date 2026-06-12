# Gemma 4 12B LoRA Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `google/gemma-4-12B-it` as a fine-tuned core test model in VRD-UQA, trained with a standalone HF `Trainer` + PEFT LoRA script in a dedicated per-job SLURM venv, reusing the existing model-agnostic SFT dataset.

**Architecture:** A standalone training script (`finetuning/gemma4/finetune_gemma4.py`) structurally mirrors the proven `finetuning/phi4mm_official/finetune_phi4mm.py` (dataset class → pre-tokenized tensors with answer-only label masking, custom collator, `Trainer` + `TrainingArguments`). It wraps the base model in a **PEFT LoRA adapter** (output ~500MB, not a full model). Gemma 4 is dense + **encoder-free unified** multimodal (35M vision embedder, raw 48×48 patches, **configurable visual-token budget** — no vision tower, no dynamic tiling, no Phi-4-style image-token explosion). Heavy model/peft imports are **lazy** (inside `create_model`) so the data path is unit-testable in the main repo venv without `transformers>=5.5.2`. A SLURM runner builds an isolated per-job venv (`transformers>=5.5.2`, `peft>=0.19.0`) on scratch, trains, and copies the adapter back.

**Tech Stack:** Python, PyTorch, HF `transformers` (`AutoModelForMultimodalLM` + `AutoProcessor`), PEFT LoRA, `accelerate`, SLURM, `uv` venvs, BeeGFS scratch.

---

## Design notes (read before starting)

- **Deliberate deviation from the spec, flagged for the executor:** the spec header says "TRL `SFTTrainer` + PEFT LoRA", but the spec's *structural* section says mirror Phi-4 (`Trainer` + `TrainingArguments`). This plan uses **plain `Trainer`** because the dataset is pre-tokenized with custom answer-only label masking and a custom multimodal collator — `SFTTrainer` adds text-field auto-processing that fights a pre-tokenized dataset. PEFT LoRA is unchanged. `trl` is therefore **dropped from the venv deps** (avoids dependency-resolution conflicts against `transformers>=5.5.2`). If the user wants `SFTTrainer` specifically, that's a follow-up swap.
- **Robust to the unknown processor key names:** Gemma 4's processor image-output key names (e.g. `pixel_values`, token-type ids) can only be confirmed on the A40. The collator **passes through every non-text key the processor emits, under its original name** — so whatever the processor produces is forwarded verbatim to the model's `forward`, which by HF convention expects those same names. This removes the main uncertainty without hardcoding key names.
- **Visual-token budget = 560** (decided with the user). Set at `AutoProcessor.from_pretrained(..., visual_token_budget=560)`. The exact kwarg name is the one open item the A40 smoke run confirms; if the kwarg differs, adjust in `main()` only.
- **Untestable-here reality (repo convention #5):** real training needs the A40 and is run on SLURM by the user. Local verification = the CI-safe data-path test + `python -m ast` parse + `bash -n`. Task 5 is the HPC smoke→full run the user executes.
- **NEVER `git add -A`/`git add .`** — the working tree holds the user's uncommitted notebooks + `run_fewshot_ksweep.sh`. Every commit step uses explicit paths.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `finetuning/gemma4/__init__.py` | **Create** | Make `finetuning.gemma4` an importable package (needed by the test). |
| `finetuning/gemma4/finetune_gemma4.py` | **Create** | TRL-style PEFT LoRA training script. Top half (dataset + collator, pure Python) is import-clean in the main venv; `create_model`/`main` lazily import `AutoModelForMultimodalLM`/`peft`. |
| `tests/test_gemma4_finetune.py` | **Create** | CI-safe test of the data path (message building, label masking, collator) with a mock processor. No GPU/model. |
| `scripts/slurm/run_finetune_gemma4.sh` | **Create** | SLURM job: per-job scratch venv (`transformers>=5.5.2`, `peft>=0.19.0`), train, rsync adapter back to `artifacts/finetuning/gemma4_12b_lora_sft/`. |
| `finetuning/readme.txt` | **Modify** | Add one Gemma 4 launch line. |

**Reused unchanged:** `artifacts/finetuning/dataset/train.json` (5920 alpaca records), `finetuning/build_paired_dataset.py`, `scripts/env.sh`.

**Out of scope (documented follow-ups, see end):** Gemma 4 *evaluator* + `run_vqa_analysis.sh` `gemma4)` case + `gemma4_finetuned` config entry + a pinned eval env. These need the trained adapter to exist first and a separate inference-side design.

---

## Task 1: Data path — dataset class + collator (CI-safe, TDD)

Build only the pure-Python parts of the training script (no model/peft), plus the test that locks their behavior. The model loader + `main()` come in Task 2.

**Files:**
- Create: `finetuning/gemma4/__init__.py`
- Create: `finetuning/gemma4/finetune_gemma4.py` (data-path portion)
- Create: `tests/test_gemma4_finetune.py`

- [ ] **Step 1: Create the package marker**

Create `finetuning/gemma4/__init__.py` (empty file):

```python
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_gemma4_finetune.py`:

```python
"""CI-safe tests for the Gemma 4 fine-tune data path (no model/GPU needed).
Run: uv run python -m tests.test_gemma4_finetune

Exercises the pure-Python parts of finetuning/gemma4/finetune_gemma4.py — record
-> chat-message text building, answer-only label masking, and the collator — with a
MOCK processor. Real training runs on the A40 via scripts/slurm/run_finetune_gemma4.sh.
"""
import json
import tempfile
from pathlib import Path

import torch
from PIL import Image

from finetuning.gemma4.finetune_gemma4 import (
    VrdUqaGemmaDataset,
    build_collate_fn,
    _IGNORE_INDEX,
)


class _MockTokenizer:
    eos_token = "<eos>"
    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        # 1 id per whitespace token; deterministic, content-independent.
        ids = torch.tensor([[5 + i for i, _ in enumerate(text.split())]])
        r = type("R", (), {})()
        r.input_ids = ids
        return r


class _MockProcessor:
    """Records the last apply_chat_template / __call__ args for assertions."""

    def __init__(self):
        self.tokenizer = _MockTokenizer()
        self.last_messages = None
        self.last_images = None
        self.last_text = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.last_messages = messages
        parts = [c["text"] for m in messages for c in m["content"] if c["type"] == "text"]
        return "PROMPT " + " ".join(parts)

    def __call__(self, images=None, text=None, return_tensors=None):
        self.last_images = images
        self.last_text = text
        n = max(1, len(text.split()))

        class _R(dict):
            def __init__(self, d):
                super().__init__(d)
                self.input_ids = d["input_ids"]

        return _R({
            "input_ids": torch.arange(100, 100 + n).unsqueeze(0),
            "pixel_values": torch.zeros(1, 3, 48, 48),
        })


def _fixture(tmp, n=2):
    img_path = Path(tmp) / "p.jpg"
    Image.new("RGB", (32, 32), (255, 255, 255)).save(img_path)
    recs = [{
        "instruction": "GUIDE",
        "input": "<image>\nWhat is X?",
        "output": "ANS WORD",
        "images": [str(img_path)],
    } for _ in range(n)]
    j = Path(tmp) / "train.json"
    j.write_text(json.dumps(recs))
    return str(j)


def test_getitem_builds_message_and_masks_labels():
    tmp = tempfile.mkdtemp()
    proc = _MockProcessor()
    ds = VrdUqaGemmaDataset(proc, _fixture(tmp))
    item = ds[0]

    # message carries an image placeholder + combined instruction/question text
    content = proc.last_messages[0]["content"]
    assert {c["type"] for c in content} == {"image", "text"}
    text_part = [c["text"] for c in content if c["type"] == "text"][0]
    assert text_part == "GUIDE\nWhat is X?"      # <image> stripped, instruction prepended
    assert len(proc.last_images) == 1            # exactly one PIL image passed

    # labels: only the answer span is supervised; everything before is masked
    ans_text = ds.data[0]["output"] + proc.tokenizer.eos_token
    n_answer = proc.tokenizer(ans_text).input_ids.shape[1]
    assert item["input_ids"].shape == item["labels"].shape
    assert (item["labels"][0, :-n_answer] == _IGNORE_INDEX).all()
    assert (item["labels"][0, -n_answer:] != _IGNORE_INDEX).all()
    assert "pixel_values" in item                # image tensor carried through


def test_collate_pads_and_stacks():
    tmp = tempfile.mkdtemp()
    proc = _MockProcessor()
    ds = VrdUqaGemmaDataset(proc, _fixture(tmp, n=2))
    batch = [ds[0], ds[1]]
    collate = build_collate_fn(pad_token_id=0)
    out = collate(batch)
    assert out["input_ids"].shape[0] == 2
    assert out["labels"].shape == out["input_ids"].shape
    assert out["attention_mask"].shape == out["input_ids"].shape
    assert out["pixel_values"].shape[0] == 2     # image tensors stacked along dim 0


if __name__ == "__main__":
    test_getitem_builds_message_and_masks_labels()
    test_collate_pads_and_stacks()
    print("OK: gemma4 finetune data path")
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run python -m tests.test_gemma4_finetune`
Expected: FAIL — `ModuleNotFoundError: No module named 'finetuning.gemma4.finetune_gemma4'` (script not created yet).

- [ ] **Step 4: Create the script's data-path portion**

Create `finetuning/gemma4/finetune_gemma4.py` with exactly this content (model loader + `main()` are added in Task 2):

```python
"""Fine-tune Gemma 4 12B (it) on the VRD-UQA SFT data via HF Trainer + PEFT LoRA.

Gemma 4 is a dense, encoder-free *unified* multimodal model: a 35M-param vision
embedder projects raw 48x48 patches straight into the decoder under a configurable
visual-token budget (no vision tower, no dynamic tiling -> no Phi-4-style image-token
explosion). We train a PEFT LoRA adapter on the frozen base; eval loads base + adapter.

LLaMA-Factory has no Gemma 4 vision template, so this is a standalone script that
mirrors finetuning/phi4mm_official/finetune_phi4mm.py in structure (dataset class,
answer-only label masking, custom collator, Trainer + TrainingArguments). It reuses the
model-agnostic alpaca train.json unchanged.

Run via scripts/slurm/run_finetune_gemma4.sh, which builds a per-job venv with the
versions Gemma 4 needs (transformers>=5.5.2, peft>=0.19.0) — isolated from the main
repo venv (~4.57.6) and the Phi-4 path (pinned 4.47.0). Output is a PEFT LoRA adapter
(~500MB); eval loads base + adapter_path.

NOTE: AutoModelForMultimodalLM and peft are imported LAZILY inside create_model() so this
module imports cleanly in the main repo venv for the CI-safe data-path test
(tests/test_gemma4_finetune.py) without requiring transformers>=5.5.2.
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, BatchFeature, Trainer, TrainingArguments

_IGNORE_INDEX = -100
_MAX_TRAINING_LENGTH = 4096  # matches Qwen/InternVL cutoff_len; Gemma 4 ctx is far larger
_TEXT_KEYS = {"input_ids", "labels", "attention_mask"}


class VrdUqaGemmaDataset(Dataset):
    """VRD-UQA alpaca records -> Gemma 4 training tensors (image + chat, answer-only labels)."""

    def __init__(self, processor, data_json, max_samples=None):
        with open(data_json) as f:
            self.data = json.load(f)
        if max_samples:
            self.data = self.data[:max_samples]
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]
        instruction = rec.get("instruction", "") or ""
        # input is '<image>\n{question}' (LLaMA-Factory placeholder); strip the placeholder.
        question = (rec.get("input", "") or "").replace("<image>", "").strip()
        text = (instruction + "\n" + question).strip()
        image = Image.open(rec["images"][0]).convert("RGB")

        # Gemma 4 chat: image placeholder + text; the PIL image is passed via images=[...].
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": text},
        ]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.processor(images=[image], text=prompt, return_tensors="pt")

        answer = f'{rec["output"]}{self.processor.tokenizer.eos_token}'
        answer_ids = self.processor.tokenizer(
            answer, add_special_tokens=False, return_tensors="pt"
        ).input_ids

        input_ids = torch.cat([prompt_inputs.input_ids, answer_ids], dim=1)
        labels = torch.full_like(input_ids, _IGNORE_INDEX)
        labels[:, -answer_ids.shape[1]:] = answer_ids
        if input_ids.size(1) > _MAX_TRAINING_LENGTH:
            input_ids = input_ids[:, :_MAX_TRAINING_LENGTH]
            labels = labels[:, :_MAX_TRAINING_LENGTH]
            if torch.all(labels == _IGNORE_INDEX).item():
                labels[:, -1] = self.processor.tokenizer.eos_token_id

        item = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }
        # Carry through every image tensor the processor produced (key names are
        # model-specific, e.g. pixel_values) without hardcoding them.
        for key, val in prompt_inputs.items():
            if key not in _TEXT_KEYS:
                item[key] = val
        return item


def pad_sequence(sequences, padding_side="right", padding_value=0):
    assert padding_side in ("right", "left")
    trailing_dims = sequences[0].size()[1:]
    max_len = max(len(seq) for seq in sequences)
    output = sequences[0].new_full((len(sequences), max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == "right":
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def cat_with_pad(tensors, dim=0, padding_value=0):
    """Concatenate tensors that may differ in non-cat dims, padding to the per-dim max."""
    ndim = tensors[0].dim()
    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)
    index = 0
    for t in tensors:
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        slices[dim] = slice(index, index + t.shape[dim])
        output[tuple(slices)] = t
        index += t.shape[dim]
    return output


def build_collate_fn(pad_token_id=0):
    """Return a collator closure. Pads input_ids/labels, builds a length-based attention
    mask (pad-id agnostic), and stacks every image key the dataset carried through."""

    def collate_fn(batch):
        seqs = [b["input_ids"][0] for b in batch]
        lengths = [int(s.size(0)) for s in seqs]
        input_ids = pad_sequence(seqs, "right", pad_token_id)
        labels = pad_sequence([b["labels"][0] for b in batch], "right", _IGNORE_INDEX)
        max_len = input_ids.size(1)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, length in enumerate(lengths):
            attention_mask[i, :length] = 1
        out = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        image_keys = [k for k in batch[0] if k not in _TEXT_KEYS]
        for key in image_keys:
            out[key] = cat_with_pad([b[key] for b in batch], dim=0)
        return BatchFeature(out)

    return collate_fn
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run python -m tests.test_gemma4_finetune`
Expected: `OK: gemma4 finetune data path`

- [ ] **Step 6: Commit**

```bash
git add finetuning/gemma4/__init__.py finetuning/gemma4/finetune_gemma4.py tests/test_gemma4_finetune.py
git commit -m "feat(gemma4): SFT data path (dataset + collator) with CI-safe test

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Model loader + LoRA + `main()` (A40-only; validated via parse + re-run test)

Append the model/training code to `finetune_gemma4.py`. Heavy imports stay lazy so the data-path test from Task 1 keeps passing in the main venv.

**Files:**
- Modify: `finetuning/gemma4/finetune_gemma4.py` (append `create_model` + `main`)

- [ ] **Step 1: Append `create_model` and `main`**

Append to the end of `finetuning/gemma4/finetune_gemma4.py`:

```python
def create_model(model_name_or_path, use_flash_attention=True, lora_r=16, lora_alpha=32):
    # Lazy imports: only needed on the A40 (per-job venv with transformers>=5.5.2 / peft>=0.19).
    from transformers import AutoModelForMultimodalLM
    from peft import LoraConfig, get_peft_model

    model = AutoModelForMultimodalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,  # bf16: fp32 weights OOM on the A40 (Phi-4 lesson)
        attn_implementation="flash_attention_2" if use_flash_attention else "sdpa",
    )
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=None,  # PEFT 0.19+ supplies Gemma 4 decoder defaults (no vision tower)
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()  # needed for gradient checkpointing + PEFT
    model.print_trainable_parameters()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="google/gemma-4-12B-it")
    p.add_argument("--data_json", required=True, help="VRD-UQA SFT train.json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--use_flash_attention", action="store_true", default=True)
    p.add_argument("--no_flash_attention", dest="use_flash_attention", action="store_false")
    p.add_argument("--visual_token_budget", type=int, default=560,
                   help="Gemma 4 per-image token budget (supported: 70/140/280/560/1120)")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=2.0e-5)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--max_samples", type=int, default=None, help="for a smoke run")
    args = p.parse_args()

    # visual_token_budget: confirm the exact processor kwarg on the A40 smoke run; if the
    # kwarg name differs, fix it here only (the dataset/collator are budget-agnostic).
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path, visual_token_budget=args.visual_token_budget
    )
    model = create_model(
        args.model_name_or_path,
        use_flash_attention=args.use_flash_attention,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    train_dataset = VrdUqaGemmaDataset(processor, args.data_json, max_samples=args.max_samples)
    print(f"Training on {len(train_dataset)} examples")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,  # keep pixel_values etc. for the model forward
        dataloader_num_workers=1,     # custom processor not picklable (Phi-4 lesson)
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    pad_token_id = processor.tokenizer.pad_token_id or 0
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=build_collate_fn(pad_token_id),
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model()           # saves the PEFT LoRA adapter only (~500MB)
    processor.save_pretrained(args.output_dir)
    print(f"Saved Gemma 4 LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the file still parses (syntax check)**

Run: `uv run python -m py_compile finetuning/gemma4/finetune_gemma4.py && echo PARSE_OK`
Expected: `PARSE_OK` (no syntax errors).

- [ ] **Step 3: Verify the module still imports without the GPU deps (lazy-import guard)**

Run: `uv run python -c "import finetuning.gemma4.finetune_gemma4 as m; print('IMPORT_OK', hasattr(m, 'create_model'), hasattr(m, 'main'))"`
Expected: `IMPORT_OK True True` — importing must NOT trigger `AutoModelForMultimodalLM`/`peft` (they are inside `create_model`). If this errors with a transformers/peft import failure, a heavy import leaked to module scope — move it back inside `create_model`.

- [ ] **Step 4: Re-run the Task 1 test (still green after the append)**

Run: `uv run python -m tests.test_gemma4_finetune`
Expected: `OK: gemma4 finetune data path`

- [ ] **Step 5: Commit**

```bash
git add finetuning/gemma4/finetune_gemma4.py
git commit -m "feat(gemma4): model loader + PEFT LoRA + Trainer main (lazy GPU imports)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: SLURM runner (per-job venv + scratch + adapter copy-back)

**Files:**
- Create: `scripts/slurm/run_finetune_gemma4.sh`

- [ ] **Step 1: Create the runner**

Create `scripts/slurm/run_finetune_gemma4.sh`:

```bash
#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-16:00:00
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-finetune-gemma4-%j.out

# Gemma 4 12B (it) LoRA fine-tuning via HF Trainer + PEFT (standalone script) —
# LLaMA-Factory has no Gemma 4 vision template, so this trains a PEFT LoRA adapter with
# finetuning/gemma4/finetune_gemma4.py.
#
# Usage:
#   sbatch run_finetune_gemma4.sh [smoke]
#
# Gemma 4 needs a NEWER transformers than the rest of the repo; this builds a dedicated
# per-job venv (pinned below) isolated from the main venv (4.57.6) and the Phi-4 path
# (4.47.0). Output is a PEFT LoRA adapter (~500MB); eval loads base + adapter.
# Gemma 4 is GATED -> needs HF_TOKEN (verify the HF account has Gemma 4 access).

set -euo pipefail
module purge
module load miniconda3/3.13.25
module load nvhpc/25.1
source "$HOME/VRD-UQA/scripts/env.sh"
if [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN="$(cat "$HOME/.hf_token")"
fi
export UV_LINK_MODE=copy

VARIANT="${1:-sft}"   # sft (full) | smoke
WORK_DIR="$SCRATCH_FLASH/finetune_gemma4_${SLURM_JOB_ID}"
VENV_DIR="$WORK_DIR/.venv"
export HF_HOME="$WORK_DIR/hf_home"
mkdir -p "$WORK_DIR"
trap 'cd "$HOME"; rm -rf "$WORK_DIR"' EXIT

# ---- Sync repo (script + dataset); skip the big eval-artifact trees ----
rsync -aq --exclude='.git' --exclude='.venv' --exclude='data' \
      --exclude='corruption-scripts/results' \
      --exclude='artifacts/evaluation_runs' --exclude='artifacts/evaluation_archive' \
      "$HOME/VRD-UQA/" "$WORK_DIR/VRD-UQA/"

# ---- Per-job venv with the versions Gemma 4 needs ----
uv venv --python 3.11 "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
uv pip install --upgrade pip
uv pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1+cu121 torchvision==0.19.1+cu121
uv pip install \
    "transformers>=5.5.2" "peft>=0.19.0" accelerate bitsandbytes \
    Pillow sentencepiece protobuf
# flash-attn is optional (perf only); training falls back to sdpa+bf16 if it is absent.
uv pip install flash-attn --no-build-isolation || echo "flash-attn unavailable; using sdpa"

# ---- Train ----
DATA_JSON="$WORK_DIR/VRD-UQA/artifacts/finetuning/dataset/train.json"
SCRATCH_OUT="$WORK_DIR/out"
SCRIPT="$WORK_DIR/VRD-UQA/finetuning/gemma4/finetune_gemma4.py"

FLASH_FLAG="--use_flash_attention"
python -c "import flash_attn" 2>/dev/null || FLASH_FLAG="--no_flash_attention"

EXTRA=""
if [ "$VARIANT" = "smoke" ]; then
    EXTRA="--max_samples 50"
fi

echo "Gemma 4 LoRA fine-tune | variant=$VARIANT | flash=$FLASH_FLAG | data=$DATA_JSON"
accelerate launch --num_processes 1 "$SCRIPT" \
    --model_name_or_path google/gemma-4-12B-it \
    --data_json "$DATA_JSON" \
    --output_dir "$SCRATCH_OUT" \
    --visual_token_budget 560 \
    --num_train_epochs 1 \
    --learning_rate 2.0e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    $FLASH_FLAG \
    $EXTRA

# ---- Copy the LoRA adapter back (~500MB) ----
DEST="$HOME/VRD-UQA/artifacts/finetuning/gemma4_12b_lora_${VARIANT}"
if [ -d "$SCRATCH_OUT" ]; then
    mkdir -p "$DEST"
    rsync -a "$SCRATCH_OUT/" "$DEST/"
    echo "Copied Gemma 4 LoRA adapter -> $DEST"
    echo "NOTE: add a gemma4_finetuned eval entry (model_name=google/gemma-4-12B-it,"
    echo "      adapter_path=$DEST) once a Gemma 4 evaluator exists; eval needs the same"
    echo "      pinned transformers (>=5.5.2)."
else
    echo "WARNING: $SCRATCH_OUT not found; nothing copied back." >&2
fi

# ---- Per-job slurm log move (scoped to THIS job id) ----
mv "$HOME"/slurm-finetune-gemma4-${SLURM_JOB_ID}.out "$HOME/VRD-UQA/" 2>/dev/null || true
```

- [ ] **Step 2: Verify the script is syntactically valid**

Run: `bash -n scripts/slurm/run_finetune_gemma4.sh && echo BASH_OK`
Expected: `BASH_OK` (no syntax errors).

- [ ] **Step 3: Commit**

```bash
git add scripts/slurm/run_finetune_gemma4.sh
git commit -m "feat(gemma4): SLURM runner (per-job venv, adapter copy-back)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Add the Gemma 4 launch line to the readme

**Files:**
- Modify: `finetuning/readme.txt`

- [ ] **Step 1: Add the launch line**

In `finetuning/readme.txt`, immediately after the Phi-4 block (the two lines starting with "Phi-4-multimodal via Microsoft's official path..."), insert:

```
Gemma 4 12B via HF Trainer+PEFT LoRA (LLaMA-Factory has no Gemma 4 vision template):
    sbatch scripts/slurm/run_finetune_gemma4.sh smoke   # then drop 'smoke' for the full run
```

- [ ] **Step 2: Verify the line is present**

Run: `grep -n "run_finetune_gemma4.sh" finetuning/readme.txt`
Expected: one match showing the new launch line.

- [ ] **Step 3: Commit**

```bash
git add finetuning/readme.txt
git commit -m "docs(gemma4): add launch line to finetuning readme

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: HPC smoke run → full run (user-executed on SLURM)

Local validation cannot exercise real training (needs the A40 + the new venv). The user runs these on SLURM; this task is the integration checkpoint.

- [ ] **Step 1: Confirm HF gating access** — verify the HF account tied to `$HOME/.hf_token` has accepted the Gemma 4 license at `https://huggingface.co/google/gemma-4-12B-it` (Gemma is gated; a 403 here is the same failure class that blocked Llama). Never print/commit the token.

- [ ] **Step 2: Submit the smoke run** (50 samples):

Run: `sbatch scripts/slurm/run_finetune_gemma4.sh smoke`
Then watch: `squeue -u $USER` and `tail -f slurm-finetune-gemma4-<JOBID>.out`

- [ ] **Step 3: Verify the smoke run** — in the log, confirm in order:
  1. venv builds and deps install (note whether `flash-attn` built or fell back to sdpa);
  2. model + processor load (no gated-403, no missing-class error for `AutoModelForMultimodalLM`);
  3. `print_trainable_parameters` reports a small trainable % (LoRA attached);
  4. `Training on 50 examples`;
  5. several optimizer steps complete with **no OOM and no image-embed shape mismatch** (the first training step is the real test of the `visual_token_budget` kwarg + the pass-through collator);
  6. adapter copied to `artifacts/finetuning/gemma4_12b_lora_smoke/`.

  **If the first step errors on the processor kwarg** (`visual_token_budget` not accepted): inspect the processor signature on the node and set the correct kwarg in `main()`. **If it OOMs:** switch `create_model` to QLoRA — add `quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)` to `from_pretrained` (`bitsandbytes` is already installed) — or drop `--visual_token_budget` to 280.

- [ ] **Step 4: Submit the full run** (after the smoke run is clean):

Run: `sbatch scripts/slurm/run_finetune_gemma4.sh`
Output adapter lands at `artifacts/finetuning/gemma4_12b_lora_sft/`.

---

## Deferred follow-ups (NOT part of this plan's execution)

These require the trained adapter to exist and a separate inference-side design (the spec's Component 3 explicitly defers the evaluator). Do **not** add a `gemma4_finetuned` config entry before the evaluator exists — `run_vqa_analysis.sh` has no `gemma4)` case and there is no `gemma4_evaluator.py`, so an entry would be inert/misleading and unrunnable.

1. **`VQA_analysis/evaluators/gemma4_evaluator.py`** — subclass `BaseVQAEvaluator`, load `AutoModelForMultimodalLM` + `AutoProcessor` + the LoRA adapter (mirror the rewritten `internvl_evaluator.py`).
2. **`run_vqa_analysis.sh`** — add a `gemma4) ENTRY="VQA_analysis/evaluators/gemma4_evaluator.py" ;;` case.
3. **Config entry** — add `gemma4_finetuned` to `VQA_analysis/config_zeroshot.json` (+ fewshot/mock) with `model_name=google/gemma-4-12B-it`, `adapter_path=artifacts/finetuning/gemma4_12b_lora_sft`.
4. **Pinned eval env** — Gemma 4 eval needs `transformers>=5.5.2`, separate from the main 4.57.6 eval pipeline (same situation as Phi-4's 4.47.0 env).

---

## Self-Review

**Spec coverage:**
- Component 1 (`finetune_gemma4.py`): Tasks 1–2 — `AutoModelForMultimodalLM` (corrected class), PEFT LoRA r=16/α=32, dataset reuse, `<image>`-strip + instruction+question text, answer-only label masking, bf16, `visual_token_budget=560`, hyperparameters matching `qwen25vl_lora_sft.yaml` (lr 2e-5, cosine, warmup 50, 1 epoch, bs 1, grad_accum 8). ✓
- Component 2 (`run_finetune_gemma4.sh`): Task 3 — SBATCH header from the Phi-4 runner, HF token load, per-job scratch venv + `HF_HOME`, rsync excludes, `transformers>=5.5.2`/`peft>=0.19.0`, `accelerate launch`, smoke arg, adapter copy-back, scoped log move. ✓
- `finetuning/readme.txt`: Task 4. ✓
- Component 3 (eval entry/evaluator): consciously deferred to the follow-ups section with rationale (no evaluator + no dispatch case yet). Documented, not silently dropped. ✓

**Placeholder scan:** No TBD/TODO/"add error handling"/"similar to Task N". All code blocks are complete; each code step shows the code.

**Type/name consistency:** `VrdUqaGemmaDataset`, `build_collate_fn`, `pad_sequence`, `cat_with_pad`, `create_model`, `main`, `_IGNORE_INDEX`, `_TEXT_KEYS`, `_MAX_TRAINING_LENGTH` are defined in Task 1/2 and referenced consistently in the test (Task 1) and the runner CLI flags (Task 3: `--visual_token_budget`, `--use_flash_attention`/`--no_flash_attention`, `--model_name_or_path`, `--data_json`, `--output_dir`, matching `main()`'s argparse). ✓

**Deviations flagged:** plain `Trainer` instead of `SFTTrainer` (+ `trl` dropped from deps) — documented in Design Notes for the user to accept or override.
