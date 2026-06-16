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
        # PEFT 0.19 has no target-module mapping for Gemma 4's multimodal arch, so it
        # cannot auto-infer from None. "all-linear" discovers every nn.Linear — same as
        # the repo's working qwen/internvl LoRA configs (lora_target: all).
        target_modules="all-linear",
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
