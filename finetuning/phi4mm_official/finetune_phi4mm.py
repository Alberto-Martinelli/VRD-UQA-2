"""Fine-tune Phi-4-multimodal-instruct on the VRD-UQA SFT data (vision path).

Adapted from Microsoft's official `sample_finetune_vision.py`
(microsoft/Phi-4-multimodal-instruct) — LLaMA-Factory cannot fine-tune Phi-4-mm's
vision path (its `phi4` template is text-only), so this is the B.4 fallback.

Differences from the sample:
- Reads our alpaca-style train.json (instruction / input='<image>\\n{question}' / output /
  images=[path]) instead of PMC-VQA; images are local paths (no zip/csv download).
- Drops the PMC eval loop (evaluation is done by the VQA pipeline, separately).
- Keeps the sample's model surgery (delete audio/speech), built-in `vision` LoRA tuning,
  collator, and label masking verbatim.

Run via scripts/slurm/run_finetune_phi4mm_official.sh, which builds a venv with the
versions Phi-4-mm's remote code needs:
    transformers==4.47.0  peft==0.13.2  accelerate==1.3.0  scipy==1.15.1  backoff==2.2.1
The output_dir is a FULL fine-tuned model (not a PEFT adapter); eval loads it via model_name.
"""
import argparse
import json
from pathlib import Path

import torch
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BatchFeature,
    Trainer,
    TrainingArguments,
)

_IGNORE_INDEX = -100
_MAX_TRAINING_LENGTH = 8192


class VrdUqaDataset(Dataset):
    """Our SFT records -> Phi-4-mm training tensors (mirrors the sample's __getitem__)."""

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
        # input is '<image>\n{question}' (LLaMA-Factory placeholder); strip it — Phi-4 uses <|image_1|>
        question = (rec.get("input", "") or "").replace("<image>", "").strip()
        text = (instruction + "\n" + question).strip()
        image = Image.open(rec["images"][0]).convert("RGB")

        user_message = {"role": "user", "content": "<|image_1|>" + text}
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        answer = f'{rec["output"]}<|end|><|endoftext|>'
        inputs = self.processor(prompt, images=[image], return_tensors="pt")
        answer_ids = self.processor.tokenizer(answer, return_tensors="pt").input_ids

        input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
        labels = torch.full_like(input_ids, _IGNORE_INDEX)
        labels[:, -answer_ids.shape[1]:] = answer_ids
        if input_ids.size(1) > _MAX_TRAINING_LENGTH:
            input_ids = input_ids[:, :_MAX_TRAINING_LENGTH]
            labels = labels[:, :_MAX_TRAINING_LENGTH]
            if torch.all(labels == _IGNORE_INDEX).item():
                labels[:, -1] = self.processor.tokenizer.eos_token_id

        return {
            "input_ids": input_ids,
            "labels": labels,
            "input_image_embeds": inputs.input_image_embeds,
            "image_attention_mask": inputs.image_attention_mask,
            "image_sizes": inputs.image_sizes,
        }


def pad_sequence(sequences, padding_side="right", padding_value=0):
    assert padding_side in ("right", "left")
    trailing_dims = sequences[0].size()[1:]
    max_len = max(len(seq) for seq in sequences)
    output = sequences[0].new_full((len(sequences), max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        output.data[i, :length] = seq if padding_side == "right" else output.data[i, -length:]
        if padding_side == "left":
            output.data[i, -length:] = seq
    return output


def cat_with_pad(tensors, dim, padding_value=0):
    ndim = tensors[0].dim()
    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)
    index = 0
    for t in tensors:
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        slices[dim] = slice(index, index + t.shape[dim])
        output[slices] = t
        index += t.shape[dim]
    return output


def collate_fn(batch):
    input_ids = pad_sequence([b["input_ids"][0] for b in batch], "right", 0)
    labels = pad_sequence([b["labels"][0] for b in batch], "right", 0)
    return BatchFeature({
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": (input_ids != 0).long(),
        "input_image_embeds": cat_with_pad([b["input_image_embeds"] for b in batch], dim=0),
        "image_attention_mask": cat_with_pad([b["image_attention_mask"] for b in batch], dim=0),
        "image_sizes": torch.cat([b["image_sizes"] for b in batch]),
        "input_mode": 1,  # vision mode
    })


def create_model(model_name_or_path, use_flash_attention=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,  # bf16 even with sdpa: fp32 (~22GB weights) OOMs on the A40
        _attn_implementation="flash_attention_2" if use_flash_attention else "sdpa",
        trust_remote_code=True,
    ).to("cuda")
    # Strip the speech/audio path — we only fine-tune vision (verbatim from the sample).
    del model.model.embed_tokens_extend.audio_embed
    for layer in model.model.layers:
        del layer.mlp.down_proj.lora_A.speech
        del layer.mlp.down_proj.lora_B.speech
        del layer.mlp.gate_up_proj.lora_A.speech
        del layer.mlp.gate_up_proj.lora_B.speech
        del layer.self_attn.o_proj.lora_A.speech
        del layer.self_attn.o_proj.lora_B.speech
        del layer.self_attn.qkv_proj.lora_A.speech
        del layer.self_attn.qkv_proj.lora_B.speech
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="microsoft/Phi-4-multimodal-instruct")
    p.add_argument("--data_json", required=True, help="VRD-UQA SFT train.json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--use_flash_attention", action="store_true")
    p.add_argument("--dynamic_hd", type=int, default=36, help="max image crops")
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=4.0e-5)  # sample default for vision-LoRA
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--batch_size_per_gpu", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=None, help="for a smoke run")
    args = p.parse_args()

    accelerator = Accelerator()
    with accelerator.local_main_process_first():
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path, trust_remote_code=True, dynamic_hd=args.dynamic_hd
        )
        model = create_model(args.model_name_or_path, use_flash_attention=args.use_flash_attention)

    # Tune only the built-in vision LoRA + the image projector.
    model.set_lora_adapter("vision")
    for param in model.model.embed_tokens_extend.image_embed.parameters():
        param.requires_grad = True

    num_gpus = accelerator.num_processes
    assert args.batch_size % (num_gpus * args.batch_size_per_gpu) == 0, \
        "batch_size must be divisible by num_gpus * batch_size_per_gpu"
    grad_accum = args.batch_size // (num_gpus * args.batch_size_per_gpu)
    bf16 = True  # bf16 regardless of flash-attn (sdpa path); fp32 OOMs on the A40

    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size_per_gpu,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_accumulation_steps=grad_accum,
        optim="adamw_torch",
        adam_beta1=0.9, adam_beta2=0.95, adam_epsilon=1e-7,
        learning_rate=args.learning_rate,
        weight_decay=args.wd,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        warmup_steps=50,
        logging_steps=10,
        output_dir=args.output_dir,
        save_strategy="no",
        save_only_model=True,
        bf16=bf16,
        fp16=not bf16,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=4,
        ddp_find_unused_parameters=True,  # unused SigLIP layers
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dataset = VrdUqaDataset(processor, args.data_json, max_samples=args.max_samples)
    print(f"Training on {len(train_dataset)} examples; grad_accum={grad_accum}")

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model()  # full fine-tuned model (load at eval via model_name)
    processor.save_pretrained(args.output_dir)
    accelerator.wait_for_everyone()
    print(f"Saved fine-tuned Phi-4-mm model to {args.output_dir}")


if __name__ == "__main__":
    main()
