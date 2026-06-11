# Phi-4-multimodal fine-tuning — official-sample fallback (ACTIVE)

LLaMA-Factory **cannot** fine-tune Phi-4-multimodal's vision path: its `phi4` template
is text-only and raises `ValueError: This model does not support image input` during
dataset preprocessing (confirmed on job 1757561). So Phi-4-mm uses this dedicated path,
adapted from Microsoft's official `sample_finetune_vision.py`.

## What this does
- `finetune_phi4mm.py` — training script adapted from the MS sample. It trains the model's
  **built-in `vision` LoRA** + the image projector (audio/speech path is stripped), on our
  alpaca-style `artifacts/finetuning/dataset/train.json`. Output is a **full fine-tuned
  model** (not a separate PEFT adapter).
- `sample_finetune_vision_reference.py` — Microsoft's original, kept for reference.
- `scripts/slurm/run_finetune_phi4mm_official.sh` — SLURM runner with a dedicated venv.

## Why a dedicated venv
Phi-4-mm's remote code is written for an **older transformers**; the runner pins:
```
transformers==4.47.0  peft==0.13.2  accelerate==1.3.0  scipy==1.15.1  backoff==2.2.1
```
This is isolated from the main repo venv (transformers 4.57.6, used by Qwen/InternVL).

## Run
```bash
sbatch scripts/slurm/run_finetune_phi4mm_official.sh smoke   # 50 samples, fast validation
sbatch scripts/slurm/run_finetune_phi4mm_official.sh         # full run
```
The trained model lands in `artifacts/finetuning/phi4mm_vision_sft/` (≈10–20 GB, full model).

## Eval wiring (do after training)
The output is a full model, not a PEFT adapter, so update the `phi4_finetuned` eval entry:
- set `model_name` = `artifacts/finetuning/phi4mm_vision_sft` (the dir above)
- remove `adapter_path`

**Caveat:** evaluating this fine-tuned Phi-4-mm model needs the **same pinned transformers
(4.47.0)** as training — the main eval pipeline runs 4.57.6. So Phi-4-mm (zero-shot *and*
fine-tuned) likely needs its own eval environment; to be resolved when wiring Phi-4 into the
eval runs.

## Hyperparameters
This path tunes the built-in vision LoRA (architecturally different from a fresh rank-16
LoRA), so it follows the MS sample's recipe (lr 4e-5, linear schedule, warmup 50, 1 epoch,
effective batch 8) rather than the verbatim Qwen YAML params — noted for the thesis writeup.

## Decision log
- [x] LLaMA-Factory `phi4` template result: **fail** ("does not support image input", job 1757561)
- [x] Fallback used: **yes** (this path)
- [ ] Smoke run passed: __pending HPC__
- [ ] Full run + eval wiring done: __pending HPC__
