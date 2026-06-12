# Gemma 4 12B LoRA Fine-Tuning — Design

**Date:** 2026-06-11 (model id + architecture verified against primary sources 2026-06-12)
**Status:** Approved (design + spec review). Repo id and architecture confirmed; two
corrections applied 2026-06-12 (model class `AutoModelForMultimodalLM`; visual-token budget
replaces the borrowed Phi-4 `dynamic_hd`/tiling reasoning). Next: implementation plan.
**Scope:** Add Gemma 4 12B (multimodal, instruction-tuned) as a new fine-tuned core test
model in VRD-UQA, trained via HF TRL + PEFT LoRA in a dedicated per-job venv.

## Goal

Add **`google/gemma-4-12B-it`** (multimodal, instruction-tuned) as a fine-tuned core test
model alongside Qwen2.5-VL, InternVL3.5-8B, and Phi-4-multimodal. Training uses **HF TRL
`SFTTrainer` + PEFT LoRA** in an **isolated per-job venv** with a newer `transformers` than
the rest of the repo can run, reusing the existing model-agnostic SFT dataset.

## Why this approach (confirmed decisions)

- **Model:** Gemma 4 12B multimodal, instruction-tuned (`google/gemma-4-12B-it`, released
  2026-06-03, **Apache-2.0** — not EU-restricted, unlike the dropped Meta Llama vision
  models). Repo id **verified to exist** (HF model card + Google developer guide, 2026-06-12).
  It is gated → needs `HF_TOKEN`.
- **Architecture (verified):** Gemma 4 12B is a **dense, encoder-free *unified* multimodal**
  model — a single decoder-only transformer. A tiny 35M-param "vision embedder" projects raw
  48×48 image patches straight into the LLM hidden dim via one matmul; there is **no separate
  vision tower and no dynamic tiling**. Images use a **configurable visual-token budget**
  (supported: 70 / 140 / 280 / 560 / 1120 tokens per image). This is materially simpler than
  Phi-4: no crop-count surgery, and the per-record sequence length is bounded (one image/
  record × ≤1120 tokens), so the Phi-4 image-token-explosion / OOM lesson does **not** apply.
- **Framework: HF TRL `SFTTrainer` + PEFT LoRA — NOT LLaMA-Factory.** LLaMA-Factory has no
  Gemma 4 vision template yet (only PaliGemma2 / Qwen / InternVL). This mirrors the Phi-4
  decision (Phi-4 also left LLaMA-Factory, for a different reason).
- **Dependency isolation (critical):** Gemma 4 needs **`transformers>=5.5.2`** (KV-sharing
  fix, Apr 2026), **`peft>=0.19.0`** (ships default Gemma 4 LoRA `target_modules`),
  **`trl>=0.13`**. These conflict with both the main repo venv (transformers ~4.57.6, for
  Qwen/InternVL) and the Phi-4 path (pinned `transformers==4.47.0`). So Gemma 4 gets its
  **own per-job scratch venv**, built fresh each job — the same isolation pattern already
  used by `run_finetune_vlm.sh` and `run_finetune_phi4mm_official.sh`.
- **Output:** a **PEFT LoRA adapter** (~500MB) copied back to
  `artifacts/finetuning/gemma4_12b_lora_sft/` — unlike the Phi-4-MM path which copies a full
  model. Eval loads base + adapter.

## Non-goals

- No changes to LLaMA-Factory configs, `run_finetune_vlm.sh`, or the main repo `.venv`.
- No new dataset: reuse `artifacts/finetuning/dataset/train.json` (built by
  `finetuning/build_paired_dataset.py`, model-agnostic) as-is.
- No quantization by default (LoRA + bf16); QLoRA is a documented OOM fallback only.

---

## Files to create / modify

| File | Action | Responsibility |
|---|---|---|
| `finetuning/gemma4/finetune_gemma4.py` | **New** | TRL+PEFT LoRA training script for Gemma 4. |
| `scripts/slurm/run_finetune_gemma4.sh` | **New** | SLURM job: per-job venv + scratch + adapter copy-back. |
| `finetuning/readme.txt` | **Modify** | Add one launch line for Gemma 4. |
| VQA eval config(s) | **Modify (post-training)** | Add a `gemma4_finetuned` entry (base + adapter_path). |

`finetuning/build_paired_dataset.py` and `train.json` are **reused unchanged**.

---

## Component 1 — `finetuning/gemma4/finetune_gemma4.py`

Structurally mirrors `finetuning/phi4mm_official/finetune_phi4mm.py` (dataset class, label
masking, collator, `TrainingArguments`), but Gemma-4-specific:

**Model load** (class corrected 2026-06-12 — the HF model card uses `AutoModelForMultimodalLM`,
**not** `AutoModelForImageTextToText`):
```python
AutoModelForMultimodalLM.from_pretrained(
    "google/gemma-4-12B-it",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

**LoRA via PEFT (our own adapter, not a built-in one):**
```python
LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32,
           target_modules=None,  # PEFT 0.19+ supplies Gemma 4 defaults
           bias="none")
model = get_peft_model(model, lora_config)
```
Save the **adapter** (`model.save_pretrained(output_dir)`); eval loads base + adapter.

**Dataset (reuse the alpaca JSON):** each record is
`{instruction, input="<image>\n{question}", output, images=[path]}`. Per example:
- strip the `<image>` placeholder from `input` to recover the question text;
- build a Gemma 4 chat message:
  `[{"role":"user","content":[{"type":"image"},{"type":"text","text":question}]}]`
  (the `instruction` guideline prepended to the text, matching how the other evaluators
  combine instruction + question);
- run through `processor.apply_chat_template(...)` + `AutoProcessor.__call__(images=[img], text=...)`,
  with the processor's **visual-token budget set to 560** (balanced default for dense VRD
  text/tables/charts; honors the project's capped-input constraint and fits the A40 easily).
  Exact processor knob name (e.g. `visual_token_budget` / a processor kwarg) is an
  implementation detail to confirm from the model's `AutoProcessor` API;
- mask labels so only the **response** tokens are supervised (same masking approach as the
  Phi-4-MM script: build prompt then answer ids, set labels = IGNORE except the answer span).

**Hyperparameters** (match `finetuning/qwen25vl_lora_sft.yaml` where sensible):
`learning_rate=2e-5`, `lr_scheduler_type="cosine"`, `warmup_steps=50`,
`num_train_epochs=1`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`,
`bf16=True`, `gradient_checkpointing=True`. CLI: `--model_name_or_path`, `--data_json`,
`--output_dir`, `--max_samples` (smoke), `--use_flash_attention` (default on),
`--visual_token_budget` (default **560**).

---

## Component 2 — `scripts/slurm/run_finetune_gemma4.sh`

Copy the structure of `run_finetune_phi4mm_official.sh`:

- **SBATCH header** from the Phi-4 runner: 1×A40, 64G, 16h, `--partition gpu_a40`,
  `--gres gpu:1`, `--output slurm-finetune-gemma4-%j.out`.
- `module load` + `source scripts/env.sh`; **load `$HOME/.hf_token` → `HF_TOKEN`** (gated).
- **Per-job scratch:** `WORK_DIR=$SCRATCH_FLASH/finetune_gemma4_${SLURM_JOB_ID}`, `.venv`
  inside it, `export HF_HOME="$WORK_DIR/hf_home"`, `trap 'rm -rf "$WORK_DIR"' EXIT`.
- **rsync repo** into scratch (exclude `.git/.venv/data/corruption-scripts/results` and the
  big `artifacts/evaluation_runs` + `artifacts/evaluation_archive` trees, like the Phi-4 runner).
- **Build venv** (`uv venv --python 3.11`):
  - `torch==2.4.1+cu121 torchvision==0.19.1+cu121` (cu121 index-url),
  - then `transformers>=5.5.2 peft>=0.19.0 trl>=0.13 accelerate bitsandbytes Pillow sentencepiece`.
- **Train:** `accelerate launch --num_processes 1 finetune_gemma4.py --model_name_or_path
  google/gemma-4-12B-it --data_json <scratch>/artifacts/finetuning/dataset/train.json
  --output_dir <scratch>/out [--max_samples 50 if smoke]`.
- Optional `smoke` positional arg → `--max_samples 50`.
- **Copy adapter back:** `rsync` `<scratch>/out` → `artifacts/finetuning/gemma4_12b_lora_sft/`
  (~500MB; LoRA adapter only).
- Per-job slurm-log move scoped to `${SLURM_JOB_ID}`.

---

## Component 3 — eval wiring (post-training)

After the adapter exists, add a `gemma4_finetuned` entry to the VQA eval config JSON(s),
following the existing `*_finetuned` pattern: `model_name` = `google/gemma-4-12B-it`,
`adapter_path` = `artifacts/finetuning/gemma4_12b_lora_sft`. (A Gemma 4 **evaluator** for
the inference side is a separate follow-up; this design covers fine-tuning + the config
entry. The evaluator would load `AutoModelForImageTextToText` + `AutoProcessor` + chat
template, like the rewritten InternVL evaluator — and would also need the Gemma-4
`transformers` version, so Gemma-4 eval likely needs its own env, same as Phi-4.)

---

## Open items to verify during implementation (not design blockers)

- **Repo id:** ✅ **CONFIRMED** — `google/gemma-4-12B-it` exists (HF model card + Google
  developer guide, 2026-06-03 release). Instruction-tuned, dense, multimodal, Apache-2.0.
- **Model class:** ✅ **CONFIRMED** — `AutoModelForMultimodalLM` + `AutoProcessor` (per the
  model card), corrected in Component 1.
- **Visual-token budget:** default **560** (decided). Confirm the exact processor knob/kwarg
  name and that 560 is a supported value (supported set: 70/140/280/560/1120). Because Gemma 4
  is encoder-free with a bounded per-image budget, there is **no `dynamic_hd`/tiling knob and no
  image-token explosion** — the Phi-4 crop-capping lesson does not transfer.
- **Memory:** Gemma 4 12B + LoRA + bf16 + gradient checkpointing on 1×A40 48GB is expected to
  fit comfortably (one image/record × ≤560 tokens; far below Phi-4's ~9345). If it still OOMs,
  fall back to **QLoRA** (`BitsAndBytesConfig` 4-bit) — `bitsandbytes` is in the dep list.
- **`target_modules=None`:** confirm PEFT 0.19+ supplies Gemma 4 defaults; if not, set them
  explicitly. NB encoder-free → targets are the **decoder** attn/MLP projections only (no
  vision-tower modules to target).
- **flash-attention-2** availability in the venv for the pinned torch 2.4.1+cu121; if the
  build is problematic, fall back to `attn_implementation="sdpa"` + bf16 (the same lesson
  learned on Phi-4: bf16 is what matters for fitting; flash is a perf bonus).

## Dependencies & isolation summary

| Path | transformers | Framework | Output |
|---|---|---|---|
| Qwen2.5 / InternVL3.5 | ~4.57.6 (main venv) | LLaMA-Factory | LoRA adapter |
| Phi-4-MM | ==4.47.0 (per-job venv) | MS official script | full model |
| **Gemma 4 12B** | **>=5.5.2 (per-job venv)** | **TRL + PEFT** | **LoRA adapter** |

Each fine-tuning path is environment-isolated; only the model-agnostic dataset is shared.
