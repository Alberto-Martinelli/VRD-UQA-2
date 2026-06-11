# Multi-model VQA Evaluators + Fine-tuning Infrastructure — Design

**Date:** 2026-06-11
**Status:** Approved (design); pending implementation plan
**Scope:** Add three new VLLM evaluators and LoRA fine-tuning infrastructure for them, extending the existing VRD-UQA framework.

## Goal

Extend the VRD-UQA evaluation + fine-tuning pipeline (currently Qwen2.5-VL only) to cover **three additional core test models**, each evaluated under zero-shot, few-shot, and fine-tuned (LoRA) conditions, with and without OCR injection — matching the paradigm already used for Qwen2.5-VL.

### Target models (checkpoints confirmed against the installed environment)

| Role | Model | HF checkpoint | Load path (transformers 4.57.6) |
|---|---|---|---|
| New test model 1 | Llama-3.2-11B-Vision-Instruct | `meta-llama/Llama-3.2-11B-Vision-Instruct` | `MllamaForConditionalGeneration` + `AutoProcessor` (native) |
| New test model 2 | Phi-4-multimodal-instruct | `microsoft/Phi-4-multimodal-instruct` | `AutoModelForCausalLM` + `AutoProcessor`, `trust_remote_code` |
| New test model 3 | InternVL3.5-8B | `OpenGVLab/InternVL3_5-8B` | `AutoModel` + `AutoTokenizer`, `trust_remote_code, use_fast=False`, `model.chat()` |

The InternVL load path is the one already proven by the answerability judge
([corruption-scripts/verification/answerability_verifier.py:17](../../../corruption-scripts/verification/answerability_verifier.py#L17)).

## Non-goals

- No retraining of Qwen2.5-VL (its adapter already exists; it is the template).
- No new datasets, corruption types, or metrics. Output schema is unchanged so the
  existing metrics steps (1–3) consume new models with no changes beyond leaf discovery.
- No quantization by default (all three models fit on the A40 48GB in bf16, run one at a
  time). QLoRA / bitsandbytes is a documented fallback only.

## Documented caveats (carry into the thesis write-up)

1. **InternVL3.5-8B is the exact same checkpoint as the answerability judge.** The model
   that filtered which corrupted questions count as "unanswerable" is also being graded on
   them. This inflates its QUR via self-consistency. Frame InternVL3.5-8B results as an
   **upper-bound / reference point**, not a peer comparison, and state this explicitly.
2. **Llama-3.2-Vision is effectively single-image per prompt** (cross-attention design).
   Its window size is pinned to `batch_size: 1`; multi-image few-shot is best-effort and
   flagged at runtime.
3. **Phi-4-multimodal requires `trust_remote_code`** and is sensitive to the transformers
   version; its generation config is verified at implementation time.

---

## Phase A — Evaluators

### A.1 Architecture: base class + thin subclasses

Today there is a single ~690-line `QwenVQAEvaluator`
([VQA_analysis/evaluators/qwen2.5_evaluator.py](../../../VQA_analysis/evaluators/qwen2.5_evaluator.py)).
~90% of it is model-agnostic. Extract a base class and reduce each model to two methods.

**New file: `VQA_analysis/evaluators/base_evaluator.py`** — `BaseVQAEvaluator` owns all
model-agnostic logic, unchanged in behavior from today's Qwen evaluator:

- `__init__` (config load, seed, model-config resolution, `images_base_path` extraction)
- `_set_seed`, `_create_prompt`
- `get_sorted_ocr_text`, `get_ocr_text`
- `generate_answer` (the sliding-window batching loop) — now delegates per-window
  inference to `self._generate(...)`
- few-shot: `_get_fs_pool`, `_fs_specific_score`, `_select_few_shot_examples`,
  `_build_few_shot_turns` (now emits a **neutral** turn list, see A.3)
- `_resolve_leaf`, `_save_results` (manifest), resume check
- `QUESTION_SIDES`, `_process_single_question`, `evaluate`, `_cleanup_model`
- mock-mode handling (short-circuits **before** model load / inference)

**Subclass contract** — each model overrides class attributes + two methods:

```python
class BaseVQAEvaluator:
    MODEL_KEY: str             # config key under open_source_models, e.g. "llama3.2"
    FINETUNED_MODEL_KEY = None # set only where a fine-tuned variant exists
    MODEL_TYPE: str            # written to vqa_result["model_type"], e.g. "llama"
    MODEL_LEAF_PREFIX = ""     # "" for Qwen (back-compat); model key for new models

    def _load_model(self):
        """Populate self.model, self.processor or self.tokenizer, self.max_tokens."""
        raise NotImplementedError

    def _generate(self, window_image_paths, question_prompt, few_shot_turns) -> str:
        """Run inference for one window; return the decoded answer string."""
        raise NotImplementedError
```

If `--finetuned` is passed but `FINETUNED_MODEL_KEY is None`, raise a clear error
("model X has no fine-tuned variant configured").

**Files (all keep the identical `--config_path / --finetuned / --questions` CLI):**

- `base_evaluator.py` — new, `BaseVQAEvaluator`
- `qwen2.5_evaluator.py` — refactored: `QwenVQAEvaluator(BaseVQAEvaluator)`, CLI unchanged
- `llama_evaluator.py` — new, `LlamaVQAEvaluator`
- `phi4_evaluator.py` — new, `Phi4VQAEvaluator`
- `internvl_evaluator.py` — new, `InternVLVQAEvaluator`

### A.2 Per-model `_load_model` / `_generate`

| Model | `_load_model` | `_generate` |
|---|---|---|
| **Qwen2.5-VL** | port of current `initialize_model` (Qwen class + processor + optional flash-attn + optional PEFT adapter) | port of current per-window block: `{"type":"image","image":"file://…"}` messages → `process_vision_info` → generate → decode |
| **Llama-3.2-Vision** | `MllamaForConditionalGeneration.from_pretrained` + `AutoProcessor`; optional flash-attn; optional PEFT adapter | chat-template messages with `{"type":"image"}` content + PIL image list → `processor(images=…, text=…)` → generate → decode. `batch_size` pinned to 1. |
| **Phi-4-multimodal** | `AutoModelForCausalLM.from_pretrained(trust_remote_code=True)` + `AutoProcessor(trust_remote_code=True)`; optional flash-attn; optional adapter (see B.5) | build `<\|user\|>…<\|end\|><\|assistant\|>` prompt with numbered `<\|image_i\|>` placeholders across all turns → `processor(text=…, images=…)` → generate (model `generation_config`) → decode |
| **InternVL3.5-8B** | `AutoModel.from_pretrained(trust_remote_code=True)` + `AutoTokenizer(trust_remote_code=True, use_fast=False)`; optional adapter (see B.5) | manual dynamic tiling (`build_transform` + `dynamic_preprocess`) → `pixel_values` + `num_patches_list`; question carries `<image>` tags; `model.chat(tokenizer, pixel_values, question, gen_cfg, num_patches_list, history=…)`; few-shot via `history` |

OCR is folded into `question_prompt` by the base **before** `_generate` is called (exactly
as Qwen does today), so `_generate` never handles OCR directly. Mock mode returns
`"Mock answer"` from the base without loading any model — all four evaluators are testable
with no GPU and no downloads.

### A.3 Neutral few-shot turn format

`_build_few_shot_turns` (base) emits a model-neutral list; each subclass converts it into
its native message/prompt format inside `_generate`:

```python
[
  {"role": "user",      "image_paths": ["/abs/p1.png", ...], "text": "<prompt incl OCR>"},
  {"role": "assistant", "text": "<answer or 'Unable to determine'>"},
  ...
]
```

This keeps few-shot **selection + anchoring + OCR formatting** (the non-trivial,
corruption-matched logic) in the base, and leaves only format translation to subclasses.

### A.4 Artifact-layout change (multi-model collision fix)

`run_layout.leaf_dir` is currently `run/dataset/<slug>` with **no model dimension**, so
running >1 model in one run collides on the same leaf. Fix:

```python
def leaf_dir(run_id, dataset, slug, model_prefix=""):
    name = f"{model_prefix}__{slug}" if model_prefix else slug
    return run_dir(run_id) / dataset / name
```

- Qwen passes `MODEL_LEAF_PREFIX=""` → keeps its **bare** path. Existing artifacts, the
  mock test, and in-flight few-shot k-sweep resume are undisturbed.
- New models namespace as `llama__zeroshot_ocr`, `phi4__zeroshot_noocr`,
  `internvl__fewshot_ocr_k4_mixed_random`, etc.
- Leaves stay **one level** under the dataset dir, so
  [3_compute_metrics.py:953](../../../VQA_analysis/metrics/3_compute_metrics.py#L953)
  iteration is unchanged; it already groups by `model_name` from the manifest, and
  `summary.csv` already carries a `model` column.
- `_resolve_leaf` threads `self.MODEL_LEAF_PREFIX` into `leaf_dir`.

*(Rejected alternative: prefix all four models for full path consistency — changes Qwen
paths and restarts in-flight ksweep runs. Not worth the disruption.)*

### A.5 Config + SLURM wiring

- **Configs** ([config_zeroshot.json](../../../VQA_analysis/config_zeroshot.json),
  [config_fewshot.json](../../../VQA_analysis/config_fewshot.json),
  [config_mock.json](../../../VQA_analysis/config_mock.json)):
  - confirm `llama3.2` → `meta-llama/Llama-3.2-11B-Vision-Instruct`, pin `batch_size: 1`
  - confirm `phi4` → `microsoft/Phi-4-multimodal-instruct`
  - add `internvl3_5` → `OpenGVLab/InternVL3_5-8B` (`name: "InternVL3.5-8B"`,
    `batch_size: 1`, `max_tokens: 1024`, tiling params as needed)
- **`run_vqa_analysis.sh`**: drive the evaluators from a small `MODELS` list mapping a
  model key → entrypoint script. **Opt-in and conservatively defaulted** (do not auto-run
  all 4 models × 4 datasets within the 20h wall-time). Exact matrix finalized in the plan.

### A.6 Testing (Phase A)

Extend the mock-test pattern ([tests/test_evaluator_mock.py](../../../tests/test_evaluator_mock.py))
to parametrize over all four entrypoints. Assert per evaluator:

- dual-answer schema (`answer_corrupted` / `answer_clean` as `[{pages, answer}]`)
- `--questions corrupted` omits the clean side
- manifest `model_name` matches the config `name`
- new models write a **model-namespaced** leaf; Qwen still writes the bare leaf
- resume skips a completed leaf

Real-model `val_5` smoke runs (downloads + GPU) are run by the user on the HPC; the plan
provides the exact commands.

---

## Phase B — Fine-tuning infrastructure

### B.1 Approach

**LLaMA-Factory LoRA SFT for all three models**, extending the existing Qwen pattern
([finetuning/qwen25vl_lora_sft.yaml](../../../finetuning/qwen25vl_lora_sft.yaml),
[scripts/slurm/run_finetune_qwen25vl.sh](../../../scripts/slurm/run_finetune_qwen25vl.sh)).
The 50/50 balanced training dataset is **model-agnostic** (alpaca + images,
[finetuning/dataset_info.json](../../../finetuning/dataset_info.json)), so all models reuse
`vrd_uqa_train` / `vrd_uqa_val` — only the per-model YAML and adapter output path differ.

### B.2 Per-model LoRA configs

Each new YAML is a **verbatim copy of the current
[finetuning/qwen25vl_lora_sft.yaml](../../../finetuning/qwen25vl_lora_sft.yaml)** with only
the intrinsically model-specific keys changed. **All training hyperparameters are kept
identical** to the Qwen config — `finetuning_type: lora`, `lora_target: all`,
`lora_rank: 16`, `lora_alpha: 32`, `cutoff_len: 4096`, `per_device_train_batch_size: 1`,
`gradient_accumulation_steps: 8`, `learning_rate: 2.0e-5`, `num_train_epochs: 1.0`,
`lr_scheduler_type: cosine`, `warmup_steps: 50`, `bf16: true`, `eval_steps: 100`,
`save_steps: 100`, `save_total_limit: 3`, `load_best_model_at_end: true`,
`overwrite_cache: true`, `preprocessing_num_workers: 4`, `report_to: none`,
`ddp_timeout: 180000000`, dataset `vrd_uqa_train` / `vrd_uqa_val`, `trust_remote_code: true`.

Only these keys change per model:

| Config (new) | `model_name_or_path` | `template` | `output_dir` |
|---|---|---|---|
| `finetuning/llama32vl_lora_sft.yaml` (+ `_smoke`) | `meta-llama/Llama-3.2-11B-Vision-Instruct` | `mllama` | `…/finetune_out/llama32vl_lora_sft` |
| `finetuning/internvl35_lora_sft.yaml` (+ `_smoke`) | `OpenGVLab/InternVL3_5-8B` | `intern_vl` | `…/finetune_out/internvl35_lora_sft` |
| `finetuning/phi4mm_lora_sft.yaml` (+ `_smoke`) | `microsoft/Phi-4-multimodal-instruct` | `phi4` (verify; see B.4) | `…/finetune_out/phi4mm_lora_sft` |

`dataset_dir` is also set per job by the parallel-safe scratch layout (B.3.1); it points at
each job's own rsync'd repo copy. No hyperparameter is tuned per model in this design —
identical params keep the comparison across models clean. LoRA fits all three on a single
A40 (largest is 11B); QLoRA is a documented memory fallback only, **not** a default change.

### B.3 Generalized, parallel-safe SLURM finetune script

Generalize `run_finetune_qwen25vl.sh` into `scripts/slurm/run_finetune_vlm.sh <model_key>`
that:

1. selects the YAML by `model_key` (`qwen25vl` | `llama32vl` | `internvl35` | `phi4mm`),
2. installs model-specific vision deps in the scratch LF venv,
3. runs `llamafactory-cli train`,
4. writes the run manifest ([finetuning/write_run_manifest.py](../../../finetuning/write_run_manifest.py)),
5. copies the adapter back to `artifacts/finetuning/<model_key>_lora_sft`.

The existing Qwen script stays working (either kept as-is or made a thin wrapper that calls
`run_finetune_vlm.sh qwen25vl`).

#### B.3.1 Running the 3 fine-tunings in parallel (3 sbatch jobs, no collision)

The three jobs are submitted independently and run concurrently (subject to the scheduler
finding GPUs — 3× `--gres=gpu:1`; if only one A40 is free they queue rather than collide,
which is correct behavior, not a data hazard):

```bash
sbatch --job-name=ft-llama    scripts/slurm/run_finetune_vlm.sh llama32vl
sbatch --job-name=ft-internvl scripts/slurm/run_finetune_vlm.sh internvl35
sbatch --job-name=ft-phi4     scripts/slurm/run_finetune_vlm.sh phi4mm
```

Every shared, writable resource is made **per-job-unique**. The current Qwen script uses a
fixed `WORK_DIR="$SCRATCH_FLASH/finetune_qwen25vl"` and a `mv slurm-finetune-*.out` glob —
both unsafe under concurrency. Required isolation in `run_finetune_vlm.sh`:

| Resource | Today (collision-prone) | Parallel-safe design |
|---|---|---|
| Scratch workspace | fixed `finetune_qwen25vl` | `WORK_DIR=$SCRATCH_FLASH/finetune_${MODEL_KEY}_${SLURM_JOB_ID}` |
| LLaMA-Factory clone | shared `$WORK_DIR/LLaMA-Factory` | per-job (inside the per-job `WORK_DIR`) — no concurrent `git clone`/`uv pip` into one tree |
| Python venv | shared `$WORK_DIR/.venv` | per-job venv inside `WORK_DIR` — no racing `uv pip install` |
| Repo rsync copy | per-`WORK_DIR` | already isolated once `WORK_DIR` is per-job (distinct `dataset_dir` ⇒ distinct tokenized cache) |
| HF caches | default shared `~/.cache/huggingface` | `export HF_HOME=$WORK_DIR/hf_home` per job (isolates the dataset/tokenizer cache that `overwrite_cache: true` rewrites). *Optional:* point `HF_HUB_CACHE` at a shared read-mostly dir to avoid re-downloading weights — safe since the 3 models pull disjoint files and the hub uses file locks. |
| Training `output_dir` | per-model in YAML | distinct per model (`finetune_out/<model_key>_lora_sft`) — the 3 different models never share an output dir |
| Adapter copy-back dest | per-model | `artifacts/finetuning/<model_key>_lora_sft` — distinct dirs under a shared parent; `mkdir -p` of distinct subdirs is concurrency-safe |
| SLURM log move | `mv slurm-finetune-*.out …` (grabs siblings!) | `mv "$HOME"/slurm-finetune-${SLURM_JOB_ID}.out …` — scoped to this job id only |
| Cleanup trap | removes fixed dir | `trap 'rm -rf "$WORK_DIR"' EXIT` removes only this job's scratch |

This guarantees no cross-job collision for **the stated use case (3 distinct models in
parallel)**. Running the *same* model twice concurrently would still share that model's YAML
`output_dir` and copy-back dest; if that case is ever needed, override `output_dir` per job
via `llamafactory-cli train <yaml> output_dir=$WORK_DIR/out` and copy back to a job-suffixed
dest — out of scope for now, noted for completeness.

### B.4 Phi-4-multimodal fallback

Phi-4-multimodal carries internal speech/vision LoRA adapters; LLaMA-Factory's vision-path
fine-tuning support for it is uncertain. Plan:

1. Attempt Phi-4-mm in LLaMA-Factory (`phi4` template) first.
2. **If LF cannot cleanly fine-tune its vision path, fall back to Microsoft's official
   Phi-4-multimodal vision finetuning sample** as a dedicated, isolated path
   (`finetuning/phi4mm_official/`), producing an adapter consumable at eval per B.5.

### B.5 Adapter consumption at eval

The base evaluator already branches on `adapter_path`: when present it attaches the LoRA via
`PeftModel.from_pretrained`; when absent it loads `model_name` plainly. After training, add a
`<model>_finetuned` entry to the eval config JSONs and set each subclass's
`FINETUNED_MODEL_KEY`. The entry's shape depends on how the adapter is consumed:

- **Qwen, Llama** (standard PEFT-attachable): the `<model>_finetuned` entry keeps
  `model_name` = the base checkpoint and sets `adapter_path` =
  `artifacts/finetuning/<model_key>_lora_sft` (a PEFT adapter dir). Current path; no eval
  code change.
- **Phi-4-mm, InternVL** (custom `trust_remote_code` classes / internal LoRA): PEFT-attach is
  attempted first; if it doesn't cleanly attach, the robust fallback is to **merge the
  adapter into base weights post-training** (export a merged checkpoint to
  `artifacts/finetuning/<model_key>_lora_merged`). In that case the `<model>_finetuned` entry
  sets `model_name` = the merged-checkpoint dir and leaves `adapter_path` unset, so the same
  `_load_model` path loads it as a plain model. Whether the artifact is an adapter or merged
  weights is therefore explicit in the config entry, not implicit.

### B.6 Testing (Phase B)

- A `_smoke` YAML per model (few steps, tiny subset) to validate the LF config end-to-end
  on the HPC before a full run.
- A lightweight manifest/config test in the spirit of
  [tests/test_finetune_manifest.py](../../../tests/test_finetune_manifest.py) asserting each
  new YAML references a valid registered dataset and a resolvable adapter `output_dir`.

---

## Phasing & dependencies

Phase A (evaluators) is independent and ships first — it is testable in mock mode with no
GPU. Phase B (fine-tuning) produces adapters that Phase A's `--finetuned` path then
consumes (B.5). Within Phase B, models are independent; Phi-4-mm may diverge to the
official sample (B.4).

## Open items to verify during implementation (not blockers)

- Exact Phi-4-multimodal prompt/placeholder + generation config on transformers 4.57.6.
- InternVL3.5 `model.chat` few-shot-via-`history` with multi-image `num_patches_list`.
- LLaMA-Factory template names/availability for `intern_vl` (InternVL3.5) and `phi4` on a
  fresh clone; A40 peak memory for Llama-3.2-11B LoRA (QLoRA fallback if it spills).
