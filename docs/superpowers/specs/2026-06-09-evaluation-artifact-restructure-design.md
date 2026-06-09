# Evaluation Artifact Restructure — Design

**Date:** 2026-06-09
**Branch:** `restructure-results`
**Status:** Approved design, pending implementation plan

## 1. Problem

The VQA analysis pipeline writes evaluation artifacts in a structure that is hard to
read, hard to query, and not reproducible. Concretely, today a run produces:

```
artifacts/evaluation/<dataset>/LLM/results_w1_OCR_fewshot_mixed_2_qwen25vl_lora_sft_answerable/
  original/   <Model>_vqa_analysis_results.json
  converted/  <Model>_..._converted.json
  augmented/  <Model>_..._converted_augmented.json
  results/    QUR.csv ... FRR.csv
```

Problems (user-reported and discovered during exploration):

1. **Vestigial `LLM/` folder** under every dataset — hardcoded in
   `qwen2.5_evaluator.py::_save_results`, carries no information.
2. **Run metadata encoded in the folder name**
   (`results_w1_OCR_fewshot_mixed_2_qwen25vl_lora_sft_answerable`) — unreadable, brittle,
   and the metrics step *parses* this string (`"_answerable" in folder.name`) to decide
   which metrics to compute.
3. **No run manifest** — nothing records dataset/split/n/seed/config/model/adapter/ocr/
   window/git_commit/created_at.
4. **QUR vs FRR is a painful manual procedure** — QUR requires running the evaluator
   *without* `--answerable`, FRR requires running it *with* `--answerable`. Two manual
   passes, unclear back-to-back behavior, and no meaningful FRR plotting to demonstrate the
   fine-tuned model is not collapsing into "always refuse".

Additional issues found during exploration:

5. **Notebook reads a different layout entirely.** `thesis_plots.ipynb` loads
   `artifacts/evaluation_archive/results_final/<dataset>/<config>/QUR.csv`
   (`config ∈ {zeroshot, fewshot, finetuned}`) — a clean layout that the pipeline never
   produces. A manual reorganization step sits between the run and the plots; this is the
   real source of friction.
6. **Nothing is seeded.** `random.sample` (sampling) and few-shot selection use bare
   `random`, so runs are not reproducible. A `seed` field in a manifest would be a lie
   unless seeding is actually wired in.
7. **`FRR()` is a byte-for-byte copy of `QUR()`.** They are the same computation on
   different question sets — confirming QUR/FRR should be two facets of one run, not two
   folder conventions.
8. **Heavy JSON is triple-stored** (`original` → `converted` → `augmented`), each a full
   re-serialization of the input (layout, OCR, patch entities).
9. **CSV model column is truncated** — `result_file.stem.split("_")[0]` yields `"Qwen"`,
   losing `2.5_7B_finetuned`, so base vs fine-tuned cannot be told apart from the CSV.

**Constraint:** Existing artifacts must not be modified or migrated. All changes target the
code that *generates* artifacts; the existing `eval_*_copy` snapshot and
`evaluation_archive/results_final` are left untouched.

## 2. Decisions (locked with user)

| # | Decision |
|---|----------|
| A | **Layout = dataset → config**, matching the notebook's existing mental model and eliminating the manual reorg. One timestamped run folder; metadata lives in manifests; no `LLM/` folder; no metadata in folder names. |
| B | **QUR + FRR in one pass.** Each item already carries both `original_question` (answerable) and `corrupted_question` (unanswerable). The evaluator asks both per item in one loop (same images, few-shot, seed). A `--questions both\|corrupted\|clean` switch (default `both`) can restrict to one side. `--answerable` is **removed**. |
| C | **Config slug encodes the axes.** `mode` (`zeroshot\|fewshot\|finetuned`) + `_ocr\|_noocr` + `_w{n}` (only when n≠1). Manifest carries the exact params. |
| D | **Raw + one cached derived file.** `predictions.json` is raw and immutable. `_cache/normalized.json` is the single cached derived file: produced by step 1 (LLM canonicalization) and **augmented in place** by step 2 (entity enrichment). No `enriched.json`. Two files per leaf. |
| E | **Keep the three numbered metrics scripts as three runnable steps**, just rewired to the new layout. |

## 3. Target structure

```
artifacts/evaluation_runs/<run_id>/
  run_manifest.json                  # run-wide metadata (one place)
  summary.csv                        # tidy long-format across all leaves
  BDocs/
    finetuned_ocr/
      manifest.json                  # leaf metadata; decodes the slug
      predictions.json               # raw model output, immutable; both answers per item
      _cache/normalized.json         # canonicalized (step 1) + enriched (step 2), cached
      metrics/
        QUR.csv  QUR_C1.csv … QUR_DE.csv …  UR*.csv  FRR.csv  FRR_C1.csv …
    finetuned_noocr/ …
  DUDE/ …  MPDocVQA/ …  SlideVQA/ …

artifacts/evaluation_runs/latest -> <run_id>   # symlink, refreshed each run
```

- **`run_id`**: `eval_<split>_<n>_<YYYYMMDD_HHMMSS>` (human-readable, sortable).
- **Config slug grammar**: `<mode>[_ocr|_noocr][_w{n}]`, e.g. `finetuned_ocr`,
  `zeroshot_noocr`, `fewshot_ocr_w2`.

### 3.1 `run_manifest.json`
```json
{
  "run_id": "eval_val_300_20260608_222213",
  "created_at": "2026-06-08T22:22:13Z",
  "git_commit": "9ee687c", "git_dirty": false,
  "split": "val", "n": 300, "seed": 42,
  "datasets": ["BDocs", "DUDE", "MPDocVQA", "SlideVQA"],
  "configs": ["finetuned_ocr"]
}
```

### 3.2 Leaf `manifest.json`
```json
{
  "run_id": "eval_val_300_20260608_222213",
  "dataset": "BDocs", "config": "finetuned_ocr",
  "label": "Fine-Tuned (LoRA) · OCR",
  "split": "val", "n": 300, "seed": 42,
  "config_path": "VQA_analysis/config_fewshot.json",
  "input_file": ".../BDocs_unanswerable_corrupted_questions_just_false.json",
  "model": "Qwen/Qwen2.5-VL-7B-Instruct", "model_name": "Qwen_2.5_7B_finetuned",
  "adapter": ".../qwen25vl_lora_sft",
  "ocr_enabled": true, "window_size": 1,
  "few_shot": {"enabled": true, "n_shots": 2, "shot_type": "mixed"},
  "questions": "both",
  "counts": {"corrupted": 300, "clean": 300},
  "min_pixels": 200704, "max_pixels": 501760,
  "git_commit": "9ee687c", "created_at": "2026-06-08T22:22:13Z"
}
```
`adapter` is `null` for base-model runs.

### 3.3 `predictions.json` (dual-answer)
Per item, one `vqa_results` record holding both sides; only the requested side(s) are populated:
```jsonc
"vqa_results": [{
  "model_type": "qwen",
  "model_config": { "batch_size": 1, "max_tokens": 1024, "use_flash_attention": false, "adapter_path": "..." },
  "analysis_type": "window_size_1",
  "question_corrupted": "...", "answer_corrupted": [{"pages": [...], "answer": "..."}],
  "question_clean":     "...", "answer_clean":     [{"pages": [...], "answer": "..."}],
  "timestamp": "..."
}]
```
After step 1, each answer dict gains `answer_converted` (canonical form). QUR/UR read
`answer_corrupted`; FRR reads `answer_clean`.

### 3.4 `summary.csv` (tidy long-format)
Columns: `dataset, config, label, model, metric, complexity, value`
(`complexity ∈ {overall, C1, C2, C3}`). One row per metric value. Headline plots can read
this single file; detailed slice plots still read the per-metric CSVs in `metrics/`.

## 4. Component changes

### 4.1 New shared module — `VQA_analysis/run_layout.py`
Single source of truth for layout, imported by evaluator and all metrics scripts:
- `make_run_id(split, n) -> str`
- `build_slug(mode, ocr_enabled, window_size) -> str`
- `leaf_dir(run_root, dataset, slug) -> Path`
- `write_manifest(path, dict)` / `read_manifest(path) -> dict`
- `git_commit() -> str`, `git_dirty() -> bool`
- `append_summary_rows(run_root, rows: list[dict])`
- `human_label(manifest) -> str`

This replaces the brittle name-building in `_save_results` and the name-*parsing* in
`3_compute_metrics.py`.

### 4.2 Evaluator — `qwen2.5_evaluator.py`
- New CLI arg `--questions {both,corrupted,clean}` (default `both`). Remove `--answerable`.
- Seed from `config["seed"]` at startup: `random.seed`, `numpy.random.seed`,
  `torch.manual_seed` (+ cuda).
- `_process_single_question` evaluates the requested side(s): for `both`, runs
  `generate_answer` twice (corrupted + clean question, same images/OCR/few-shot), stores
  `answer_corrupted` / `answer_clean`.
- `_save_results` → uses `run_layout` to write `predictions.json` into
  `evaluation_runs/<run_id>/<dataset>/<slug>/` and writes the leaf `manifest.json` with
  `counts`. No `LLM/`, no `original/`.
- `run_id` resolution: read from env (`VQA_RUN_ID`, exported by the orchestrator) or
  generate if absent (single-dataset local runs still work).

### 4.3 Metrics — three rewired steps
- **`1_normalize_unanswerable_responses.py`**: walk `evaluation_runs/<run_id>` for leaves
  with `predictions.json`. Canonicalize **both** answer arrays → write
  `_cache/normalized.json` (skip if present). Same Qwen2.5-7B classifier as today.
- **`2_enrich_metadata.py`**: read `_cache/normalized.json`, apply current enrichment logic,
  **write back into `_cache/normalized.json` in place** (no `enriched.json`). Idempotent
  (skip if already enriched, e.g. via a flag key in the file).
- **`3_compute_metrics.py`**: read `_cache/normalized.json` + leaf `manifest.json`. Decide
  metrics from `manifest["questions"]` — `corrupted` present → QUR/UR suite; `clean` present
  → FRR. `_get_answers(res, side)` selects the right array. Write CSVs to `metrics/` with
  the CSV column = `manifest["model_name"]`. Append rows to run-level `summary.csv`.
  Remove folder-name string matching.

### 4.4 Orchestration — `scripts/slurm/run_vqa_analysis.sh`
- Compute `run_id` once, `export VQA_RUN_ID`, write `run_manifest.json`, refresh `latest`
  symlink.
- Evaluator writes the **final clean tree directly** under `evaluation_runs/<run_id>/`
  (on scratch); the existing copy-back becomes a 1:1 copy of the clean tree — no separate
  `evaluation/` shape, no post-hoc reorg.

### 4.5 Notebook — `VQA_analysis/notebooks/thesis_plots.ipynb`
- `RESULTS_DIR = artifacts/evaluation_runs/latest`.
- `load_csv()` path → `<dataset>/<config>/metrics/<file>`.
- Discover config leaves from disk and read each `manifest.json` for the legend `label`,
  instead of the hardcoded `CONFIGS` list (headline plots may read `summary.csv`).
- **New FRR section (4 plots):**
  1. **FRR by complexity** — C1/C2/C3 grouped bars per dataset.
  2. **Answer-behavior breakdown** — stacked bars: *clean* = Answered `(1−FRR)` vs Falsely
     refused `(FRR)`; *corrupted* = Correctly refused `(QUR)` vs Wrongly answered `(1−QUR)`.
  3. **QUR↑ vs FRR↓ scatter with fine-tuning trajectory** — arrow `zeroshot → finetuned`
     toward/away from the ideal corner (high QUR, low FRR). Generalizes current plot 5c.
  4. **Refusal calibration summary** — balanced refusal accuracy `(QUR + (1−FRR)) / 2` per
     dataset/config (1.0 = perfect discrimination, 0.5 = refuse/answer everything).

## 5. Non-goals
- No migration of existing runs; old artifacts are read-only and untouched.
- No change to corruption generation, dataset building, or the model/inference logic beyond
  the dual-question loop and seeding.
- No new metrics beyond wiring FRR through cleanly (FRR computation already exists).

## 6. Risks & mitigations
- **Dual-question loop doubles inference per item.** Acceptable at val_300; `--questions`
  lets you opt out. Few-shot context is built once and reused across both questions.
- **`predictions.json` schema change** breaks readers of the old `answer`/`answers` keys.
  Mitigated by the clean-break new layout (old artifacts untouched, new readers only).
- **Notebook config discovery** must handle missing leaves gracefully (some configs absent
  in a given run) — keep the current `[MISSING]` skip behavior.
```