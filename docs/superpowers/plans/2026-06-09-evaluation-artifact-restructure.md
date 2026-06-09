# Evaluation Artifact Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unreadable, name-encoded evaluation artifact tree with a manifest-driven `dataset/config` layout, compute QUR and FRR from a single dual-question evaluation pass, and give the notebook clean inputs plus new FRR plots.

**Architecture:** A new installed module `config/run_layout.py` is the single source of truth for run IDs, config slugs, paths, and manifests. The evaluator writes `predictions.json` (both answers per item) + a leaf `manifest.json` into `artifacts/evaluation_runs/<run_id>/<dataset>/<slug>/`. The three metrics steps are rewired: step 1 canonicalizes both answer arrays into `_cache/normalized.json`, step 2 augments that file in place with entity metadata, step 3 reads the manifest to decide QUR/UR (corrupted side) and/or FRR (clean side) and writes per-metric CSVs plus a run-level tidy `summary.csv`. The notebook reads `evaluation_runs/latest`.

**Tech Stack:** Python 3.12, `uv`, plain `assert`-based tests (no pytest) run via `uv run python -m tests.test_x`, transformers/torch (evaluator + step-1 classifier), pandas (metrics/CSV), matplotlib (notebook). Spec: `docs/superpowers/specs/2026-06-09-evaluation-artifact-restructure-design.md`.

**Conventions for every test in this plan:**
- File lives in `tests/`, mirrors `tests/test_paths.py`: top-level `def test_*()` functions using bare `assert`, plus an `if __name__ == "__main__":` block calling each and printing `OK: <name>`.
- Run with `uv run python -m tests.test_<name>` from the repo root.
- "Verify it fails" means the run raises (ImportError / AssertionError / AttributeError); "verify it passes" means it prints the `OK:` line and exits 0.
- Tests must write only into `tempfile.mkdtemp()` dirs — never into `artifacts/`.

Helper used by several tests to import run-by-path modules (numbered metrics scripts, evaluator):
```python
import importlib.util
from config.paths import REPO_ROOT

def _load(rel_path, name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
```

---

## File Structure

**Create:**
- `config/run_layout.py` — run IDs, slugs, manifest I/O, git helpers, summary.csv writer, `latest` symlink.
- `tests/test_run_layout.py`, `tests/test_evaluator_mock.py`, `tests/test_metrics_normalize.py`, `tests/test_metrics_enrich.py`, `tests/test_metrics_compute.py`, `tests/test_pipeline_integration.py`
- `tests/fixtures/normalized_min.json` — tiny hand-written normalized fixture for metric tests.

**Modify:**
- `VQA_analysis/evaluators/qwen2.5_evaluator.py` — seeding, `--questions`, dual-answer loop, new save path + manifest.
- `VQA_analysis/metrics/1_normalize_unanswerable_responses.py` — walk new tree, canonicalize both arrays, `--no-model`.
- `VQA_analysis/metrics/2_enrich_metadata.py` — augment `normalized.json` in place, idempotent.
- `VQA_analysis/metrics/3_compute_metrics.py` — manifest-driven side selection, model-named columns, `summary.csv`.
- `VQA_analysis/config_zeroshot.json`, `config_fewshot.json`, `config_mock.json` — add `"seed": 42`.
- `scripts/slurm/run_vqa_analysis.sh` — run_id env, run_manifest, `latest`, 1:1 copy-back.
- `VQA_analysis/notebooks/thesis_plots.ipynb` — repoint paths, config discovery, 4 new FRR plots.

---

## Phase 1 — Shared layout module

### Task 1: run_id, mode, slug, label helpers

**Files:**
- Create: `config/run_layout.py`
- Test: `tests/test_run_layout.py`

- [ ] **Step 1: Write the failing test**

```python
"""Validates config/run_layout.py. Run: uv run python -m tests.test_run_layout"""
import datetime
import json
import tempfile
from pathlib import Path
from config import run_layout as rl


def test_make_run_id_format():
    when = datetime.datetime(2026, 6, 8, 22, 22, 13)
    assert rl.make_run_id("val", 300, when) == "eval_val_300_20260608_222213"


def test_derive_mode():
    assert rl.derive_mode(finetuned=True, few_shot_enabled=True) == "finetuned"
    assert rl.derive_mode(finetuned=False, few_shot_enabled=True) == "fewshot"
    assert rl.derive_mode(finetuned=False, few_shot_enabled=False) == "zeroshot"


def test_build_slug():
    assert rl.build_slug("finetuned", ocr_enabled=True, window_size=1) == "finetuned_ocr"
    assert rl.build_slug("zeroshot", ocr_enabled=False, window_size=1) == "zeroshot_noocr"
    assert rl.build_slug("fewshot", ocr_enabled=True, window_size=2) == "fewshot_ocr_w2"


def test_human_label():
    m = {"config": "finetuned_ocr", "ocr_enabled": True, "window_size": 1}
    assert rl.human_label(m) == "Fine-Tuned (LoRA) · OCR"
    m2 = {"config": "zeroshot_noocr", "ocr_enabled": False, "window_size": 1}
    assert rl.human_label(m2) == "Zero-Shot · no-OCR"


if __name__ == "__main__":
    test_make_run_id_format()
    test_derive_mode()
    test_build_slug()
    test_human_label()
    print("OK: config/run_layout.py (ids/slugs/labels)")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m tests.test_run_layout`
Expected: FAIL — `ModuleNotFoundError: No module named 'config.run_layout'`.

- [ ] **Step 3: Write minimal implementation**

Create `config/run_layout.py`:
```python
"""Single source of truth for the evaluation-run artifact layout.

Imported by the evaluator, every metrics step, and the plotting notebook so the
folder structure and manifests are defined in exactly one place. Mirrors the
packaging intent in pyproject.toml: `config` is installed editable, so this
imports cleanly from SLURM by-path scripts, `-m`, and notebooks.
"""
from __future__ import annotations

import csv
import datetime
import json
import subprocess
from pathlib import Path

from config.paths import REPO_ROOT

EVAL_RUNS_DIR = REPO_ROOT / "artifacts" / "evaluation_runs"

MODE_LABELS = {
    "zeroshot": "Zero-Shot",
    "fewshot": "Few-Shot",
    "finetuned": "Fine-Tuned (LoRA)",
}

SUMMARY_COLUMNS = ["dataset", "config", "label", "model", "metric", "complexity", "value"]


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id(split: str, n: int, when: datetime.datetime | None = None) -> str:
    when = when or datetime.datetime.now()
    return f"eval_{split}_{n}_{when:%Y%m%d_%H%M%S}"


def derive_mode(finetuned: bool, few_shot_enabled: bool) -> str:
    if finetuned:
        return "finetuned"
    if few_shot_enabled:
        return "fewshot"
    return "zeroshot"


def build_slug(mode: str, ocr_enabled: bool, window_size: int = 1) -> str:
    slug = f"{mode}_ocr" if ocr_enabled else f"{mode}_noocr"
    if window_size and window_size != 1:
        slug += f"_w{window_size}"
    return slug


def human_label(manifest: dict) -> str:
    mode = (manifest.get("config", "") or "").split("_")[0]
    label = MODE_LABELS.get(mode, mode or "?")
    label += " · OCR" if manifest.get("ocr_enabled") else " · no-OCR"
    w = manifest.get("window_size", 1)
    if w and w != 1:
        label += f" · w{w}"
    return label
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m tests.test_run_layout`
Expected: PASS — prints `OK: config/run_layout.py (ids/slugs/labels)`.

- [ ] **Step 5: Commit**

```bash
git add config/run_layout.py tests/test_run_layout.py
git commit -m "feat(run_layout): run_id, mode, slug, label helpers"
```

---

### Task 2: paths, manifests, git, summary, latest symlink

**Files:**
- Modify: `config/run_layout.py`
- Test: `tests/test_run_layout.py:1` (extend)

- [ ] **Step 1: Add failing tests**

Append these functions to `tests/test_run_layout.py` (and add their calls to the `__main__` block before the final print):
```python
def test_paths_and_manifest_roundtrip():
    import os
    tmp = Path(tempfile.mkdtemp())
    # Point EVAL_RUNS_DIR at a temp location for this test only.
    rl.EVAL_RUNS_DIR = tmp
    leaf = rl.leaf_dir("eval_val_5_20260101_000000", "BDocs", "finetuned_ocr")
    assert leaf == tmp / "eval_val_5_20260101_000000" / "BDocs" / "finetuned_ocr"
    mpath = leaf / "manifest.json"
    rl.write_manifest(mpath, {"dataset": "BDocs", "config": "finetuned_ocr"})
    assert mpath.is_file()
    assert rl.read_manifest(mpath)["dataset"] == "BDocs"


def test_git_helpers_return_types():
    assert isinstance(rl.git_commit(), str) and rl.git_commit()
    assert isinstance(rl.git_dirty(), bool)


def test_append_summary_rows_writes_header_once():
    tmp = Path(tempfile.mkdtemp())
    rl.EVAL_RUNS_DIR = tmp
    run_id = "eval_val_5_20260101_000000"
    rl.append_summary_rows(run_id, [{"dataset": "BDocs", "config": "finetuned_ocr",
        "label": "Fine-Tuned (LoRA) · OCR", "model": "Qwen_2.5_7B_finetuned",
        "metric": "QUR", "complexity": "overall", "value": 0.77}])
    rl.append_summary_rows(run_id, [{"dataset": "BDocs", "config": "finetuned_ocr",
        "label": "Fine-Tuned (LoRA) · OCR", "model": "Qwen_2.5_7B_finetuned",
        "metric": "FRR", "complexity": "overall", "value": 0.10}])
    text = (tmp / run_id / "summary.csv").read_text()
    assert text.count("dataset,config,label,model,metric,complexity,value") == 1
    assert "QUR" in text and "FRR" in text


def test_update_latest_symlink():
    tmp = Path(tempfile.mkdtemp())
    rl.EVAL_RUNS_DIR = tmp
    (tmp / "eval_a").mkdir(parents=True)
    rl.update_latest_symlink("eval_a")
    link = tmp / "latest"
    assert link.is_symlink()
    assert (link.resolve()) == (tmp / "eval_a").resolve()
    # Re-pointing replaces the old target without error.
    (tmp / "eval_b").mkdir()
    rl.update_latest_symlink("eval_b")
    assert (link.resolve()) == (tmp / "eval_b").resolve()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m tests.test_run_layout`
Expected: FAIL — `AttributeError: module 'config.run_layout' has no attribute 'leaf_dir'`.

- [ ] **Step 3: Implement**

Append to `config/run_layout.py`:
```python
def run_dir(run_id: str) -> Path:
    return EVAL_RUNS_DIR / run_id


def leaf_dir(run_id: str, dataset: str, slug: str) -> Path:
    return run_dir(run_id) / dataset / slug


def write_manifest(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def read_manifest(path) -> dict:
    with open(path) as f:
        return json.load(f)


def _git(args) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(REPO_ROOT), text=True, stderr=subprocess.DEVNULL
    ).strip()


def git_commit() -> str:
    try:
        return _git(["rev-parse", "--short", "HEAD"])
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(_git(["status", "--porcelain"]))
    except Exception:
        return False


def append_summary_rows(run_id: str, rows) -> None:
    path = run_dir(run_id) / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if not exists:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in SUMMARY_COLUMNS})


def update_latest_symlink(run_id: str) -> None:
    EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    link = EVAL_RUNS_DIR / "latest"
    if link.is_symlink() or link.exists():
        try:
            link.unlink()
        except OSError:
            pass
    link.symlink_to(run_id)  # relative target inside EVAL_RUNS_DIR
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run python -m tests.test_run_layout`
Expected: PASS — prints `OK: config/run_layout.py (ids/slugs/labels)`.

- [ ] **Step 5: Commit**

```bash
git add config/run_layout.py tests/test_run_layout.py
git commit -m "feat(run_layout): paths, manifests, git, summary, latest symlink"
```

---

## Phase 2 — Evaluator: seeding, --questions, dual-answer, new save path

### Task 3: Add seed to configs + seed the evaluator

**Files:**
- Modify: `VQA_analysis/config_zeroshot.json`, `config_fewshot.json`, `config_mock.json`
- Modify: `VQA_analysis/evaluators/qwen2.5_evaluator.py:24-52` (`__init__`)

- [ ] **Step 1: Add `"seed": 42` to all three configs**

In each of `VQA_analysis/config_zeroshot.json`, `config_fewshot.json`, `config_mock.json`, add a top-level key alongside `"sampling_percentage"`:
```json
    "seed": 42,
```

- [ ] **Step 2: Seed in `__init__`**

In `qwen2.5_evaluator.py`, change the constructor signature and add seeding. Replace lines 24-37 region:
```python
    def __init__(self, config_path, finetuned, questions="both"):
        with open(config_path) as f:
            self.config = json.load(f)

        self.finetuned = finetuned
        self.questions = questions
        self.seed = self.config.get("seed", 42)
        self._set_seed()
        if self.finetuned:
            self.model_config = self.config["open_source_models"]["qwen2.5_finetuned"]
        else:
            self.model_config = self.config["open_source_models"]["qwen2.5"]

        self.sampling_percentage = self.config.get("sampling_percentage", 100)
        self.unable_to_respond_aware = self.config.get("unable_to_respond_aware", True)
```

Add this method just below `__init__` (before `_create_prompt`):
```python
    def _set_seed(self):
        random.seed(self.seed)
        try:
            import numpy as np
            np.random.seed(self.seed)
        except Exception:
            pass
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
```

- [ ] **Step 3: Update `main()` to drop `--answerable`, add `--questions`**

Replace the `main()` body (lines 598-609) with:
```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument(
        "--questions", choices=["both", "corrupted", "clean"], default="both",
        help="Which question side(s) to evaluate: both (default), corrupted (QUR), or clean (FRR).",
    )
    args = parser.parse_args()

    evaluator = QwenVQAEvaluator(args.config_path, args.finetuned, questions=args.questions)
    print(f"Starting QWEN 2.5 evaluator (questions={args.questions}, seed={evaluator.seed})")
    evaluator.evaluate()
```

Also delete the now-unused `self.answerable` references — they are removed in Task 4.

- [ ] **Step 4: Smoke-check it imports and parses args**

Run: `uv run python VQA_analysis/evaluators/qwen2.5_evaluator.py --help`
Expected: help text lists `--questions {both,corrupted,clean}` and NO `--answerable`.

- [ ] **Step 5: Commit**

```bash
git add VQA_analysis/config_*.json VQA_analysis/evaluators/qwen2.5_evaluator.py
git commit -m "feat(evaluator): seed from config, replace --answerable with --questions"
```

---

### Task 4: Dual-answer evaluation loop

**Files:**
- Modify: `VQA_analysis/evaluators/qwen2.5_evaluator.py:483-539` (`_process_single_question`)
- Test: `tests/test_evaluator_mock.py`

- [ ] **Step 1: Write the failing test (mock subprocess run)**

```python
"""Mock-mode evaluator checks. Run: uv run python -m tests.test_evaluator_mock"""
import json
import os
import subprocess
import tempfile
from pathlib import Path
from config.paths import REPO_ROOT

SAMPLE = REPO_ROOT / "VQA_analysis" / "evaluation_files" / "BDocs_sample15.json"


def _write_mock_config(tmp, questions_n=4):
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


def _run(cfg, run_id, questions, run_root):
    env = dict(os.environ)
    env["VQA_RUN_ID"] = run_id
    env["VQA_EVAL_RUNS_DIR"] = str(run_root)
    subprocess.check_call(
        ["uv", "run", "python", "VQA_analysis/evaluators/qwen2.5_evaluator.py",
         "--config_path", str(cfg), "--finetuned", "--questions", questions],
        cwd=str(REPO_ROOT), env=env,
    )


def test_dual_answer_predictions_and_manifest():
    tmp = tempfile.mkdtemp()
    run_root = Path(tmp) / "runs"
    cfg = _write_mock_config(tmp)
    run_id = "eval_val_15_20260101_000000"
    _run(cfg, run_id, "both", run_root)

    leaf = run_root / run_id / "BDocs" / "finetuned_noocr"
    preds = json.load(open(leaf / "predictions.json"))
    item0 = preds["corrupted_questions"][0]["verification_result"]["vqa_results"][0]
    assert "answer_corrupted" in item0 and "answer_clean" in item0
    assert item0["question_corrupted"] != item0["question_clean"]

    man = json.load(open(leaf / "manifest.json"))
    assert man["dataset"] == "BDocs"
    assert man["config"] == "finetuned_noocr"
    assert man["questions"] == "both"
    assert man["seed"] == 42
    assert man["model_name"] == "Qwen_2.5_7B_finetuned"
    assert man["adapter"]  # finetuned -> non-null
    assert "git_commit" in man and "created_at" in man


def test_questions_corrupted_only_omits_clean():
    tmp = tempfile.mkdtemp()
    run_root = Path(tmp) / "runs"
    cfg = _write_mock_config(tmp)
    run_id = "eval_val_15_20260101_000001"
    _run(cfg, run_id, "corrupted", run_root)
    leaf = run_root / run_id / "BDocs" / "finetuned_noocr"
    item0 = json.load(open(leaf / "predictions.json"))["corrupted_questions"][0]["verification_result"]["vqa_results"][0]
    assert "answer_corrupted" in item0 and "answer_clean" not in item0


if __name__ == "__main__":
    test_dual_answer_predictions_and_manifest()
    test_questions_corrupted_only_omits_clean()
    print("OK: evaluator mock dual-answer")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m tests.test_evaluator_mock`
Expected: FAIL — predictions land in the old path / `answer_corrupted` missing (KeyError or FileNotFound). (Save-path changes land in Task 5; this task makes the dual-answer keys appear.)

- [ ] **Step 3: Implement the dual-answer loop**

Replace `_process_single_question` (lines 483-539) with:
```python
    QUESTION_SIDES = {
        "both": ["corrupted", "clean"],
        "corrupted": ["corrupted"],
        "clean": ["clean"],
    }

    def _process_single_question(self, item, dataset_pool):
        """Evaluate the requested question side(s) for one item, storing both
        answers in a single vqa_result (corrupted -> QUR/UR, clean -> FRR)."""
        if "verification_result" not in item:
            item["verification_result"] = {}
        if "vqa_results" not in item["verification_result"]:
            item["verification_result"]["vqa_results"] = []

        pages = item["layout_analysis"]["pages"]
        image_paths = [
            os.path.join(self.images_base_path, os.path.basename(page_id))
            for page_id in pages
        ]
        ocr_text = self.get_ocr_text(pages) if self.config.get("ocr_enabled", False) else None

        few_shot_turns = None
        few_shot_config = self.config.get("few_shot", {})
        if few_shot_config.get("enabled", False):
            shots = self._select_few_shot_examples(dataset_pool, item)
            few_shot_turns = self._build_few_shot_turns(shots)

        question_text = {
            "corrupted": item["corrupted_question"],
            "clean": item["original_question"],
        }

        vqa_result = {
            "model_type": "qwen",
            "model_config": {
                "batch_size": self.model_config.get("batch_size", 1),
                "max_tokens": self.max_tokens,
                "use_flash_attention": self.model_config.get("use_flash_attention", False),
                "adapter_path": self.model_config.get("adapter_path", None),
            },
            "ocr_enabled": bool(ocr_text),
            "few_shot_config": self.config.get("few_shot", {"enabled": False}),
            "timestamp": datetime.datetime.now().isoformat(),
        }

        success = True
        for side in self.QUESTION_SIDES[self.questions]:
            result = self.generate_answer(
                question_text[side], image_paths, ocr_text, few_shot_turns=few_shot_turns
            )
            vqa_result[f"question_{side}"] = question_text[side]
            vqa_result[f"answer_{side}"] = result.get("answer", "Unable to determine")
            vqa_result["analysis_type"] = result.get("analysis_type", "")
            if "error" in result:
                vqa_result[f"error_{side}"] = result["error"]
                vqa_result[f"traceback_{side}"] = result.get("traceback", "")
                success = False

        item["verification_result"]["vqa_results"].append(vqa_result)
        return success
```

Note: `self.answerable` no longer exists; `evaluate()` already passes `data["corrupted_questions"]` as the pool and does not reference `answerable`. Confirm no remaining `self.answerable` references with `grep -n answerable VQA_analysis/evaluators/qwen2.5_evaluator.py` (should be empty).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run python -m tests.test_evaluator_mock`
Expected: still FAIL at this point on the path/manifest assertions (save path lands in Task 5), but `answer_corrupted`/`answer_clean` are now produced. To confirm just the loop, temporarily inspect: the FAIL message should now be a missing-file/manifest error, not a missing `answer_corrupted` key. Proceed to Task 5.

- [ ] **Step 5: Commit**

```bash
git add VQA_analysis/evaluators/qwen2.5_evaluator.py
git commit -m "feat(evaluator): dual-answer loop (corrupted + clean per item)"
```

---

### Task 5: New save path + leaf manifest

**Files:**
- Modify: `VQA_analysis/evaluators/qwen2.5_evaluator.py:304-356` (`_save_results`), imports near top.
- Test: `tests/test_evaluator_mock.py` (from Task 4)

- [ ] **Step 1: Add the import**

Near the existing `from config.paths import REPO_ROOT` (line 20), add:
```python
from config import run_layout as rl
```

- [ ] **Step 2: Run the Task-4 test to confirm it still fails on path/manifest**

Run: `uv run python -m tests.test_evaluator_mock`
Expected: FAIL — `FileNotFoundError` for the new `predictions.json` path (old `_save_results` still writes `evaluation/<dataset>/LLM/.../original/...`).

- [ ] **Step 3: Replace `_save_results`**

Replace the whole `_save_results` method (lines 304-356) with:
```python
    def _save_results(self, data):
        few_shot_enabled = self.config.get("few_shot", {}).get("enabled", False)
        mode = rl.derive_mode(self.finetuned, few_shot_enabled)
        ocr_enabled = bool(self.config.get("ocr_enabled", False))
        window_size = self.model_config.get("batch_size", 1)
        slug = rl.build_slug(mode, ocr_enabled, window_size)

        dataset = self.config["dataset"]
        split = self.config.get("split", "val")
        n_items = len(data.get("corrupted_questions", []))
        run_id = os.environ.get("VQA_RUN_ID") or rl.make_run_id(split, n_items)

        # Allow tests/orchestration to redirect the runs root.
        runs_dir_override = os.environ.get("VQA_EVAL_RUNS_DIR")
        if runs_dir_override:
            rl.EVAL_RUNS_DIR = __import__("pathlib").Path(runs_dir_override)

        leaf = rl.leaf_dir(run_id, dataset, slug)
        leaf.mkdir(parents=True, exist_ok=True)

        predictions_path = leaf / "predictions.json"
        try:
            with open(predictions_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Predictions saved to {predictions_path}")
        except Exception as e:
            print(f"Error saving predictions: {str(e)}")
            return

        counts = {}
        if self.questions in ("both", "corrupted"):
            counts["corrupted"] = n_items
        if self.questions in ("both", "clean"):
            counts["clean"] = n_items

        manifest = {
            "run_id": run_id,
            "dataset": dataset,
            "config": slug,
            "split": split,
            "n": n_items,
            "seed": self.seed,
            "config_path": os.environ.get("VQA_CONFIG_PATH", ""),
            "input_file": self.config.get("input_file"),
            "model": self.model_config["model_name"],
            "model_name": self.model_config["name"],
            "adapter": self.model_config.get("adapter_path"),
            "ocr_enabled": ocr_enabled,
            "window_size": window_size,
            "few_shot": self.config.get("few_shot", {"enabled": False}),
            "questions": self.questions,
            "counts": counts,
            "min_pixels": self.model_config.get("min_pixels"),
            "max_pixels": self.model_config.get("max_pixels"),
            "git_commit": rl.git_commit(),
            "git_dirty": rl.git_dirty(),
            "created_at": rl.utc_now_iso(),
        }
        manifest["label"] = rl.human_label(manifest)
        rl.write_manifest(leaf / "manifest.json", manifest)
        print(f"Manifest saved to {leaf / 'manifest.json'}")
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run python -m tests.test_evaluator_mock`
Expected: PASS — prints `OK: evaluator mock dual-answer`.

- [ ] **Step 5: Commit**

```bash
git add VQA_analysis/evaluators/qwen2.5_evaluator.py
git commit -m "feat(evaluator): write predictions+manifest into evaluation_runs/<run_id>/<dataset>/<slug>"
```

---

## Phase 3 — Metrics: three rewired steps

### Task 6: Step 1 — canonicalize both answer arrays into `_cache/normalized.json`

**Files:**
- Modify: `VQA_analysis/metrics/1_normalize_unanswerable_responses.py`
- Test: `tests/test_metrics_normalize.py`

- [ ] **Step 1: Write the failing test**

```python
"""Step-1 normalize checks. Run: uv run python -m tests.test_metrics_normalize"""
import json
import tempfile
from pathlib import Path
from config.paths import REPO_ROOT
import importlib.util


def _load(rel_path, name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


STEP1 = _load("VQA_analysis/metrics/1_normalize_unanswerable_responses.py", "step1")


def test_canonicalize_rule_based():
    ident = lambda a: a  # identity classifier; must not be called for these
    assert STEP1.canonicalize_answer("Not available", ident) == "unable to determine"
    assert STEP1.canonicalize_answer("", ident) == "unable to determine"
    assert STEP1.canonicalize_answer("$100", ident) == "$100"  # numeric passthrough


def test_canonicalize_uses_classifier_for_freetext():
    calls = []
    def stub(a):
        calls.append(a)
        return "stubbed"
    assert STEP1.canonicalize_answer("Paris", stub) == "stubbed"
    assert calls == ["Paris"]


def test_label_vqa_answers_both_sides():
    item = {"verification_result": {"vqa_results": [{
        "answer_corrupted": [{"pages": ["p1"], "answer": "Not available"}],
        "answer_clean": [{"pages": ["p1"], "answer": "$42"}],
    }]}}
    src = {"corrupted_questions": [item]}
    tmp = Path(tempfile.mkdtemp())
    inp, out = tmp / "predictions.json", tmp / "normalized.json"
    inp.write_text(json.dumps(src))
    STEP1.label_vqa_answers(str(inp), str(out), classify_fn=lambda a: a)
    r = json.load(open(out))["corrupted_questions"][0]["verification_result"]["vqa_results"][0]
    assert r["answer_corrupted"][0]["answer_converted"] == "unable to determine"
    assert r["answer_clean"][0]["answer_converted"] == "$42"


if __name__ == "__main__":
    test_canonicalize_rule_based()
    test_canonicalize_uses_classifier_for_freetext()
    test_label_vqa_answers_both_sides()
    print("OK: metrics step 1 normalize")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m tests.test_metrics_normalize`
Expected: FAIL — `AttributeError: module 'step1' has no attribute 'canonicalize_answer'`.

- [ ] **Step 3: Implement**

In `1_normalize_unanswerable_responses.py`:

(a) Add the import at the top alongside the existing imports:
```python
from config.run_layout import EVAL_RUNS_DIR, run_dir
```

(b) Extract the per-answer logic into a pure function (place above `label_vqa_answers`):
```python
UNABLE_PHRASES = [
    "unable to determine", "not answerable", "not provided", "not available",
    "not in the image", "not in the document", "not found", "not contain",
    "not include", "cannot determine", "cannot answer", "cannot provide",
    "cannot find", "i don ' t know", "unknown",
]


def _is_numeric(text):
    text = text.replace("$", "").replace("€", "").replace(",", "").strip()
    text = "".join(text.split()).rstrip("%").rstrip(".")
    try:
        float(text)
        return True
    except ValueError:
        return False


def canonicalize_answer(answer, classify_fn=None):
    """Map a free-text answer to a canonical form. Rule-based first (cheap),
    falling back to the injected classifier only for ambiguous free text."""
    if classify_fn is None:
        classify_fn = classify_unanswerable_answer
    low = answer.lower()
    if any(p in low for p in UNABLE_PHRASES) or low == "":
        return "unable to determine"
    if _is_numeric(answer):
        return answer
    return classify_fn(answer)
```

(c) Rewrite `label_vqa_answers` to canonicalize both arrays:
```python
def label_vqa_answers(input_file, output_file, classify_fn=None):
    with open(input_file, "r") as f:
        data = json.load(f)

    for question in tqdm(data.get("corrupted_questions", []), mininterval=30):
        for result in question.get("verification_result", {}).get("vqa_results", []):
            for side_key in ("answer_corrupted", "answer_clean", "answers", "answer"):
                answers = result.get(side_key)
                if not isinstance(answers, list):
                    continue
                for answer_obj in answers:
                    if not isinstance(answer_obj, dict):
                        continue
                    answer_obj["answer_converted"] = canonicalize_answer(
                        answer_obj.get("answer", ""), classify_fn
                    )

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Processed file saved to {output_file}")
```

(d) Rewrite `process_all_folders` to walk the new tree and add a `--no-model` switch:
```python
def process_all_folders(run_id=None, use_model=True):
    root = run_dir(run_id) if run_id else (EVAL_RUNS_DIR / "latest")
    print(f"{'='*100}\nNORMALIZE — scanning leaves under: {root}\n{'='*100}")
    if not root.exists():
        print(f"ERROR: run directory does not exist: {root}")
        return

    classify_fn = (lambda a: a) if not use_model else classify_unanswerable_answer
    total_processed = total_skipped = total_errors = 0

    for manifest_path in root.rglob("manifest.json"):
        leaf = manifest_path.parent
        preds = leaf / "predictions.json"
        if not preds.exists():
            continue
        cache = leaf / "_cache"
        cache.mkdir(exist_ok=True)
        out = cache / "normalized.json"
        if out.exists():
            print(f"Skipping (already normalized): {leaf}")
            total_skipped += 1
            continue
        try:
            label_vqa_answers(str(preds), str(out), classify_fn=classify_fn)
            total_processed += 1
        except Exception as e:
            print(f"ERROR normalizing {preds}: {e}")
            total_errors += 1

    print(f"Normalize complete — processed: {total_processed}, skipped: {total_skipped}, errors: {total_errors}")


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=os.environ.get("VQA_RUN_ID"))
    parser.add_argument("--no-model", action="store_true", help="Skip LLM classifier (rule-based only); for tests/debug.")
    args = parser.parse_args()
    process_all_folders(run_id=args.run_id, use_model=not args.no_model)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run python -m tests.test_metrics_normalize`
Expected: PASS — prints `OK: metrics step 1 normalize`.

- [ ] **Step 5: Commit**

```bash
git add VQA_analysis/metrics/1_normalize_unanswerable_responses.py tests/test_metrics_normalize.py
git commit -m "feat(metrics): step 1 canonicalizes both answer sides into _cache/normalized.json"
```

---

### Task 7: Step 2 — enrich `normalized.json` in place, idempotently

**Files:**
- Modify: `VQA_analysis/metrics/2_enrich_metadata.py`
- Test: `tests/test_metrics_enrich.py`

- [ ] **Step 1: Write the failing test**

```python
"""Step-2 enrich checks. Run: uv run python -m tests.test_metrics_enrich"""
import json
import tempfile
from pathlib import Path
from config.paths import REPO_ROOT
import importlib.util


def _load(rel_path, name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


STEP2 = _load("VQA_analysis/metrics/2_enrich_metadata.py", "step2")


def _minimal_normalized():
    return {"corrupted_questions": [{
        "complexity": 1,
        "entity_type": ["year_number_information"],
        "original_entity": [{"text": "2019"}],
        "corrupted_entities": [{"text": "2019"}],
        "question_entities": [{"text": "2019"}],
        "patch_entities": {},
        "verification_result": {"vqa_results": [{
            "answer_corrupted": [{"pages": ["p1"], "answer": "x", "answer_converted": "unable to determine"}],
        }]},
    }]}


def test_enrich_in_place_and_idempotent():
    tmp = Path(tempfile.mkdtemp())
    norm = tmp / "normalized.json"
    norm.write_text(json.dumps(_minimal_normalized()))

    STEP2.enrich_file(str(norm))
    data = json.load(open(norm))
    assert data.get("_enriched") is True
    # question_entities rebuilt with positions key
    assert "positions" in data["corrupted_questions"][0]["question_entities"][0]
    # answers untouched
    vr = data["corrupted_questions"][0]["verification_result"]["vqa_results"][0]
    assert vr["answer_corrupted"][0]["answer_converted"] == "unable to determine"

    # Idempotent: second run is a no-op (flag already set), file still valid.
    mtime_marker = json.dumps(json.load(open(norm)), sort_keys=True)
    STEP2.enrich_file(str(norm))
    assert json.dumps(json.load(open(norm)), sort_keys=True) == mtime_marker


if __name__ == "__main__":
    test_enrich_in_place_and_idempotent()
    print("OK: metrics step 2 enrich")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m tests.test_metrics_enrich`
Expected: FAIL — `AttributeError: module 'step2' has no attribute 'enrich_file'`.

- [ ] **Step 3: Implement**

In `2_enrich_metadata.py`:

(a) Add import at top:
```python
from config.run_layout import EVAL_RUNS_DIR, run_dir
```

(b) Keep `enrich_entity`, `find_patch_matches`, and the body of `process_vqa_file` as the enrichment logic, but refactor `process_vqa_file` so its core works on an in-memory `data` dict. Rename the existing `process_vqa_file(input_file, output_file)` to an internal `_enrich_data(data)` that takes and returns the dict (move the load/save out). Then add:
```python
def enrich_file(path):
    """Enrich a normalized.json in place. No-op if already enriched."""
    with open(path) as f:
        data = json.load(f)
    if data.get("_enriched"):
        return
    _enrich_data(data)
    data["_enriched"] = True
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Enriched in place: {path}")
```
(Concretely: take the current `process_vqa_file` body, delete its first `with open(input_file) as f: data = json.load(f)` and its trailing `with open(output_file, "w")...` block, wrap the remaining transformation as `def _enrich_data(data):` returning nothing — it mutates `data`.)

(c) Replace `process_all_folders` with:
```python
def process_all_folders(run_id=None):
    root = run_dir(run_id) if run_id else (EVAL_RUNS_DIR / "latest")
    print(f"{'='*100}\nENRICH — scanning normalized.json under: {root}\n{'='*100}")
    if not root.exists():
        print(f"ERROR: run directory does not exist: {root}")
        return
    processed = skipped = errors = 0
    for norm in root.rglob("_cache/normalized.json"):
        try:
            before = json.load(open(norm)).get("_enriched", False)
            enrich_file(str(norm))
            if before:
                skipped += 1
            else:
                processed += 1
        except Exception as e:
            print(f"ERROR enriching {norm}: {e}")
            errors += 1
    print(f"Enrich complete — processed: {processed}, skipped: {skipped}, errors: {errors}")


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=os.environ.get("VQA_RUN_ID"))
    args = parser.parse_args()
    process_all_folders(run_id=args.run_id)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run python -m tests.test_metrics_enrich`
Expected: PASS — prints `OK: metrics step 2 enrich`.

- [ ] **Step 5: Commit**

```bash
git add VQA_analysis/metrics/2_enrich_metadata.py tests/test_metrics_enrich.py
git commit -m "feat(metrics): step 2 enriches normalized.json in place, idempotent"
```

---

### Task 8: Step 3 — manifest-driven QUR/FRR, model-named columns, summary.csv

**Files:**
- Modify: `VQA_analysis/metrics/3_compute_metrics.py`
- Create: `tests/fixtures/normalized_min.json`
- Test: `tests/test_metrics_compute.py`

- [ ] **Step 1: Create the fixture**

`tests/fixtures/normalized_min.json`:
```json
{
  "_enriched": true,
  "corrupted_questions": [
    {
      "complexity": 1,
      "is_corrupted": true,
      "verification_result": {"vqa_results": [{
        "answer_corrupted": [{"pages": ["p1"], "answer": "n/a", "answer_converted": "unable to determine"}],
        "answer_clean": [{"pages": ["p1"], "answer": "Rome", "answer_converted": "Rome"}]
      }]}
    },
    {
      "complexity": 2,
      "is_corrupted": true,
      "verification_result": {"vqa_results": [{
        "answer_corrupted": [{"pages": ["p1"], "answer": "n/a", "answer_converted": "unable to determine"}],
        "answer_clean": [{"pages": ["p1"], "answer": "n/a", "answer_converted": "unable to determine"}]
      }]}
    }
  ]
}
```

- [ ] **Step 2: Write the failing test**

```python
"""Step-3 compute checks. Run: uv run python -m tests.test_metrics_compute"""
import json
from pathlib import Path
from config.paths import REPO_ROOT
import importlib.util


def _load(rel_path, name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


STEP3 = _load("VQA_analysis/metrics/3_compute_metrics.py", "step3")
FIX = REPO_ROOT / "tests" / "fixtures" / "normalized_min.json"


def _results():
    return json.load(open(FIX))["corrupted_questions"]


def test_qur_reads_corrupted_side():
    az = STEP3.VQAAnalyzer(_results(), None, "BDocs", side="corrupted")
    qur = az.QUR()
    assert qur[0] == 1.0  # both items fully refused on corrupted side


def test_frr_reads_clean_side():
    az = STEP3.VQAAnalyzer(_results(), None, "BDocs", side="clean")
    frr = az.FRR()
    # item0 answered (Rome) -> not refused; item1 refused -> FRR = 1/2
    assert abs(frr[0] - 0.5) < 1e-9


def test_get_answers_side_selection():
    az = STEP3.VQAAnalyzer(_results(), None, "BDocs", side="corrupted")
    assert az._get_answers(az.valid_results[0])[0]["answer_converted"] == "unable to determine"
    az2 = STEP3.VQAAnalyzer(_results(), None, "BDocs", side="clean")
    assert az2._get_answers(az2.valid_results[0])[0]["answer_converted"] == "Rome"


if __name__ == "__main__":
    test_qur_reads_corrupted_side()
    test_frr_reads_clean_side()
    test_get_answers_side_selection()
    print("OK: metrics step 3 compute")
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run python -m tests.test_metrics_compute`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'side'`.

- [ ] **Step 4: Implement — add `side` and rewire selection**

In `3_compute_metrics.py`:

(a) Add import at top:
```python
from config import run_layout as rl
```

(b) `VQAAnalyzer.__init__`: add `side="corrupted"` parameter and store it; keep the `valid_results` filter unchanged:
```python
    def __init__(self, results, entity_verifier, dataset, debug=False, images_path=None, side="corrupted"):
        self.results = results
        self.debug = debug
        self.entity_identifier = entity_verifier
        self.dataset = dataset
        self.images_path = images_path
        self.side = side
        self.valid_results = [
            r for r in results
            if r.get("is_corrupted")
            and "verification_result" in r
            and "vqa_results" in r["verification_result"]
            and len(r["verification_result"]["vqa_results"]) > 0
        ]
```

(c) Rewrite `_get_answers` to use the side, with backward-compatible fallback:
```python
    def _get_answers(self, res):
        vqa_result = res["verification_result"]["vqa_results"][0]
        return vqa_result.get(
            f"answer_{self.side}",
            vqa_result.get("answers", vqa_result.get("answer", [])),
        )
```
(Make it an instance method — remove the `@staticmethod` decorator above it. The existing call sites already use `self._get_answers(res)`, so no other call-site edits are needed.)

(d) Replace `generate_analysis_report` so it walks the run tree, reads each leaf manifest, and computes the right side(s). Replace the whole function with:
```python
def generate_analysis_report(run_id=None, dataset=None, images_path=None):
    entity_verifier = None
    root = rl.run_dir(run_id) if run_id else (rl.EVAL_RUNS_DIR / "latest")
    if not root.exists():
        print(f"ERROR: run directory does not exist: {root}")
        return

    dataset_dirs = [root / dataset] if dataset else [p for p in root.iterdir() if p.is_dir() and p.name != "latest"]

    for dataset_dir in dataset_dirs:
        if not dataset_dir.is_dir():
            print(f"Skipping missing dataset dir: {dataset_dir}")
            continue
        ds_name = dataset_dir.name
        for leaf in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            manifest_path = leaf / "manifest.json"
            norm_path = leaf / "_cache" / "normalized.json"
            if not manifest_path.exists() or not norm_path.exists():
                continue
            manifest = rl.read_manifest(manifest_path)
            model_name = manifest.get("model_name", "model")
            label = manifest.get("label", rl.human_label(manifest))
            questions = manifest.get("questions", "both")
            slug = manifest.get("config", leaf.name)

            with open(norm_path) as f:
                data = json.load(f)
            results = data.get("corrupted_questions", [])
            images_path_local = data.get("base_image_dir", images_path)

            metrics_dir = leaf / "metrics"
            metrics_dir.mkdir(exist_ok=True)
            summary_rows = []

            print(f"\n{'#'*100}\n{ds_name}/{slug}  (questions={questions}, model={model_name})")

            if questions in ("both", "corrupted"):
                az = VQAAnalyzer(results, entity_verifier, ds_name, images_path=images_path_local, side="corrupted")
                _save_qur_suite(az, metrics_dir, model_name)
                qur = az.QUR()
                ur = az.UR()
                for tag, vals in [("QUR", qur[:4]), ("UR", ur[:4])]:
                    for comp, v in zip(["overall", "C1", "C2", "C3"], vals):
                        summary_rows.append({"dataset": ds_name, "config": slug, "label": label,
                                             "model": model_name, "metric": tag, "complexity": comp, "value": v})

            if questions in ("both", "clean"):
                az_c = VQAAnalyzer(results, entity_verifier, ds_name, images_path=images_path_local, side="clean")
                frr = az_c.FRR()
                save_metric(metrics_dir, "FRR", {model_name: frr}, ["FRR", "FRR_C1", "FRR_C2", "FRR_C3"])
                for comp, v in zip(["overall", "C1", "C2", "C3"], frr):
                    summary_rows.append({"dataset": ds_name, "config": slug, "label": label,
                                         "model": model_name, "metric": "FRR", "complexity": comp, "value": v})

            rl.append_summary_rows(manifest["run_id"], summary_rows)
            print(f"metrics written to {metrics_dir}")
```

(e) Extract the existing per-folder QUR/UR CSV saving block (current lines ~1032-1046) into a helper so both the new function and tests can reuse it. Add above `generate_analysis_report`:
```python
def _save_qur_suite(az, metrics_dir, model_name):
    m = az.calculate_metrics()
    *qur_pl_dicts, list_len = m["QUR_PL"]
    m["QUR_PL"] = qur_pl_dicts

    def col(values):
        return {model_name: list(values)}

    save_metric(metrics_dir, "QUR", col(m["QUR"][:5]), ["QUR", "QUR_C1", "QUR_C2", "QUR_C3", "QUR_weighted"])
    save_metric(metrics_dir, "UR",  col(m["UR"]),       ["UR", "UR_C1", "UR_C2", "UR_C3"])
    for name, index in [("QUR_DE", LAYOUT_TYPES), ("QUR_NLPE", MACRO_ENTITY_TYPES), ("QUR_QP", PAGE_LAYOUT),
                        ("QUR_PL", list_len), ("QUR_DED", ["<15", "15-25", ">25"]),
                        ("UR_DE", LAYOUT_TYPES), ("UR_NLPE", MACRO_ENTITY_TYPES),
                        ("UR_PAGE_DE", LAYOUT_TYPES), ("UR_PAGE_QP", PAGE_LAYOUT), ("UR_PAGE_DED", ["0", "1", ">1"])]:
        base, c1, c2, c3 = m[name]
        save_metric(metrics_dir, name, {model_name: list(base.values())}, index,
                    [{model_name: list(c1.values())}, {model_name: list(c2.values())}, {model_name: list(c3.values())}])
    inpage, ip1, ip2, ip3, outpage, op1, op2, op3 = m["UR_PAGE"]
    save_metric(metrics_dir, "UR_PAGE_inpage",  {model_name: [inpage, ip1, ip2, ip3]}, ["UR_inpage", "UR_inpage_C1", "UR_inpage_C2", "UR_inpage_C3"])
    save_metric(metrics_dir, "UR_PAGE_outpage", {model_name: [outpage, op1, op2, op3]}, ["UR_outpage", "UR_outpage_C1", "UR_outpage_C2", "UR_outpage_C3"])
```

(f) Replace `main()`:
```python
if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser(description="Generate VQA analysis report for an evaluation run")
    parser.add_argument("--run-id", default=os.environ.get("VQA_RUN_ID"))
    parser.add_argument("--dataset", type=str, default=None, help="Limit to one dataset; default = all in the run.")
    parser.add_argument("--images_path", type=str, default=None)
    args = parser.parse_args()
    print("\n" + "="*100 + "\n3. COMPUTE METRICS\n" + "="*100)
    generate_analysis_report(run_id=args.run_id, dataset=args.dataset, images_path=args.images_path)
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run python -m tests.test_metrics_compute`
Expected: PASS — prints `OK: metrics step 3 compute`.

- [ ] **Step 6: Commit**

```bash
git add VQA_analysis/metrics/3_compute_metrics.py tests/test_metrics_compute.py tests/fixtures/normalized_min.json
git commit -m "feat(metrics): step 3 manifest-driven QUR/FRR, model-named columns, summary.csv"
```

---

## Phase 4 — Orchestration

### Task 9: Rewire `run_vqa_analysis.sh`

**Files:**
- Modify: `scripts/slurm/run_vqa_analysis.sh:51-103`

- [ ] **Step 1: Replace the eval/metrics/copy-back section**

Replace lines 51-103 (from `ZS_CONFIG=...` through `echo "Results copied to $DEST"`) with:
```bash
ZS_CONFIG="VQA_analysis/config_zeroshot.json"
FS_CONFIG="VQA_analysis/config_fewshot.json"

# One run_id for the whole job; the evaluator + metrics steps all write under it.
N="${SPLIT##*_}"                       # e.g. val_300 -> 300
SPLIT_NAME="${SPLIT%_*}"               # e.g. val_300 -> val
export VQA_RUN_ID="eval_${SPLIT_NAME}_${N}_$(date +%Y%m%d_%H%M%S)"
export VQA_CONFIG_PATH="$FS_CONFIG"
echo "Run id: $VQA_RUN_ID"

# ---- Evaluate each dataset (QUR+FRR in one pass via --questions both) ----
for D in "${DATASETS[@]}"; do
    printf "\n\n########## DATASET: %s  (split=%s) ##########\n" "$D" "$SPLIT"
    uv run python -c "
import json, sys
dataset, split = sys.argv[1], sys.argv[2]
input_file = f'/home/amartinelli/VRD-UQA/data/{dataset}/{dataset}_{split}/{dataset}_unanswerable_corrupted_questions_just_false.json'
for path in ['VQA_analysis/config_zeroshot.json', 'VQA_analysis/config_fewshot.json']:
    cfg = json.load(open(path))
    cfg['dataset'] = dataset
    cfg['split'] = split.split('_')[0]
    cfg['input_file'] = input_file
    json.dump(cfg, open(path, 'w'), indent=4)
" "$D" "$SPLIT"

    printf "\n=== QWEN2.5 — FEW-SHOT FINETUNED — QUR+FRR — %s ===\n" "$D"
    uv run python VQA_analysis/evaluators/qwen2.5_evaluator.py --config_path "$FS_CONFIG" --finetuned --questions both
done

# ---- Metrics (all operate on $VQA_RUN_ID) ----
uv run python VQA_analysis/metrics/1_normalize_unanswerable_responses.py --run-id "$VQA_RUN_ID"
uv run python VQA_analysis/metrics/2_enrich_metadata.py            --run-id "$VQA_RUN_ID"
uv run python VQA_analysis/metrics/3_compute_metrics.py            --run-id "$VQA_RUN_ID"

# ---- Run manifest + latest symlink ----
uv run python -c "
import json
from config import run_layout as rl
run_id = '$VQA_RUN_ID'
datasets = '${DATASETS[*]}'.split()
run = rl.run_dir(run_id)
configs = sorted({leaf.name for ds in datasets for leaf in (run / ds).glob('*') if leaf.is_dir()}) if run.exists() else []
rl.write_manifest(run / 'run_manifest.json', {
    'run_id': run_id, 'created_at': rl.utc_now_iso(),
    'git_commit': rl.git_commit(), 'git_dirty': rl.git_dirty(),
    'split': '${SPLIT_NAME}', 'n': int('${N}'), 'seed': 42,
    'datasets': datasets, 'configs': configs,
})
rl.update_latest_symlink(run_id)
print('Wrote run_manifest + latest ->', run_id)
"

mv $HOME/slurm* $HOME/VRD-UQA/ 2>/dev/null || true

# Copy the clean run tree back to $HOME (1:1 — no reshaping).
SRC="$WORK_DIR/artifacts/evaluation_runs/$VQA_RUN_ID"
DEST="$HOME/VRD-UQA/artifacts/evaluation_runs/$VQA_RUN_ID"
if [ -d "$SRC" ]; then
    mkdir -p "$DEST"
    cp -r "$SRC/." "$DEST/"
    ln -sfn "$VQA_RUN_ID" "$HOME/VRD-UQA/artifacts/evaluation_runs/latest"
    echo "Results copied to $DEST"
else
    echo "WARNING: no run dir produced at $SRC"
fi
```

- [ ] **Step 2: Shellcheck-style sanity parse**

Run: `bash -n scripts/slurm/run_vqa_analysis.sh`
Expected: no output (syntax OK).

- [ ] **Step 3: Commit**

```bash
git add scripts/slurm/run_vqa_analysis.sh
git commit -m "feat(orchestration): single run_id, run_manifest, latest symlink, 1:1 copy-back"
```

---

### Task 10: End-to-end mock pipeline integration test

**Files:**
- Test: `tests/test_pipeline_integration.py`

- [ ] **Step 1: Write the test (CPU-only, mock eval + `--no-model` normalize)**

```python
"""End-to-end mock pipeline. Run: uv run python -m tests.test_pipeline_integration"""
import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path
from config.paths import REPO_ROOT

SAMPLE = REPO_ROOT / "VQA_analysis" / "evaluation_files" / "BDocs_sample15.json"


def test_full_mock_run_produces_clean_tree():
    tmp = Path(tempfile.mkdtemp())
    run_root = tmp / "runs"
    run_id = "eval_val_15_20260101_010101"

    base = json.load(open(REPO_ROOT / "VQA_analysis" / "config_mock.json"))
    base.update({"dataset": "BDocs", "input_file": str(SAMPLE), "ocr_enabled": False,
                 "sampling_percentage": 100, "seed": 42, "split": "val",
                 "few_shot": {"enabled": False}})
    cfg = tmp / "cfg.json"
    cfg.write_text(json.dumps(base))

    env = dict(os.environ)
    env.update({"VQA_RUN_ID": run_id, "VQA_EVAL_RUNS_DIR": str(run_root)})

    def run(*cmd):
        subprocess.check_call(["uv", "run", "python", *cmd], cwd=str(REPO_ROOT), env=env)

    run("VQA_analysis/evaluators/qwen2.5_evaluator.py", "--config_path", str(cfg), "--finetuned", "--questions", "both")
    run("VQA_analysis/metrics/1_normalize_unanswerable_responses.py", "--run-id", run_id, "--no-model")
    run("VQA_analysis/metrics/2_enrich_metadata.py", "--run-id", run_id)
    run("VQA_analysis/metrics/3_compute_metrics.py", "--run-id", run_id)

    leaf = run_root / run_id / "BDocs" / "finetuned_noocr"
    assert (leaf / "predictions.json").is_file()
    assert (leaf / "manifest.json").is_file()
    assert (leaf / "_cache" / "normalized.json").is_file()
    assert not (leaf / "_cache" / "enriched.json").exists()  # no third heavy file
    assert (leaf / "metrics" / "QUR.csv").is_file()
    assert (leaf / "metrics" / "FRR.csv").is_file()

    norm = json.load(open(leaf / "_cache" / "normalized.json"))
    vr = norm["corrupted_questions"][0]["verification_result"]["vqa_results"][0]
    assert "answer_converted" in vr["answer_corrupted"][0]
    assert "answer_converted" in vr["answer_clean"][0]
    assert norm.get("_enriched") is True

    summary = list(csv.DictReader(open(run_root / run_id / "summary.csv")))
    metrics_seen = {r["metric"] for r in summary}
    assert {"QUR", "UR", "FRR"} <= metrics_seen
    assert all(set(r.keys()) >= {"dataset", "config", "label", "model", "metric", "complexity", "value"} for r in summary)


if __name__ == "__main__":
    test_full_mock_run_produces_clean_tree()
    print("OK: end-to-end mock pipeline")
```

- [ ] **Step 2: Run to verify it passes**

Run: `uv run python -m tests.test_pipeline_integration`
Expected: PASS — prints `OK: end-to-end mock pipeline`. (If it fails, fix the responsible step's task before continuing — this is the gate for Phase 5.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline_integration.py
git commit -m "test: end-to-end mock pipeline integration"
```

---

## Phase 5 — Notebook

### Task 11: Repoint paths + discover configs from manifests

**Files:**
- Modify: `VQA_analysis/notebooks/thesis_plots.ipynb` (cell 1, cell 3 loaders)

- [ ] **Step 1: Replace the Paths + loader block (cell 1)**

Replace the `# ── Paths ──` block and `load_csv` with:
```python
# ── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "/home/amartinelli/VRD-UQA/artifacts/evaluation_runs/latest"
OUTPUT_DIR  = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = ["DUDE", "MPDocVQA", "SlideVQA", "BDocs"]
DATASET_LABELS = {"DUDE": "DUDE", "MPDocVQA": "MPDocVQA",
                  "SlideVQA": "SlideVQA", "BDocs": "BoundingDocs"}

import json, glob

def discover_configs(dataset):
    """Return {config_slug: label} for a dataset by reading leaf manifests."""
    out = {}
    for man_path in sorted(glob.glob(os.path.join(RESULTS_DIR, dataset, "*", "manifest.json"))):
        man = json.load(open(man_path))
        out[man["config"]] = man.get("label", man["config"])
    return out

# Union of configs across datasets, stable order.
CONFIGS, CONFIG_LABELS = [], {}
for _ds in DATASETS:
    for _slug, _lab in discover_configs(_ds).items():
        if _slug not in CONFIG_LABELS:
            CONFIGS.append(_slug); CONFIG_LABELS[_slug] = _lab

# Color per config (stable hashing onto a palette).
_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
COLORS = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(CONFIGS)}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.linestyle": "--", "grid.alpha": 0.5,
})

def load_csv(dataset, config, filename):
    """Load a results CSV from <dataset>/<config>/metrics/<filename>."""
    path = os.path.join(RESULTS_DIR, dataset, config, "metrics", filename)
    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        return None
    df = pd.read_csv(path, index_col=0)
    df.columns = ["value"]
    return df["value"]
```

- [ ] **Step 2: Verify config discovery against the integration run**

Temporarily set `RESULTS_DIR` to the integration test's run root leaf parent (or run the real pipeline once). Then in a scratch cell run:
```python
print(CONFIGS, CONFIG_LABELS)
print(load_csv("BDocs", CONFIGS[0], "QUR.csv"))
```
Expected: `CONFIGS` non-empty, `QUR.csv` Series prints with `QUR`/`QUR_C1`... index, no `[MISSING]`.

- [ ] **Step 3: Commit**

```bash
git add VQA_analysis/notebooks/thesis_plots.ipynb
git commit -m "feat(notebook): read evaluation_runs/latest, discover configs from manifests"
```

---

### Task 12: Four new FRR plots

**Files:**
- Modify: `VQA_analysis/notebooks/thesis_plots.ipynb` (replace the FRR section, cells 31-33; append new cells)

- [ ] **Step 1: Replace the FRR data-load cell (cell 32) to use discovered finetuned config**

```python
# Find the fine-tuned config slug(s) present in the run.
FT_CONFIGS = [c for c in CONFIGS if c.startswith("finetuned")]
FT_CONFIG  = FT_CONFIGS[0] if FT_CONFIGS else None

frr_records = []
if FT_CONFIG:
    for dataset in DATASETS:
        s_frr = load_csv(dataset, FT_CONFIG, "FRR.csv")
        s_qur = load_csv(dataset, FT_CONFIG, "QUR.csv")
        if s_frr is None or s_qur is None:
            continue
        has_c3 = s_qur.get("QUR_C3", 0.0) != 0.0
        frr_records.append({
            "dataset": dataset,
            "QUR": s_qur.get("QUR", np.nan), "QUR_C1": s_qur.get("QUR_C1", np.nan),
            "QUR_C2": s_qur.get("QUR_C2", np.nan),
            "QUR_C3": s_qur.get("QUR_C3", np.nan) if has_c3 else np.nan,
            "FRR": s_frr.get("FRR", np.nan), "FRR_C1": s_frr.get("FRR_C1", np.nan),
            "FRR_C2": s_frr.get("FRR_C2", np.nan),
            "FRR_C3": s_frr.get("FRR_C3", np.nan) if has_c3 else np.nan,
        })
frr_df = pd.DataFrame(frr_records).set_index("dataset") if frr_records else pd.DataFrame()
frr_df
```

- [ ] **Step 2: Plot 5c stays (QUR vs FRR paired bars).** Keep the existing cell 33 as-is (it already reads `frr_df`).

- [ ] **Step 3: Append Plot 5d — FRR by complexity (new cell)**

```python
### Plot 5d — FRR by Complexity (is refusal level-dependent, or flat = collapse?)
if not frr_df.empty:
    COMP = [("FRR_C1", "C1"), ("FRR_C2", "C2"), ("FRR_C3", "C3")]
    x = np.arange(len(frr_df.index)); w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (col, lab) in enumerate(COMP):
        ax.bar(x + (i - 1) * w, frr_df[col].values, width=w, label=lab,
               edgecolor="white", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels([DATASET_LABELS[d] for d in frr_df.index])
    ax.set_ylabel("FRR (↓ better)"); ax.set_ylim(0, 1)
    ax.set_title("False Refusal Rate by Corruption Complexity — Fine-Tuned", fontweight="bold")
    ax.legend(title="Complexity")
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "plot5d_frr_by_complexity.pdf"), bbox_inches="tight"); plt.show()
```

- [ ] **Step 4: Append Plot 5e — answer-behavior breakdown (new cell)**

```python
### Plot 5e — Answer Behavior: refuse-everything check
if not frr_df.empty:
    fig, axes = plt.subplots(1, len(frr_df.index), figsize=(3.2 * len(frr_df.index), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, frr_df.index):
        r = frr_df.loc[dataset]
        # Clean bar: answered vs falsely refused; Corrupted bar: correctly refused vs wrongly answered.
        clean_ans, clean_ref = 1 - r["FRR"], r["FRR"]
        corr_ref, corr_ans  = r["QUR"], 1 - r["QUR"]
        ax.bar(["Clean", "Corrupted"], [clean_ans, corr_ref], color="#55A868", label="Answered / Correctly refused")
        ax.bar(["Clean", "Corrupted"], [clean_ref, corr_ans], bottom=[clean_ans, corr_ref],
               color="#C44E52", label="Falsely refused / Wrongly answered")
        ax.set_ylim(0, 1); ax.set_title(DATASET_LABELS[dataset], fontweight="bold")
    axes[0].set_ylabel("Fraction of questions")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Answer Behavior — Fine-Tuned (a refuse-everything model fills the Clean bar red)", fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 1]); plt.savefig(os.path.join(OUTPUT_DIR, "plot5e_answer_behavior.pdf"), bbox_inches="tight"); plt.show()
```

- [ ] **Step 5: Append Plot 5f — QUR↑ vs FRR↓ scatter with fine-tuning trajectory (new cell)**

```python
### Plot 5f — QUR vs FRR with zero-shot -> fine-tuned trajectory
ZS_CONFIG_SLUG = next((c for c in CONFIGS if c.startswith("zeroshot")), None)
fig, ax = plt.subplots(figsize=(6.5, 6))
MARKERS = {"DUDE": "o", "MPDocVQA": "s", "SlideVQA": "^", "BDocs": "D"}
for dataset in DATASETS:
    ft_q, ft_f = load_csv(dataset, FT_CONFIG, "QUR.csv"), load_csv(dataset, FT_CONFIG, "FRR.csv")
    if ft_q is None or ft_f is None:
        continue
    ax.scatter(ft_f["FRR"], ft_q["QUR"], marker=MARKERS.get(dataset, "o"),
               s=90, color="#55A868", edgecolor="black", zorder=3, label=f"{DATASET_LABELS[dataset]} (FT)")
    if ZS_CONFIG_SLUG:
        zq, zf = load_csv(dataset, ZS_CONFIG_SLUG, "QUR.csv"), load_csv(dataset, ZS_CONFIG_SLUG, "FRR.csv")
        if zq is not None and zf is not None:
            ax.annotate("", xy=(ft_f["FRR"], ft_q["QUR"]), xytext=(zf["FRR"], zq["QUR"]),
                        arrowprops=dict(arrowstyle="->", color="#999999", lw=1.3))
            ax.scatter(zf["FRR"], zq["QUR"], marker=MARKERS.get(dataset, "o"),
                       s=70, facecolor="white", edgecolor="#4C72B0", zorder=3)
ax.scatter([0], [1], marker="*", s=320, color="gold", edgecolor="black", zorder=4, label="Ideal")
ax.set_xlabel("FRR — false refusal on answerable (↓ better)")
ax.set_ylabel("QUR — correct refusal on unanswerable (↑ better)")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_title("Refusal Calibration: Zero-Shot → Fine-Tuned Trajectory", fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "plot5f_qur_frr_trajectory.pdf"), bbox_inches="tight"); plt.show()
```

- [ ] **Step 6: Append Plot 5g — balanced refusal accuracy (new cell)**

```python
### Plot 5g — Balanced Refusal Accuracy = (QUR + (1 - FRR)) / 2
if not frr_df.empty:
    bra = ((frr_df["QUR"] + (1 - frr_df["FRR"])) / 2).reindex(DATASETS).dropna()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([DATASET_LABELS[d] for d in bra.index], bra.values,
                  color="#55A868", edgecolor="white")
    for b, v in zip(bars, bra.values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(0.5, color="#C44E52", linestyle="--", lw=1, label="0.5 = refuse/answer everything")
    ax.axhline(1.0, color="#999999", linestyle=":", lw=1, label="1.0 = perfect discrimination")
    ax.set_ylim(0, 1.1); ax.set_ylabel("Balanced refusal accuracy")
    ax.set_title("Balanced Refusal Accuracy — Fine-Tuned", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "plot5g_balanced_refusal_accuracy.pdf"), bbox_inches="tight"); plt.show()
```

- [ ] **Step 7: Manual verification — Restart & Run All**

After a real pipeline run exists at `evaluation_runs/latest` (or point `RESULTS_DIR` at the integration run), Restart & Run All.
Expected: no `[MISSING]` lines for the configs present; plots 5c–5g render; PDFs appear under `output/`.

- [ ] **Step 8: Commit**

```bash
git add VQA_analysis/notebooks/thesis_plots.ipynb
git commit -m "feat(notebook): FRR plots — by complexity, answer behavior, trajectory, balanced accuracy"
```

---

## Self-Review

**1. Spec coverage:**
- §2.A dataset/config layout, no `LLM/`, no name-encoded metadata → Tasks 1,2,5 (paths/slug), 5 (save path). ✓
- §2.B single-pass QUR+FRR with `--questions` → Tasks 3 (arg), 4 (dual loop), 8 (manifest-driven split). ✓
- §2.C config slug encodes axes → Task 1 `build_slug`; mode via `derive_mode`. ✓
- §2.D raw + one cached derived file; no `enriched.json`; enrich in place → Tasks 6,7; integration asserts no `enriched.json`. ✓
- §2.E keep three numbered scripts → Tasks 6,7,8 modify the three files, kept runnable. ✓
- §3.1/§3.2 run + leaf manifests (incl. git_dirty, input_file, counts, pixels) → Task 5 manifest dict; Task 9 run_manifest. ✓
- §3.3 dual-answer predictions schema → Task 4; canonicalize both → Task 6. ✓
- §3.4 summary.csv tidy long-format → Tasks 2 (writer), 8 (rows). ✓
- §4.1 shared module → Tasks 1,2 (located at `config/run_layout.py`, deviation noted in header). ✓
- §4.2 seeding + run_id resolution from env → Tasks 3,5. ✓
- §4.3 three steps rewired, model-named CSV column, remove folder-name matching → Tasks 6,7,8. ✓
- §4.4 orchestration run_id/run_manifest/latest/1:1 copy → Task 9. ✓
- §4.5 notebook repoint + config discovery + 4 FRR plots → Tasks 11,12 (5d,5e,5f,5g; 5c retained). ✓
- §6 risk "dual loop doubles inference" → `--questions` opt-out (Task 3); few-shot built once per item (Task 4). ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output. Notebook/shell verification steps are explicit manual runs with stated expectations (acceptable — these artifacts are not unit-testable in this repo).

**3. Type/name consistency:** `make_run_id/derive_mode/build_slug/human_label/leaf_dir/run_dir/read_manifest/write_manifest/git_commit/git_dirty/append_summary_rows/update_latest_symlink/EVAL_RUNS_DIR/utc_now_iso` — defined in Tasks 1-2, used identically in Tasks 5,8,9 and tests. `QwenVQAEvaluator(config_path, finetuned, questions=...)` — Task 3 signature matches Task-4/5 tests. `VQAAnalyzer(..., side=...)` and instance `_get_answers(self, res)` — Task 8 definition matches Task-8 tests. `canonicalize_answer(answer, classify_fn)`, `label_vqa_answers(..., classify_fn=None)`, `enrich_file(path)`, `_enrich_data(data)` — defined and used consistently. Manifest keys (`config`, `model_name`, `questions`, `run_id`, `ocr_enabled`, `window_size`) align across producer (Task 5) and consumers (Tasks 8,11,12). ✓

No gaps found.
