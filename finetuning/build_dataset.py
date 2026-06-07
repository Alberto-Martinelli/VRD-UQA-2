from __future__ import annotations

import json
import random
import time
from pathlib import Path

import datasets

# Silence HuggingFace dataset progress bars (Filter/Map) — in a non-TTY SLURM
# log they emit hundreds of redraw lines and bury the meaningful output.
try:
    datasets.disable_progress_bars()
except AttributeError:  # older datasets versions
    from datasets.utils.logging import disable_progress_bar

    disable_progress_bar()

from datasets_api.BDocs_dataset import get_BDocs_split, sample_BDocs_different_from, standardize_BDocs_for_corruption_pipeline
from datasets_api.DUDE_dataset import get_DUDE_split, sample_DUDE_different_from, standardize_DUDE_for_corruption_pipeline
from datasets_api.MPDocVQA_dataset import get_MPDocVQA_split, sample_MPDocVQA_different_from, standardize_MPDocVQA_for_corruption_pipeline
from datasets_api.SlideVQA_dataset import get_SlideVQA_split, sample_SlideVQA_different_from, standardize_SlideVQA_for_corruption_pipeline
from config.paths import REPO_ROOT

TRAIN_SOURCES = {
    "MPDocVQA": REPO_ROOT / "data/MPDocVQA/MPDocVQA_train_750/MPDocVQA_unanswerable_corrupted_questions_just_false.json",
    "DUDE":     REPO_ROOT / "data/DUDE/DUDE_train_750/DUDE_unanswerable_corrupted_questions_just_false.json",
    "SlideVQA": REPO_ROOT / "data/SlideVQA/SlideVQA_train_750/SlideVQA_unanswerable_corrupted_questions_just_false.json",
    "BDocs":    REPO_ROOT / "data/BDocs/BDocs_train_750/BDocs_unanswerable_corrupted_questions_just_false.json",
}

VAL_SOURCES = {
    "MPDocVQA": REPO_ROOT / "data/MPDocVQA/MPDocVQA_val_250/MPDocVQA_unanswerable_corrupted_questions_just_false.json",
    "DUDE":     REPO_ROOT / "data/DUDE/DUDE_val_250/DUDE_unanswerable_corrupted_questions_just_false.json",
    "SlideVQA": REPO_ROOT / "data/SlideVQA/SlideVQA_val_250/SlideVQA_unanswerable_corrupted_questions_just_false.json",
    "BDocs":    REPO_ROOT / "data/BDocs/BDocs_val_250/BDocs_unanswerable_corrupted_questions_just_false.json",
}

INSTRUCTION = "Analyze the image and answer the question precisely."


def log(msg: str = "") -> None:
    """Print with an explicit flush so lines appear in SLURM logs immediately."""
    print(msg, flush=True)


def resolve_image_path(raw: str) -> str:
    if raw.startswith("./"):
        return str((REPO_ROOT / raw[2:]).resolve())
    return raw

def build_corrupted_sample(entry: dict) -> list[dict] | None:
    vr = entry.get("verification_result")
    if not isinstance(vr, dict):
        return None
    image_path = vr.get("image_path")
    if not image_path:
        return None
    image_path = resolve_image_path(image_path)

    corrupted_q = entry.get("corrupted_question")
    if not corrupted_q:
        return None

    corrupted_entities = entry.get("corrupted_entities") or []
    if not corrupted_entities:
        return None
    corrupted_text = corrupted_entities[0].get("text")
    if not corrupted_text:
        return None

    refusal = (
        f"Unable to determine. The term '{corrupted_text}' refers to a "
        f"corrupted entity not present as a data point in this context."
    )

    return [
        {
            "instruction": INSTRUCTION,
            "input": f"<image>\n{corrupted_q}",
            "output": refusal,
            "images": [image_path],
        }
    ]

def build_clean_sample(entry: dict) -> list[dict]:
    """Format a standardized entry (shared schema across all 4 datasets).
    Expects: 'question' (str), 'answers' (list or str), 'document' (list), 'answer_page_idx' (int).
    """
    answers = entry["answers"]
    answer = answers[0] if isinstance(answers, list) else str(answers)
    doc = entry["document"]
    if not doc:
        return []
    page_idx = min(entry["answer_page_idx"], len(doc) - 1)
    image_path = doc[page_idx]
    return [
        {
            "instruction": INSTRUCTION,
            "input": f"<image>\n{entry['question']}",
            "output": answer,
            "images": [image_path],
        }
    ]

def build_split(sources: dict[str, Path], rng: random.Random, split_type: str) -> tuple[list[dict], dict]:
    """Build one split. Returns (examples, per-dataset stats).

    stats maps dataset_name -> {"clean": int, "corrupted": int}.
    """
    all_examples: list[dict] = []
    stats: dict[str, dict[str, int]] = {}

    for dataset_name, corrupted_questions_file_path in sources.items():
        log(f"\n[{split_type}] ----- {dataset_name} -----")
        corrupted_questions_file = json.loads(corrupted_questions_file_path.read_text())
        corrupted_entries = corrupted_questions_file["corrupted_questions"]
        log(f"[{split_type}] {dataset_name}: {len(corrupted_entries)} corrupted questions in source file")

        if dataset_name == "MPDocVQA":
            get_dataset_split = get_MPDocVQA_split
            sample_dataset_different_from = sample_MPDocVQA_different_from
            standardize_for_corruption_pipeline = standardize_MPDocVQA_for_corruption_pipeline
        elif dataset_name == "DUDE":
            get_dataset_split = get_DUDE_split
            sample_dataset_different_from = sample_DUDE_different_from
            standardize_for_corruption_pipeline = standardize_DUDE_for_corruption_pipeline
        elif dataset_name == "SlideVQA":
            get_dataset_split = get_SlideVQA_split
            sample_dataset_different_from = sample_SlideVQA_different_from
            standardize_for_corruption_pipeline = standardize_SlideVQA_for_corruption_pipeline
        elif dataset_name == "BDocs":
            get_dataset_split = get_BDocs_split
            sample_dataset_different_from = sample_BDocs_different_from
            standardize_for_corruption_pipeline = standardize_BDocs_for_corruption_pipeline
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        # CLEAN QUESTIONS — sample gold pairs, excluding originals already used for corruption
        original_questions = [e["original_question"] for e in corrupted_entries]
        dataset_split = get_dataset_split(split_type=split_type)
        sampled = sample_dataset_different_from(
            dataset_split, num_questions=len(corrupted_entries), exclude_questions=original_questions
        )
        # Clean gold examples don't need bounding boxes; relaxing this for DUDE
        # avoids dropping ~55% of its sampled questions.
        if dataset_name == "DUDE":
            sampled = standardize_DUDE_for_corruption_pipeline(sampled, split_type, require_bbox=False)
        else:
            sampled = standardize_for_corruption_pipeline(sampled, split_type)

        clean_count, clean_dropped = 0, 0
        for entry in sampled["data"]:
            built = build_clean_sample(entry)
            if built:
                all_examples.extend(built)
                clean_count += len(built)
            else:
                clean_dropped += 1
        log(f"[{split_type}] {dataset_name}: {clean_count} clean examples kept ({clean_dropped} dropped — no image/pages)")

        # CORRUPTED QUESTIONS — targets a structured refusal string
        corrupted_count, skipped = 0, 0
        for corrupted_entry in corrupted_entries:
            corrupted_sample = build_corrupted_sample(corrupted_entry)
            if corrupted_sample is None:
                skipped += 1
                continue
            all_examples.extend(corrupted_sample)
            corrupted_count += len(corrupted_sample)
        log(f"[{split_type}] {dataset_name}: {corrupted_count} corrupted examples kept ({skipped} skipped — missing fields)")

        stats[dataset_name] = {"clean": clean_count, "corrupted": corrupted_count}

    rng.shuffle(all_examples)
    return all_examples, stats


def print_recap(split_name: str, examples: list[dict], stats: dict) -> None:
    """Print a per-dataset breakdown of the produced artifact (clean vs corrupted)."""
    width = 60
    log("\n" + "=" * width)
    log(f"RECAP — {split_name} split")
    log("=" * width)
    log(f"{'Dataset':<14}{'Clean':>9}{'Corrupted':>12}{'Total':>9}{'% of split':>12}")
    log("-" * width)

    total = len(examples)
    sum_clean = sum_corrupted = 0
    for name, s in stats.items():
        clean, corrupted = s["clean"], s["corrupted"]
        ds_total = clean + corrupted
        sum_clean += clean
        sum_corrupted += corrupted
        pct = f"{ds_total / total * 100:.1f}%" if total else "0.0%"
        log(f"{name:<14}{clean:>9}{corrupted:>12}{ds_total:>9}{pct:>12}")

    log("-" * width)
    grand = sum_clean + sum_corrupted
    log(f"{'TOTAL':<14}{sum_clean:>9}{sum_corrupted:>12}{grand:>9}{'100.0%':>12}")
    if grand:
        log(f"\nBalance: {sum_clean / grand * 100:.1f}% clean / {sum_corrupted / grand * 100:.1f}% corrupted")
    log(f"Examples in produced artifact: {total}")
    if total != grand:
        log(f"WARNING: artifact size ({total}) != summed dataset contributions ({grand})")
    log("=" * width)


def main() -> None:
    out_dir = REPO_ROOT / "artifacts" / "finetuning" / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    rng = random.Random(seed)

    log("#" * 60)
    log("Building TRAIN split")
    log("#" * 60)
    t0 = time.time()
    train, train_stats = build_split(TRAIN_SOURCES, rng, split_type="train")
    train_path = out_dir / "train.json"
    train_path.write_text(json.dumps(train, indent=2, ensure_ascii=False))
    log(f"\nWrote {len(train)} examples -> {train_path}  ({time.time() - t0:.1f}s)")
    print_recap("TRAIN", train, train_stats)

    log("\n\n" + "#" * 60)
    log("Building VAL split")
    log("#" * 60)
    t1 = time.time()
    val, val_stats = build_split(VAL_SOURCES, rng, split_type="val")
    val_path = out_dir / "val.json"
    val_path.write_text(json.dumps(val, indent=2, ensure_ascii=False))
    log(f"\nWrote {len(val)} examples -> {val_path}  ({time.time() - t1:.1f}s)")
    print_recap("VAL", val, val_stats)

    log(f"\nAll done in {time.time() - t0:.1f}s total.")


if __name__ == "__main__":
    main()
