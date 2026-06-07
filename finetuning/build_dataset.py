from __future__ import annotations

import json
import random
from pathlib import Path
from tqdm import tqdm

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

def build_split(sources: dict[str, Path], rng: random.Random, split_type: str) -> list[dict]:
    all_examples: list[dict] = []
    for dataset_name, corrupted_questions_file_path in tqdm(sources.items(), desc="Datasets", unit="dataset"):
        corrupted_questions_file = json.loads(corrupted_questions_file_path.read_text())
        corrupted_entries = corrupted_questions_file["corrupted_questions"]

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

        # CLEAN QUESTIONS — exclude original questions already used in the corrupted set
        original_questions = [e["original_question"] for e in corrupted_entries]
        dataset_split = get_dataset_split(split_type=split_type)
        sampled = sample_dataset_different_from(
            dataset_split, num_questions=len(corrupted_entries), exclude_questions=original_questions
        )
        sampled = standardize_for_corruption_pipeline(sampled, split_type)
        for entry in tqdm(sampled["data"], desc=f"  {dataset_name} clean", leave=False):
            all_examples.extend(build_clean_sample(entry))

        # CORRUPTED QUESTIONS
        ok, skipped = 0, 0
        for corrupted_entry in tqdm(corrupted_entries, desc=f"  {dataset_name} corrupted", leave=False):
            corrupted_sample = build_corrupted_sample(corrupted_entry)
            if corrupted_sample is None:
                skipped += 1
                continue
            all_examples.extend(corrupted_sample)
            ok += 1
        print(f"  {dataset_name}: {ok} usable / {skipped} skipped")

    rng.shuffle(all_examples)
    return all_examples


def main() -> None:
    out_dir = REPO_ROOT / "finetuning" / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    rng = random.Random(seed)

    print("Building train split...")
    train = build_split(TRAIN_SOURCES, rng, split_type="train")
    train_path = out_dir / "train.json"
    train_path.write_text(json.dumps(train, indent=2, ensure_ascii=False))
    print(f"  -> {len(train)} examples written to {train_path}\n")

    print("Building val split...")
    val = build_split(VAL_SOURCES, rng, split_type="val")
    val_path = out_dir / "val.json"
    val_path.write_text(json.dumps(val, indent=2, ensure_ascii=False))
    print(f"  -> {len(val)} examples written to {val_path}")


if __name__ == "__main__":
    main()
