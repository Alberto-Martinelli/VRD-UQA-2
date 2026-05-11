"""Build a small finetuning dataset (train.json) from the four VRD-UQA datasets.

For each of the 4 source datasets we sample 50 source entries. Every source entry
yields two training examples:
  - "Corrupted": input is the corrupted_question, output is a refusal string.
  - "Gold": input is the original_question, output is the first original answer.

Final dataset: 4 * 50 * 2 = 400 shuffled examples written to train.json.
"""

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"

DATASETS = {
    "MPDocVQA": REPO_ROOT / "data/MPDocVQA/MPDocVQA_reduced/MPDocVQA_unanswerable_corrupted_questions_cleaned.json",
    "DUDE":     REPO_ROOT / "data/DUDE/DUDE_reduced/DUDE_unanswerable_corrupted_questions_cleaned.json",
    "SlideVQA": REPO_ROOT / "data/SlideVQA/SlideVQA_reduced/SlideVQA_unanswerable_corrupted_questions_cleaned.json",
    "BDocs":    REPO_ROOT / "data/BDocs/BDocs_reduced/BDocs_unanswerable_corrupted_questions_cleaned.json",
}

INSTRUCTION = "Analyze the image and answer the question precisely."
ENTRIES_PER_DATASET = 50  # produces 100 examples (50 corrupted + 50 gold) per dataset


def get_image_path(entry: dict, page_id: str) -> str | None:
    pages = entry.get("layout_analysis", {}).get("pages", {})
    page = pages.get(page_id)
    if not page:
        return None
    raw = page.get("image_path")
    if not raw:
        return None
    if raw.startswith("./"):
        return str((DATA_ROOT / raw[2:]).resolve())
    return raw


def build_examples_from_entry(entry: dict) -> list[dict] | None:
    """Return [corrupted_example, gold_example] or None if the entry is unusable."""
    answer_locations = entry.get("original_answer_locations") or []
    if not answer_locations:
        return None

    first_loc = answer_locations[0]
    page_id = first_loc.get("page_id")
    answer = first_loc.get("answer")
    if not page_id or answer is None:
        return None

    image_path = get_image_path(entry, page_id)
    if not image_path:
        return None

    corrupted_q = entry.get("corrupted_question")
    original_q = entry.get("original_question")
    if not corrupted_q or not original_q:
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

    corrupted_example = {
        "instruction": INSTRUCTION,
        "input": f"<image>\n{corrupted_q}",
        "output": refusal,
        "images": [image_path],
    }
    gold_example = {
        "instruction": INSTRUCTION,
        "input": f"<image>\n{original_q}",
        "output": str(answer),
        "images": [image_path],
    }
    return [corrupted_example, gold_example]


def sample_dataset(path: Path, n_entries: int, rng: random.Random) -> list[dict]:
    data = json.loads(path.read_text())
    entries = data["corrupted_questions"]
    rng.shuffle(entries)

    examples: list[dict] = []
    used = 0
    for entry in entries:
        if used >= n_entries:
            break
        pair = build_examples_from_entry(entry)
        if pair is None:
            continue
        examples.extend(pair)
        used += 1

    if used < n_entries:
        raise RuntimeError(f"Only found {used} usable entries in {path.name}, needed {n_entries}.")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "train.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--entries-per-dataset", type=int, default=ENTRIES_PER_DATASET)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    all_examples: list[dict] = []
    for name, path in DATASETS.items():
        print(f"Sampling {args.entries_per_dataset} entries from {name} ...")
        examples = sample_dataset(path, args.entries_per_dataset, rng)
        print(f"  -> produced {len(examples)} examples")
        all_examples.extend(examples)

    rng.shuffle(all_examples)
    args.output.write_text(json.dumps(all_examples, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(all_examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
