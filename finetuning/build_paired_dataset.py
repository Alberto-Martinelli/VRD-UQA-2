"""Build finetuning datasets (train.json and val.json) from the four VRD-UQA datasets.

Sources
-------
Train : *_train_750 folders — 750 entries per dataset, all used → 6 000 examples
Val   : *_val_250  folders — 250 entries per dataset, all used → 2 000 examples

Each source entry yields two examples:
  - Corrupted : input = corrupted_question,  output = refusal string
  - Gold      : input = original_question,   output = original_answer_locations[0]['answer']
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

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
    """Convert relative ./data/... paths to absolute; leave absolute paths unchanged."""
    if raw.startswith("./"):
        return str((REPO_ROOT / raw[2:]).resolve())
    return raw


def build_examples_from_entry(entry: dict) -> list[dict] | None:
    """Return [corrupted_example, gold_example] or None if the entry is unusable."""
    # Image path — new format stores it directly in verification_result
    vr = entry.get("verification_result")
    if not isinstance(vr, dict):
        return None
    image_path = vr.get("image_path")
    if not image_path:
        return None
    image_path = resolve_image_path(image_path)

    # Questions
    corrupted_q = entry.get("corrupted_question")
    original_q = entry.get("original_question")
    if not corrupted_q or not original_q:
        return None

    # Corrupted entity text for the refusal string
    corrupted_entities = entry.get("corrupted_entities") or []
    if not corrupted_entities:
        return None
    corrupted_text = corrupted_entities[0].get("text")
    if not corrupted_text:
        return None

    # Gold answer
    answer_locations = entry.get("original_answer_locations") or []
    if not answer_locations:
        return None
    answer = answer_locations[0].get("answer")
    if answer is None:
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
        },
        {
            "instruction": INSTRUCTION,
            "input": f"<image>\n{original_q}",
            "output": str(answer),
            "images": [image_path],
        },
    ]


def build_split(sources: dict[str, Path], rng: random.Random) -> list[dict]:
    """Convert all entries from every source file into a shuffled example list."""
    all_examples: list[dict] = []
    for name, path in sources.items():
        data = json.loads(path.read_text())
        entries = data["corrupted_questions"]
        ok, skipped = 0, 0
        for entry in entries:
            pair = build_examples_from_entry(entry)
            if pair is None:
                skipped += 1
                continue
            all_examples.extend(pair)
            ok += 1
        print(f"  {name}: {ok} usable / {skipped} skipped → {ok * 2} examples")
    rng.shuffle(all_examples)
    return all_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("Building train split...")
    train = build_split(TRAIN_SOURCES, rng)
    train_path = args.out_dir / "train.json"
    train_path.write_text(json.dumps(train, indent=2, ensure_ascii=False))
    print(f"  → {len(train)} examples written to {train_path}\n")

    print("Building val split...")
    val = build_split(VAL_SOURCES, rng)
    val_path = args.out_dir / "val.json"
    val_path.write_text(json.dumps(val, indent=2, ensure_ascii=False))
    print(f"  → {len(val)} examples written to {val_path}")


if __name__ == "__main__":
    main()
