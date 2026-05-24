import os
import json
from datasets import load_dataset

OUT_DIR = "data/DUDE/DUDE_val_300/qas"


def load_DUDE(split_type: str, max_questions: int = 300, offset: int = 0, out_dir: str = OUT_DIR):
    dataset = load_dataset("jordyvl/DUDE_loader", split=split_type, trust_remote_code=True)

    print("First example:\n", dataset[0])
    print("\nColumn names:\n", dataset.column_names)
    print("\nDataset length:\n", len(dataset))

    small_dataset = dataset.shuffle(seed=42).select(range(offset, offset + max_questions))

    output_data = {
        "dataset_name": "DUDE",
        "data": list(small_dataset)
    }

    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"{split_type}.json")
    with open(out_json, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"\nSaved {len(small_dataset)} questions to {out_json}")


if __name__ == "__main__":
    split = "val"
    corrupted_questions_desired = 250
    questions_to_process = 3 * corrupted_questions_desired
    offset = 0
    load_DUDE(split, max_questions=questions_to_process, offset=offset, out_dir=f"data/DUDE/DUDE_{split}_{corrupted_questions_desired}/qas")
