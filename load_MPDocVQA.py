import json
import random
import os

SOURCE_DIR = "/home/amartinelli/MPDocVQA/MPDocVQA_complete/qas"


def load_MPDocVQA(split_type: str, max_questions: int = 300):
    source = os.path.join(SOURCE_DIR, f"{split_type}.json")
    with open(source) as f:
        full = json.load(f)

    print(f"Total questions in {split_type} split: {len(full['data'])}")
    print("First example:\n", full["data"][0])

    random.seed(42)
    sampled = random.sample(full["data"], min(max_questions, len(full["data"])))

    output = {**full, "data": sampled}
    with open(f"mpdocvqa_{split_type}.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(sampled)} questions to mpdocvqa_{split_type}.json")


if __name__ == "__main__":
    load_MPDocVQA("val", max_questions=300)
