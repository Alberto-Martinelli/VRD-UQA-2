from datasets import load_dataset

def load_DUDE(split_type: str, max_questions: int = 300):
    dataset = load_dataset("jordyvl/DUDE_loader", split=split_type, trust_remote_code=True)

    print("First example:\n", dataset[0])      # first example
    print("\nColumn names:\n", dataset.column_names)
    print("\nDataset length:\n", len(dataset))

    import json

    small_dataset = dataset.shuffle(seed=42).select(range(max_questions))

    # Create the wrapper structure
    output_data = {
        "dataset_name": "DUDE",
        "data": list(small_dataset)
    }

    # Save the wrapped data to your JSON file
    with open(f"dude_{split_type}.json", "w") as f:
        json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    load_DUDE("val", max_questions=300)