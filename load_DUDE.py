from datasets import load_dataset

dataset = load_dataset("jordyvl/DUDE_loader", split="train", trust_remote_code=True)

print("First example:\n", dataset[0])      # first example
print("\nColumn names:\n", dataset.column_names)
print("\nDataset length:\n", len(dataset))

import json

small_dataset = dataset.select(range(18))

# Create the wrapper structure
output_data = {
    "dataset_name": "DUDE",
    "data": list(small_dataset)
}

# Save the wrapped data to your JSON file
with open("dude_train.json", "w") as f:
    json.dump(output_data, f, indent=4)