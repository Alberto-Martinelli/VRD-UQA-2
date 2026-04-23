from datasets import load_dataset
import json

dataset = load_dataset("NTT-hil-insight/SlideVQA", split="train")

print("First example:\n", dataset[0])      # first example
print("\nColumn names:\n", dataset.column_names)
print("\nDataset length:\n", len(dataset))

small_dataset = dataset.select(range(18))

# Create the wrapper structure
output_data = {
    "dataset_name": "SlideVQA Dataset",
    "data": list(small_dataset)
}

# Save the wrapped data to your JSON file
with open("slidevqa_train.json", "w") as f:
    json.dump(output_data, f, indent=4)
