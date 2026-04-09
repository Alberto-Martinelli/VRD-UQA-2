from datasets import load_dataset

dataset = load_dataset("jordyvl/DUDE_loader", split="train[:100]")

print(dataset[0])      # first example
print(dataset.column_names)
len(dataset)           # should be 100