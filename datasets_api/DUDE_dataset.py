from datasets import load_dataset
from datasets_api.datasets_utils import save_sample

DUDE_IMAGE_DIR = ""

def get_DUDE_image_dir():
    return DUDE_IMAGE_DIR

def get_DUDE_split(split_type: str):
    dataset_split = load_dataset("jordyvl/DUDE_loader", split=split_type, trust_remote_code=True)

    print("First example:\n", dataset_split[0])
    print("\nColumn names:\n", dataset_split.column_names)
    print("\nDataset length:\n", len(dataset_split))
    return dataset_split

def sample_DUDE(dataset_split, num_questions: int, offset: int = 0, shuffle: bool = True):
    if shuffle:
        sampled = dataset_split.shuffle(seed=42).select(range(offset, offset + num_questions))
    else:
        sampled = dataset_split.select(range(offset, offset + num_questions))
    
    output_wrapper = {
        "dataset_name": "DUDE",
        "data": sampled
    }
    return output_wrapper

def sample_DUDE_different_from(dataset_split, num_questions: int, exclude_questions: list[str], offset: int = 0, shuffle: bool = True):
    exclude_set = set(exclude_questions)
    filtered = dataset_split.filter(lambda item: item["question"] not in exclude_set)
    if shuffle:
        sampled = filtered.shuffle(seed=42).select(range(offset, min(offset + num_questions, len(filtered))))
    else:
        sampled = filtered.select(range(offset, min(offset + num_questions, len(filtered))))

    output_wrapper = {
        "dataset_name": "DUDE",
        "data": sampled
    }
    return output_wrapper

if __name__ == "__main__":
    split = "train"
    num_questions = 10
    dude_split = get_DUDE_split(split)
    sample_dude = sample_DUDE(dude_split, num_questions)
    save_sample("DUDE", split, num_questions, sample_dude)