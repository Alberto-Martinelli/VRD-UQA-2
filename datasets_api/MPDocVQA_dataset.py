import os
import json
import random
from datasets_api.datasets_utils import save_sample

SOURCE_DIR = "/home/amartinelli/MPDocVQA/MPDocVQA_complete/qas"
MPDOCVQA_IMAGE_DIR = "/home/amartinelli/MPDocVQA/MPDocVQA_complete/qas/images"

def get_MPDocVQA_image_dir():
    return MPDOCVQA_IMAGE_DIR

def get_MPDocVQA_split(split_type: str, shuffle: bool = True):
    source = os.path.join(SOURCE_DIR, f"{split_type}.json")
    with open(source) as f:
        full = json.load(f)

    print(f"Total questions in {split_type} split: {len(full['data'])}")

    if shuffle:
        # Shuffle in-place
        random.seed(42)
        random.shuffle(full["data"])
        return full["data"]
    else:
        return full["data"]

def sample_MPDocVQA(mpdocvqa_full_data, num_questions: int, offset: int = 0):
    sampled = mpdocvqa_full_data[offset : min(offset + num_questions, len(mpdocvqa_full_data))]
    output_wrapper = {
        "dataset_name": "MPDocVQA",
        "data": sampled
    }
    return output_wrapper

def sample_MPDocVQA_different_from(mpdocvqa_full_data, num_questions: int, exclude_questions: list[str], offset: int = 0):
    exclude_set = set(exclude_questions)
    filtered = [item for item in mpdocvqa_full_data if item["question"] not in exclude_set]
    sampled = filtered[offset : min(offset + num_questions, len(filtered))]
    
    output_wrapper = {
        "dataset_name": "MPDocVQA",
        "data": sampled
    }
    return output_wrapper

if __name__ == "__main__":
    split = "train"
    num_questions = 10
    mpdocvqa_full_data = get_MPDocVQA_split(split)
    sample_mpdocvqa = sample_MPDocVQA(mpdocvqa_full_data, num_questions)
    save_sample("MPDocVQA", split, num_questions, sample_mpdocvqa)