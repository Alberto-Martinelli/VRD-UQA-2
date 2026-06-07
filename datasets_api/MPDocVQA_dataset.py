import os
import json
import random
from datasets_api.datasets_utils import save_sample
from config import paths
import pandas as pd


def get_MPDocVQA_image_dir():
    return paths.image_dir("MPDocVQA")

def get_MPDocVQA_split(split_type: str, shuffle: bool = True):
    source = os.path.join(str(paths.MPDOCVQA_SOURCE_QAS), f"{split_type}.json")
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
        "base_image_dir": get_MPDocVQA_image_dir(),
        "data": sampled
    }
    return output_wrapper

def sample_MPDocVQA_different_from(mpdocvqa_full_data, num_questions: int, exclude_questions: list[str], offset: int = 0):
    exclude_set = set(exclude_questions)
    filtered = [item for item in mpdocvqa_full_data if item["question"] not in exclude_set]
    sampled = filtered[offset : min(offset + num_questions, len(filtered))]
    
    output_wrapper = {
        "dataset_name": "MPDocVQA",
        "base_image_dir": get_MPDocVQA_image_dir(),
        "data": sampled
    }
    return output_wrapper

def standardize_MPDocVQA_for_corruption_pipeline(data, split_type):
    df = pd.DataFrame(data["data"])
    df = df.rename(columns={"doc_id": "docId"})
    df["questionId"] = df["questionId"].astype(str)
    # The returned dataframe must contain a field 'document' with absolute image paths
    df["document"] = df["page_ids"].apply(
        lambda x: [
            os.path.join(get_MPDocVQA_image_dir(), f"{page_id}.jpg") for page_id in x
        ]
    )

    data["data"] = df.to_dict(orient="records")
    return data


if __name__ == "__main__":
    split = "train"
    num_questions = 10
    mpdocvqa_full_data = get_MPDocVQA_split(split)
    sample_mpdocvqa = sample_MPDocVQA(mpdocvqa_full_data, num_questions)
    sample_mpdocvqa = standardize_MPDocVQA_for_corruption_pipeline(sample_mpdocvqa, split)
    save_sample("MPDocVQA", split, num_questions, sample_mpdocvqa)