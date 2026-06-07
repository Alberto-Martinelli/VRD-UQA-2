from datasets import load_dataset
import os
from datasets_api.datasets_utils import save_sample
from config import paths
import pandas as pd


def get_SlideVQA_image_dir():
    return paths.image_dir("SlideVQA")

def get_SlideVQA_split(split_type: str):
    print(f"[SlideVQA] Loading '{split_type}' split from HuggingFace (NTT-hil-insight/SlideVQA)...", flush=True)
    dataset_split = load_dataset("NTT-hil-insight/SlideVQA", split=split_type, trust_remote_code=True)
    print(f"[SlideVQA] Loaded {len(dataset_split)} raw questions.", flush=True)
    return dataset_split

def _process_samples(sampled_data, image_dir):
    total = len(sampled_data)
    print(f"[SlideVQA] Extracting images for {total} questions...", flush=True)
    processed_data = []
    # Track processed decks to avoid redundant image saving
    processed_decks = set()

    for i, sample in enumerate(sampled_data, 1):
        # ["page_3", "page_1", "page_10", "page_2"] -> ["page_1", "page_2", "page_3", "page_10"]
        page_keys = sorted(
            [k for k in sample.keys() if k.startswith('page_')], 
            key=lambda x: int(x.split('_')[1])
        )
        
        deck_name = sample['deck_name']
        
        # Get absolute paths of images of the sample
        page_absolute_paths = []
        for key in page_keys:
            page_num = key.split('_')[1]

            image = sample[key]
            if image is None:
                continue
                
            # Construct the absolute path for this specific image
            filename = f"{deck_name}_{page_num}.jpg"
            abs_image_path = os.path.join(image_dir, filename)
            
            # Save image if it doesn't exist yet
            if deck_name not in processed_decks or not os.path.exists(abs_image_path):
                image.save(abs_image_path)
                
            # Store the absolute path in the list
            page_absolute_paths.append(abs_image_path)
        
        processed_decks.add(deck_name)
        
        # Create the record for the JSON
        record = {k: v for k, v in sample.items() if not k.startswith('page_')}
        
        # Add the absolute document paths list
        record['document'] = page_absolute_paths
        
        if record.get('evidence_pages'):
            record['answers_page_bounding_boxes'] = {
                "page": [p - 1 for p in record['evidence_pages']]
            }
        
        processed_data.append(record)

        if i % 100 == 0 or i == total:
            print(f"[SlideVQA]   processed {i}/{total} questions "
                  f"({len(processed_decks)} unique decks so far)", flush=True)

    print(f"[SlideVQA] Done: {len(processed_data)} questions across {len(processed_decks)} decks.", flush=True)
    return processed_data

def sample_SlideVQA(dataset_split, num_questions: int, offset: int = 0, shuffle: bool = True):
    if shuffle:
        sampled_data = dataset_split.shuffle(seed=42).select(range(offset, offset + num_questions))
    else:
        sampled_data = dataset_split.select(range(offset, offset + num_questions))

    # We store images in absolute paths
    image_dir = get_SlideVQA_image_dir()
    os.makedirs(image_dir, exist_ok=True)

    processed_data = _process_samples(sampled_data, image_dir)

    output_wrapper = {
        "dataset_name": "SlideVQA",
        "base_image_dir": get_SlideVQA_image_dir(),
        "data": processed_data
    }
    return output_wrapper

def sample_SlideVQA_different_from(dataset_split, num_questions: int, exclude_questions: list[str], offset: int = 0, shuffle: bool = True):
    exclude_set = set(exclude_questions)
    print(f"[SlideVQA] Filtering out {len(exclude_set)} excluded questions "
          f"(this can take several minutes for image datasets)...", flush=True)
    filtered = dataset_split.filter(lambda item: item["question"] not in exclude_set)
    print(f"[SlideVQA] {len(filtered)} questions remain after exclusion; sampling up to {num_questions}.", flush=True)
    if shuffle:
        sampled_data = filtered.shuffle(seed=42).select(range(offset, min(offset + num_questions, len(filtered))))
    else:
        sampled_data = filtered.select(range(offset, min(offset + num_questions, len(filtered))))
    
    # We store images in absolute paths
    image_dir = get_SlideVQA_image_dir()
    os.makedirs(image_dir, exist_ok=True)

    processed_data = _process_samples(sampled_data, image_dir)

    output_wrapper = {
        "dataset_name": "SlideVQA",
        "base_image_dir": get_SlideVQA_image_dir(),
        "data": processed_data
    }
    return output_wrapper

def standardize_SlideVQA_for_corruption_pipeline(data, split_type):
    # Create DataFrame with same structure as MPDocVQA
    df = pd.DataFrame(data["data"])

    # For SlideVQA we provide in input absolute image paths, so no change is needed here

    # Derive base_image_dir from the first document entry (absolute paths in train.json)
    first_doc = df.iloc[0]["document"] if len(df) > 0 else None
    first_page = first_doc[0] if isinstance(first_doc, list) and first_doc else (first_doc if isinstance(first_doc, str) else "")
    base_image_dir = os.path.dirname(first_page) if first_page else ""

    # Map the SlideVQA specific fields to the pipeline's expected column names
    df["questionId"] = df["qa_id"].astype(str)
    df["answers"] = df["answer"]
    
    # SlideVQA has answers_page_bounding_boxes.page which contains the index
    # and evidence_pages which also contains it. We'll use the bounding box one for consistency.
    df["answer_page_idx"] = df["answers_page_bounding_boxes"].apply(
        lambda x: x.get("page", [0])[0] if isinstance(x, dict) and x.get("page") else 0
    )

    # Filter out questions with empty answers and correct data split
    def check_answers(x):
        if isinstance(x, float):  # Handle NaN values
            return False
        return bool(x) and len(x) > 0

    df["data_split"] = split_type
    df = df[
        (df["answers"].apply(check_answers))
    ]

    # Select and reorder only the columns needed for the pipeline
    df = df[
        [
            "questionId",
            "question",
            "answers",
            "answer_page_idx",
            "data_split",
            "document",
        ]
    ]
    data["data"] = df.to_dict(orient="records")
    return data

if __name__ == "__main__":
    split = "train"
    num_questions = 10
    slidevqa_split = get_SlideVQA_split(split)
    sample_slidevqa = sample_SlideVQA(slidevqa_split, num_questions)
    sample_slidevqa = standardize_SlideVQA_for_corruption_pipeline(sample_slidevqa, split)
    save_sample("SlideVQA", split, num_questions, sample_slidevqa)