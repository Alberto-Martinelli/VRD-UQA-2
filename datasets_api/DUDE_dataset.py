from datasets import load_dataset
from datasets_api.datasets_utils import save_sample
from config import paths
import pandas as pd
import os
import logging


def get_DUDE_image_dir():
    return paths.image_dir("DUDE")

def get_DUDE_split(split_type: str):
    print(f"[DUDE] Loading '{split_type}' split from HuggingFace (jordyvl/DUDE_loader)...", flush=True)
    dataset_split = load_dataset("jordyvl/DUDE_loader", split=split_type, trust_remote_code=True)
    print(f"[DUDE] Loaded {len(dataset_split)} raw entries.", flush=True)
    return dataset_split

def sample_DUDE(dataset_split, num_questions: int, offset: int = 0, shuffle: bool = True):
    if shuffle:
        sampled = dataset_split.shuffle(seed=42).select(range(offset, offset + num_questions))
    else:
        sampled = dataset_split.select(range(offset, offset + num_questions))
    
    output_wrapper = {
        "dataset_name": "DUDE",
        "base_image_dir": get_DUDE_image_dir(),
        "data": sampled
    }
    return output_wrapper

def sample_DUDE_different_from(dataset_split, num_questions: int, exclude_questions: list[str], offset: int = 0, shuffle: bool = True):
    exclude_set = set(exclude_questions)
    filtered = dataset_split.filter(lambda item: item["question"] not in exclude_set)
    print(f"[DUDE] {len(filtered)} entries after excluding {len(exclude_set)} originals; "
          f"sampling up to {num_questions}.", flush=True)
    if shuffle:
        sampled = filtered.shuffle(seed=42).select(range(offset, min(offset + num_questions, len(filtered))))
    else:
        sampled = filtered.select(range(offset, min(offset + num_questions, len(filtered))))

    output_wrapper = {
        "dataset_name": "DUDE",
        "base_image_dir": get_DUDE_image_dir(),
        "data": sampled
    }
    return output_wrapper

def standardize_DUDE_for_corruption_pipeline(data, split_type, require_bbox: bool = True):
    # Create DataFrame with same structure as MPDocVQA
    df = pd.DataFrame(data["data"])

    # Filter out questions with empty bounding boxes, empty answers, and train split
    def check_bounding_boxes(x):
        # Handle NaN or non-dictionary values
        if not isinstance(x, dict):
            return False

        # Check if 'left' exists and has at least one coordinate
        # Use any key: "left", "top", "width", "height", or "page"
        return "left" in x and len(x["left"]) > 0

    def check_answers(x):
        if isinstance(x, float):  # Handle NaN values
            return False
        return bool(x) and len(x) > 0

    # The corruption pipeline needs bounding boxes, but clean (gold) SFT examples
    # do not — pass require_bbox=False there to avoid dropping ~55% of DUDE.
    mask = (df["data_split"] == split_type) & (df["answers"].apply(check_answers))
    if require_bbox:
        mask &= df["answers_page_bounding_boxes"].apply(check_bounding_boxes)
    df = df[mask]

    base_image_dir = get_DUDE_image_dir()

    # Get document pages using directory scanning
    def get_document_pages(doc_id):
        pages = []
        if os.path.exists(base_image_dir):
            for filename in os.listdir(base_image_dir):
                # Look for any image file starting with doc_id (even without an underscore)
                if filename.startswith(doc_id) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    pages.append(filename)
            pages.sort(key=lambda f: int(f.rsplit('_', 1)[-1].split('.')[0]))
        else:
            logging.error(f"Warning: base_image_dir does not exist at {base_image_dir}!")
        return pages

    # Create necessary columns
    df["page_ids"] = df["docId"].apply(get_document_pages)
    
    # Warn if 0 page_ids are found
    empty_docs = df[df["page_ids"].map(len) == 0]
    if not empty_docs.empty:
        logging.warning(f"Found {len(empty_docs)} documents with 0 page_ids in {base_image_dir}!")
        if os.path.exists(base_image_dir):
            sample_files = os.listdir(base_image_dir)[:10]
            logging.warning(f"Sample files actually present in directory: {sample_files}")
            logging.warning(f"We were looking for files starting with doc_id like: {df.iloc[0]['docId']}")
    
    df["document"] = df["page_ids"].apply(
        lambda x: [
            os.path.join(base_image_dir, pid)
            for pid in x
        ]
    )
    df["answer_page_idx"] = df["answers_page_bounding_boxes"].apply(
        lambda x: x.get("page", [0])[0] if isinstance(x, dict) and x.get("page") else 0
    )
    df["questionId"] = df["questionId"].astype(str)

    # Select and reorder columns
    df = df[
        [
            "questionId",
            "question",
            "answers",
            "answer_page_idx",
            "data_split",
            "docId",
            "document",
        ]
    ]

    _filt = "non-empty answers" + (" + bbox" if require_bbox else "")
    print(f"[DUDE] {len(df)} entries survive standardization "
          f"(data_split=='{split_type}' + {_filt}).", flush=True)
    data["data"] = df.to_dict(orient="records")
    return data

if __name__ == "__main__":
    split = "train"
    num_questions = 10
    dude_split = get_DUDE_split(split)
    sample_dude = sample_DUDE(dude_split, num_questions)
    sample_dude = standardize_DUDE_for_corruption_pipeline(sample_dude, split)
    save_sample("DUDE", split, num_questions, sample_dude)