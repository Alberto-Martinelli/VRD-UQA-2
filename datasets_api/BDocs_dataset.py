import os
from datasets import load_dataset
from tqdm import tqdm
import json
from langdetect import detect, LangDetectException
from datasets_api.datasets_utils import save_sample
import pandas as pd

SCRATCH_FLASH = '/mnt/beegfs/amartinelli/'
IMAGES_PATH = 'BDocs_images/'

def get_BDocs_image_dir():
    return os.path.join(SCRATCH_FLASH, IMAGES_PATH)

def get_BDocs_split(split_type: str):
    # HuggingFace uses "validation" but internally we use "val" to match all other datasets
    hf_split = "validation" if split_type == "val" else split_type
    dataset_split = load_dataset("letxbe/BoundingDocs", split=hf_split, trust_remote_code=True)
    return dataset_split

def _flatten_documents(dataset_to_process, num_questions, image_dir, offset):
    flattened_data = []
    skipped_count = 0

    for doc in tqdm(dataset_to_process, desc="Flattening Documents"):
        if len(flattened_data) >= num_questions:
            break
        doc_id = doc['doc_id']
        
        # --- Step A: Save images and store absolute paths ---
        page_paths = []
        for i, img in enumerate(doc['doc_images']):
            filename = f"{doc_id.replace('/', '_')}_p{i}.jpg"
            abs_path = os.path.join(image_dir, filename)
            if not os.path.exists(abs_path):
                img.save(abs_path, "JPEG")
            page_paths.append(abs_path)
        
        # --- Step B: Extract nested questions ---
        try:
            qa_dict = json.loads(doc['Q&A'])
        except:
            continue # Skip if JSON is malformed
            
        for qa_id, content in qa_dict.items():
            if len(flattened_data) >= num_questions:
                break
            question_text = content.get("question", "")
            
            # --- Step B1: Filter for English ---
            try:
                if not question_text or detect(question_text) != 'en':
                    continue
            except LangDetectException:
                continue

            raw_answers = content.get("answers", [])
            clean_answers = []
            for ans in raw_answers:
                if isinstance(ans, dict):
                    ans_copy = ans.copy()
                    if "location" in ans_copy:
                        del ans_copy["location"]
                    
                    # BoundingDocs pages are 1-indexed, but the pipeline expects 0-indexed
                    if "page" in ans_copy and isinstance(ans_copy["page"], int):
                        ans_copy["page"] = max(0, ans_copy["page"] - 1)
                    else:
                        ans_copy["page"] = 0
                        
                    clean_answers.append(ans_copy)
                else:
                    # Wrap simple strings in a dict so data_loader.py can safely extract 'value' and 'page'
                    clean_answers.append({"value": str(ans), "page": 0})

            # This is exactly the format your pipeline expects:
            entry = {
                "question_id": qa_id,
                "question": content.get("question"),
                "document": page_paths,  # List of image paths
                "answers": clean_answers,
                "doc_id": doc_id,
                "source": doc.get("source")
            }
            
            # Step 2: Skip the first 'offset' valid questions for disjoint sets
            if skipped_count < offset:
                skipped_count += 1
                continue

            flattened_data.append(entry)

    return flattened_data

def sample_BDocs(dataset_split, num_questions: int, offset: int = 0, shuffle: bool = True):
    if shuffle:
        dataset_to_process = dataset_split.shuffle(seed=42)
    else:
        dataset_to_process = dataset_split

    image_dir = get_BDocs_image_dir()
    os.makedirs(image_dir, exist_ok=True)

    flattened_data = _flatten_documents(dataset_to_process, num_questions, image_dir, offset)

    output_wrapper = {
        "dataset_name": "Bounding Docs",
        "base_image_dir": get_BDocs_image_dir(),
        "data": flattened_data
    }
    return output_wrapper

def sample_BDocs_different_from(dataset_split, num_questions: int, exclude_questions: list[str], offset: int = 0, shuffle: bool = True):
    image_dir = get_BDocs_image_dir()
    os.makedirs(image_dir, exist_ok=True)

    if shuffle:
        dataset_to_process = dataset_split.shuffle(seed=42)
    else:
        dataset_to_process = dataset_split

    # BDocs is document-level — must flatten before filtering by question text
    exclude_set = set(exclude_questions)
    flattened_data = _flatten_documents(dataset_to_process, num_questions * 10, image_dir, 0)
    filtered = [item for item in flattened_data if item["question"] not in exclude_set]
    sampled = filtered[offset : offset + num_questions]

    output_wrapper = {
        "dataset_name": "Bounding Docs",
        "base_image_dir": get_BDocs_image_dir(),
        "data": sampled
    }
    return output_wrapper

def standardize_BDocs_for_corruption_pipeline(data, split_type):
    df = pd.DataFrame(data["data"])

    # For Bounding Docs we provide in input absolute image paths, so no change is needed here

    # Derive base_image_dir from the first document entry (absolute paths in train.json)
    first_doc = df.iloc[0]["document"] if len(df) > 0 else None
    first_page = first_doc[0] if isinstance(first_doc, list) and first_doc else (first_doc if isinstance(first_doc, str) else "")
    base_image_dir = os.path.dirname(first_page) if first_page else ""

    df["questionId"] = df["question_id"].astype(str)
    df["docId"] = df["doc_id"]
    df["data_split"] = split_type
    
    df["answer_page_idx"] = df["answers"].apply(
        lambda x: x[0].get("page", 0) if isinstance(x, list) and len(x) > 0 else 0
    )
    df["answers"] = df["answers"].apply(
        lambda x: [ans.get("value", "") for ans in x] if isinstance(x, list) else []
    )
    
    def check_answers(x):
        if isinstance(x, float):
            return False
        return bool(x) and len(x) > 0

    df = df[df["answers"].apply(check_answers)]
    
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
    data["data"] = df.to_dict(orient="records")
    return data

if __name__ == "__main__":
    split = "train"
    num_questions = 10
    bdocs_split = get_BDocs_split(split)
    sample_bdocs = sample_BDocs(bdocs_split, num_questions)
    sample_bdocs = standardize_BDocs_for_corruption_pipeline(sample_bdocs, split)
    save_sample("BDocs", split, num_questions, sample_bdocs)
