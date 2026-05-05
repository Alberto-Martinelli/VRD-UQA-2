import os
import json
from datasets import load_dataset
from tqdm import tqdm
from langdetect import detect, LangDetectException

# 1. HPC Path Management
# Use absolute paths so the corruption pipeline can find them from anywhere
base_dir = os.path.abspath(os.getcwd())
image_dir = os.path.join(base_dir, "data/BoundingDocs/images/train")
os.makedirs(image_dir, exist_ok=True)

# 2. Load Dataset
print("Loading BoundingDocs...")
dataset = load_dataset("letxbe/BoundingDocs", split="train", trust_remote_code=True)

# Use a sample for testing, or full dataset: dataset
# Language filtering is now handled during the flattening process below.
small_dataset = dataset.select(range(10)) 

flattened_data = []

for doc in tqdm(small_dataset, desc="Flattening Documents"):
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
        flattened_data.append(entry)

# 3. Wrap and Save
output_wrapper = {
    "dataset_name": "Bounding Docs",
    "data": flattened_data
}

with open("bounding_docs_train.json", "w", encoding="utf-8") as f:
    json.dump(output_wrapper, f, indent=4)

print(f"\nDone! Produced {len(flattened_data)} question-level entries.")

print("First example:\n", dataset[0])      # first example
print("\nColumn names:\n", dataset.column_names)
print("\nDataset length:\n", len(dataset))