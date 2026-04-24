import os
import json
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

# 1. Load the dataset
print("Loading SlideVQA dataset from HuggingFace...")
dataset = load_dataset("NTT-hil-insight/SlideVQA", split="train")

# 2. Setup directories for images
image_dir = "data/SlideVQA/images/train"
os.makedirs(image_dir, exist_ok=True)

# 3. Select a small sample
small_dataset = dataset.select(range(18))

processed_data = []
print(f"Processing {len(small_dataset)} questions and extracting images...")

# Track processed decks to avoid redundant image saving
processed_decks = set()

for sample in tqdm(small_dataset):
    deck_name = sample['deck_name']
    
    # Identify all page keys (SlideVQA uses page_1, page_2, etc.)
    page_keys = sorted(
        [k for k in sample.keys() if k.startswith('page_')], 
        key=lambda x: int(x.split('_')[1])
    )
    
    page_paths = []
    
    for key in page_keys:
        page_num = key.split('_')[1]
        image = sample[key]
        
        # Some samples might have empty page slots if they vary in length
        if image is None:
            continue
            
        filename = f"{deck_name}_{page_num}.jpg"
        # We store relative paths that the pipeline can resolve
        relative_path = os.path.join(image_dir, filename)
        
        # Save image if it doesn't exist yet
        if deck_name not in processed_decks or not os.path.exists(relative_path):
            image.save(relative_path)
            
        page_paths.append(relative_path)
    
    processed_decks.add(deck_name)
    
    # Create the record for the JSON (removing the heavy PIL objects)
    record = {k: v for k, v in sample.items() if not k.startswith('page_')}
    
    # Add the document filenames list
    record['document'] = page_paths
    
    # Prepare metadata for our pipeline (similar to DUDE structure)
    # SlideVQA evidence_pages is 1-indexed. We map it to answers_page_bounding_boxes.
    if record.get('evidence_pages'):
        # Usually SlideVQA has one evidence page, but it's a list. 
        # We take the first one and convert to 0-indexed for internal consistency if needed,
        # but let's keep the raw metadata and let the data_loader handle it.
        record['answers_page_bounding_boxes'] = {
            "page": [p - 1 for p in record['evidence_pages']] # Converting to 0-indexed for our pipeline
        }
    
    processed_data.append(record)

# 4. Save the wrapped data
output_file = "slidevqa_train.json"
output_wrapper = {
    "dataset_name": "SlideVQA",
    "data": processed_data
}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_wrapper, f, indent=4)

print(f"\nSuccess! Processed data saved to {output_file}")
print(f"Images extracted to {image_dir}")
