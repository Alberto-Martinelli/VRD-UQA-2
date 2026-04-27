import os
import json
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

# 1. Get the Absolute Base Directory 
# This will resolve to /mnt/beegfs/amartinelli/download_SlideVQA/VRD-UQA/
base_dir = os.path.abspath(os.getcwd())

# 2. Load the dataset
print("Loading SlideVQA dataset from HuggingFace...")
dataset = load_dataset("NTT-hil-insight/SlideVQA", split="train")

# 3. Setup directories for images using absolute paths
image_dir = os.path.join(base_dir, "data/SlideVQA/images/train")
os.makedirs(image_dir, exist_ok=True)

# 4. Select a small sample
small_dataset = dataset.select(range(18))

processed_data = []
print(f"Processing {len(small_dataset)} questions and extracting images...")

# Track processed decks to avoid redundant image saving
processed_decks = set()

for sample in tqdm(small_dataset):
    deck_name = sample['deck_name']
    
    page_keys = sorted(
        [k for k in sample.keys() if k.startswith('page_')], 
        key=lambda x: int(x.split('_')[1])
    )
    
    page_paths = []
    
    for key in page_keys:
        page_num = key.split('_')[1]
        image = sample[key]
        
        if image is None:
            continue
            
        filename = f"{deck_name}_{page_num}.jpg"
        
        # Construct the absolute path for this specific image
        abs_image_path = os.path.join(image_dir, filename)
        
        # Save image if it doesn't exist yet
        if deck_name not in processed_decks or not os.path.exists(abs_image_path):
            image.save(abs_image_path)
            
        # Store the absolute path in the list
        page_paths.append(abs_image_path)
    
    processed_decks.add(deck_name)
    
    # Create the record for the JSON
    record = {k: v for k, v in sample.items() if not k.startswith('page_')}
    
    # Add the absolute document paths list
    record['document'] = page_paths
    
    if record.get('evidence_pages'):
        record['answers_page_bounding_boxes'] = {
            "page": [p - 1 for p in record['evidence_pages']]
        }
    
    processed_data.append(record)

# 5. Save the wrapped data
output_file = "slidevqa_train.json"
output_wrapper = {
    "dataset_name": "SlideVQA",
    "data": processed_data
}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_wrapper, f, indent=4)

print(f"\nSuccess! Processed data saved to {os.path.join(base_dir, output_file)}")
print(f"Absolute Image paths saved in JSON start with: {image_dir}")