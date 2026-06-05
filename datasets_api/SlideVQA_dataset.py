from datasets import load_dataset
import os
from tqdm import tqdm
from datasets_api.datasets_utils import save_sample

SCRATCH_FLASH = '/mnt/beegfs/amartinelli/'
IMAGES_PATH = 'SlideVQA_images/'

def get_SlideVQA_image_dir():
    return os.path.join(SCRATCH_FLASH, IMAGES_PATH)

def get_SlideVQA_split(split_type: str):
    print("Loading SlideVQA dataset from HuggingFace...")
    dataset_split = load_dataset("NTT-hil-insight/SlideVQA", split=split_type, trust_remote_code=True)
    return dataset_split

def _process_samples(sampled_data, image_dir):
    print(f"Processing {len(sampled_data)} questions and extracting images...")
    processed_data = []
    # Track processed decks to avoid redundant image saving
    processed_decks = set()

    for sample in tqdm(sampled_data):
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
        "data": processed_data
    }
    return output_wrapper

def sample_SlideVQA_different_from(dataset_split, num_questions: int, exclude_questions: list[str], offset: int = 0, shuffle: bool = True):
    exclude_set = set(exclude_questions)
    filtered = dataset_split.filter(lambda item: item["question"] not in exclude_set)
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
        "data": processed_data
    }
    return output_wrapper

if __name__ == "__main__":
    split = "train"
    num_questions = 10
    slidevqa_split = get_SlideVQA_split(split)
    sample_slidevqa = sample_SlideVQA(slidevqa_split, num_questions)
    save_sample("SlideVQA", split, num_questions, sample_slidevqa)