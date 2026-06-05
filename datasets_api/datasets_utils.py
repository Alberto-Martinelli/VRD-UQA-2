from pathlib import Path
import os
import json

PROJECT_DIR = "/home/amartinelli/VRD-UQA/"

def save_sample(dataset_name: str, split_type: str, num_questions: int, output_data):
    out_dir = Path(PROJECT_DIR) / f"{dataset_name}_{split_type}_{num_questions}" / "qas"
    os.makedirs(out_dir, exist_ok=True)

    out_json = out_dir / f"{split_type}.json"
    with open(out_json, "w") as f:
        json.dump(output_data, f, indent=4)
    
    print(f"\nSaved {len(output_data)} questions to {out_json}")