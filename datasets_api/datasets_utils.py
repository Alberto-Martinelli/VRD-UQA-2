import os
import json
from config.paths import REPO_ROOT

def save_sample(dataset_name: str, split_type: str, num_questions: int, output_data):
    # REPO_ROOT self-locates to the running copy of the repo. On a relocated SLURM
    # work-dir this now writes into the work-dir copy (the previously-hardcoded
    # $HOME path wrote to the wrong copy on the compute node). See design doc.
    out_dir = REPO_ROOT / f"{dataset_name}_{split_type}_{num_questions}" / "qas"
    os.makedirs(out_dir, exist_ok=True)

    out_json = out_dir / f"{split_type}.json"
    with open(out_json, "w") as f:
        json.dump(output_data, f, indent=4)
    
    print(f"\nSaved {len(output_data)} questions to {out_json}")