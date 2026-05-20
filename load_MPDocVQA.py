import json
import random
import os
import shutil
from tqdm import tqdm

SOURCE_DIR = "/home/amartinelli/MPDocVQA/MPDocVQA_complete/qas"
OUT_DIR = "data/MPDocVQA/MPDocVQA_test_250/qas"


def load_MPDocVQA(split_type: str, max_questions: int = 300, source_dir: str = SOURCE_DIR, out_dir: str = OUT_DIR):
    source = os.path.join(source_dir, f"{split_type}.json")
    with open(source) as f:
        full = json.load(f)

    print(f"Total questions in {split_type} split: {len(full['data'])}")

    random.seed(42)
    sampled = random.sample(full["data"], min(max_questions, len(full["data"])))

    os.makedirs(out_dir, exist_ok=True)
    output = {**full, "data": sampled}
    out_json = os.path.join(out_dir, f"{split_type}.json")
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(sampled)} questions to {out_json}")

    # Copy only the images referenced by the sampled questions
    src_images = os.path.join(source_dir, "images")
    dst_images = os.path.join(out_dir, "images")
    os.makedirs(dst_images, exist_ok=True)

    needed = set()
    for item in sampled:
        for page_id in item.get("page_ids", []):
            needed.add(f"{page_id}.jpg")

    print(f"Copying {len(needed)} images...")
    
    for filename in tqdm(needed):
        src = os.path.join(src_images, filename)
        dst = os.path.join(dst_images, filename)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    print(f"Done. Images saved to {dst_images}")


if __name__ == "__main__":
    split = "train"
    questions_to_process = 300
    corrupted_questions_desired = 50
    load_MPDocVQA(split, max_questions=questions_to_process, source_dir=SOURCE_DIR, out_dir=f"data/MPDocVQA/MPDocVQA_{split}_{corrupted_questions_desired}/qas")
