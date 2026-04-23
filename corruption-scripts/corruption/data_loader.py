from datasets import load_from_disk
import pandas as pd
import os
import json
from pathlib import Path


class DataLoader:
    @staticmethod
    def load_dataset(base_path: str, split_type: str, dataset_name: str, dataset_json_path: str = None) -> dict:
        if dataset_name == "DUDE":
            path = Path(base_path) / dataset_json_path / f"{split_type}.json"
            try:
                with open(path, "r") as file:
                    return json.load(file)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Dataset not found at {path}. Please check the path and ensure the dataset is in the correct format."
                )
        elif dataset_name == "MPDocVQA":
            path = Path(base_path) / dataset_json_path / f"{split_type}.json"
            with open(path, "r") as file:
                return json.load(file)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_name}")

    @staticmethod
    def create_dataframe(raw_dataset_dict: dict, dataset_name: str, base_path: str, dataset_json_path: str) -> pd.DataFrame:
        if dataset_name == "MPDocVQA":
            path = os.path.join(base_path, dataset_json_path)
            df = pd.DataFrame(raw_dataset_dict["data"])
            df["docId"] = df["doc_id"] 
            df["questionId"] = df["questionId"].astype(str)
            df["document"] = df["page_ids"].apply(
                lambda x: [
                    os.path.join(path, "images", f"{page_id}.jpg") for page_id in x
                ]
            )
            df["data_split"] = df["data_split"]
            df["answers"] = df["answers"]
            df["answers_page_idx"] = df["answer_page_idx"]

        elif dataset_name == "DUDE":
            # Create DataFrame with same structure as MPDocVQA
            df = pd.DataFrame(raw_dataset_dict["data"])

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

            df = df[
                (df["data_split"] == "train")
                & (df["answers_page_bounding_boxes"].apply(check_bounding_boxes))
                & (df["answers"].apply(check_answers))
            ]

            # Derive image_dir dynamically based on the environment (Mac vs Linux HPC)
            sample_doc = df.iloc[0]["document"] if len(df) > 0 else ""
            if "PDF" in str(sample_doc):
                # Works perfectly for standard HPC/Huggingface downloaded layouts
                base_extracted_path = sample_doc.rsplit("PDF", 1)[0]
                dude_images_dir = os.path.join(base_extracted_path, "DUDE_train-val-test_binaries", "images", "train")
            else:
                # Fallback for Mac setups using local images strictly within dataset_json_path
                dude_images_dir = os.path.join(base_path, dataset_json_path, "images", "train")

            # Get document pages using directory scanning
            def get_document_pages(doc_id):
                pages = []
                if os.path.exists(dude_images_dir):
                    for filename in sorted(os.listdir(dude_images_dir)):
                        if filename.startswith(f"{doc_id}_") and filename.endswith(
                            ".jpg"
                        ):
                            # Extract just the page ID without extension
                            page_id = filename[:-4]  # Remove .jpg
                            pages.append(page_id)
                return pages

            # Create necessary columns
            df["doc_id"] = df["docId"]
            df["page_ids"] = df["docId"].apply(get_document_pages)
            df["document"] = df["page_ids"].apply(
                lambda x: [
                    os.path.join(dude_images_dir, f"{pid}.jpg")
                    for pid in x
                ]
            )
            df["answer_page_idx"] = df["answers_page_bounding_boxes"].apply(
                lambda x: x.get("page", [0])[0] if isinstance(x, dict) and x.get("page") else 0
            )
            df["answers_page_idx"] = df["answer_page_idx"]
            df["questionId"] = df["questionId"].astype(str)

            # Select and reorder columns
            df = df[
                [
                    "questionId",
                    "question",
                    "doc_id",
                    "page_ids",
                    "answers",
                    "answer_page_idx",
                    "data_split",
                    "docId",
                    "document",
                    "answers_page_idx",
                ]
            ]

        else:
            raise ValueError(f"Unsupported dataset type: {dataset_name}")

        df["image_path"] = df["document"]
        return df
