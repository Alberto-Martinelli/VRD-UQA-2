from datasets import load_from_disk
import pandas as pd
import os
import json


class DataLoader:
    @staticmethod
    def load_dataset(dataset_path, split_type, dataset_type, dataset_json_path=None):
        if dataset_type == "DUDE":
            path = (
                dataset_json_path
                if dataset_json_path
                else os.path.join(
                    dataset_path,
                    "data/DUDE_train-val-test_binaries",
                    "2023-03-23_DUDE_gt_test_PUBLIC.json",
                )
            )
            try:
                with open(path, "r") as file:
                    return json.load(file)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Dataset not found at {path}. Please check the path and ensure the dataset is in the correct format."
                )
        elif dataset_type == "MPDocVQA":
            path = os.path.join(dataset_path, "data/qas", f"{split_type}.json")
            with open(path, "r") as file:
                return json.load(file)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

    @staticmethod
    def create_dataframe(data, dataset_type, base_path):
        if dataset_type == "MPDocVQA":
            path = os.path.join(base_path, "data/qas")
            df = pd.DataFrame(data["data"])
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

        elif dataset_type == "DUDE":
            # Create DataFrame with same structure as MPDocVQA
            df = pd.DataFrame(data["data"])

            # Filter out questions with empty bounding boxes, empty answers, and train split
            def check_bounding_boxes(x):
                if isinstance(x, float):  # Handle NaN values
                    return False
                return bool(x) and len(x) > 0 and len(x[0]) > 0

            def check_answers(x):
                if isinstance(x, float):  # Handle NaN values
                    return False
                return bool(x) and len(x) > 0

            df = df[
                (df["data_split"] == "train")
                & (df["answers_page_bounding_boxes"].apply(check_bounding_boxes))
                & (df["answers"].apply(check_answers))
            ]

            # Get document pages using directory scanning
            def get_document_pages(doc_id):
                image_dir = os.path.join(
                    base_path, "data", "DUDE_train-val-test_binaries", "images", "train"
                )
                pages = []
                if os.path.exists(image_dir):
                    for filename in sorted(os.listdir(image_dir)):
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
                    os.path.join(
                        base_path,
                        "data",
                        "DUDE_train-val-test_binaries",
                        "images",
                        "train",
                        f"{pid}.jpg",
                    )
                    for pid in x
                ]
            )
            df["answer_page_idx"] = df["answers_page_bounding_boxes"].apply(
                lambda x: x[0][0]["page"] if x and len(x) > 0 else 0
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
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        df["image_path"] = df["document"]
        return df
