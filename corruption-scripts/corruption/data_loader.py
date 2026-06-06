from datasets import load_from_disk
import pandas as pd
import os
import json
import logging
from pathlib import Path


class DataLoader:
    @staticmethod
    def load_dataset(base_path: str, split_type: str, dataset_name: str, dataset_json_path: str = None) -> dict:
        path = Path(base_path) / dataset_json_path / f"{split_type}.json"
        try:
            with open(path, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset {dataset_name} not found at {path}. Please check the path and ensure the dataset is in the correct format."
            )
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_name}")

    @staticmethod
    def create_dataframe(raw_dataset_dict: dict, dataset_name: str, base_path: str, dataset_json_path: str, split_type: str):
        if dataset_name not in ["MPDocVQA", "DUDE", "SlideVQA", "Bounding Docs"]:
            raise ValueError(f"Unsupported dataset type: {dataset_name}")
        else:
            base_image_dir = raw_dataset_dict.get("base_image_dir", "")
            df = pd.DataFrame(raw_dataset_dict["data"])

        df["image_path"] = df["document"]
        return df, base_image_dir
