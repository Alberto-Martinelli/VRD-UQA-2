import json

def get_default_config():
    return {
        "paths": {
            "base_path": "./",
            "augmented_dataset": "MPDocVQA_augmented.json",
            "output_corrupted": "unanswerable_corrupted_questions.json",
            "output_corrupted_cleaned": "unanswerable_corrupted_questions_cleaned.json",
            "patch_saving_dir": "./patches",
            "layout_saving_dir": "./layouts"
        },
        "dataset": {"type": "MPDocVQA", "split": "train"},
        "corruption": {
            "percentage": 10,
            "complexity": 3,
            "generated_sample_per_complexity_greater_than_1": 5,
            "types": {
                "numerical": True,
                "temporal": True,
                "entity": True,
                "location": True,
                "document": True,
            },
        },
        "layout_analysis": {
            "model": "Qwen/Qwen2-VL-2B-Instruct",
        },
        "model": {
            "provider": "ollama",
            "name": "llama3.2",
        }
    }

def load_config(config_path="code/corruption-scripts/config.json"):
    """Load configuration from JSON file."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            if config is None:
                return get_default_config()
        return config
    except FileNotFoundError:
        print(f"Config file not found at {config_path}. Using default configuration.")
        return get_default_config()
    except json.JSONDecodeError:
        print(
            f"Error parsing config file at {config_path}. Using default configuration."
        )
        return get_default_config()

def extract_config(config):
    paths = config["paths"]
    dataset = config["dataset"]
    corruption = config["corruption"]
    layout = config["layout_analysis"]
    model_cfg = config["model"]
    types = corruption["types"]

    return {
        "base_path": paths["base_path"],
        "dataset_name": dataset["name"],
        "dataset_json_path": dataset.get("dataset_json_path"),
        "augmented_dataset_path": paths["augmented_dataset"],
        "output_corrupted": paths["output_corrupted"],
        "output_corrupted_cleaned": paths["output_corrupted_cleaned"],
        "patch_saving_dir": paths["patch_saving_dir"],
        "layout_saving_dir": paths["layout_saving_dir"],
        "percentage": float(corruption["percentage"]),
        "complexity": int(corruption["complexity"]),
        "generated_sample_per_complexity_greater_than_1": int(corruption["generated_sample_per_complexity_greater_than_1"]),
        "layout_model": layout["model"],
        "model_provider": model_cfg["provider"],
        "model_name": model_cfg["name"],
        "numerical": types["numerical"],
        "temporal": types["temporal"],
        "entity": types["entity"],
        "location": types["location"],
        "document": types["document"],
        "split": dataset.get("split", "train"),
    }

def print_parameters(params):
    print("\nUsing the following parameters:")
    for k, v in params.items():
        print(f"{k.replace('_', ' ').capitalize()}: {v}")
