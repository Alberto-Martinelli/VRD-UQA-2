import warnings
import os

# Suppress noisy third-party warnings before importing libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import logging
import nltk
import argparse

nltk.download("punkt_tab", quiet=True)
from utils.config_utils import load_config, extract_config, print_parameters
from pipeline import load_data, identify_all_entities, create_augmented_dataset, corrupt_questions

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Silence noisy third-party loggers
for logger_name in ["httpx", "httpcore", "gliner", "transformers", "sentence_transformers", "sentencepiece"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

def get_env_bool(key, default=False):
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")

def main(config_path=None):
    print("\nStarting the question corruption and verification process...")

    # Load configuration
    config = load_config(config_path)
    params = extract_config(config)
    print_parameters(params)

    # Load data
    print("\n")
    print(
        "----------------------------------- 1. Loading data -----------------------------------"
    )
    print("\n")

    df_to_corrupt = load_data(params)

    print("\n")
    print(
        "----------------------------------- 2. Identifying entities (Optional)-----------------------------------"
    )
    print("\n")

    questions_list, question_with_entities, entity_identifier = identify_all_entities(params, df_to_corrupt)

    print(f"\nQuestions with identified entities: {sum(1 for entities in question_with_entities if entities)}")

    print("\nExample of question with identified entities:")
    for i, (question, entities) in enumerate(zip(questions_list[:3], question_with_entities[:3])):
        print(f"\nQuestion {i+1}: {question}")
        print(f"Entities: {entities}")

    # Process layout analysis
    print("\n")
    print(
        "----------------------------------- 3. Analyzing document layout -----------------------------------"
    )
    print("\n")

    df_to_corrupt = create_augmented_dataset(params, df_to_corrupt)

    print("\n")
    print(
        "----------------------------------- 4. In-context corruption -----------------------------------"
    )
    print("\n")

    corrupt_questions(params, entity_identifier)

    print("\n")
    print(
        "----------------------------------- Process completed successfully! -----------------------------------"
    )
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run question corruption.')
    parser.add_argument('--config', type=str, help='Path to the configuration file', default="code/corruption-scripts/config.json")
    args = parser.parse_args()
    
    main(config_path=args.config)
