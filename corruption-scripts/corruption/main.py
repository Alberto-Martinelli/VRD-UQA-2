import warnings
import os

# Suppress noisy third-party warnings before importing libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import logging
import nltk
import argparse

nltk.download("punkt_tab", quiet=True)
from utils.config_utils import load_config, extract_config, print_parameters
from pipeline import load_data, identify_all_entities, create_augmented_dataset, corrupt_questions

logging.basicConfig(
    # level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    level=logging.INFO, format="%(levelname)s - %(message)s"
)
# Silence noisy third-party loggers
for logger_name in ["httpx", "httpcore", "gliner", "transformers", "sentence_transformers", "sentencepiece"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

from transformers.cache_utils import DynamicCache
if not hasattr(DynamicCache, "seen_tokens"):
    @property
    def seen_tokens(self):
        return self.get_seq_length()
    DynamicCache.seen_tokens = seen_tokens

if not hasattr(DynamicCache, "get_max_length"):
    def get_max_length(self):
        return getattr(self, "_max_cache_len", None) or getattr(self, "max_cache_len", 4096)
    DynamicCache.get_max_length = get_max_length

def get_env_bool(key, default=False):
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")

def main(config_path=None):
    logging.info("\nStarting the question corruption and verification process...")

    # Load configuration
    config = load_config(config_path)
    params = extract_config(config)
    print_parameters(params)

    # Load data
    logging.info("\n")
    logging.info(
        "----------------------------------- 1. Loading data -----------------------------------"
    )
    logging.info("\n")

    df_to_corrupt = load_data(params)

    logging.info("\n")
    logging.info(
        "----------------------------------- 2. Identifying entities (Optional)-----------------------------------"
    )
    logging.info("\n")

    questions_list, question_with_entities, entity_identifier = identify_all_entities(params, df_to_corrupt)

    logging.info(f"\nQuestions with identified entities: {sum(1 for entities in question_with_entities if entities)}")

    logging.info("\nExample of question with identified entities:")
    for i, (question, entities) in enumerate(zip(questions_list[:3], question_with_entities[:3])):
        logging.info(f"\nQuestion {i+1}: {question}")
        logging.info(f"Entities: {entities}")

    # Process layout analysis
    logging.info("\n")
    logging.info(
        "----------------------------------- 3. Analyzing document layout -----------------------------------"
    )
    logging.info("\n")

    df_to_corrupt = create_augmented_dataset(params, df_to_corrupt)

    logging.info("\n")
    logging.info(
        "----------------------------------- 4. In-context corruption -----------------------------------"
    )
    logging.info("\n")

    corrupt_questions(params, entity_identifier)

    logging.info("\n")
    logging.info(
        "----------------------------------- Process completed successfully! -----------------------------------"
    )
    logging.info("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run question corruption.')
    parser.add_argument('--config', type=str, help='Path to the configuration file', default="code/corruption-scripts/config.json")
    args = parser.parse_args()
    
    main(config_path=args.config)
