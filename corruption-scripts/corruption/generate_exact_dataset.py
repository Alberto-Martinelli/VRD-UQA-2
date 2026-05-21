import os
import sys
import json
import logging
import argparse
import copy
from pathlib import Path
import pandas as pd

# Resolve Python Path for Sibling Directory Imports
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

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
nltk.download("punkt_tab", quiet=True)

from utils.config_utils import load_config, extract_config
from pipeline import (
    load_data, 
    identify_all_entities, 
    create_augmented_dataset, 
    corrupt_questions,
    clean_corrupted_questions
)
from verification.answerability_verifier import AnswerabilityVerifier
from verification.just_false import filter_false_verifications

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


def get_already_processed_texts(output_pool_path):
    """Load already processed original question texts to resume safely if interrupted."""
    if os.path.exists(output_pool_path):
        try:
            with open(output_pool_path, "r") as f:
                data = json.load(f)
                return {q.get("original_question") 
                        for q in data.get("corrupted_questions", []) 
                        if q.get("original_question")}
        except Exception as e:
            logging.warning(f"Could not parse existing pool file: {e}. Starting fresh.")
    return set()

def main():
    parser = argparse.ArgumentParser(description='Run incremental exact question generation.')
    parser.add_argument('--config', type=str, help='Path config', default="corruption-scripts/config.json")
    parser.add_argument('--target', type=int, help='Exact number of verified questions needed', default=300)
    parser.add_argument('--batch_size', type=int, help='Size of each incremental batch', default=30)
    parser.add_argument('--output', type=str, help='Custom path to save final verified pool', default=None)
    args = parser.parse_args()

    # Load Configurations
    config = load_config(args.config)
    params = extract_config(config)
    
    # Target output determination
    if args.output:
        final_pool_path = args.output
    else:
        final_pool_path = params["output_corrupted_cleaned"].replace("_cleaned.json", "_just_false.json")
        
    dataset_name = params["dataset_name"].replace(" ", "_")
    logging.info(f"Targeting exactly {args.target} verified unanswerable questions for {dataset_name}...")

    # Load already processed question texts for fault tolerance
    verified_false_pool = []
    processed_question_texts = get_already_processed_texts(final_pool_path)
    
    if processed_question_texts:
        with open(final_pool_path, "r") as f:
            verified_false_pool = json.load(f).get("corrupted_questions", [])
        logging.info(f"Resuming run. Already have {len(verified_false_pool)} verified questions.")
        
    if len(verified_false_pool) >= args.target:
        logging.info("Target already met! Exiting.")
        return

    # Force load data to sample from
    params["percentage"] = 100.0
    raw_questions_df, base_image_dir = load_data(params)
    
    # Shuffle the dataset with a fixed seed for reproducible randomness
    shuffled_df = raw_questions_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Filter out questions that have already contributed to our pool
    if processed_question_texts:
        shuffled_df = shuffled_df[~shuffled_df["question"].isin(processed_question_texts)].reset_index(drop=True)

    current_row_idx = 0
    total_available_rows = len(shuffled_df)
    
    # Isolated temp paths to avoid collisions during concurrent Slurm jobs
    batch_corrupted_path = f"./corruption-scripts/results/temp_{dataset_name}_corrupted.json"
    batch_cleaned_path = f"./corruption-scripts/results/temp_{dataset_name}_cleaned.json"
    batch_verified_path = f"./corruption-scripts/results/temp_{dataset_name}_verified.json"
    batch_augmented_path = f"./corruption-scripts/results/temp_{dataset_name}_augmented.json"

    # Inject temporary paths into config parameters for the batch runs
    batch_params = copy.deepcopy(params)
    batch_params["output_corrupted"] = batch_corrupted_path
    batch_params["output_corrupted_cleaned"] = batch_verified_path
    batch_params["augmented_dataset_path"] = batch_augmented_path

    # Initialize the Verifier in memory (prevents model reloading on every batch)
    logging.info("Initializing Answerability Verifier...")
    verifier = AnswerabilityVerifier(config_path=args.config)
    verifier.input_file = batch_cleaned_path
    verifier.output_file = batch_verified_path

    while len(verified_false_pool) < args.target and current_row_idx < total_available_rows:
        needed = args.target - len(verified_false_pool)
        logging.info(f"\n=================== PROGRESS: {len(verified_false_pool)}/{args.target} (Needed: {needed}) ===================")
        
        # Take the next slice/batch of shuffled questions
        batch_df = shuffled_df.iloc[current_row_idx : current_row_idx + args.batch_size].copy()
        current_row_idx += args.batch_size
        
        if len(batch_df) == 0:
            logging.warning("No more questions left in the dataset to process!")
            break
            
        logging.info(f"Processing batch of {len(batch_df)} original questions...")

        # Clear batch cache if it exists to ensure new layout analyses are compiled
        if os.path.exists(batch_augmented_path):
            os.remove(batch_augmented_path)

        # --- Pipeline Step A: Entity Identification ---
        _, _, entity_identifier = identify_all_entities(batch_params, batch_df)
        
        # --- Pipeline Step B: Document Layout & OCR ---
        batch_df = create_augmented_dataset(batch_params, batch_df)
        
        # --- Pipeline Step C: LLM Rephrasing/Corruption ---
        corrupt_questions(batch_params, entity_identifier, base_image_dir)
        
        if not os.path.exists(batch_corrupted_path) or os.path.getsize(batch_corrupted_path) == 0:
            logging.info("Batch yielded 0 corrupted candidates. Moving to next batch.")
            continue

        # --- Filter out failed corruptions and invalid formats before verification ---
        clean_corrupted_questions(batch_corrupted_path, batch_cleaned_path)

        if not os.path.exists(batch_cleaned_path) or os.path.getsize(batch_cleaned_path) == 0:
            logging.info("Batch yielded 0 valid corrupted candidates after cleaning. Moving to next batch.")
            continue

        # --- Pipeline Step D: Verification ---
        logging.info("Running verification on current batch...")
        verifier.verify_questions_from_file()
        
        # --- Pipeline Step E: Filter "False" (Unanswerable) questions ---
        if os.path.exists(batch_verified_path):
            with open(batch_verified_path, "r") as f:
                batch_verified_data = json.load(f)
            
            batch_filtered = filter_false_verifications(batch_verified_data)
            batch_false_questions = batch_filtered.get("corrupted_questions", [])
            
            logging.info(f"Batch completed! Yielded {len(batch_false_questions)} verified false questions.")
            
            # Append verified false questions to our master pool
            for q in batch_false_questions:
                verified_false_pool.append(q)
                
                # Break early the moment we hit the exact target
                if len(verified_false_pool) == args.target:
                    logging.info(f"Exactly {args.target} target reached! Ending execution.")
                    break
            
            # Save progress incrementally (HPC Slurm-resilient checkpointing)
            master_data = {
                "base_image_dir": base_image_dir,
                "corrupted_questions": verified_false_pool
            }
            with open(final_pool_path, "w") as f:
                json.dump(master_data, f, indent=2)
                
            logging.info(f"Intermediate master pool saved to {final_pool_path} ({len(verified_false_pool)} items)")
            
        # Clean up temporary batch files
        for temp_file in [batch_corrupted_path, batch_cleaned_path, batch_verified_path, batch_augmented_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    logging.info(f"\nProcess completed successfully! Output pool contains exactly {len(verified_false_pool)} verified unanswerable questions at: {final_pool_path}")

if __name__ == "__main__":
    main()
