import warnings
import os

# Suppress noisy third-party warnings before importing libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import json
import pandas as pd
import logging
from tqdm import tqdm
import nltk
import argparse

nltk.download("punkt_tab", quiet=True)
from in_context_modifier import InContextModifier
from model_loader import ModelLoader
import numpy as np
from utils.config_utils import load_config, extract_config, print_parameters
from pipeline import load_data, identify_all_entities, create_augmented_dataset

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

    # Load and enrich the augmented dataset created in the previous step
    with open(params["augmented_dataset_path"], "r", encoding="utf-8") as file:
        augmented_dataset = json.load(file)
    print(f"Total number of questions in dataset: {len(augmented_dataset.keys())}")

    # Convert the augmented dataset to DataFrame and extract its questions
    df_augmented = pd.DataFrame(augmented_dataset).T
    df_augmented["question"] = df_augmented["question_data"].apply(
        lambda x: x["question"]
    )

    # Locate all the original answers
    def find_answer_bbox(row):
        """
        Locates where the original answer appears in the layout (page, object type, bounding box)
        """
        answers = row["question_data"]["answers"]
        answer_page_idx = row["question_data"]["answer_page_idx"]
        document = row["question_data"]["document"]
        original_answer_locations = []

        # Get the correct page filename using the answer_page_idx
        if answer_page_idx < len(document):
            answer_page_path = document[answer_page_idx]
            # Extract just the filename from the path
            page_filename = answer_page_path.split("/")[-1]

            # Check if this page exists in the layout analysis
            if page_filename in row["layout_analysis"]["pages"]:
                page_data = row["layout_analysis"]["pages"][page_filename]
                layout_objects = page_data.get("layout_analysis", {})

                # Iterate through each object in the page
                for obj_id, obj_data in layout_objects.items():
                    ocr_text = obj_data.get("OCR", "")

                    # Check if any of the answers appear in the OCR text
                    for answer in answers:
                        if str(answer) in ocr_text:
                            original_answer_locations.append(
                                {
                                    "page_id": page_filename,
                                    "object_type": obj_data.get("ObjectType"),
                                    "object_typeID": obj_data.get("ObjectTypeID"),
                                    "bbox": obj_data.get("BBOX"),
                                    "answer": answer,
                                }
                            )

        return original_answer_locations if original_answer_locations else None

    df_augmented["original_answer_locations"] = df_augmented.apply(
        find_answer_bbox, axis=1
    )

    # Identify entities for questions and layout objects (performs NER via GLiNER on the OCR text of every layout object across all pages) 
    def process_layout_objects(row):
        """
        Function to process layout objects (tables, figures, ...) and identify entities in OCR text
        """
        layout_analysis = row.get("layout_analysis", {}).get("pages", {})
        patch_entities = {}

        for page_id, page_data in layout_analysis.items():
            page_objects = page_data.get("layout_analysis", {})
            page_entities = {}

            for obj_id, obj_data in page_objects.items():
                ocr_text = obj_data.get("OCR", "")
                entities = entity_identifier.identify_entities(ocr_text)
                if entities:  # Only add if entities were found
                    page_entities[obj_id] = {
                        "bbox": obj_data.get("BBOX"),
                        "type": obj_data.get("ObjectType"),
                        "typeID": obj_data.get("ObjectTypeID"),
                        "entities": entities,
                    }

            if page_entities:  # Only add if any entities were found on the page
                patch_entities[page_id] = page_entities

        return patch_entities

    df_augmented["question_entities"] = df_augmented["question"].apply(
        entity_identifier.identify_entities
    )
    df_augmented["patch_entities"] = df_augmented.apply(process_layout_objects, axis=1)

    # CONFIGURE THE CORRUPTION ENGINE
    # Load the model once
    model_loader = ModelLoader.get_instance()
    model_loader.load_model(params["model_provider"], params["model_name"])

    # Set the model_loader for InContextModifier (no need to reload)
    InContextModifier.set_model_loader(model_loader)

    # Set other parameters for InContextModifier
    InContextModifier.set_parameters(
        complexity=params["complexity"], in_document=True, out_document=True, generated_sample_per_complexity_greater_than_1=params["generated_sample_per_complexity_greater_than_1"]
    )

    # CORRUPT EACH QUESTION
    tqdm.pandas(desc="Corrupting and Verifying (In-Context)")
    df_in_context = df_augmented.copy()

    def process_corruption(row):
        # Create a dictionary with all the necessary information
        question_data = {
            "question": row["question"],
            "question_entities": row["question_entities"] or [],
            "original_answer_locations": row["original_answer_locations"] or [],
            "patch_entities": row["patch_entities"] or [],
            "context": row,  # Pass the entire row as context if needed
        }

        # Create an instance of InContextModifier if needed
        modifier = InContextModifier()

        # Call corrupt_question as an instance method
        corrupted_questions = modifier.corrupt_question(question_data)

        if corrupted_questions:
            return pd.Series([corrupted_questions])
        else:
            return pd.Series([None])

    # Apply the corruption process
    df_in_context["corrupted_data"] = df_in_context.progress_apply(
        process_corruption, axis=1
    )

    # Explode the dataframe to create separate rows for each corrupted question
    df_exploded = df_in_context.explode("corrupted_data").reset_index(drop=True)

    # Create new columns based on the corrupted data
    df_exploded["is_corrupted"] = df_exploded["corrupted_data"].notnull()
    df_exploded["corrupted_question"] = df_exploded["corrupted_data"].apply(
        lambda x: (
            x["corruption"]["corrupted_question"] if x and "corruption" in x else None
        )
    )
    df_exploded["original_entity"] = df_exploded["corrupted_data"].apply(
        lambda x: x["corruption"]["original"] if x and "corruption" in x else None
    )
    df_exploded["corrupted_entities"] = df_exploded["corrupted_data"].apply(
        lambda x: (
            x["corruption"]["corrupted_entities"]
            if x and "corruption" in x and x["corruption"]["corrupted_entities"]
            else None
        )
    )

    # Rename layout_type to objectType in corrupted_entities
    def update_layout_type(entities):
        if entities is None:
            return None

        # If it's a list of entities
        if isinstance(entities, list):
            return [
                {
                    "text": e["text"],
                    "page_id": e["page_id"],
                    "bbox": e["bbox"],
                    "obj_id": e["obj_id"],
                    "objectType": e["layout_type"],
                    "layout_type_id": e["layout_type_id"],
                }
                for e in entities
                if isinstance(e, dict)
            ]

        # If it's a single entity
        if isinstance(entities, dict):
            return {
                "text": entities["text"],
                "page_id": entities["page_id"],
                "bbox": entities["bbox"],
                "obj_id": entities["obj_id"],
                "objectType": entities["layout_type"],
                "layout_type_id": entities["layout_type_id"],
            }

        return entities

    # Apply the update to corrupted_entities
    df_exploded["corrupted_entities"] = df_exploded["corrupted_entities"].apply(
        update_layout_type
    )
    df_exploded["entity_type"] = df_exploded["corrupted_data"].apply(
        lambda x: x["corruption"]["entity_type"] if x and "corruption" in x else None
    )
    df_exploded["complexity"] = df_exploded["corrupted_data"].apply(
        lambda x: x["complexity"] if x else 0
    )
    df_exploded["question_entities"] = df_exploded["corrupted_data"].apply(
        lambda x: x["question_entities"] if x else None
    )
    df_exploded["entity_types"] = df_exploded["corrupted_data"].apply(
        lambda x: [x["corruption"]["entity_type"]] if x and "corruption" in x else None
    )
    df_exploded["verification_result"] = (
        "Not Applicable"  # You can update this if you implement verification
    )

    # Remove the temporary column
    df_exploded = df_exploded.drop(columns=["corrupted_data"])

    # Rename columns if needed
    df_exploded = df_exploded.rename(
        columns={
            "question": "original_question",
            "corrupted_question": "corrupted_question",
        }
    )

    # For non-corrupted questions, use the original question
    df_exploded.loc[~df_exploded["is_corrupted"], "corrupted_question"] = df_exploded[
        "original_question"
    ]

    # Before saving, let's check what columns are actually available
    print("\nAvailable columns in DataFrame:")
    print(df_exploded.columns.tolist())

    # Add OCR information to the columns to keep
    columns_to_keep = [
        "corrupted_question",
        "original_question",
        "complexity",
        "verification_result",
        "is_corrupted",
        "question_entities",
        "original_entity",
        "corrupted_entities",
        "entity_type",
        "original_answer_locations",
        "patch_entities",
        "layout_analysis",
    ]

    # Filter to keep only columns that exist in the DataFrame
    columns_to_keep = [col for col in columns_to_keep if col in df_exploded.columns]

    df_exploded = df_exploded[columns_to_keep]

    # Before saving to JSON, convert DataFrame to records and handle any non-serializable types
    def convert_to_serializable(obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Convert DataFrame to records
    records = df_exploded.to_dict(orient="records")
    serializable_records = []

    for record in records:
        serializable_record = {}
        for key, value in record.items():
            serializable_record[key] = convert_to_serializable(value)
        serializable_records.append(serializable_record)

    # Create metadata
    def flatten_list(lst):
        #Flatten a list of lists or single items into a single list
        result = []
        for item in lst:
            if isinstance(item, list):
                result.extend(item)
            else:
                result.append(item)
        return result

    metadata = {
        "total_questions": int(len(df_exploded)),
        "corrupted_questions": int(df_exploded["is_corrupted"].sum()),
        "entity_types": pd.Series(
            flatten_list(df_exploded["entity_type"].dropna().tolist())
        )
        .value_counts()
        .to_dict(),
    }

    # Create the output data structure with serializable data
    output_data = {"corrupted_questions": serializable_records, "metadata": metadata}

    # Save to JSON with all the information
    with open(params["output_corrupted"], "w") as f:
        json.dump(output_data, f, indent=2)

    def clean_corrupted_questions(input_file, output_file):
        # Read the JSON file
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Keep track of removed questions
        removed_count = {'duplicates': 0, 'invalid_format': 0}
        
        # Filter out entries where:
        # 1. corrupted_question equals original_question
        # 2. corrupted_question contains unwanted pattern
        filtered_questions = []
        for question in data['corrupted_questions']:
            corrupted = question['corrupted_question']
            original = question['original_question']
            
            # Check for duplicate questions
            if corrupted.lower() == original.lower():
                removed_count['duplicates'] += 1
                continue
            
            # Check for unwanted pattern
            if '\nuser' in corrupted:
                removed_count['invalid_format'] += 1
                continue
            
            filtered_questions.append(question)
        
        # Create new JSON with filtered questions
        cleaned_data = {'corrupted_questions': filtered_questions}
        
        # Save to new JSON file
        with open(output_file, 'w') as f:
            json.dump(cleaned_data, f, indent=2)
        
        return removed_count

    # Update the print statement after calling the function
    removed = clean_corrupted_questions(params["output_corrupted"], params["output_corrupted_cleaned"])
    print(f"Removed {removed['duplicates']} questions where corrupted matched original")
    print(f"Removed {removed['invalid_format']} questions with invalid format")

    print(f"Total questions processed: {len(df_augmented)}")
    print(f"Number of corrupted questions: {df_exploded['is_corrupted'].sum()}")

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
