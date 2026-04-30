import os
from data_loader import DataLoader
from entity_identifier import EntityIdentifier
from layout_with_ocr import DocumentAnalyzer
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from model_loader import ModelLoader
from in_context_modifier import InContextModifier
import logging

# ------------ Helpers ------------
def sample_questions_to_corrupt(questions_df, percentage):
    # Calculate the number of questions to corrupt
    num_questions_to_corrupt = int(len(questions_df) * percentage / 100)
    # Ensure we don't try to sample more questions than available
    num_questions_to_corrupt = min(num_questions_to_corrupt, len(questions_df))

    # Sample questions with non-null answers
    df_to_corrupt = questions_df.sample(n=num_questions_to_corrupt)

    # Print information about the sampled data
    logging.info(
        f"Number of questions selected for corruption: {len(df_to_corrupt)}/{len(questions_df)}"
    )

    return df_to_corrupt

def verify_all_images_present(questions_df):
    # Check that all page_ids mentioned in the dataset have corresponding image files
    logging.info("Verifying image file existence...")
    all_images_exist = True
    for idx, row in questions_df.iterrows():
        image_paths = row["image_path"]
        # Handle case where image_path is a list
        if isinstance(image_paths, list):
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    logging.warning(f"Image file not found at path: {image_path}")
                    all_images_exist = False
        # Handle case where image_path is a single string
        else:
            if not os.path.exists(image_paths):
                logging.warning(f"Image file not found at path: {image_paths}")
                all_images_exist = False

    if all_images_exist:
        logging.info("All images are present!")
    else:
        logging.warning("Some images are missing!")
    return all_images_exist

# HELPERS FOR CORRUPTION STEP
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

def process_layout_objects(row, entity_identifier):
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

def flatten_list(lst):
    #Flatten a list of lists or single items into a single list
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result

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

def extract_corruption_fields(x):
    """Extract all fields from a single corrupted_data dict."""
    if not x or "corruption" not in x:
        return {
            "corrupted_question": None, "original_entity": None,
            "corrupted_entities": None, "entity_type": None,
            "complexity": 0, "question_entities": None, "entity_types": None,
        }
    c = x["corruption"]
    return {
        "corrupted_question": c["corrupted_question"],
        "original_entity": c["original"],
        "corrupted_entities": update_layout_type(c.get("corrupted_entities")),
        "entity_type": c["entity_type"],
        "complexity": x["complexity"],
        "question_entities": x["question_entities"],
        "entity_types": [c["entity_type"]],
    }

# ------------ Pipeline Steps ------------
# STEP 1 (data_loader.py)
def load_data(params):
    raw_dataset_dict = DataLoader.load_dataset(params["base_path"], params["split"], params["dataset_name"], params["dataset_json_path"])
    logging.info(f"Total questions in original dataset: {len(raw_dataset_dict['data'])}")
    questions_df = DataLoader.create_dataframe(raw_dataset_dict, params["dataset_name"], params["base_path"], params["dataset_json_path"], params["split"])
    logging.info(f"Total questions loaded: {len(questions_df)}")

    df_to_corrupt = sample_questions_to_corrupt(questions_df, params["percentage"])
    # df_to_corrupt.to_csv("df_to_corrupt.csv")
    return df_to_corrupt

# STEP 2 (entity_identifier.py)
def identify_all_entities(params, df_to_corrupt):
    logging.info("Setting up Entity Identifier...")
    entity_identifier = EntityIdentifier(
        dataset_name=params["dataset_name"],
        # Five boolean flags from the config that act as filters for which categories of entities to detect
        numerical=params["numerical"], # looks for numbers, quantities, percentages, etc.
        temporal=params["temporal"], # looks for dates, times, periods, etc.
        entity=params["entity"], # looks for people, organizations, products, etc.
        location=params["location"], # looks for places, addresses, etc.
        document=params["document"], # looks for document structural elements (Table 2, Section 3.1, page 4 etc.)
    )

    logging.info("Identifying entities for each question...")

    questions_list = df_to_corrupt["question"].tolist()
    question_with_entities = []
    for question in questions_list:
        entities = entity_identifier.identify_entities(question)
        question_with_entities.append(entities)
    return questions_list, question_with_entities, entity_identifier

# STEP 3 (layout_with_ocr.py)
def create_augmented_dataset(params, df_to_corrupt):
    if not os.path.exists(params["augmented_dataset_path"]):
        # Model configuration
        model_config = {
            "model_name": params["layout_model"],
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 720 * 28 * 28,
        }

        # Initialize DocumentAnalyzer with config
        document_analyzer = DocumentAnalyzer(model_config, params["patch_saving_dir"], params["layout_saving_dir"])
        logging.info("Loading layout analysis models...")
        document_analyzer.load_models()
        logging.info("Models loaded successfully.")
        
        # Process the dataframe to add layout analysis
        df_to_corrupt = document_analyzer.process_dataset_questions(
            df_to_corrupt, params["augmented_dataset_path"]
        )
    else:
        logging.info("Augmented dataset already exists. Skipping layout analysis.")
    return df_to_corrupt

# STEP 4 CORRUPTION (in_context_modifier.py)
def corrupt_questions(params, entity_identifier):
    """Step 4: Load augmented dataset, corrupt questions, save results."""

    # --- 4a. Load augmented dataset and enrich with answer locations + entities ---
    with open(params["augmented_dataset_path"], "r", encoding="utf-8") as file:
        augmented_dataset = json.load(file)
    logging.info(f"Total number of questions in augmented dataset: {len(augmented_dataset.keys())}")

    df_augmented = pd.DataFrame(augmented_dataset).T
    df_augmented["question"] = df_augmented["question_data"].apply(lambda x: x["question"])

    # For each question, find where its answer appears in the document layout
    df_augmented["original_answer_locations"] = df_augmented.apply(find_answer_bbox, axis=1)

    # Run NER on each question and on every layout object's OCR text
    df_augmented["question_entities"] = df_augmented["question"].apply(
        entity_identifier.identify_entities
    )
    df_augmented["patch_entities"] = df_augmented.apply(
        process_layout_objects, axis=1, entity_identifier=entity_identifier
    )

    # --- 4b. Configure the corruption engine (LLM for question rewriting) ---
    model_loader = ModelLoader.get_instance()
    model_loader.load_model(params["model_provider"], params["model_name"])
    InContextModifier.set_model_loader(model_loader)
    InContextModifier.set_parameters(
        complexity=params["complexity"],
        in_document=True,
        out_document=True,
        generated_sample_per_complexity_greater_than_1=params["generated_sample_per_complexity_greater_than_1"],
    )

    # --- 4c. Run corruption on each question ---
    tqdm.pandas(desc="Corrupting questions")
    df_augmented["corrupted_data"] = df_augmented.progress_apply(process_corruption, axis=1)

    # --- 4d. Flatten results: one row per corruption variant ---
    # Each question may produce multiple corruptions; explode into separate rows
    logging.info("\nCorruption completed. Flattening results...")
    df_results = df_augmented.explode("corrupted_data").reset_index(drop=True)

    # Unpack the nested corruption dicts into flat columns
    extracted = df_results["corrupted_data"].apply(extract_corruption_fields).apply(pd.Series)
    df_results = pd.concat([df_results, extracted], axis=1)
    df_results["is_corrupted"] = df_results["corrupted_data"].notnull()
    df_results["verification_result"] = "Not Applicable"
    df_results = df_results.drop(columns=["corrupted_data"])

    # Rename for clarity in the output
    df_results = df_results.rename(columns={"question": "original_question"})

    # For questions that failed corruption, keep the original question text
    # Cast to object to avoid dtype mismatch when all corruptions are None (float64 vs str)
    df_results["corrupted_question"] = df_results["corrupted_question"].astype(object)
    df_results.loc[~df_results["is_corrupted"], "corrupted_question"] = df_results["original_question"]

    # --- 4e. Save results and clean up ---
    OUTPUT_COLUMNS = [
        "corrupted_question", "original_question", "complexity",
        "verification_result", "is_corrupted", "question_entities",
        "original_entity", "corrupted_entities", "entity_type",
        "original_answer_locations", "patch_entities", "layout_analysis",
    ]
    df_results = df_results[[col for col in OUTPUT_COLUMNS if col in df_results.columns]]

    # Serialize to JSON (convert numpy types to native Python)
    records = [
        {k: convert_to_serializable(v) for k, v in record.items()}
        for record in df_results.to_dict(orient="records")
    ]

    metadata = {
        "total_questions": int(len(df_results)),
        "corrupted_questions": int(df_results["is_corrupted"].sum()),
        "entity_types": pd.Series(
            flatten_list(df_results["entity_type"].dropna().tolist())
        ).value_counts().to_dict(),
    }

    logging.info(f"\nSaving corrupted questions to file {params['output_corrupted']}...")
    with open(params["output_corrupted"], "w") as f:
        json.dump({"corrupted_questions": records, "metadata": metadata}, f, indent=2)

    # Remove corruptions that are identical to the original or have bad formatting
    removed = clean_corrupted_questions(params["output_corrupted"], params["output_corrupted_cleaned"])
    
    total_variants = int(len(df_results))
    failed_gracefully = total_variants - int(df_results["is_corrupted"].sum())
    llm_duplicate_removals = removed['duplicates'] - failed_gracefully
    final_cleaned_count = total_variants - removed['duplicates'] - removed['invalid_format']

    logging.info(f"From an initial {len(augmented_dataset.keys())} questions (total number of questions in augmented dataset)")
    logging.info(f"Total questions processed: {len(df_augmented)}")
    logging.info(f"We identified {total_variants} corruption variants")
    logging.info(f"  -> {failed_gracefully} variants failed gracefully (e.g., couldn't find matching entity in OCR to swap)")
    logging.info(f"  -> {llm_duplicate_removals} questions were removed because the LLM rewrote them identically to the original")
    logging.info(f"  -> {removed['invalid_format']} questions were removed for invalid format (e.g., hallucinated tags)")
    logging.info(f"=== Final cleaned corrupted questions available: {final_cleaned_count} ===")
