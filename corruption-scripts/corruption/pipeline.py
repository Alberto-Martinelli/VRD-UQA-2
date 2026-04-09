import os
from data_loader import DataLoader
from entity_identifier import EntityIdentifier
from layout_with_ocr import DocumentAnalyzer

def sample_questions_to_corrupt(questions_df, percentage):
    # Calculate the number of questions to corrupt
    num_questions_to_corrupt = int(len(questions_df) * percentage / 100)
    # Ensure we don't try to sample more questions than available
    num_questions_to_corrupt = min(num_questions_to_corrupt, len(questions_df))

    # Sample questions with non-null answers
    df_to_corrupt = questions_df.sample(n=num_questions_to_corrupt)

    # Print information about the sampled data
    print(
        f"Number of questions selected for corruption: {len(df_to_corrupt)}/{len(questions_df)}"
    )

    return df_to_corrupt

def verify_all_images_present(questions_df):
    # Check that all page_ids mentioned in the dataset have corresponding image files
    print("\nVerifying image file existence...")
    all_images_exist = True
    for idx, row in questions_df.iterrows():
        image_paths = row["image_path"]
        # Handle case where image_path is a list
        if isinstance(image_paths, list):
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    print(f"Warning: Image file not found at path: {image_path}")
                    all_images_exist = False
        # Handle case where image_path is a single string
        else:
            if not os.path.exists(image_paths):
                print(f"Warning: Image file not found at path: {image_paths}")
                all_images_exist = False

    if all_images_exist:
        print("\nAll images are present!\n")
    else:
        print("\nSome images are missing!\n")
    return all_images_exist

# STEP 1
def load_data(params):
    raw_dataset_dict = DataLoader.load_dataset(params["base_path"], params["split"], params["dataset_name"], params["dataset_json_path"])
    questions_df = DataLoader.create_dataframe(raw_dataset_dict, params["dataset_name"], params["base_path"], params["dataset_json_path"])
    print(f"Total questions loaded: {len(questions_df)}")

    df_to_corrupt = sample_questions_to_corrupt(questions_df, params["percentage"])
    # df_to_corrupt.to_csv("df_to_corrupt.csv")
    return df_to_corrupt

# STEP 2
def identify_all_entities(params, df_to_corrupt):
    print("Setting up Entity Identifier...")
    entity_identifier = EntityIdentifier(
        dataset_name=params["dataset_name"],
        # Five boolean flags from the config that act as filters for which categories of entities to detect
        numerical=params["numerical"], # looks for numbers, quantities, percentages, etc.
        temporal=params["temporal"], # looks for dates, times, periods, etc.
        entity=params["entity"], # looks for people, organizations, products, etc.
        location=params["location"], # looks for places, addresses, etc.
        document=params["document"], # looks for document structural elements (Table 2, Section 3.1, page 4 etc.)
    )

    print("\nIdentifying entities for each question...")

    questions_list = df_to_corrupt["question"].tolist()
    question_with_entities = []
    for question in questions_list:
        entities = entity_identifier.identify_entities(question)
        question_with_entities.append(entities)
    return questions_list, question_with_entities, entity_identifier

# STEP 3
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
        print("Loading layout analysis models...")
        document_analyzer.load_models()
        print("Models loaded successfully.")
        
        # Process the dataframe to add layout analysis
        df_to_corrupt = document_analyzer.process_dataset_questions(
            df_to_corrupt, params["augmented_dataset_path"]
        )
    else:
        print("Augmented dataset already exists. Skipping layout analysis.")
    return df_to_corrupt

