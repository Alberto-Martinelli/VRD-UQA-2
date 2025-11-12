import json
import time
import os
from pathlib import Path


max_tokens = 1024
print("Gemini model initialized successfully")


def enrich_entity(entity):
    """
    Normalizes and returns an entity object with complete details.
    It picks a layout type from either 'layout_type' or 'layout_type_id'
    and includes object type if available.
    """
    layout = entity.get("layout_type") or entity.get("layout_type_id")
    object_type = entity.get(
        "objectType"
    )  # might be defined only for corrupted entities
    return {
        "text": entity.get("text", ""),
        "entity_type": entity.get("entity_type"),
        "page_id": entity.get("page_id"),
        "bbox": entity.get("bbox"),
        "obj_id": entity.get("obj_id"),
        "layout_type": layout,
        "object_type": object_type,
    }


def find_patch_matches(patch_entities, search_text, page_id=None):
    """
    Searches through the patch_entities dictionary for all occurrences
    of an entity with matching text (case-insensitive). If page_id is provided and exists,
    the search is limited to that page; otherwise, it searches all pages.

    Returns a list of matching entries. Each match contains:
      - page_id: the page where the match was found
      - object_id: the patch object key (e.g., "object0")
      - object_bbox: the bbox of that patch object in the page
      - entity_text, label, score, start and end: the details of the matching patch entity
    """
    matches = []
    if page_id and page_id in patch_entities:
        pages_to_search = {page_id: patch_entities[page_id]}
    else:
        pages_to_search = patch_entities
    for p_id, objects in pages_to_search.items():
        for obj_key, obj_info in objects.items():
            if "entities" in obj_info:
                for ent in obj_info["entities"]:
                    if ent.get("text", "").strip().lower() == search_text:
                        match = {
                            "page_id": p_id,
                            "object_id": obj_key,
                            "object_bbox": obj_info.get("bbox"),
                            "entity_text": ent.get("text"),
                            "label": ent.get("label"),
                            "score": ent.get("score"),
                            "start": ent.get("start"),
                            "end": ent.get("end"),
                        }
                        matches.append(match)
    return matches


def process_vqa_file(input_file, output_file):
    """
    Processes the VQA json file:
      1. For complexity‑1 questions, assigns the sole entity type to all original entities.
      2. Builds a global mapping of enriched entity info from all complexity‑1 questions' original_entity objects.
      3. For every question (regardless of complexity), updates its original_entity objects using the global mapping.
         (We leave the corrupted_entities list completely untouched to preserve the original data.)
      3.5. Adds a "label" field to every original_entity and corrupted_entity if not already present.
           For each entity the patch_entities are searched using its text and page_id, and if any match is found,
           the entity's "label" is set to the first match's label.
      4. Rebuilds the question_entities field so that each entry contains the enriched information.
      5. Searches through patch_entities to find all positional occurrences of each entity by its text.
         If multiple positions are found, they are added in a list under the key "positions".
      6. Finally, to avoid redundancy, only the basic information ("text") remains at the top level,
         while all detailed entity data (including entity type and positional info) appears inside "positions".
    """
    # Load JSON data from the input file
    with open(input_file, "r") as f:
        data = json.load(f)

    # Step 1: For complexity‑1 questions, update each original_entity with the sole entity type declared
    for question in data.get("corrupted_questions", []):
        if question.get("complexity") == 1 and question.get("entity_type"):
            for orig in question.get("original_entity", []):
                text = orig.get("text", "").strip()
                if text:
                    orig["entity_type"] = question["entity_type"][0]

    # Step 2: Build a global mapping (keyed by lowercase text) from complexity‑1 questions' original_entity objects
    global_entity_info = {}
    for question in data.get("corrupted_questions", []):
        if question.get("complexity") == 1:
            for orig in question.get("original_entity", []):
                token = orig.get("text", "").strip()
                if token:
                    key = token.lower()
                    global_entity_info[key] = enrich_entity(orig)

    # Step 3: For every question, update its own original_entity objects using the global mapping.
    # ---------------------------------------------------------------------------
    # Note: We intentionally do not update the corrupted_entities list so that it remains
    #       exactly the same as in the initial file, including any duplicates.
    for question in data.get("corrupted_questions", []):
        for orig in question.get("original_entity", []):
            token = orig.get("text", "").strip().lower()
            if token and token in global_entity_info:
                orig.update(global_entity_info[token])

        # Step 3.5: Add "label" to original_entity and corrupted_entities if missing.
        patch_entities = question.get("patch_entities", {})
        for orig in question.get("original_entity", []):
            if "label" not in orig or orig.get("label") is None:
                search_text = orig.get("text", "").strip().lower()
                page_id = orig.get("page_id")
                matches = find_patch_matches(patch_entities, search_text, page_id)
                if not matches:
                    matches = find_patch_matches(patch_entities, search_text)
                if matches:
                    orig["label"] = matches[0].get("label")
        for corr in question.get("corrupted_entities", []):
            if "label" not in corr or corr.get("label") is None:
                search_text = corr.get("text", "").strip().lower()
                page_id = corr.get("page_id")
                matches = find_patch_matches(patch_entities, search_text, page_id)
                if not matches:
                    matches = find_patch_matches(patch_entities, search_text)
                if matches:
                    corr["label"] = matches[0].get("label")

        # Step 4: Rebuild the question_entities field using the global mapping and local fallback searches.
        new_entities = []
        if isinstance(question.get("question_entities"), list):
            for ent in question["question_entities"]:
                token = ent.get("text", "").strip()
                key = token.lower()
                if token:
                    if key in global_entity_info:
                        new_entities.append(global_entity_info[key])
                    else:
                        # Fallback: search in current question's original_entity and corrupted_entities
                        found = None
                        for orig in question.get("original_entity", []):
                            if orig.get("text", "").strip().lower() == key:
                                found = enrich_entity(orig)
                                break
                        if not found:
                            for corr in question.get("corrupted_entities", []):
                                if corr.get("text", "").strip().lower() == key:
                                    found = enrich_entity(corr)
                                    break
                        if found:
                            new_entities.append(found)
                        else:
                            new_entities.append(
                                {
                                    "text": token,
                                    "entity_type": None,
                                    "page_id": None,
                                    "bbox": None,
                                    "obj_id": None,
                                    "layout_type": None,
                                    "object_type": None,
                                }
                            )
                else:
                    new_entities.append(
                        {
                            "text": token,
                            "entity_type": None,
                            "page_id": None,
                            "bbox": None,
                            "obj_id": None,
                            "layout_type": None,
                            "object_type": None,
                        }
                    )
            question["question_entities"] = new_entities
        elif isinstance(question.get("question_entities"), str):
            tokens = [
                token.strip()
                for token in question.get("question_entities", "").split(",")
                if token.strip()
            ]
            for token in tokens:
                key = token.lower()
                if key in global_entity_info:
                    new_entities.append(global_entity_info[key])
                else:
                    new_entities.append(
                        {
                            "text": token,
                            "entity_type": None,
                            "page_id": None,
                            "bbox": None,
                            "obj_id": None,
                            "layout_type": None,
                            "object_type": None,
                        }
                    )
            question["question_entities"] = new_entities
        else:
            # Fallback: build question_entities from original_entity if no information is available.
            new_entities = []
            for orig in question.get("original_entity", []):
                token = orig.get("text", "").strip()
                if token:
                    new_entities.append(enrich_entity(orig))
            question["question_entities"] = new_entities

        # Step 5: For each entity, search patch_entities to find all matching occurrences.
        patch_entities = question.get("patch_entities", {})
        for entity in question.get("question_entities", []):
            search_text = entity.get("text", "").strip().lower()
            page_id = entity.get("page_id")
            matches = find_patch_matches(patch_entities, search_text, page_id)
            if not matches:
                matches = find_patch_matches(patch_entities, search_text)
            # Add the list (even if empty) as "positions"
            entity["positions"] = matches

        # Step 6: Eliminate duplicate information at the top-level.
        # Only keep basic info ("text") and "positions". We remove the top-level "entity_type"
        # because it is already available in the label of the position.
        cleaned_entities = []
        for entity in question["question_entities"]:
            cleaned_entities.append(
                {"text": entity.get("text"), "positions": entity.get("positions", [])}
            )
        question["question_entities"] = cleaned_entities

    # Save the updated JSON data to the output file
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Processed file saved to {output_file}")


def process_all_folders():
    """
    Processes all JSON files in subdirectories beneath the script directory.
    Files in an "original" folder are processed and the output is written into a sibling folder "converted".
    Files that have already been converted are skipped.
    """
    script_dir = Path(__file__).parent

    for root, dirs, files in os.walk(script_dir):
        root_path = Path(root)
        if root_path.name == "converted":
            parent_dir = root_path.parent
            augmented_dir = parent_dir / "augmented"
            augmented_dir.mkdir(exist_ok=True)

            json_files = [f for f in files if f.endswith(".json")]
            for json_file in json_files:
                input_path = root_path / json_file
                output_filename = json_file.replace(".json", "_augmented.json")
                output_path = augmented_dir / output_filename

                if output_path.exists():
                    print(f"Skipping already converted file: {output_path}")
                    continue

                print(f"Processing: {input_path}")
                print(f"Saving to: {output_path}")
                try:
                    process_vqa_file(str(input_path), str(output_path))
                    print(f"Successfully processed: {input_path}")
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")
                    continue


if __name__ == "__main__":
    process_all_folders()
