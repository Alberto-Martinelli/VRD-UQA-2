import os
import logging
import itertools
from model_loader import ModelLoader
import pandas as pd


class InContextModifier:
    complexity = 1
    in_document = True
    out_document = True
    max_attempts = 5
    model_loader = None
    generated_sample_per_complexity_greater_than_1 = 5

    @classmethod
    def set_model_loader(cls, model_loader):
        cls.model_loader = model_loader

    @classmethod
    def set_parameters(
        cls,
        complexity,
        in_document,
        out_document,
        generated_sample_per_complexity_greater_than_1,
    ):
        cls.complexity = complexity
        cls.in_document = in_document
        cls.out_document = out_document
        cls.generated_sample_per_complexity_greater_than_1 = (
            generated_sample_per_complexity_greater_than_1
        )

    @classmethod
    def generate_text(cls, prompt):
        if cls.model_loader is None:
            raise ValueError("ModelLoader not set. Call set_model_loader first.")
        return cls.model_loader.generate_text(prompt)

    @classmethod
    def corrupt_entity(
        cls,
        entity,
        question,
        original_answer_locations,
        patch_entities,
        row,
    ):
        entity_text = entity["text"] if isinstance(entity, dict) else str(entity)
        entity_label = (
            entity.get("label", "unknown") if isinstance(entity, dict) else "unknown"
        )

        # Get answer info
        answer_text = ""
        original_obj_id = ""
        original_bbox = []
        answer_page_key = ""

        for answer_loc in original_answer_locations:
            answer_text = answer_loc.get("answer", "")
            answer_page_key = answer_loc.get("page_id", "")
            original_obj_id = answer_loc.get("object_typeID", "")
            original_bbox = answer_loc.get("bbox", [])

            print(f"Found answer: {answer_text} on page: {answer_page_key}")

        print(f"Processing entity: {entity_text} of type {entity_label}")
        corruptions = []

        if cls.in_document and original_answer_locations:
            # Get the first answer location
            answer_loc = original_answer_locations[0]
            answer_page_key = answer_loc["page_id"]
            answer_text = answer_loc["answer"]
            original_layout_type = answer_loc["object_type"]
            original_bbox = answer_loc["bbox"]

            print(
                f"Found answer: {answer_text} on page: {answer_page_key} in layout type: {original_layout_type}"
            )

            # Collect ALL entities of the same type
            all_matching_entities = []
            entity_counter = 0  # Counter for unique IDs

            for page_id, page_data in patch_entities.items():
                for obj_id, obj_data in page_data.items():
                    matching_entities = [
                        {
                            "text": e["text"],
                            "label": e["label"],
                            "page_id": page_id,
                            "layout_type": obj_data["type"],
                            "layout_type_id": obj_data["typeID"],
                            "bbox": obj_data["bbox"],
                            "obj_id": f"{obj_id}_entity_{entity_counter}",  # Make obj_id unique
                        }
                        for e in obj_data["entities"]
                        if e.get("label") == entity_label
                        and e["text"].lower() != entity_text.lower()
                    ]

                    # Increment counter for each entity
                    entity_counter += len(matching_entities)
                    all_matching_entities.extend(matching_entities)

            corruptions.extend(
                cls._generate_corruptions(
                    all_matching_entities,
                    entity_text,
                    entity_label,
                    question,
                    answer_page_key,
                    original_bbox,
                    original_obj_id,
                    original_layout_type,
                    answer_text,
                )
            )

        return corruptions

    @classmethod
    def _generate_corruptions(
        cls,
        candidate_entities,
        entity_text,
        entity_label,
        question,
        answer_page_key,
        original_bbox,
        original_obj_id,
        original_layout_type,
        answer_text,
    ):
        corruptions = []
        remaining_entities = candidate_entities.copy()
        attempts = 0

        print(f"Generating corruptions for {entity_text} of type {entity_label}")
        print(f"Remaining entities: {remaining_entities}")

        while remaining_entities and attempts < cls.max_attempts:
            candidate_texts = [e["text"] for e in remaining_entities]

            if not candidate_texts:
                break

            # Pick the first candidate text (no LLM)
            selected_entity = candidate_texts[0].strip().lower()

            matching_entities = [
                e for e in remaining_entities if e["text"].lower() == selected_entity
            ]

            if matching_entities:
                # Remove duplicates while preserving all unique occurrences
                seen = set()
                unique_matching_entities = []

                for e in candidate_entities:
                    if e["text"].lower() == selected_entity:
                        entity_key = (
                            e["text"],
                            e["page_id"],
                            tuple(e["bbox"]),
                            e["obj_id"],
                            e["layout_type"],
                            e["layout_type_id"],
                        )

                        if entity_key not in seen:
                            seen.add(entity_key)
                            unique_matching_entities.append(
                                {
                                    "text": e["text"],
                                    "page_id": e["page_id"],
                                    "bbox": e["bbox"],
                                    "obj_id": e["obj_id"],
                                    "layout_type": e["layout_type"],
                                    "layout_type_id": e["layout_type_id"],
                                }
                            )

                # Simple replacement
                corrupted_question = question.replace(entity_text, selected_entity)

                corruptions.append(
                    {
                        "entity_type": entity_label,
                        "original": {
                            "text": entity_text,
                            "page_id": answer_page_key,
                            "bbox": original_bbox,
                            "obj_id": original_obj_id,
                            "answer": answer_text,
                            "layout_type": original_layout_type,
                        },
                        "corrupted_entities": unique_matching_entities,
                        "original_question": question,
                        "corrupted_question": corrupted_question,
                    }
                )

                remaining_entities = [
                    e
                    for e in remaining_entities
                    if e["text"].lower() != selected_entity
                ]

            attempts += 1

        return corruptions

    @classmethod
    def corrupt_question(cls, row):
        print("---------------------------- New Corruption ----------------------")
        question = row["question"]
        question_entities = row["question_entities"]

        if pd.isna(question):
            print("Skipping corruption due to missing question")
            return None

        max_complexity = min(cls.complexity, len(question_entities))
        print(f"Using max complexity: {max_complexity}")

        corrupted_questions = []
        # Dictionary tracking how many samples we have for each complexity
        complexity_samples = {}

        for current_complexity in range(1, max_complexity + 1):
            print(f"Attempting corruptions with complexity {current_complexity}")

            # Initialize counter for this complexity if not exists
            complexity_samples.setdefault(current_complexity, 0)

            # Build all entity combinations
            entity_combinations = itertools.combinations(
                question_entities, current_complexity
            )

            for entity_combination in entity_combinations:
                # If we already have 2 samples for complexity > 1, skip further
                if (
                    current_complexity > 1
                    and complexity_samples[current_complexity]
                    >= cls.generated_sample_per_complexity_greater_than_1
                ):
                    print(
                        f"Skipping remaining combinations for complexity {current_complexity} - already have {cls.generated_sample_per_complexity_greater_than_1} samples"
                    )
                    break

                print(f"Processing combination: {entity_combination}")

                # Get corruptions for all entities in the combination
                all_entity_corruptions = []
                success = True

                for entity in entity_combination:
                    try:
                        entity_corruptions = cls.corrupt_entity(
                            entity,
                            question,
                            row.get("original_answer_locations", []),
                            row.get("patch_entities", {}),
                            row,
                        )

                        if entity_corruptions:
                            all_entity_corruptions.append(entity_corruptions)
                        else:
                            success = False
                            break

                    except Exception as e:
                        print(f"Error corrupting entity {entity}: {str(e)}")
                        success = False
                        break

                if not success or not all_entity_corruptions:
                    continue

                # Generate all possible combinations of corruptions
                corruption_combinations = itertools.product(*all_entity_corruptions)

                for corruption_combination in corruption_combinations:
                    # If we already have 2 samples for complexity > 1, skip further
                    if (
                        current_complexity > 1
                        and complexity_samples[current_complexity]
                        >= cls.generated_sample_per_complexity_greater_than_1
                    ):
                        break

                    # Apply all corruptions sequentially
                    current_question = question
                    all_originals = []
                    all_corrupted_entities = []
                    all_entity_types = []

                    for corruption in corruption_combination:
                        current_question = current_question.replace(
                            corruption["original"]["text"],
                            corruption["corrupted_entities"][0]["text"],
                        )
                        all_originals.append(corruption["original"])
                        all_corrupted_entities.extend(corruption["corrupted_entities"])
                        all_entity_types.append(corruption["entity_type"])

                    # Create a string of all question entities for reference
                    entities_string = ", ".join(
                        [
                            str(e["text"]) if isinstance(e, dict) else str(e)
                            for e in question_entities
                        ]
                    )

                    # Combine
                    combined_corruption = {
                        "original_question": question,
                        "corrupted_question": current_question,
                        "corruptions": list(corruption_combination),
                        "entity_type": all_entity_types,
                        "original": all_originals,
                        "corrupted_entities": all_corrupted_entities,
                    }

                    corrupted_questions.append(
                        {
                            "corruption": combined_corruption,
                            "complexity": current_complexity,
                            "question_entities": entities_string,
                        }
                    )

                    # Increment our sample count
                    complexity_samples[current_complexity] += 1

                    # If complexity > 1 and we have 2 samples, stop
                    if (
                        current_complexity > 1
                        and complexity_samples[current_complexity]
                        >= cls.generated_sample_per_complexity_greater_than_1
                    ):
                        print(
                            f"Collected {cls.generated_sample_per_complexity_greater_than_1} samples for complexity {current_complexity}, stopping."
                        )
                        break

        # After building the corrupted questions, rewrite them with the final prompt
        if corrupted_questions:
            for cq in corrupted_questions:
                all_corrupted_entities = set(
                    entity["text"] for entity in cq["corruption"]["corrupted_entities"]
                )
                original_question = cq["corruption"]["original_question"]
                current_corrupted_question = cq["corruption"]["corrupted_question"]

                prompt = f"""You are given two questions. The first one is the original one, the second one is the corrupted one.
The corruption is done based on entities extracted from the original question.

Original question: "{original_question}"
Corrupted question: "{current_corrupted_question}"

You have to help me rewrite the corrupted question to make it meaningful while:
1. Making it coherent and natural, while strictly keeping the exact same meaning
2. Ensuring it makes sense in the context of the original question
3. Never changing the corrupted entities: {list(all_corrupted_entities)}
4. Editing the question minimally - only what's needed to make it coherent
5. Guaranteeing that the final output is meaningful

Original: "What is the highest temperature recorded?"
Bad corruption: "What is the 85°F temperature recorded?"
Correct rewrite: "Was 85°F the highest temperature recorded?"

Good Examples:
Original: "Which year is mentioned first in the x axis?"
Bad corruption: "Which 1975 is mentioned first in the x axis?"
Good rewrite: "Is 1975 the first year mentioned in the x axis?"

Original: "Which company had the most sales in 2022?"
Bad corruption: "Which Microsoft had the most sales in 2022?"
Correct rewrite: "Did Microsoft have the most sales in 2022?"

Important: The following corrupted entities must be preserved in the rewritten question: {list(all_corrupted_entities)}
Important: Return only the rewritten question, without any explanation or introductions."""

                final_rewritten_question = cls.generate_text(prompt).strip()
                cq["corruption"]["corrupted_question"] = final_rewritten_question

        return corrupted_questions if corrupted_questions else None
