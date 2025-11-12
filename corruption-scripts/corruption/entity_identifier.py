import json
import os
import logging
from gliner import GLiNER
import nltk
from nltk.tokenize import sent_tokenize
import torch
from tqdm import tqdm
import pandas as pd
import re


class EntityIdentifier:
    # Class-level DataFrame to store processed OCR pages
    ocr_cache = pd.DataFrame(columns=["page_id", "ocr_entities"])

    def __init__(
        self,
        dataset_type,
        numerical=True,
        temporal=True,
        entity=True,
        location=True,
        document=True,
    ):
        self.dataset_type = dataset_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"EntityIdentifier using device: {self.device}")
        self.model = GLiNER.from_pretrained("urchade/gliner_largev2").to(self.device)
        self.labels = {
            "Numerical Corruption": [
                "numerical_value_number",
                "measure_unit",
                "price_number_information",
                "price_numerical_value",
                "percentage",
                "temperature",
                "currency",
            ],
            "Temporal Corruption": [
                "date_information",
                "date_numerical_value",
                "time_information",
                "time_numerical_value",
                "year_number_information",
                "year_numerical_value",
            ],
            "Entity Corruption": [
                "person_name",
                "company_name",
                "event",
                "product",
                "food",
                "chemical_element",
                "job_title_name",
                "job_title_information",
                "animal",
                "plant",
                "movie",
                "book",
                "transport_means",
            ],
            "Location Corruption": [
                "country",
                "city",
                "street",
                "spatial_information",
                "continent",
                "postal_code_information",
                "postal_code_numerical_value",
            ],
            "Document Structure Corruption": [
                "document_position_information",
                "page_number_information",
                "page_number_numerical_value",
                "document_element_type",
                "document_element_information",
                "document_structure_information",
            ],
        }
        self.flat_labels = []
        if numerical:
            self.flat_labels.extend(self.labels["Numerical Corruption"])
        if temporal:
            self.flat_labels.extend(self.labels["Temporal Corruption"])
        if entity:
            self.flat_labels.extend(self.labels["Entity Corruption"])
        if location:
            self.flat_labels.extend(self.labels["Location Corruption"])
        if document:
            self.flat_labels.extend(self.labels["Document Structure Corruption"])
        nltk.download("punkt", quiet=True)  # Download punkt tokenizer data
        self.thresholds = {
            "document_position_information": 0.75,
            "page_number_information": 0.75,
            "page_number_numerical_value": 0.8,
            "document_element_type": 0.8,
            "document_element_information": 0.8,
            "document_structure_information": 0.8,
            "postal_code_information": 0.8,
            "postal_code_numerical_value": 0.78,
            "date_information": 0.75,
            "year_numerical_value": 0.7,
            "job_title_information": 0.8,
            "job_title_name": 0.9,
            "default": 0.75,
        }

    def identify_entities(self, text):
        sentences = sent_tokenize(text)
        all_entities = []

        for sentence in sentences:
            # If a single sentence is too long (>350 chars), split on semicolons and line breaks first
            if len(sentence) > 350:
                # Split primarily on semicolons and line breaks
                sub_sentences = re.split(r"([;\n])", sentence)
                # If still too long, then consider splitting on commas
                if any(len(s) > 350 for s in sub_sentences):
                    sub_sentences = re.split(r"([,;\n])", sentence)
                # Recombine the delimiters with the text
                sub_sentences = [
                    "".join(i)
                    for i in zip(sub_sentences[::2], sub_sentences[1::2] + [""])
                ]
                # If any subsection is still too long, use a simple character cut as last resort
                sub_sentences = [
                    s[i : i + 350]
                    for s in sub_sentences
                    for i in range(0, len(s), 350)
                    if len(s) > 350
                ]
            else:
                sub_sentences = [sentence]

            for sub_sentence in sub_sentences:
                entities = self.model.predict_entities(sub_sentence, self.flat_labels)
                # Filter entities based on their type-specific threshold
                high_confidence_entities = [
                    entity
                    for entity in entities
                    if entity.get("score", 0)
                    > self.thresholds.get(
                        entity.get("label"), self.thresholds["default"]
                    )
                ]

                # Clean up entities: remove double spaces and punctuation
                for entity in high_confidence_entities:
                    # Remove double or more spaces
                    cleaned_text = re.sub(r"\s+", " ", entity["text"])
                    # Remove punctuation except for specific cases (e.g., decimal points in numbers)
                    cleaned_text = re.sub(r"[^\w\s.-]", "", cleaned_text)
                    # Remove leading/trailing spaces and hyphens
                    cleaned_text = cleaned_text.strip(" -")
                    # To lowercase
                    cleaned_text = cleaned_text.lower()
                    # Update the entity text
                    entity["text"] = cleaned_text

                all_entities.extend(high_confidence_entities)

        return all_entities

    def unify_ocr_text(self, ocr_file_paths):
        if isinstance(ocr_file_paths, str):
            ocr_file_paths = [ocr_file_paths]

        text = []
        for ocr_file_path in ocr_file_paths:
            try:
                with open(ocr_file_path, "r") as f:
                    ocr_data = json.load(f)

                if self.dataset_type == "MPDocVQA":
                    self.process_mpdocvqa_ocr(ocr_data, text, ocr_file_path)
                elif self.dataset_type == "DUDE":
                    self.process_dude_ocr(ocr_data, text, ocr_file_path)
                else:
                    logging.error(f"Unknown dataset type: {self.dataset_type}")
            except Exception as e:
                logging.error(f"Error processing OCR file {ocr_file_path}: {str(e)}")
        return text

    def process_mpdocvqa_ocr(self, ocr_data, text, ocr_file_path):
        if "LINE" in ocr_data:
            for item in ocr_data["LINE"]:
                if (
                    "Text" in item
                    and "Geometry" in item
                    and "BoundingBox" in item["Geometry"]
                ):
                    text.append(
                        [item["Text"], item["Geometry"]["BoundingBox"], ocr_file_path]
                    )

    def process_dude_ocr(self, ocr_data, text, ocr_file_path):
        if isinstance(ocr_data, list):
            ocr_data = ocr_data[0]
        if "Blocks" in ocr_data:
            for block in ocr_data["Blocks"]:
                if block["BlockType"] == "LINE":
                    if (
                        "Text" in block
                        and "Geometry" in block
                        and "BoundingBox" in block["Geometry"]
                    ):
                        text.append(
                            [
                                block["Text"],
                                block["Geometry"]["BoundingBox"],
                                ocr_file_path,
                            ]
                        )

    def process_row(self, row):
        """Process a single row with layout analysis information."""
        question_entities = self.identify_entities(row["question"])
        if not question_entities:
            print(f"No entities found for question: {row['question']}")
            return [], []

        all_ocr_entities = []

        # Get layout analysis results if available
        layout_analysis = row.get("layout_analysis", {})

        if layout_analysis and "layout_elements" in layout_analysis:
            # Process each layout element's text
            for element in layout_analysis["layout_elements"]:
                page_id = element["page_id"]
                element_text = element["text"]
                element_type = element["type"]
                bbox = element["bbox"]

                # Check cache first
                cached_entities = EntityIdentifier.ocr_cache[
                    EntityIdentifier.ocr_cache["page_id"] == page_id
                ]

                if not cached_entities.empty:
                    # Use cached entities
                    element_entities = cached_entities.iloc[0]["ocr_entities"]
                else:
                    # Process new entities
                    element_entities = self.identify_entities(element_text)

                    # Add layout information to entities
                    for entity in element_entities:
                        entity["page_id"] = page_id
                        entity["bounding_box"] = bbox
                        entity["layout_type"] = element_type

                    # Cache the results
                    new_row = pd.DataFrame(
                        {"page_id": [page_id], "ocr_entities": [element_entities]}
                    )
                    EntityIdentifier.ocr_cache = pd.concat(
                        [EntityIdentifier.ocr_cache, new_row], ignore_index=True
                    )

                all_ocr_entities.extend(element_entities)
        else:
            # Fallback to original OCR processing if no layout analysis
            ocr_file_paths = row["ocr_path"]
            if isinstance(ocr_file_paths, str):
                ocr_file_paths = [ocr_file_paths]

            for ocr_file_path in ocr_file_paths:
                page_id = os.path.basename(ocr_file_path).split(".")[0]

                cached_entities = EntityIdentifier.ocr_cache[
                    EntityIdentifier.ocr_cache["page_id"] == page_id
                ]

                if not cached_entities.empty:
                    all_ocr_entities.extend(cached_entities.iloc[0]["ocr_entities"])
                else:
                    ocr_text = self.unify_ocr_text(ocr_file_path)
                    page_entities = []

                    for t in ocr_text:
                        ocr_entities = self.identify_entities(t[0])
                        for entity in ocr_entities:
                            entity["page_id"] = page_id
                            entity["bounding_box"] = t[1]
                        page_entities.extend(ocr_entities)

                    all_ocr_entities.extend(page_entities)

                    new_row = pd.DataFrame(
                        {"page_id": [page_id], "ocr_entities": [page_entities]}
                    )
                    EntityIdentifier.ocr_cache = pd.concat(
                        [EntityIdentifier.ocr_cache, new_row], ignore_index=True
                    )

        print(
            f"Processed question: {row['question']} and document: {row['page_ids'][0]}"
        )
        return question_entities, all_ocr_entities

    def process_dataframe(self, df):
        tqdm.pandas(desc="Processing rows")

        def safe_process_row(row):
            result = self.process_row(row)
            if result is None:
                return [], []  # Return empty lists if process_row returns None
            return result

        processed_data = df.progress_apply(safe_process_row, axis=1)

        df["question_entities"] = processed_data.apply(lambda x: x[0])
        df["ocr_entities"] = processed_data.apply(lambda x: x[1])

        return df
