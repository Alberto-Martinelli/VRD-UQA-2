import json
import os
import logging
import re
from typing import Optional

import torch
import nltk
from nltk.tokenize import sent_tokenize
from gliner import GLiNER
from tqdm import tqdm
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENTITY_LABELS = {
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

# Per-label confidence thresholds (labels not listed here use DEFAULT_THRESHOLD)
LABEL_THRESHOLDS = {
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
}
DEFAULT_THRESHOLD = 0.75

MAX_CHUNK_LENGTH = 350


# ---------------------------------------------------------------------------
# Helpers (pure functions, no state)
# ---------------------------------------------------------------------------

def _build_flat_labels(
    numerical: bool,
    temporal: bool,
    entity: bool,
    location: bool,
    document: bool,
) -> list[str]:
    """Return the flat list of GLiNER label strings enabled by the flags."""
    flag_category_pairs = [
        (numerical, "Numerical Corruption"),
        (temporal, "Temporal Corruption"),
        (entity, "Entity Corruption"),
        (location, "Location Corruption"),
        (document, "Document Structure Corruption"),
    ]
    labels: list[str] = []
    for flag, category in flag_category_pairs:
        if flag:
            labels.extend(ENTITY_LABELS[category])
    return labels


def _chunk_text(text: str, max_len: int = MAX_CHUNK_LENGTH) -> list[str]:
    """
    Split *text* into sentence-level chunks, further breaking long sentences
    on semicolons / newlines / commas, and as a last resort on character count.
    """
    nltk.download("punkt", quiet=True)
    sentences = sent_tokenize(text)
    chunks: list[str] = []

    for sentence in sentences:
        if len(sentence) <= max_len:
            chunks.append(sentence)
            continue

        # Try splitting on semicolons / newlines first
        parts = re.split(r"([;\n])", sentence)
        if any(len(p) > max_len for p in parts):
            parts = re.split(r"([,;\n])", sentence)

        # Re-attach delimiters to preceding segment
        parts = [
            "".join(pair)
            for pair in zip(parts[::2], parts[1::2] + [""])
        ]

        # Hard cut anything still over the limit
        for part in parts:
            if len(part) <= max_len:
                chunks.append(part)
            else:
                chunks.extend(part[i : i + max_len] for i in range(0, len(part), max_len))

    return chunks


def _clean_entity_text(text: str) -> str:
    """Normalise whitespace, strip punctuation, lowercase."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.-]", "", text)
    text = text.strip(" -").lower()
    return text


def _passes_threshold(entity: dict) -> bool:
    """Return True if the entity's score exceeds its label-specific threshold."""
    label = entity.get("label", "")
    score = entity.get("score", 0.0)
    threshold = LABEL_THRESHOLDS.get(label, DEFAULT_THRESHOLD)
    return score > threshold


# ---------------------------------------------------------------------------
# OCR file readers (one function per dataset format)
# ---------------------------------------------------------------------------

def _read_mpdocvqa_ocr(ocr_data: dict, ocr_file_path: str) -> list[list]:
    """Return list of [text, bounding_box, file_path] from MPDocVQA OCR JSON."""
    results = []
    for item in ocr_data.get("LINE", []):
        if "Text" in item and "Geometry" in item and "BoundingBox" in item["Geometry"]:
            results.append([item["Text"], item["Geometry"]["BoundingBox"], ocr_file_path])
    return results


def _read_dude_ocr(ocr_data, ocr_file_path: str) -> list[list]:
    """Return list of [text, bounding_box, file_path] from DUDE OCR JSON."""
    if isinstance(ocr_data, list):
        ocr_data = ocr_data[0]
    results = []
    for block in ocr_data.get("Blocks", []):
        if block.get("BlockType") == "LINE":
            if "Text" in block and "Geometry" in block and "BoundingBox" in block["Geometry"]:
                results.append([block["Text"], block["Geometry"]["BoundingBox"], ocr_file_path])
    return results


OCR_READERS = {
    "MPDocVQA": _read_mpdocvqa_ocr,
    "DUDE": _read_dude_ocr,
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EntityIdentifier:
    """
    Wraps a GLiNER model to detect named entities in short texts
    (questions, OCR fragments).

    Responsibilities
    ----------------
    1. Run NER on arbitrary text  → ``identify_entities()``
    2. Parse dataset-specific OCR files into unified text → ``read_ocr_text()``
    """

    def __init__(
        self,
        dataset_name: str,
        numerical: bool = True,
        temporal: bool = True,
        entity: bool = True,
        location: bool = True,
        document: bool = True,
    ):
        self.dataset_name = dataset_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("EntityIdentifier using device: %s", self.device)

        self.model = GLiNER.from_pretrained("urchade/gliner_largev2").to(self.device)
        self.flat_labels = _build_flat_labels(numerical, temporal, entity, location, document)

    # ---- public API --------------------------------------------------------

    def identify_entities(self, text: str) -> list[dict]:
        """
        Detect entities in *text*.

        Returns a list of dicts, each with at least ``text``, ``label``,
        and ``score`` keys.
        """
        chunks = _chunk_text(text)
        entities: list[dict] = []

        for chunk in chunks:
            predictions = self.model.predict_entities(chunk, self.flat_labels)
            for ent in predictions:
                if _passes_threshold(ent):
                    ent["text"] = _clean_entity_text(ent["text"])
                    entities.append(ent)

        return entities

    def read_ocr_text(self, ocr_file_paths) -> list[list]:
        """
        Read one or more OCR JSON files and return a unified list of
        ``[text, bounding_box, file_path]`` entries.
        """
        if isinstance(ocr_file_paths, str):
            ocr_file_paths = [ocr_file_paths]

        reader = OCR_READERS.get(self.dataset_name)
        if reader is None:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        results: list[list] = []
        for path in ocr_file_paths:
            try:
                with open(path, "r") as f:
                    ocr_data = json.load(f)
                results.extend(reader(ocr_data, path))
            except Exception as e:
                logging.error("Error processing OCR file %s: %s", path, e)

        return results