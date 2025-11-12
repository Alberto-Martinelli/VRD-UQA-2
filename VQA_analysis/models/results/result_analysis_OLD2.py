

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from collections import Counter
import math
import re  # Added for window size extraction
from anls_star import anls_score
from gliner import GLiNER
import pandas as pd
import os
import torch
from PIL import Image

ENTITY_TYPES = [
    # Numerical Corruption
    "numerical_value_number",
    "measure_unit",
    "price_number_information",
    "price_numerical_value",
    "percentage",
    "temperature",
    "currency",
    # Temporal Corruption
    "date_information",
    "date_numerical_value",
    "time_information",
    "time_numerical_value",
    "year_number_information",
    "year_numerical_value",
    # Entity Corruption
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
    # Location Corruption
    "country",
    "city",
    "street",
    "spatial_information",
    "continent",
    "postal_code_information",
    "postal_code_numerical_value",
    # Document Structure Corruption
    "document_position_information",
    "page_number_information",
    "page_number_numerical_value",
    "document_element_type",
    "document_element_information",
    "document_structure_information",
]

LAYOUT_TYPES = [
    "title",
    "plain text",
    "abandon",
    "figure",
    "figure_caption",
    "table",
    "table_caption",
    "table_footnote",
    "isolate_formula",
    "formula_caption",
]


def get_sorted_ocr_text_and_layout(layout_analysis):
    """Extract and sort OCR text by layout patches and their bounding boxes

    Returns:
        list: List of dictionaries containing layout type, formatted OCR text, and bbox
        [{
            'layout': str,  # Layout type (table, text, title, etc.)
            'ocr_text_formatted': str,  # Sorted OCR text for this layout
            'bbox': list  # Bounding box coordinates [y1, x1, y2, x2]
        }]
    """
    layout_texts = {}
    layout_bboxes = {}  # Store bounding boxes for each layout type

    # Group texts by layout type
    for obj_id, obj in layout_analysis.items():
        if isinstance(obj, dict):
            # Get layout type if available
            layout_type = obj.get("type", "unknown")

            # Initialize this layout type if not exists
            if layout_type not in layout_texts:
                layout_texts[layout_type] = []
                layout_bboxes[layout_type] = []

            # If this object has OCR and BBOX
            if "OCR" in obj and "BBOX" in obj:
                bbox = obj["BBOX"]
                layout_texts[layout_type].append((bbox[1], bbox[0], obj["OCR"]))
                layout_bboxes[layout_type].append(bbox)

    # Create list of layout objects
    layout_objects = []

    for layout_type, texts in layout_texts.items():
        if texts:
            # Sort texts within this layout type by y, then x coordinate
            texts.sort()
            # Create formatted text for this layout
            formatted_text = "\n".join(item[2] for item in texts)

            # Calculate combined bbox for this layout type
            bboxes = layout_bboxes[layout_type]
            if bboxes:
                y1 = min(bbox[0] for bbox in bboxes)
                x1 = min(bbox[1] for bbox in bboxes)
                y2 = max(bbox[2] for bbox in bboxes)
                x2 = max(bbox[3] for bbox in bboxes)
                combined_bbox = [y1, x1, y2, x2]
            else:
                combined_bbox = None

            # Add layout object to list
            layout_objects.append(
                {
                    "layout": layout_type,
                    "ocr_text_formatted": formatted_text,
                    "bbox": combined_bbox,
                }
            )

    return layout_objects


class EntityIdentifier:
    def __init__(self, labels):
        self.labels = labels
        self.model = GLiNER.from_pretrained("urchade/gliner_largev2")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def identify_entities(self, text):
        return self.model.predict_entities(text, self.labels)


class VQAAnalyzer:
    def __init__(
        self, results, entity_verifier, dataset, debug=False, images_path=None
    ):
        self.results = results
        self.entity_types = [
            # Numerical Corruption
            "numerical_value_number",
            "measure_unit",
            "price_number_information",
            "price_numerical_value",
            "percentage",
            "temperature",
            "currency",
            # Temporal Corruption
            "date_information",
            "date_numerical_value",
            "time_information",
            "time_numerical_value",
            "year_number_information",
            "year_numerical_value",
            # Entity Corruption
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
            # Location Corruption
            "country",
            "city",
            "street",
            "spatial_information",
            "continent",
            "postal_code_information",
            "postal_code_numerical_value",
            # Document Structure Corruption
            "document_position_information",
            "page_number_information",
            "page_number_numerical_value",
            "document_element_type",
            "document_element_information",
            "document_structure_information",
        ]
        self.debug = debug
        self.entity_identifier = entity_verifier
        self.dataset = dataset
        self.images_path = images_path

    def calculate_metrics(self):
        metrics = {
            "CEPAR": self.CEPAR(),
            "OPAR_ANSL": self.OPAR_ANSL(),
            "QUR": self.QUR(),
            "UR": self.UR(),
            "AEMR_ALMR_HR": self.AEMR_ALMR_HR(),
            "QEWR": self.QEWR(),
            "QEPR": self.QEPR(),
            "CEBBOX": self.CEBBOX(self.images_path),
            "CEBBOX_UTD": self.CEBBOX_UTD(self.images_path),
            "UTD_LAYOUT": self.UTD_LAYOUT(),
            "LWP": self.LWP(self.images_path),
            "DEWP": self.DEWP(),
        }
        return metrics

    def CEPAR(self):
        # print('-'*100)
        print("CEPAR - ALL")
        tot = 0
        ans_page_corr_ent_page = 0
        aggregation_layout_type = {}
        aggregation_entity_type = {}

        # initial balancing layout and entity type
        total_layout_type = {}
        total_entity_type = {}

        tot_complexity_1 = 0
        tot_complexity_2 = 0
        tot_complexity_3 = 0
        ans_page_corr_ent_page_complexity_1 = 0
        ans_page_corr_ent_page_complexity_2 = 0
        ans_page_corr_ent_page_complexity_3 = 0
        aggregation_layout_type_complexity_1 = {}
        aggregation_layout_type_complexity_2 = {}
        aggregation_layout_type_complexity_3 = {}

        aggregation_entity_type_complexity_1 = {}
        aggregation_entity_type_complexity_2 = {}
        aggregation_entity_type_complexity_3 = {}

        CLOSEST = []
        COUNTER = 0

        C = 0
        TOT = 0

        for res in self.results:
            if (
                res["is_corrupted"]
                and "verification_result" in res
                and "vqa_results" in res["verification_result"]
                and len(res["verification_result"]["vqa_results"]) > 0
            ):

                corrupted_entities = res["corrupted_entities"]
                complexity = res["complexity"]
                # vqa_results_ans=res["verification_result"]["vqa_results"][0].get("answer", [])
                vqa_result = res["verification_result"]["vqa_results"][0]
                vqa_results_ans = vqa_result.get(
                    "answers", vqa_result.get("answer", [])
                )

                for ans in vqa_results_ans:
                    generated_answer = ans["answer_converted"].lower()
                    answer_converted = ans["answer_converted"].lower()
                    if generated_answer != "unable to determine":
                        tot += 1
                        if complexity == 1:
                            tot_complexity_1 += 1
                        if complexity == 2:
                            tot_complexity_2 += 1
                        if complexity == 3:
                            tot_complexity_3 += 1
                        page_path = ans["pages"]

                        ans_pages = [
                            full_page_path.split("/")[-1]
                            for full_page_path in ans.get("pages", [])
                        ]

                        seen_corrupted_entities = []
                        seen_corrupted_entities_text = []
                        flag = False
                        for corr_ent in corrupted_entities:

                            corr_ent_layout_type = corr_ent["objectType"]
                            if self.dataset == "DUDE":
                                corr_ent_page_number = int(
                                    corr_ent["page_id"].split(".")[0].split("_")[1]
                                )
                                # print(corr_ent_page_number)
                            else:
                                corr_ent_page_number = int(
                                    corr_ent["page_id"]
                                    .split(".")[0]
                                    .split("_")[1]
                                    .replace("p", "")
                                )
                            entity_type = corr_ent["label"]

                            for pp in page_path:
                                if self.dataset == "DUDE":
                                    page_number = int(pp.split(".")[0].split("_")[3])
                                else:
                                    page_number = int(
                                        pp.split(".")[0].split("_")[1].replace("p", "")
                                    )

                                if (
                                    page_number == corr_ent_page_number
                                    and flag == False
                                ):

                                    flag = True

                                    ans_page_corr_ent_page += 1
                                    if complexity == 1:
                                        ans_page_corr_ent_page_complexity_1 += 1
                                    if complexity == 2:
                                        ans_page_corr_ent_page_complexity_2 += 1
                                    if complexity == 3:
                                        ans_page_corr_ent_page_complexity_3 += 1
                                    # break

                                if page_number == corr_ent_page_number:
                                    if (
                                        corr_ent["text"]
                                        not in seen_corrupted_entities_text
                                    ):
                                        seen_corrupted_entities_text.append(
                                            corr_ent["text"]
                                        )
                                        seen_corrupted_entities.append(corr_ent)
                                    C += 1
                                    if (
                                        corr_ent_layout_type
                                        not in aggregation_layout_type
                                    ):
                                        aggregation_layout_type[
                                            corr_ent_layout_type
                                        ] = 0
                                    aggregation_layout_type[corr_ent_layout_type] += 1

                                    if entity_type not in aggregation_entity_type:
                                        aggregation_entity_type[entity_type] = 0
                                    aggregation_entity_type[entity_type] += 1

                                    if complexity == 1:
                                        if (
                                            corr_ent_layout_type
                                            not in aggregation_layout_type_complexity_1
                                        ):
                                            aggregation_layout_type_complexity_1[
                                                corr_ent_layout_type
                                            ] = 0
                                        aggregation_layout_type_complexity_1[
                                            corr_ent_layout_type
                                        ] += 1

                                        if (
                                            entity_type
                                            not in aggregation_entity_type_complexity_1
                                        ):
                                            aggregation_entity_type_complexity_1[
                                                entity_type
                                            ] = 0
                                        aggregation_entity_type_complexity_1[
                                            entity_type
                                        ] += 1

                                    if complexity == 2:
                                        if (
                                            corr_ent_layout_type
                                            not in aggregation_layout_type_complexity_2
                                        ):
                                            aggregation_layout_type_complexity_2[
                                                corr_ent_layout_type
                                            ] = 0
                                        aggregation_layout_type_complexity_2[
                                            corr_ent_layout_type
                                        ] += 1

                                        if (
                                            entity_type
                                            not in aggregation_entity_type_complexity_2
                                        ):
                                            aggregation_entity_type_complexity_2[
                                                entity_type
                                            ] = 0
                                        aggregation_entity_type_complexity_2[
                                            entity_type
                                        ] += 1

                                    if complexity == 3:
                                        if (
                                            corr_ent_layout_type
                                            not in aggregation_layout_type_complexity_3
                                        ):
                                            aggregation_layout_type_complexity_3[
                                                corr_ent_layout_type
                                            ] = 0
                                        aggregation_layout_type_complexity_3[
                                            corr_ent_layout_type
                                        ] += 1

                                        if (
                                            entity_type
                                            not in aggregation_entity_type_complexity_3
                                        ):
                                            aggregation_entity_type_complexity_3[
                                                entity_type
                                            ] = 0
                                        aggregation_entity_type_complexity_3[
                                            entity_type
                                        ] += 1

                        if len(seen_corrupted_entities) > 1:
                            # print("More than one corrupted entity in the same page", len(seen_corrupted_entities))
                            answer_position = []
                            for page in ans_pages:
                                if page in res.get("layout_analysis", {}).get(
                                    "pages", {}
                                ):
                                    layout_page = res["layout_analysis"]["pages"][page]
                                    layout_objs = layout_page.get("layout_analysis", {})
                                    flag = False
                                    for obj in layout_objs.values():
                                        ocr_text = obj.get("OCR", "").lower()
                                        ocr_label = obj.get("ObjectType", "").lower()
                                        if (
                                            answer_converted in ocr_text
                                            or ocr_text in answer_converted
                                        ):
                                            answer_position.append(
                                                obj.get("BBOX", None)
                                            )

                            if len(answer_position) == 0:
                                # HALLUCINATION
                                # print("HALLUCINATION")
                                continue

                            DISTANCE = 1000000
                            COUNTER += 1
                            for ce in seen_corrupted_entities:
                                ce_position = ce.get("bbox", [])
                                ce_entity_type = ce.get("label", "")
                                for ap in answer_position:
                                    dist = self.center_distance(ce_position, ap)
                                    if dist < DISTANCE:
                                        DISTANCE = dist
                                        closest = ce_entity_type
                                    elif dist == DISTANCE:
                                        closest = "SAME"
                            CLOSEST.append(closest)

        freq = Counter(CLOSEST)
        if self.debug:
            print("Counter of closest entities: ", COUNTER, len(CLOSEST))
            print(f"Frequency of closest entities. {freq}")

        percentage_layout_type = {}
        for key, value in aggregation_layout_type.items():
            percentage_layout_type[key] = round(value / C, 2)
        percentage_entity_type = {}
        for key, value in aggregation_entity_type.items():
            percentage_entity_type[key] = round(value / C, 2)

        percentage_layout_type_complexity_1 = {}
        for key, value in aggregation_layout_type_complexity_1.items():
            percentage_layout_type_complexity_1[key] = round(value / C, 2)
        percentage_entity_type_complexity_1 = {}
        for key, value in aggregation_entity_type_complexity_1.items():
            percentage_entity_type_complexity_1[key] = round(value / C, 2)

        percentage_layout_type_complexity_2 = {}
        for key, value in aggregation_layout_type_complexity_2.items():
            percentage_layout_type_complexity_2[key] = round(value / C, 2)
        percentage_entity_type_complexity_2 = {}
        for key, value in aggregation_entity_type_complexity_2.items():
            percentage_entity_type_complexity_2[key] = round(value / C, 2)

        percentage_layout_type_complexity_3 = {}
        for key, value in aggregation_layout_type_complexity_3.items():
            percentage_layout_type_complexity_3[key] = round(value / C, 2)
        percentage_entity_type_complexity_3 = {}
        for key, value in aggregation_entity_type_complexity_3.items():
            percentage_entity_type_complexity_3[key] = round(value / C, 2)

        return [
            ans_page_corr_ent_page,
            tot,
            ans_page_corr_ent_page_complexity_1,
            ans_page_corr_ent_page_complexity_2,
            ans_page_corr_ent_page_complexity_3,
            tot_complexity_1,
            tot_complexity_2,
            tot_complexity_3,
            percentage_layout_type,
            percentage_entity_type,
            percentage_layout_type_complexity_1,
            percentage_layout_type_complexity_2,
            percentage_layout_type_complexity_3,
            percentage_entity_type_complexity_1,
            percentage_entity_type_complexity_2,
            percentage_entity_type_complexity_3,
            freq,
            COUNTER,
        ]

    def CEPAR_OLD2(self):
        # print('-'*100)
        print("CEPAR - ALL")
        tot = 0
        ans_page_corr_ent_page = 0
        aggregation_layout_type = {}
        aggregation_entity_type = {}

        # initial balancing layout and entity type
        total_layout_type = {}
        total_entity_type = {}

        tot_complexity_1 = 0
        tot_complexity_2 = 0
        tot_complexity_3 = 0
        ans_page_corr_ent_page_complexity_1 = 0
        ans_page_corr_ent_page_complexity_2 = 0
        ans_page_corr_ent_page_complexity_3 = 0
        aggregation_layout_type_complexity_1 = {}
        aggregation_layout_type_complexity_2 = {}
        aggregation_layout_type_complexity_3 = {}

        aggregation_entity_type_complexity_1 = {}
        aggregation_entity_type_complexity_2 = {}
        aggregation_entity_type_complexity_3 = {}

        total_layout_type_complexity_1 = {}
        total_layout_type_complexity_2 = {}
        total_layout_type_complexity_3 = {}

        total_entity_type_complexity_1 = {}
        total_entity_type_complexity_2 = {}
        total_entity_type_complexity_3 = {}

        CLOSEST = []
        COUNTER = 0

        C = 0
        TOT = 0

        for res in self.results:
            if (
                res["is_corrupted"]
                and "verification_result" in res
                and "vqa_results" in res["verification_result"]
                and len(res["verification_result"]["vqa_results"]) > 0
            ):

                corrupted_entities = res["corrupted_entities"]
                complexity = res["complexity"]
                # vqa_results_ans=res["verification_result"]["vqa_results"][0].get("answer", [])
                vqa_result = res["verification_result"]["vqa_results"][0]
                vqa_results_ans = vqa_result.get(
                    "answers", vqa_result.get("answer", [])
                )

                for ans in vqa_results_ans:
                    generated_answer = ans["answer_converted"].lower()
                    answer_converted = ans["answer_converted"].lower()
                    if generated_answer != "unable to determine":
                        tot += 1
                        if complexity == 1:
                            tot_complexity_1 += 1
                        if complexity == 2:
                            tot_complexity_2 += 1
                        if complexity == 3:
                            tot_complexity_3 += 1
                        page_path = ans["pages"]
                        ans_pages = [
                            full_page_path.split("/")[-1]
                            for full_page_path in ans.get("pages", [])
                        ]

                        seen_corrupted_entities = []
                        seen_corrupted_entities_text = []
                        for corr_ent in corrupted_entities:

                            corr_ent_layout_type = corr_ent["objectType"]
                            corr_ent_page_number = int(
                                corr_ent["page_id"]
                                .split(".")[0]
                                .split("_")[1]
                                .replace("p", "")
                            )
                            entity_type = corr_ent["label"]

                            total_layout_type[corr_ent_layout_type] = (
                                total_layout_type.get(corr_ent_layout_type, 0) + 1
                            )
                            total_entity_type[entity_type] = (
                                total_entity_type.get(entity_type, 0) + 1
                            )

                            if complexity == 1:
                                total_layout_type_complexity_1[corr_ent_layout_type] = (
                                    total_layout_type_complexity_1.get(
                                        corr_ent_layout_type, 0
                                    )
                                    + 1
                                )
                                total_entity_type_complexity_1[entity_type] = (
                                    total_entity_type_complexity_1.get(entity_type, 0)
                                    + 1
                                )

                            if complexity == 2:
                                total_layout_type_complexity_2[corr_ent_layout_type] = (
                                    total_layout_type_complexity_2.get(
                                        corr_ent_layout_type, 0
                                    )
                                    + 1
                                )
                                total_entity_type_complexity_2[entity_type] = (
                                    total_entity_type_complexity_2.get(entity_type, 0)
                                    + 1
                                )

                            if complexity == 3:
                                total_layout_type_complexity_3[corr_ent_layout_type] = (
                                    total_layout_type_complexity_3.get(
                                        corr_ent_layout_type, 0
                                    )
                                    + 1
                                )
                                total_entity_type_complexity_3[entity_type] = (
                                    total_entity_type_complexity_3.get(entity_type, 0)
                                    + 1
                                )

                            flag = False
                            for pp in page_path:
                                page_number = int(
                                    pp.split(".")[0].split("_")[1].replace("p", "")
                                )

                                if (
                                    page_number == corr_ent_page_number
                                    and flag == False
                                ):

                                    flag = True

                                    if (
                                        corr_ent["text"]
                                        not in seen_corrupted_entities_text
                                    ):
                                        seen_corrupted_entities_text.append(
                                            corr_ent["text"]
                                        )
                                        seen_corrupted_entities.append(corr_ent)

                                    ans_page_corr_ent_page += 1
                                    if complexity == 1:
                                        ans_page_corr_ent_page_complexity_1 += 1
                                    if complexity == 2:
                                        ans_page_corr_ent_page_complexity_2 += 1
                                    if complexity == 3:
                                        ans_page_corr_ent_page_complexity_3 += 1
                                    # break

                                if page_number == corr_ent_page_number:
                                    C += 1
                                    if (
                                        corr_ent_layout_type
                                        not in aggregation_layout_type
                                    ):
                                        aggregation_layout_type[
                                            corr_ent_layout_type
                                        ] = 0
                                    aggregation_layout_type[corr_ent_layout_type] += 1

                                    if entity_type not in aggregation_entity_type:
                                        aggregation_entity_type[entity_type] = 0
                                    aggregation_entity_type[entity_type] += 1

                                    if complexity == 1:
                                        if (
                                            corr_ent_layout_type
                                            not in aggregation_layout_type_complexity_1
                                        ):
                                            aggregation_layout_type_complexity_1[
                                                corr_ent_layout_type
                                            ] = 0
                                        aggregation_layout_type_complexity_1[
                                            corr_ent_layout_type
                                        ] += 1

                                        if (
                                            entity_type
                                            not in aggregation_entity_type_complexity_1
                                        ):
                                            aggregation_entity_type_complexity_1[
                                                entity_type
                                            ] = 0
                                        aggregation_entity_type_complexity_1[
                                            entity_type
                                        ] += 1

                                    if complexity == 2:
                                        if (
                                            corr_ent_layout_type
                                            not in aggregation_layout_type_complexity_2
                                        ):
                                            aggregation_layout_type_complexity_2[
                                                corr_ent_layout_type
                                            ] = 0
                                        aggregation_layout_type_complexity_2[
                                            corr_ent_layout_type
                                        ] += 1

                                        if (
                                            entity_type
                                            not in aggregation_entity_type_complexity_2
                                        ):
                                            aggregation_entity_type_complexity_2[
                                                entity_type
                                            ] = 0
                                        aggregation_entity_type_complexity_2[
                                            entity_type
                                        ] += 1

                                    if complexity == 3:
                                        if (
                                            corr_ent_layout_type
                                            not in aggregation_layout_type_complexity_3
                                        ):
                                            aggregation_layout_type_complexity_3[
                                                corr_ent_layout_type
                                            ] = 0
                                        aggregation_layout_type_complexity_3[
                                            corr_ent_layout_type
                                        ] += 1

                                        if (
                                            entity_type
                                            not in aggregation_entity_type_complexity_3
                                        ):
                                            aggregation_entity_type_complexity_3[
                                                entity_type
                                            ] = 0
                                        aggregation_entity_type_complexity_3[
                                            entity_type
                                        ] += 1
                                    # break

                        if len(seen_corrupted_entities) > 1:
                            # print("More than one corrupted entity in the same page", len(seen_corrupted_entities))
                            answer_position = []
                            for page in ans_pages:
                                if page in res.get("layout_analysis", {}).get(
                                    "pages", {}
                                ):
                                    layout_page = res["layout_analysis"]["pages"][page]
                                    layout_objs = layout_page.get("layout_analysis", {})
                                    flag = False
                                    for obj in layout_objs.values():
                                        ocr_text = obj.get("OCR", "").lower()
                                        ocr_label = obj.get("ObjectType", "").lower()
                                        if (
                                            answer_converted in ocr_text
                                            or ocr_text in answer_converted
                                        ):
                                            answer_position.append(
                                                obj.get("BBOX", None)
                                            )

                            if len(answer_position) == 0:
                                # HALLUCINATION
                                # print("HALLUCINATION")
                                continue

                            DISTANCE = 1000000
                            COUNTER += 1
                            for ce in seen_corrupted_entities:
                                ce_position = ce.get("bbox", [])
                                ce_entity_type = ce.get("label", "")
                                for ap in answer_position:
                                    dist = self.center_distance(ce_position, ap)
                                    if dist < DISTANCE:
                                        DISTANCE = dist
                                        closest = ce_entity_type
                                    elif dist == DISTANCE:
                                        closest = "SAME"
                            CLOSEST.append(closest)

        freq = Counter(CLOSEST)
        if self.debug:
            print("Counter of closest entities: ", COUNTER, len(CLOSEST))
            print(f"Frequency of closest entities. {freq}")

        # make total layout and entity type a percentage
        percentage_layout_type = {}
        for key, value in total_layout_type.items():
            percentage_layout_type[key] = round(
                100 * value / sum(total_layout_type.values()), 2
            )
        percentage_entity_type = {}
        for key, value in total_entity_type.items():
            percentage_entity_type[key] = round(
                100 * value / sum(total_layout_type.values()), 2
            )

        percentage_layout_type_complexity_1 = {}
        for key, value in total_layout_type_complexity_1.items():
            percentage_layout_type_complexity_1[key] = round(
                100 * value / sum(total_layout_type_complexity_1.values()), 2
            )
        percentage_entity_type_complexity_1 = {}
        for key, value in total_entity_type_complexity_1.items():
            percentage_entity_type_complexity_1[key] = round(
                100 * value / sum(total_layout_type_complexity_1.values()), 2
            )

        percentage_layout_type_complexity_2 = {}
        for key, value in total_layout_type_complexity_2.items():
            percentage_layout_type_complexity_2[key] = round(
                100 * value / sum(total_layout_type_complexity_2.values()), 2
            )
        percentage_entity_type_complexity_2 = {}
        for key, value in total_entity_type_complexity_2.items():
            percentage_entity_type_complexity_2[key] = round(
                100 * value / sum(total_layout_type_complexity_2.values()), 2
            )

        percentage_layout_type_complexity_3 = {}
        for key, value in total_layout_type_complexity_3.items():
            percentage_layout_type_complexity_3[key] = round(
                100 * value / sum(total_layout_type_complexity_3.values()), 2
            )
        percentage_entity_type_complexity_3 = {}
        for key, value in total_entity_type_complexity_3.items():
            percentage_entity_type_complexity_3[key] = round(
                100 * value / sum(total_layout_type_complexity_3.values()), 2
            )

        return [
            ans_page_corr_ent_page,
            tot,
            aggregation_layout_type,
            aggregation_entity_type,
            percentage_layout_type,
            percentage_entity_type,
            aggregation_layout_type_complexity_1,
            aggregation_layout_type_complexity_2,
            aggregation_layout_type_complexity_3,
            aggregation_entity_type_complexity_1,
            aggregation_entity_type_complexity_2,
            aggregation_entity_type_complexity_3,
            percentage_layout_type_complexity_1,
            percentage_layout_type_complexity_2,
            percentage_layout_type_complexity_3,
            percentage_entity_type_complexity_1,
            percentage_entity_type_complexity_2,
            percentage_entity_type_complexity_3,
            tot_complexity_1,
            tot_complexity_2,
            tot_complexity_3,
            ans_page_corr_ent_page_complexity_1,
            ans_page_corr_ent_page_complexity_2,
            ans_page_corr_ent_page_complexity_3,
            freq,
            COUNTER,
        ]

    def CEPAR_OLD(self):
        # print('-'*100)
        print("CEPAR - ALL")
        tot = 0
        ans_page_corr_ent_page = 0
        aggregation_layout_type = {}
        aggregation_entity_type = {}

        # initial balancing layout and entity type
        total_layout_type = {}
        total_entity_type = {}

        tot_complexity_1 = 0
        tot_complexity_2 = 0
        tot_complexity_3 = 0
        ans_page_corr_ent_page_complexity_1 = 0
        ans_page_corr_ent_page_complexity_2 = 0
        ans_page_corr_ent_page_complexity_3 = 0
        aggregation_layout_type_complexity_1 = {}
        aggregation_layout_type_complexity_2 = {}
        aggregation_layout_type_complexity_3 = {}

        aggregation_entity_type_complexity_1 = {}
        aggregation_entity_type_complexity_2 = {}
        aggregation_entity_type_complexity_3 = {}

        total_layout_type_complexity_1 = {}
        total_layout_type_complexity_2 = {}
        total_layout_type_complexity_3 = {}

        total_entity_type_complexity_1 = {}
        total_entity_type_complexity_2 = {}
        total_entity_type_complexity_3 = {}

        for res in self.results:
            if (
                res["is_corrupted"]
                and "verification_result" in res
                and "vqa_results" in res["verification_result"]
                and len(res["verification_result"]["vqa_results"]) > 0
            ):

                corrupted_entities = res["corrupted_entities"]
                complexity = res["complexity"]
                # vqa_results_ans=res["verification_result"]["vqa_results"][0].get("answer", [])
                vqa_result = res["verification_result"]["vqa_results"][0]
                vqa_results_ans = vqa_result.get(
                    "answers", vqa_result.get("answer", [])
                )

                for corr_ent in corrupted_entities:

                    corr_ent_layout_type = corr_ent["objectType"]
                    corr_ent_page = corr_ent["page_id"]
                    corr_ent_page_number = int(
                        corr_ent_page.split(".")[0].split("_")[1].replace("p", "")
                    )
                    entity_type = corr_ent["label"]

                    total_layout_type[corr_ent_layout_type] = (
                        total_layout_type.get(corr_ent_layout_type, 0) + 1
                    )
                    total_entity_type[entity_type] = (
                        total_entity_type.get(entity_type, 0) + 1
                    )

                    if complexity == 1:
                        total_layout_type_complexity_1[corr_ent_layout_type] = (
                            total_layout_type_complexity_1.get(corr_ent_layout_type, 0)
                            + 1
                        )
                        total_entity_type_complexity_1[entity_type] = (
                            total_entity_type_complexity_1.get(entity_type, 0) + 1
                        )

                    if complexity == 2:
                        total_layout_type_complexity_2[corr_ent_layout_type] = (
                            total_layout_type_complexity_2.get(corr_ent_layout_type, 0)
                            + 1
                        )
                        total_entity_type_complexity_2[entity_type] = (
                            total_entity_type_complexity_2.get(entity_type, 0) + 1
                        )

                    if complexity == 3:
                        total_layout_type_complexity_3[corr_ent_layout_type] = (
                            total_layout_type_complexity_3.get(corr_ent_layout_type, 0)
                            + 1
                        )
                        total_entity_type_complexity_3[entity_type] = (
                            total_entity_type_complexity_3.get(entity_type, 0) + 1
                        )

                    for ans in vqa_results_ans:
                        generated_answer = ans["answer_converted"].lower()
                        if generated_answer != "unable to determine":
                            tot += 1
                            if complexity == 1:
                                tot_complexity_1 += 1
                            if complexity == 2:
                                tot_complexity_2 += 1
                            if complexity == 3:
                                tot_complexity_3 += 1
                            page_path = ans["pages"]
                            for pp in page_path:
                                page_number = int(
                                    pp.split(".")[0].split("_")[1].replace("p", "")
                                )

                                if page_number == corr_ent_page_number:
                                    ans_page_corr_ent_page += 1
                                    if complexity == 1:
                                        ans_page_corr_ent_page_complexity_1 += 1
                                    if complexity == 2:
                                        ans_page_corr_ent_page_complexity_2 += 1
                                    if complexity == 3:
                                        ans_page_corr_ent_page_complexity_3 += 1

                                    if (
                                        corr_ent_layout_type
                                        not in aggregation_layout_type
                                    ):
                                        aggregation_layout_type[
                                            corr_ent_layout_type
                                        ] = 0
                                    aggregation_layout_type[corr_ent_layout_type] += 1

                                    if entity_type not in aggregation_entity_type:
                                        aggregation_entity_type[entity_type] = 0
                                    aggregation_entity_type[entity_type] += 1

                                    if complexity == 1:
                                        if (
                                            corr_ent_layout_type
                                            not in aggregation_layout_type_complexity_1
                                        ):
                                            aggregation_layout_type_complexity_1[
                                                corr_ent_layout_type
                                            ] = 0
                                        aggregation_layout_type_complexity_1[
                                            corr_ent_layout_type
                                        ] += 1

                                        if (
                                            entity_type
                                            not in aggregation_entity_type_complexity_1
                                        ):
                                            aggregation_entity_type_complexity_1[
                                                entity_type
                                            ] = 0
                                        aggregation_entity_type_complexity_1[
                                            entity_type
                                        ] += 1

                                    if complexity == 2:
                                        if (
                                            corr_ent_layout_type
                                            not in aggregation_layout_type_complexity_2
                                        ):
                                            aggregation_layout_type_complexity_2[
                                                corr_ent_layout_type
                                            ] = 0
                                        aggregation_layout_type_complexity_2[
                                            corr_ent_layout_type
                                        ] += 1

                                        if (
                                            entity_type
                                            not in aggregation_entity_type_complexity_2
                                        ):
                                            aggregation_entity_type_complexity_2[
                                                entity_type
                                            ] = 0
                                        aggregation_entity_type_complexity_2[
                                            entity_type
                                        ] += 1

                                    if complexity == 3:
                                        if (
                                            corr_ent_layout_type
                                            not in aggregation_layout_type_complexity_3
                                        ):
                                            aggregation_layout_type_complexity_3[
                                                corr_ent_layout_type
                                            ] = 0
                                        aggregation_layout_type_complexity_3[
                                            corr_ent_layout_type
                                        ] += 1

                                        if (
                                            entity_type
                                            not in aggregation_entity_type_complexity_3
                                        ):
                                            aggregation_entity_type_complexity_3[
                                                entity_type
                                            ] = 0
                                        aggregation_entity_type_complexity_3[
                                            entity_type
                                        ] += 1
                                    break

        # make total layout and entity type a percentage
        percentage_layout_type = {}
        for key, value in total_layout_type.items():
            percentage_layout_type[key] = round(
                100 * value / sum(total_layout_type.values()), 2
            )
        percentage_entity_type = {}
        for key, value in total_entity_type.items():
            percentage_entity_type[key] = round(
                100 * value / sum(total_layout_type.values()), 2
            )

        percentage_layout_type_complexity_1 = {}
        for key, value in total_layout_type_complexity_1.items():
            percentage_layout_type_complexity_1[key] = round(
                100 * value / sum(total_layout_type_complexity_1.values()), 2
            )
        percentage_entity_type_complexity_1 = {}
        for key, value in total_entity_type_complexity_1.items():
            percentage_entity_type_complexity_1[key] = round(
                100 * value / sum(total_layout_type_complexity_1.values()), 2
            )

        percentage_layout_type_complexity_2 = {}
        for key, value in total_layout_type_complexity_2.items():
            percentage_layout_type_complexity_2[key] = round(
                100 * value / sum(total_layout_type_complexity_2.values()), 2
            )
        percentage_entity_type_complexity_2 = {}
        for key, value in total_entity_type_complexity_2.items():
            percentage_entity_type_complexity_2[key] = round(
                100 * value / sum(total_layout_type_complexity_2.values()), 2
            )

        percentage_layout_type_complexity_3 = {}
        for key, value in total_layout_type_complexity_3.items():
            percentage_layout_type_complexity_3[key] = round(
                100 * value / sum(total_layout_type_complexity_3.values()), 2
            )
        percentage_entity_type_complexity_3 = {}
        for key, value in total_entity_type_complexity_3.items():
            percentage_entity_type_complexity_3[key] = round(
                100 * value / sum(total_layout_type_complexity_3.values()), 2
            )

        return [
            ans_page_corr_ent_page,
            tot,
            aggregation_layout_type,
            aggregation_entity_type,
            percentage_layout_type,
            percentage_entity_type,
            aggregation_layout_type_complexity_1,
            aggregation_layout_type_complexity_2,
            aggregation_layout_type_complexity_3,
            aggregation_entity_type_complexity_1,
            aggregation_entity_type_complexity_2,
            aggregation_entity_type_complexity_3,
            percentage_layout_type_complexity_1,
            percentage_layout_type_complexity_2,
            percentage_layout_type_complexity_3,
            percentage_entity_type_complexity_1,
            percentage_entity_type_complexity_2,
            percentage_entity_type_complexity_3,
            tot_complexity_1,
            tot_complexity_2,
            tot_complexity_3,
            ans_page_corr_ent_page_complexity_1,
            ans_page_corr_ent_page_complexity_2,
            ans_page_corr_ent_page_complexity_3,
        ]

    def OPAR_ANSL(self):
        # print('-'*100)
        print("OPAR + ANSL")
        answer_same_page_original = 0
        answer_same_page_original_complexity_1 = 0
        answer_same_page_original_complexity_2 = 0
        answer_same_page_original_complexity_3 = 0
        answer_same_page_same_text_original = 0
        answer_same_page_same_text_original_complexity_1 = 0
        answer_same_page_same_text_original_complexity_2 = 0
        answer_same_page_same_text_original_complexity_3 = 0
        tot = 0
        anls = []
        anls_complexity_1 = []
        anls_complexity_2 = []
        anls_complexity_3 = []
        for res in self.results:
            if (
                res["is_corrupted"]
                and "verification_result" in res
                and "vqa_results" in res["verification_result"]
                and len(res["verification_result"]["vqa_results"]) > 0
            ):

                tot += 1
                original_answer_location = res["original_answer_locations"][0]
                original_answer_page = original_answer_location["page_id"]
                if self.dataset == "DUDE":
                    original_answer_page_number = int(
                        original_answer_page.split(".")[0].split("_")[1]
                    )
                else:
                    original_answer_page_number = int(
                        original_answer_page.split(".")[0]
                        .split("_")[1]
                        .replace("p", "")
                    )
                original_answer_text = original_answer_location["answer"].lower()
                complexity = res["complexity"]

                vqa_result = res["verification_result"]["vqa_results"][0]
                vqa_results_ans = vqa_result.get(
                    "answers", vqa_result.get("answer", [])
                )
                # print(len(vqa_results_ans))
                for ans in vqa_results_ans:
                    page_path = ans["pages"]
                    generated_answer = ans["answer_converted"].lower()

                    for pp in page_path:
                        # print(pp)
                        if self.dataset == "DUDE":
                            page_number = int(pp.split(".")[0].split("_")[3])
                        else:
                            page_number = int(
                                pp.split(".")[0].split("_")[1].replace("p", "")
                            )
                        if generated_answer != "unable to determine":
                            if original_answer_page_number == page_number:
                                answer_same_page_original += 1
                                if complexity == 1:
                                    answer_same_page_original_complexity_1 += 1
                                if complexity == 2:
                                    answer_same_page_original_complexity_2 += 1
                                if complexity == 3:
                                    answer_same_page_original_complexity_3 += 1
                                # break

                                if (
                                    generated_answer in original_answer_text
                                    or original_answer_text in generated_answer
                                ):
                                    answer_same_page_same_text_original += 1
                                    if complexity == 1:
                                        answer_same_page_same_text_original_complexity_1 += (
                                            1
                                        )
                                    if complexity == 2:
                                        answer_same_page_same_text_original_complexity_2 += (
                                            1
                                        )
                                    if complexity == 3:
                                        answer_same_page_same_text_original_complexity_3 += (
                                            1
                                        )
                                    # break
                                else:
                                    score = anls_score(
                                        generated_answer, original_answer_text
                                    )
                                    anls.append(score)
                                    if complexity == 1:
                                        anls_complexity_1.append(score)
                                    if complexity == 2:
                                        anls_complexity_2.append(score)
                                    if complexity == 3:
                                        anls_complexity_3.append(score)

        return [
            answer_same_page_original,
            answer_same_page_same_text_original,
            tot,
            answer_same_page_original_complexity_1,
            answer_same_page_original_complexity_2,
            answer_same_page_original_complexity_3,
            answer_same_page_same_text_original_complexity_1,
            answer_same_page_same_text_original_complexity_2,
            answer_same_page_same_text_original_complexity_3,
            anls,
            anls_complexity_1,
            anls_complexity_2,
            anls_complexity_3,
        ]

    def QUR(self):
        print("QUR")
        correct_unable = 0
        correct_unable_complexity_1 = 0
        correct_unable_complexity_2 = 0
        correct_unable_complexity_3 = 0
        total_corrupted = 0

        for res in self.results:
            if (
                res["is_corrupted"]
                and "verification_result" in res
                and "vqa_results" in res["verification_result"]
                and len(res["verification_result"]["vqa_results"]) > 0
            ):
                total_corrupted += 1
                # Get all answers for this question
                vqa_result = res["verification_result"]["vqa_results"][0]
                all_answers = vqa_result.get("answers", vqa_result.get("answer", []))
                # print(len(all_answers))

                complexity = res["complexity"]

                unable_count = 0
                unable_count_complexity_1 = 0
                unable_count_complexity_2 = 0
                unable_count_complexity_3 = 0
                tot_ans = 0
                tot_ans_complexity_1 = 0
                tot_ans_complexity_2 = 0
                tot_ans_complexity_3 = 0
                for ans in all_answers:
                    if ans.get("answer_converted", "").lower() == "unable to determine":
                        unable_count += 1
                        if complexity == 1:
                            unable_count_complexity_1 += 1
                        if complexity == 2:
                            unable_count_complexity_2 += 1
                        if complexity == 3:
                            unable_count_complexity_3 += 1

                    tot_ans += 1
                    if complexity == 1:
                        tot_ans_complexity_1 += 1
                    if complexity == 2:
                        tot_ans_complexity_2 += 1
                    if complexity == 3:
                        tot_ans_complexity_3 += 1
                # print(tot_ans)
                # print(tot_ans_complexity_1)
                # print(tot_ans_complexity_2)
                # print(tot_ans_complexity_3)

                # If majority of answers are "unable to determine", count this as correct
                if tot_ans > 0 and unable_count / tot_ans == 1:
                    correct_unable += 1

                if (
                    tot_ans_complexity_1 > 0
                    and unable_count_complexity_1 / tot_ans_complexity_1 == 1
                ):
                    correct_unable_complexity_1 += 1

                if (
                    tot_ans_complexity_2 > 0
                    and unable_count_complexity_2 / tot_ans_complexity_2 == 1
                ):
                    correct_unable_complexity_2 += 1

                if (
                    tot_ans_complexity_3 > 0
                    and unable_count_complexity_3 / tot_ans_complexity_3 == 1
                ):
                    correct_unable_complexity_3 += 1

        if self.debug:
            print(f"Total corrupted questions: {total_corrupted}")
            print(
                f"Correct unable to determine: {correct_unable} ({(correct_unable/total_corrupted)*100:.2f}%)"
            )
            print(
                f"Correct unable to determine (Complexity 1): {correct_unable_complexity_1} ({(correct_unable_complexity_1/total_corrupted)*100:.2f}%)"
            )
            print(
                f"Correct unable to determine (Complexity 2): {correct_unable_complexity_2} ({(correct_unable_complexity_2/total_corrupted)*100:.2f}%)"
            )
            print(
                f"Correct unable to determine (Complexity 3): {correct_unable_complexity_3} ({(correct_unable_complexity_3/total_corrupted)*100:.2f}%)"
            )
        return [
            correct_unable / total_corrupted,
            correct_unable_complexity_1 / total_corrupted,
            correct_unable_complexity_2 / total_corrupted,
            correct_unable_complexity_3 / total_corrupted,
        ]

    def UR(self):
        print("UR")
        total_answers = 0
        total_answers_complexity_1 = 0
        total_answers_complexity_2 = 0
        total_answers_complexity_3 = 0
        tot_unable_count = 0
        tot_unable_count_complexity_1 = 0
        tot_unable_count_complexity_2 = 0
        tot_unable_count_complexity_3 = 0
        for res in self.results:
            if (
                res["is_corrupted"]
                and "verification_result" in res
                and "vqa_results" in res["verification_result"]
                and len(res["verification_result"]["vqa_results"]) > 0
            ):
                vqa_result = res["verification_result"]["vqa_results"][0]
                all_answers = vqa_result.get("answers", vqa_result.get("answer", []))
                complexity = res["complexity"]

                for ans in all_answers:
                    if ans.get("answer_converted", "").lower() == "unable to determine":
                        tot_unable_count += 1
                        if complexity == 1:
                            tot_unable_count_complexity_1 += 1
                        if complexity == 2:
                            tot_unable_count_complexity_2 += 1
                        if complexity == 3:
                            tot_unable_count_complexity_3 += 1

                    total_answers += 1
                    if complexity == 1:
                        total_answers_complexity_1 += 1
                    if complexity == 2:
                        total_answers_complexity_2 += 1
                    if complexity == 3:
                        total_answers_complexity_3 += 1

        if self.debug:
            print(f"Total answers: {total_answers}")
            print(
                f"Unable to determine/errors: {tot_unable_count} ({(tot_unable_count/total_answers)*100:.2f}%)"
            )
            print(f"Total answers (Complexity 1): {total_answers_complexity_1}")
            print(
                f"Unable to determine/errors (Complexity 1): {tot_unable_count_complexity_1} ({(tot_unable_count_complexity_1/total_answers_complexity_1)*100:.2f}%)"
            )
            print(f"Total answers (Complexity 2): {total_answers_complexity_2}")
            print(
                f"Unable to determine/errors (Complexity 2): {tot_unable_count_complexity_2} ({(tot_unable_count_complexity_2/total_answers_complexity_2)*100:.2f}%)"
            )
            print(f"Total answers (Complexity 3): {total_answers_complexity_3}")
            print(
                f"Unable to determine/errors (Complexity 3): {tot_unable_count_complexity_3} ({(tot_unable_count_complexity_3/total_answers_complexity_3)*100:.2f}%)"
            )

        # print(f"Total unable to determine/errors: {tot_unable_count} ({(tot_unable_count/total_answers)*100:.2f}%)")
        # print(f"Total unable to determine/errors (Complexity 1): {tot_unable_count_complexity_1} {total_answers_complexity_1})")
        # print(f"Total unable to determine/errors (Complexity 2): {tot_unable_count_complexity_2} {total_answers_complexity_2})")
        # print(f"Total unable to determine/errors (Complexity 3): {tot_unable_count_complexity_3} {total_answers_complexity_3})")
        if total_answers == 0:
            return [0, 0, 0, 0]
        v1 = 0
        if total_answers_complexity_1 != 0:
            v1 = tot_unable_count_complexity_1 / total_answers_complexity_1
        v2 = 0
        if total_answers_complexity_2 != 0:
            v2 = tot_unable_count_complexity_2 / total_answers_complexity_2
        v3 = 0
        if total_answers_complexity_3 != 0:
            v3 = tot_unable_count_complexity_3 / total_answers_complexity_3
        return [tot_unable_count / total_answers, v1, v2, v3]

        # return [tot_unable_count / total_answers, tot_unable_count_complexity_1 / total_answers_complexity_1, tot_unable_count_complexity_2 / total_answers_complexity_2, tot_unable_count_complexity_3 / total_answers_complexity_3]

    def AEMR_ALMR_HR(self):
        print("AEMR + ALMR + HR")
        match_entity = 0
        match_layout = 0
        match_entity_layout = 0
        total_processed_answers = 0
        hallucination_count = 0

        match_entity_complexity_1 = 0
        match_layout_complexity_1 = 0
        match_entity_layout_complexity_1 = 0
        total_processed_answers_complexity_1 = 0
        hallucination_count_complexity_1 = 0

        match_entity_complexity_2 = 0
        match_layout_complexity_2 = 0
        match_entity_layout_complexity_2 = 0
        total_processed_answers_complexity_2 = 0
        hallucination_count_complexity_2 = 0

        match_entity_complexity_3 = 0
        match_layout_complexity_3 = 0
        match_entity_layout_complexity_3 = 0
        total_processed_answers_complexity_3 = 0
        hallucination_count_complexity_3 = 0

        for res in self.results:
            if (
                res.get("is_corrupted")
                and "complexity" in res
                and "verification_result" in res
            ):
                if (
                    "vqa_results" in res["verification_result"]
                    and len(res["verification_result"]["vqa_results"]) > 0
                ):
                    vqa_result = res["verification_result"]["vqa_results"][0]
                    all_answers = vqa_result.get(
                        "answers", vqa_result.get("answer", [])
                    )
                    if not isinstance(all_answers, list):
                        all_answers = [all_answers]

                    complexity = res["complexity"]

                    for ans in all_answers:
                        answer_text = ans.get("answer", "").lower()
                        answer_converted = ans.get("answer_converted", "").lower()

                        if not (
                            "error" in answer_text
                            or "unable to determine" in answer_text
                            or "unable to determine" in answer_converted
                        ):
                            # IF HAS PROVIDE AN ANSWER
                            total_processed_answers += 1
                            if complexity == 1:
                                total_processed_answers_complexity_1 += 1
                            if complexity == 2:
                                total_processed_answers_complexity_2 += 1
                            if complexity == 3:
                                total_processed_answers_complexity_3 += 1

                            final_answer = answer_converted

                            # generated answer entity
                            ans_entity = []
                            ans_layout = None
                            for full_page_path in ans.get("pages", []):
                                page = full_page_path.split("/")[-1]

                                # Check patch entities
                                if page in res.get("patch_entities", {}):
                                    for obj in res.get("patch_entities", {})[
                                        page
                                    ].values():
                                        # obj_layout = obj.get("type", "").lower()
                                        for entity in obj.get("entities", []):
                                            patch_text = entity.get("text", "").lower()
                                            patch_label = entity.get(
                                                "label", ""
                                            ).lower()
                                            if (
                                                final_answer in patch_text
                                                or patch_text in final_answer
                                            ):
                                                ans_entity.append(patch_label)
                                                # ans_layout = obj_layout
                                                break

                                # Check layout OCR
                                if page in res.get("layout_analysis", {}).get(
                                    "pages", {}
                                ):
                                    layout_page = res["layout_analysis"]["pages"][page]
                                    layout_objs = layout_page.get("layout_analysis", {})
                                    for obj in layout_objs.values():
                                        ocr_text = obj.get("OCR", "").lower()
                                        ocr_label = obj.get("ObjectType", "").lower()
                                        if (
                                            final_answer in ocr_text
                                            or ocr_text in final_answer
                                        ):
                                            ans_layout = ocr_label
                                            break
                            if ans_layout == None:
                                hallucination_count += 1
                                if complexity == 1:
                                    hallucination_count_complexity_1 += 1
                                if complexity == 2:
                                    hallucination_count_complexity_2 += 1
                                if complexity == 3:
                                    hallucination_count_complexity_3 += 1
                            else:

                                if len(ans_entity) == 0:
                                    entities = []

                                    entities = self.entity_identifier.identify_entities(
                                        res["corrupted_question"]
                                        + " Answer:"
                                        + final_answer
                                    )
                                    for entity in entities:
                                        if (
                                            entity.get("text", "").lower()
                                            in final_answer
                                        ):
                                            ans_entity.append(entity.get("label", ""))

                                # Check original answers
                                og_ans_entity = []
                                original_answer = None
                                for original in res.get(
                                    "original_answer_locations", []
                                ):
                                    original_answer = original.get("answer", "").lower()
                                    original_page = original.get(
                                        "page_id", ""
                                    )  # GIVEN AN ORGINAL ANSWER

                                    original_answer_entity = None
                                    original_answer_layout = original.get(
                                        "object_type", ""
                                    ).lower()
                                    # Check patch entities and OCR for every page

                                    # Check patch entities
                                    if original_page in res.get("patch_entities", {}):
                                        for obj in res.get("patch_entities", {})[
                                            original_page
                                        ].values():
                                            for entity in obj.get("entities", []):
                                                patch_text = entity.get(
                                                    "text", ""
                                                ).lower()
                                                patch_label = entity.get(
                                                    "label", ""
                                                ).lower()
                                                if (
                                                    original_answer in patch_text
                                                    or patch_text in original_answer
                                                ):
                                                    original_answer_entity = patch_label

                                                    break
                                    if original_answer_entity is not None:
                                        og_ans_entity.append(original_answer_entity)
                                        break

                                if len(og_ans_entity) == 0:
                                    entities = []

                                    entities = self.entity_identifier.identify_entities(
                                        res["original_question"]
                                        + " Answer:"
                                        + original_answer
                                    )
                                    for entity in entities:
                                        if (
                                            entity.get("text", "").lower()
                                            in original_answer
                                        ):
                                            og_ans_entity.append(
                                                entity.get("label", "")
                                            )

                                if any(og in og_ans_entity for og in ans_entity):
                                    match_entity += 1
                                    if complexity == 1:
                                        match_entity_complexity_1 += 1
                                    if complexity == 2:
                                        match_entity_complexity_2 += 1
                                    if complexity == 3:
                                        match_entity_complexity_3 += 1

                                if ans_layout == original_answer_layout:
                                    match_layout += 1
                                    if complexity == 1:
                                        match_layout_complexity_1 += 1
                                    if complexity == 2:
                                        match_layout_complexity_2 += 1
                                    if complexity == 3:
                                        match_layout_complexity_3 += 1

                                if (
                                    any(og in og_ans_entity for og in ans_entity)
                                    and ans_layout == original_answer_layout
                                ):
                                    match_entity_layout += 1
                                    if complexity == 1:
                                        match_entity_layout_complexity_1 += 1
                                    if complexity == 2:
                                        match_entity_layout_complexity_2 += 1
                                    if complexity == 3:
                                        match_entity_layout_complexity_3 += 1

        """ print(f"total_processed_answers: {total_processed_answers}")
        print(f"match_entity: {match_entity}")
        print(f"match_layout: {match_layout}")
        print(f"match_entity_layout: {match_entity_layout}")
        print(f"hallucination_count: {hallucination_count}")
        print(f"total_processed_answers_complexity_1: {total_processed_answers_complexity_1}")
        print(f"match_entity_complexity_1: {match_entity_complexity_1}")
        print(f"match_layout_complexity_1: {match_layout_complexity_1}")
        print(f"match_entity_layout_complexity_1: {match_entity_layout_complexity_1}")
        print(f"hallucination_count_complexity_1: {hallucination_count_complexity_1}")
        print(f"total_processed_answers_complexity_2: {total_processed_answers_complexity_2}")
        print(f"match_entity_complexity_2: {match_entity_complexity_2}")
        print(f"match_layout_complexity_2: {match_layout_complexity_2}")
        print(f"match_entity_layout_complexity_2: {match_entity_layout_complexity_2}")
        print(f"hallucination_count_complexity_2: {hallucination_count_complexity_2}")
        print(f"total_processed_answers_complexity_3: {total_processed_answers_complexity_3}")
        print(f"match_entity_complexity_3: {match_entity_complexity_3}")
        print(f"match_layout_complexity_3: {match_layout_complexity_3}")
        print(f"match_entity_layout_complexity_3: {match_entity_layout_complexity_3}")
        print(f"hallucination_count_complexity_3: {hallucination_count_complexity_3}") """

        return [
            total_processed_answers,
            match_entity,
            match_layout,
            match_entity_layout,
            hallucination_count,
            total_processed_answers_complexity_1,
            match_entity_complexity_1,
            match_layout_complexity_1,
            match_entity_layout_complexity_1,
            hallucination_count_complexity_1,
            total_processed_answers_complexity_2,
            match_entity_complexity_2,
            match_layout_complexity_2,
            match_entity_layout_complexity_2,
            hallucination_count_complexity_2,
            total_processed_answers_complexity_3,
            match_entity_complexity_3,
            match_layout_complexity_3,
            match_entity_layout_complexity_3,
            hallucination_count_complexity_3,
        ]

    def QEWR(self):
        print("QEWR")
        qur = 0
        qur_complexity_1 = 0
        qur_complexity_2 = 0
        qur_complexity_3 = 0
        qur_ratio = []
        qur_ratio_complexity_1 = []
        qur_ratio_complexity_2 = []
        qur_ratio_complexity_3 = []
        TOT = 0
        for res in self.results:
            if (
                res.get("is_corrupted")
                and "complexity" in res
                and "verification_result" in res
            ):
                if (
                    "vqa_results" in res["verification_result"]
                    and len(res["verification_result"]["vqa_results"]) > 0
                ):
                    vqa_result = res["verification_result"]["vqa_results"][0]
                    all_answers = vqa_result.get(
                        "answers", vqa_result.get("answer", [])
                    )

                    question_entities = res.get("question_entities", [])
                    original_entities = res.get("original_entity", [])
                    corrupted_entities = res.get("corrupted_entities", [])
                    complexity = res["complexity"]

                    untouched_entities = []
                    for qe in question_entities:
                        qe_text = qe.get("text", "").lower()
                        for oe in original_entities:
                            oe_text = oe.get("text", "").lower()
                            if oe_text != qe_text:
                                untouched_entities.append(qe)
                    tot_ue = len(untouched_entities)

                    for ans in all_answers:

                        answer_text = ans.get("answer", "").lower()
                        answer_converted = ans.get("answer_converted", "").lower()

                        if not (
                            "error" in answer_text
                            or "unable to determine" in answer_text
                            or "unable to determine" in answer_converted
                        ):
                            TOT += 1
                            count_match = 0
                            for ue in untouched_entities:
                                ue_text = ue.get("text", "").lower()
                                for full_page_path in ans.get("pages", []):
                                    page = full_page_path.split("/")[-1]
                                    if page in res.get("layout_analysis", {}).get(
                                        "pages", {}
                                    ):
                                        layout_page = res["layout_analysis"]["pages"][
                                            page
                                        ]
                                        layout_objs = layout_page.get(
                                            "layout_analysis", {}
                                        )
                                        flag = False
                                        for obj in layout_objs.values():
                                            ocr_text = obj.get("OCR", "").lower()
                                            ocr_label = obj.get(
                                                "ObjectType", ""
                                            ).lower()
                                            if (
                                                ue_text in ocr_text
                                                or ocr_text in ue_text
                                            ):
                                                count_match += 1
                                                flag = True
                                                break
                                        if flag:
                                            break
                            if tot_ue == count_match:
                                qur += 1
                                if complexity == 1:
                                    qur_complexity_1 += 1
                                if complexity == 2:
                                    qur_complexity_2 += 1
                                if complexity == 3:
                                    qur_complexity_3 += 1
                            else:
                                qur_ratio.append(count_match / tot_ue)
                                if complexity == 1:
                                    qur_ratio_complexity_1.append(count_match / tot_ue)
                                if complexity == 2:
                                    qur_ratio_complexity_2.append(count_match / tot_ue)
                                if complexity == 3:
                                    qur_ratio_complexity_3.append(count_match / tot_ue)
        if self.debug:
            print(f"Total corrupted answers: {TOT}")
            print(f"QUR: {qur}")
            print(f"QUR (Complexity 1): {qur_complexity_1}")
            print(f"QUR (Complexity 2): {qur_complexity_2}")
            print(f"QUR (Complexity 3): {qur_complexity_3}")
            print(f"QUR Ratio: {np.mean(qur_ratio)}")
            print(f"QUR Ratio (Complexity 1): {np.mean(qur_ratio_complexity_1)}")
            print(f"QUR Ratio (Complexity 2): {np.mean(qur_ratio_complexity_2)}")
            print(f"QUR Ratio (Complexity 3): {np.mean(qur_ratio_complexity_3)}")
        return [
            qur,
            qur_complexity_1,
            qur_complexity_2,
            qur_complexity_3,
            np.mean(qur_ratio),
            np.mean(qur_ratio_complexity_1),
            np.mean(qur_ratio_complexity_2),
            np.mean(qur_ratio_complexity_3),
            TOT,
        ]

    def center_distance(self, bbox1, bbox2):
        # Calculate centers
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        center2_y = (bbox2[1] + bbox2[3]) / 2

        # Euclidean distance between centers
        return ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5

    def QEPR(self):
        print("QEPR")
        COUNTER = 0
        CLOSEST = []
        for res in self.results:
            if (
                res.get("is_corrupted")
                and "complexity" in res
                and "verification_result" in res
            ):
                if (
                    "vqa_results" in res["verification_result"]
                    and len(res["verification_result"]["vqa_results"]) > 0
                ):
                    vqa_result = res["verification_result"]["vqa_results"][0]
                    all_answers = vqa_result.get(
                        "answers", vqa_result.get("answer", [])
                    )

                    question_entities = res.get("question_entities", [])
                    original_entities = res.get("original_entity", [])
                    corrupted_entities = res.get("corrupted_entities", [])
                    complexity = res["complexity"]

                    untouched_entities = []
                    for qe in question_entities:
                        qe_text = qe.get("text", "").lower()
                        for oe in original_entities:
                            oe_text = oe.get("text", "").lower()
                            if oe_text != qe_text:
                                untouched_entities.append(qe)
                    tot_ue = len(untouched_entities)
                    if tot_ue == 0:
                        continue

                    for ans in all_answers:
                        DISTANCE = 10000000
                        answer_text = ans.get("answer", "").lower()
                        answer_converted = ans.get("answer_converted", "").lower()
                        ans_pages = [
                            full_page_path.split("/")[-1]
                            for full_page_path in ans.get("pages", [])
                        ]

                        if not (
                            "error" in answer_text
                            or "unable to determine" in answer_text
                            or "unable to determine" in answer_converted
                        ):

                            answer_position = []
                            for page in ans_pages:
                                if page in res.get("layout_analysis", {}).get(
                                    "pages", {}
                                ):
                                    layout_page = res["layout_analysis"]["pages"][page]
                                    layout_objs = layout_page.get("layout_analysis", {})
                                    flag = False
                                    for obj in layout_objs.values():
                                        ocr_text = obj.get("OCR", "").lower()
                                        ocr_label = obj.get("ObjectType", "").lower()
                                        if (
                                            answer_converted in ocr_text
                                            or ocr_text in answer_converted
                                        ):
                                            answer_position.append(
                                                obj.get("BBOX", None)
                                            )

                            if len(answer_position) == 0:
                                # HALLUCINATION
                                continue

                            # print("ANSWER POSITION: ", answer_position)

                            count_match = 0
                            untouched_entities_positions = []
                            for ue in untouched_entities:
                                ue_text = ue.get("text", "").lower()
                                for page2 in ans_pages:
                                    if page2 in res.get("layout_analysis", {}).get(
                                        "pages", {}
                                    ):
                                        layout_page = res["layout_analysis"]["pages"][
                                            page
                                        ]
                                        layout_objs = layout_page.get(
                                            "layout_analysis", {}
                                        )
                                        flag = False
                                        for obj in layout_objs.values():
                                            ocr_text = obj.get("OCR", "").lower()
                                            ocr_label = obj.get(
                                                "ObjectType", ""
                                            ).lower()
                                            ocr_position = obj.get("BBOX", None)
                                            if (
                                                ue_text in ocr_text
                                                or ocr_text in ue_text
                                            ):
                                                count_match += 1
                                                flag = True
                                                untouched_entities_positions.append(
                                                    ocr_position
                                                )
                                                break
                                        if flag:
                                            break
                            # if tot_ue == count_match: # all unmodified entities are in the answer pages
                            if count_match > 0:
                                # print("All unmodified entities are in the answer pages")
                                evaluated_entities = []
                                # evaluated_entities_text=[]
                                evaluated_entities_positions = []
                                for ce in corrupted_entities:
                                    ce_page = ce.get("page_id", "")
                                    ce_text = ce.get("text", "").lower()
                                    if (
                                        ce_page in ans_pages
                                    ):  # corrupted entity is in the answer pages
                                        for page2 in ans_pages:
                                            if page2 in res.get(
                                                "layout_analysis", {}
                                            ).get("pages", {}):
                                                layout_page = res["layout_analysis"][
                                                    "pages"
                                                ][page]
                                                layout_objs = layout_page.get(
                                                    "layout_analysis", {}
                                                )
                                                for obj in layout_objs.values():
                                                    ocr_text = obj.get(
                                                        "OCR", ""
                                                    ).lower()
                                                    ocr_label = obj.get(
                                                        "ObjectType", ""
                                                    ).lower()
                                                    if (
                                                        ce_text in ocr_text
                                                        or ocr_text in ce_text
                                                    ):
                                                        if ce not in evaluated_entities:
                                                            # evaluated_entities_text.append(ce_text)
                                                            evaluated_entities_positions.append(
                                                                obj.get("BBOX", None)
                                                            )
                                                            evaluated_entities.append(
                                                                ce
                                                            )
                                if (
                                    len(evaluated_entities) > 0
                                ):  # len(evaluated_entities) >= complexity: # all entities are in the answer pages
                                    # print("All entities are in the answer pages")
                                    COUNTER += 1
                                    closest = None
                                    # print("UNTOUCHED")
                                    for ue in untouched_entities:
                                        ue_positions = ue.get("positions", [])
                                        for ue_pos in ue_positions:
                                            ue_bbox = ue_pos.get("object_bbox", [])
                                            for ap in answer_position:
                                                # print(ue_bbox, ap)
                                                dist = self.center_distance(ue_bbox, ap)
                                                if dist < DISTANCE:
                                                    DISTANCE = dist
                                                    closest = "UE"
                                    # print("EVALUATED")
                                    for ce in evaluated_entities:
                                        ce_position = ce.get("bbox", [])
                                        for ap in answer_position:
                                            # print(ce_position, ap)
                                            dist = self.center_distance(ce_position, ap)
                                            if dist < DISTANCE:
                                                DISTANCE = dist
                                                closest = "CE"
                                            elif dist == DISTANCE:
                                                closest = "SAME"
                                    CLOSEST.append(closest)

        # compute the frequency of the closest entities
        freq = Counter(CLOSEST)
        if self.debug:
            print("Counter of closest entities: ", COUNTER, len(CLOSEST))
            print(f"Frequency of closest entities. {freq}")
            # print(f"Frequency of closest entities. UE:{freq.get("UE", 0)} CE:{freq.get("CE", 0)}, SAME:{freq.get("SAME", 0)}")
        return COUNTER, CLOSEST, freq

    def CEBBOX(
        self, images_path
    ):  # corrupted entity bbox analysis --> misuro quante volte il modello risponde guardando la posizione della bbox dell'entità corrotta
        print("CEBBOX")
        wrong_answer_count = 0
        wrong_answer_count_c1 = 0
        wrong_answer_count_c2 = 0
        wrong_answer_count_c3 = 0

        # discretizzazione in 3 parti
        wrong_answer_count_top = 0
        wrong_answer_count_center = 0
        wrong_answer_count_bottom = 0
        wrong_answer_top_c1 = 0
        wrong_answer_bottom_c1 = 0
        wrong_answer_center_c1 = 0
        wrong_answer_top_c2 = 0
        wrong_answer_bottom_c2 = 0
        wrong_answer_center_c2 = 0
        wrong_answer_top_c3 = 0
        wrong_answer_bottom_c3 = 0
        wrong_answer_center_c3 = 0
        # discretizzazione in 6 parti
        wrong_answer_count_top_left = 0
        wrong_answer_count_top_right = 0
        wrong_answer_count_bottom_left = 0
        wrong_answer_count_bottom_right = 0
        wrong_answer_count_center_left = 0
        wrong_answer_count_center_right = 0
        wrong_answer_top_left_c1 = 0
        wrong_answer_top_right_c1 = 0
        wrong_answer_bottom_left_c1 = 0
        wrong_answer_bottom_right_c1 = 0
        wrong_answer_center_left_c1 = 0
        wrong_answer_center_right_c1 = 0
        wrong_answer_top_left_c2 = 0
        wrong_answer_top_right_c2 = 0
        wrong_answer_bottom_left_c2 = 0
        wrong_answer_bottom_right_c2 = 0
        wrong_answer_center_left_c2 = 0
        wrong_answer_center_right_c2 = 0
        wrong_answer_top_left_c3 = 0
        wrong_answer_top_right_c3 = 0
        wrong_answer_bottom_left_c3 = 0
        wrong_answer_bottom_right_c3 = 0
        wrong_answer_center_left_c3 = 0
        wrong_answer_center_right_c3 = 0
        # discretizzazione in 4 parti
        wrong_answer_count_top_left_quarter = 0
        wrong_answer_count_top_right_quarter = 0
        wrong_answer_count_bottom_left_quarter = 0
        wrong_answer_count_bottom_right_quarter = 0
        wrong_answer_top_left_quarter_c1 = 0
        wrong_answer_top_right_quarter_c1 = 0
        wrong_answer_bottom_left_quarter_c1 = 0
        wrong_answer_bottom_right_quarter_c1 = 0
        wrong_answer_top_left_quarter_c2 = 0
        wrong_answer_top_right_quarter_c2 = 0
        wrong_answer_bottom_left_quarter_c2 = 0
        wrong_answer_bottom_right_quarter_c2 = 0
        wrong_answer_top_left_quarter_c3 = 0
        wrong_answer_top_right_quarter_c3 = 0
        wrong_answer_bottom_left_quarter_c3 = 0
        wrong_answer_bottom_right_quarter_c3 = 0

        for res in self.results:
            vqa_result = res["verification_result"]["vqa_results"][0]
            vqa_results_ans = vqa_result.get("answers", vqa_result.get("answer", []))
            for ans in vqa_results_ans:
                if not (
                    "error" in ans.get("answer", "")
                    or "unable to determine" in ans.get("answer_converted", "")
                ):
                    pages = ans.get("pages", [])
                    corrupted_entities = res.get("corrupted_entities", [])
                    for corrupted_entity in corrupted_entities:
                        corrupted_entity_pageId = corrupted_entity.get("page_id", "")
                        for page in pages:
                            ans_page_id = page.split("/")[-1]
                            if ans_page_id == corrupted_entity_pageId:
                                corrupted_entity_bbox = corrupted_entity.get("bbox", [])
                                bbox_center = (
                                    (
                                        corrupted_entity_bbox[0]
                                        + corrupted_entity_bbox[2]
                                    )
                                    / 2,
                                    (
                                        corrupted_entity_bbox[1]
                                        + corrupted_entity_bbox[3]
                                    )
                                    / 2,
                                )
                                image_path = images_path + "/" + ans_page_id
                                x_size, y_size = Image.open(image_path).size
                                # discretizzazione in 3 parti
                                if res.get("complexity") == 1:
                                    wrong_answer_count_c1 += 1
                                    top = y_size / 3
                                    center = top * 2
                                    if bbox_center[1] < top:
                                        wrong_answer_top_c1 += 1
                                    elif bbox_center[1] < center:
                                        wrong_answer_center_c1 += 1
                                    else:
                                        wrong_answer_bottom_c1 += 1

                                    # discretizzazione in 6 parti
                                    left = x_size / 2
                                    if bbox_center[1] < top:
                                        if bbox_center[0] < left:
                                            wrong_answer_top_left_c1 += 1
                                        else:
                                            wrong_answer_top_right_c1 += 1
                                    elif bbox_center[1] < center:
                                        if bbox_center[0] < left:
                                            wrong_answer_center_left_c1 += 1
                                        else:
                                            wrong_answer_center_right_c1 += 1
                                    else:
                                        if bbox_center[0] < left:
                                            wrong_answer_bottom_left_c1 += 1
                                        else:
                                            wrong_answer_bottom_right_c1 += 1

                                    # discretizzazione in 4 parti
                                    top = y_size / 2
                                    if bbox_center[1] < top:
                                        if bbox_center[0] < left:
                                            wrong_answer_top_left_quarter_c1 += 1
                                        else:
                                            wrong_answer_top_right_quarter_c1 += 1
                                    else:
                                        if bbox_center[0] < left:
                                            wrong_answer_bottom_left_quarter_c1 += 1
                                        else:
                                            wrong_answer_bottom_right_quarter_c1 += 1

                                if res.get("complexity") == 2:
                                    top = y_size / 3
                                    center = top * 2
                                    wrong_answer_count_c2 += 1
                                    if bbox_center[1] < top:
                                        wrong_answer_top_c2 += 1
                                    elif bbox_center[1] < center:
                                        wrong_answer_center_c2 += 1
                                    else:
                                        wrong_answer_bottom_c2 += 1

                                    # discretizzazione in 6 parti
                                    left = x_size / 2
                                    if bbox_center[1] < top:
                                        if bbox_center[0] < left:
                                            wrong_answer_top_left_c2 += 1
                                        else:
                                            wrong_answer_top_right_c2 += 1
                                    elif bbox_center[1] < center:
                                        if bbox_center[0] < left:
                                            wrong_answer_center_left_c2 += 1
                                        else:
                                            wrong_answer_center_right_c2 += 1
                                    else:
                                        if bbox_center[0] < left:
                                            wrong_answer_bottom_left_c2 += 1
                                        else:
                                            wrong_answer_bottom_right_c2 += 1

                                    # discretizzazione in 4 parti
                                    top = y_size / 2
                                    if bbox_center[1] < top:
                                        if bbox_center[0] < left:
                                            wrong_answer_top_left_quarter_c2 += 1
                                        else:
                                            wrong_answer_top_right_quarter_c2 += 1
                                    else:
                                        if bbox_center[0] < left:
                                            wrong_answer_bottom_left_quarter_c2 += 1
                                        else:
                                            wrong_answer_bottom_right_quarter_c2 += 1

                                if res.get("complexity") == 3:
                                    wrong_answer_count_c3 += 1
                                    top = y_size / 3
                                    center = top * 2
                                    if bbox_center[1] < top:
                                        wrong_answer_top_c3 += 1
                                    elif bbox_center[1] < center:
                                        wrong_answer_center_c3 += 1
                                    else:
                                        wrong_answer_bottom_c3 += 1

                                    # discretizzazione in 6 parti
                                    left = x_size / 2
                                    if bbox_center[1] < top:
                                        if bbox_center[0] < left:
                                            wrong_answer_top_left_c3 += 1
                                        else:
                                            wrong_answer_top_right_c3 += 1
                                    elif bbox_center[1] < center:
                                        if bbox_center[0] < left:
                                            wrong_answer_center_left_c3 += 1
                                        else:
                                            wrong_answer_center_right_c3 += 1
                                    else:
                                        if bbox_center[0] < left:
                                            wrong_answer_bottom_left_c3 += 1
                                        else:
                                            wrong_answer_bottom_right_c3 += 1

                                    # discretizzazione in 4 parti
                                    top = y_size / 2
                                    if bbox_center[1] < top:
                                        if bbox_center[0] < left:
                                            wrong_answer_top_left_quarter_c3 += 1
                                        else:
                                            wrong_answer_top_right_quarter_c3 += 1
                                    else:
                                        if bbox_center[0] < left:
                                            wrong_answer_bottom_left_quarter_c3 += 1
                                        else:
                                            wrong_answer_bottom_right_quarter_c3 += 1

        wrong_answer_count = (
            wrong_answer_count_c1 + wrong_answer_count_c2 + wrong_answer_count_c3
        )
        wrong_answer_count_top = (
            wrong_answer_top_c1 + wrong_answer_top_c2 + wrong_answer_top_c3
        )
        wrong_answer_count_center = (
            wrong_answer_center_c1 + wrong_answer_center_c2 + wrong_answer_center_c3
        )
        wrong_answer_count_bottom = (
            wrong_answer_bottom_c1 + wrong_answer_bottom_c2 + wrong_answer_bottom_c3
        )
        wrong_answer_count_top_left = (
            wrong_answer_top_left_c1
            + wrong_answer_top_left_c2
            + wrong_answer_top_left_c3
        )
        wrong_answer_count_top_right = (
            wrong_answer_top_right_c1
            + wrong_answer_top_right_c2
            + wrong_answer_top_right_c3
        )
        wrong_answer_count_bottom_left = (
            wrong_answer_bottom_left_c1
            + wrong_answer_bottom_left_c2
            + wrong_answer_bottom_left_c3
        )
        wrong_answer_count_bottom_right = (
            wrong_answer_bottom_right_c1
            + wrong_answer_bottom_right_c2
            + wrong_answer_bottom_right_c3
        )
        wrong_answer_count_center_left = (
            wrong_answer_center_left_c1
            + wrong_answer_center_left_c2
            + wrong_answer_center_left_c3
        )
        wrong_answer_count_center_right = (
            wrong_answer_center_right_c1
            + wrong_answer_center_right_c2
            + wrong_answer_center_right_c3
        )
        wrong_answer_count_top_left_quarter = (
            wrong_answer_top_left_quarter_c1
            + wrong_answer_top_left_quarter_c2
            + wrong_answer_top_left_quarter_c3
        )
        wrong_answer_count_top_right_quarter = (
            wrong_answer_top_right_quarter_c1
            + wrong_answer_top_right_quarter_c2
            + wrong_answer_top_right_quarter_c3
        )
        wrong_answer_count_bottom_left_quarter = (
            wrong_answer_bottom_left_quarter_c1
            + wrong_answer_bottom_left_quarter_c2
            + wrong_answer_bottom_left_quarter_c3
        )
        wrong_answer_count_bottom_right_quarter = (
            wrong_answer_bottom_right_quarter_c1
            + wrong_answer_bottom_right_quarter_c2
            + wrong_answer_bottom_right_quarter_c3
        )

        return (
            wrong_answer_count,
            wrong_answer_count_top,
            wrong_answer_count_center,
            wrong_answer_count_bottom,
            wrong_answer_count_top_left,
            wrong_answer_count_top_right,
            wrong_answer_count_bottom_left,
            wrong_answer_count_bottom_right,
            wrong_answer_count_center_left,
            wrong_answer_count_center_right,
            wrong_answer_count_top_left_quarter,
            wrong_answer_count_top_right_quarter,
            wrong_answer_count_bottom_left_quarter,
            wrong_answer_count_bottom_right_quarter,
            wrong_answer_count_c1,
            wrong_answer_count_c2,
            wrong_answer_count_c3,
            wrong_answer_top_c1,
            wrong_answer_bottom_c1,
            wrong_answer_center_c1,
            wrong_answer_top_c2,
            wrong_answer_bottom_c2,
            wrong_answer_center_c2,
            wrong_answer_top_c3,
            wrong_answer_bottom_c3,
            wrong_answer_center_c3,
            wrong_answer_top_left_c1,
            wrong_answer_top_right_c1,
            wrong_answer_bottom_left_c1,
            wrong_answer_bottom_right_c1,
            wrong_answer_center_left_c1,
            wrong_answer_center_right_c1,
            wrong_answer_top_left_c2,
            wrong_answer_top_right_c2,
            wrong_answer_bottom_left_c2,
            wrong_answer_bottom_right_c2,
            wrong_answer_center_left_c2,
            wrong_answer_center_right_c2,
            wrong_answer_top_left_c3,
            wrong_answer_top_right_c3,
            wrong_answer_bottom_left_c3,
            wrong_answer_bottom_right_c3,
            wrong_answer_center_left_c3,
            wrong_answer_center_right_c3,
            wrong_answer_top_left_quarter_c1,
            wrong_answer_top_right_quarter_c1,
            wrong_answer_bottom_left_quarter_c1,
            wrong_answer_bottom_right_quarter_c1,
            wrong_answer_top_left_quarter_c2,
            wrong_answer_top_right_quarter_c2,
            wrong_answer_bottom_left_quarter_c2,
            wrong_answer_bottom_right_quarter_c2,
            wrong_answer_top_left_quarter_c3,
            wrong_answer_top_right_quarter_c3,
            wrong_answer_bottom_left_quarter_c3,
            wrong_answer_bottom_right_quarter_c3,
        )

    def LWP(
        self, images_path
    ):  # corrupted entity bbox analysis --> misuro quante volte il modello risponde guardando la posizione della bbox dell'entità corrotta
        print("LWP")
        wrong_answer_count = 0
        wrong_answer_count_c1 = 0
        wrong_answer_count_c2 = 0
        wrong_answer_count_c3 = 0
        right_answer_count = 0
        right_answer_count_c1 = 0
        right_answer_count_c2 = 0
        right_answer_count_c3 = 0

        # # discretizzazione in 3 parti
        # wrong_answer_count_top = 0
        # wrong_answer_count_center = 0
        # wrong_answer_count_bottom = 0
        # wrong_answer_top_c1 = 0
        # wrong_answer_bottom_c1 = 0
        # wrong_answer_center_c1 = 0
        # wrong_answer_top_c2 = 0
        # wrong_answer_bottom_c2 = 0
        # wrong_answer_center_c2 = 0
        # wrong_answer_top_c3 = 0
        # wrong_answer_bottom_c3 = 0
        # wrong_answer_center_c3 = 0
        # wrong_answer_count_top_UTD = 0
        # wrong_answer_count_center_UTD = 0
        # wrong_answer_count_bottom_UTD = 0
        # wrong_answer_top_c1_UTD = 0
        # wrong_answer_bottom_c1_UTD = 0
        # wrong_answer_center_c1_UTD = 0
        # wrong_answer_top_c2_UTD = 0
        # wrong_answer_bottom_c2_UTD = 0
        # wrong_answer_center_c2_UTD = 0
        # wrong_answer_top_c3_UTD = 0
        # wrong_answer_bottom_c3_UTD = 0
        # wrong_answer_center_c3_UTD = 0
        
        # # discretizzazione in 6 parti
        # wrong_answer_count_top_left = 0
        # wrong_answer_count_top_right = 0
        # wrong_answer_count_bottom_left = 0
        # wrong_answer_count_bottom_right = 0
        # wrong_answer_count_center_left = 0
        # wrong_answer_count_center_right = 0
        # wrong_answer_top_left_c1 = 0
        # wrong_answer_top_right_c1 = 0
        # wrong_answer_bottom_left_c1 = 0
        # wrong_answer_bottom_right_c1 = 0
        # wrong_answer_center_left_c1 = 0
        # wrong_answer_center_right_c1 = 0
        # wrong_answer_top_left_c2 = 0
        # wrong_answer_top_right_c2 = 0
        # wrong_answer_bottom_left_c2 = 0
        # wrong_answer_bottom_right_c2 = 0
        # wrong_answer_center_left_c2 = 0
        # wrong_answer_center_right_c2 = 0
        # wrong_answer_top_left_c3 = 0
        # wrong_answer_top_right_c3 = 0
        # wrong_answer_bottom_left_c3 = 0
        # wrong_answer_bottom_right_c3 = 0
        # wrong_answer_center_left_c3 = 0
        # wrong_answer_center_right_c3 = 0
        # wrong_answer_count_top_left_UTD = 0
        # wrong_answer_count_top_right_UTD = 0
        # wrong_answer_count_bottom_left_UTD = 0
        # wrong_answer_count_bottom_right_UTD = 0
        # wrong_answer_count_center_left_UTD = 0
        # wrong_answer_count_center_right_UTD = 0
        # wrong_answer_top_left_c1_UTD = 0
        # wrong_answer_top_right_c1_UTD = 0
        # wrong_answer_bottom_left_c1_UTD = 0
        # wrong_answer_bottom_right_c1_UTD = 0
        # wrong_answer_center_left_c1_UTD = 0
        # wrong_answer_center_right_c1_UTD = 0
        # wrong_answer_top_left_c2_UTD = 0
        # wrong_answer_top_right_c2_UTD = 0
        # wrong_answer_bottom_left_c2_UTD = 0
        # wrong_answer_bottom_right_c2_UTD = 0
        # wrong_answer_center_left_c2_UTD = 0
        # wrong_answer_center_right_c2_UTD = 0
        # wrong_answer_top_left_c3_UTD = 0
        # wrong_answer_top_right_c3_UTD = 0
        # wrong_answer_bottom_left_c3_UTD = 0
        # wrong_answer_bottom_right_c3_UTD = 0
        # wrong_answer_center_left_c3_UTD = 0
        # wrong_answer_center_right_c3_UTD = 0


        # discretizzazione in 4 parti
        wrong_answer_count_top_left_quarter = 0
        wrong_answer_count_top_right_quarter = 0
        wrong_answer_count_bottom_left_quarter = 0
        wrong_answer_count_bottom_right_quarter = 0
        wrong_answer_top_left_quarter_c1 = 0
        wrong_answer_top_right_quarter_c1 = 0
        wrong_answer_bottom_left_quarter_c1 = 0
        wrong_answer_bottom_right_quarter_c1 = 0
        wrong_answer_top_left_quarter_c2 = 0
        wrong_answer_top_right_quarter_c2 = 0
        wrong_answer_bottom_left_quarter_c2 = 0
        wrong_answer_bottom_right_quarter_c2 = 0
        wrong_answer_top_left_quarter_c3 = 0
        wrong_answer_top_right_quarter_c3 = 0
        wrong_answer_bottom_left_quarter_c3 = 0
        wrong_answer_bottom_right_quarter_c3 = 0
        right_answer_count_top_left_quarter = 0
        right_answer_count_top_right_quarter = 0
        right_answer_count_bottom_left_quarter = 0
        right_answer_count_bottom_right_quarter = 0
        right_answer_top_left_quarter_c1 = 0
        right_answer_top_right_quarter_c1 = 0
        right_answer_bottom_left_quarter_c1 = 0
        right_answer_bottom_right_quarter_c1 = 0
        right_answer_top_left_quarter_c2 = 0
        right_answer_top_right_quarter_c2 = 0
        right_answer_bottom_left_quarter_c2 = 0
        right_answer_bottom_right_quarter_c2 = 0
        right_answer_top_left_quarter_c3 = 0
        right_answer_top_right_quarter_c3 = 0
        right_answer_bottom_left_quarter_c3 = 0
        right_answer_bottom_right_quarter_c3 = 0



        for res in self.results:
            vqa_result = res["verification_result"]["vqa_results"][0]
            vqa_results_ans = vqa_result.get("answers", vqa_result.get("answer", []))
            for ans in vqa_results_ans:
                    pages = ans.get("pages", [])
                    corrupted_entities = res.get("corrupted_entities", [])
                    for corrupted_entity in corrupted_entities:
                        corrupted_entity_pageId = corrupted_entity.get("page_id", "")
                        for page in pages:
                            ans_page_id = page.split("/")[-1]
                            if ans_page_id == corrupted_entity_pageId:
                                corrupted_entity_bbox = corrupted_entity.get("bbox", [])
                                bbox_center = (
                                    (
                                        corrupted_entity_bbox[0]
                                        + corrupted_entity_bbox[2]
                                    )
                                    / 2,
                                    (
                                        corrupted_entity_bbox[1]
                                        + corrupted_entity_bbox[3]
                                    )
                                    / 2,
                                )
                                image_path = images_path + "/" + ans_page_id
                                x_size, y_size = Image.open(image_path).size
                                top = y_size / 2
                                left = x_size / 2
                                if not (
                                        "error" in ans.get("answer", "")
                                        or "unable to determine" in ans.get("answer_converted", "")
                                    ):
                                    wrong_answer_count += 1
                                    
                                    if bbox_center[1] < top:
                                        if bbox_center[0] < left:
                                            wrong_answer_count_top_left_quarter += 1
                                        else:
                                            wrong_answer_count_top_right_quarter += 1
                                    else:
                                        if bbox_center[0] < left:
                                            wrong_answer_count_bottom_left_quarter += 1
                                        else:
                                            wrong_answer_count_bottom_right_quarter += 1
                                else:
                                    right_answer_count += 1

                                    if bbox_center[1] < top:
                                        if bbox_center[0] < left:
                                            right_answer_count_top_left_quarter += 1
                                        else:
                                            right_answer_count_top_right_quarter += 1
                                    else:
                                        if bbox_center[0] < left:
                                            right_answer_count_bottom_left_quarter += 1
                                        else:
                                            right_answer_count_bottom_right_quarter += 1
                                
                                if res.get("complexity") == 1:
                                    # discretizzazione in 3 parti
                                    
                                    if not (
                                        "error" in ans.get("answer", "")
                                        or "unable to determine" in ans.get("answer_converted", "")
                                    ):
                                    # discretizzazione in 4 parti
                                        wrong_answer_count_c1 += 1

                                        if bbox_center[1] < top:
                                            if bbox_center[0] < left:
                                                wrong_answer_top_left_quarter_c1 += 1
                                            else:
                                                wrong_answer_top_right_quarter_c1 += 1
                                        else:
                                            if bbox_center[0] < left:
                                                wrong_answer_bottom_left_quarter_c1 += 1
                                            else:
                                                wrong_answer_bottom_right_quarter_c1 += 1
                                    else:
                                        right_answer_count_c1 += 1
                                        if bbox_center[1] < top:
                                            if bbox_center[0] < left:
                                                right_answer_top_left_quarter_c1 += 1
                                            else:
                                                right_answer_top_right_quarter_c1 += 1
                                        else:
                                            if bbox_center[0] < left:
                                                right_answer_bottom_left_quarter_c1 += 1
                                            else:
                                                right_answer_bottom_right_quarter_c1 += 1

                                if res.get("complexity") == 2:
                                    if not (
                                        "error" in ans.get("answer", "")
                                        or "unable to determine" in ans.get("answer_converted", "")
                                    ):
                                        wrong_answer_count_c2 += 1
                                        if bbox_center[1] < top:
                                            if bbox_center[0] < left:
                                                wrong_answer_top_left_quarter_c2 += 1
                                            else:
                                                wrong_answer_top_right_quarter_c2 += 1
                                        else:
                                            if bbox_center[0] < left:
                                                wrong_answer_bottom_left_quarter_c2 += 1
                                            else:
                                                wrong_answer_bottom_right_quarter_c2 += 1
                                    else:
                                        right_answer_count_c2 += 1
                                        if bbox_center[1] < top:
                                            if bbox_center[0] < left:
                                                right_answer_top_left_quarter_c2 += 1
                                            else:
                                                right_answer_top_right_quarter_c2 += 1
                                        else:
                                            if bbox_center[0] < left:
                                                right_answer_bottom_left_quarter_c2 += 1
                                            else:
                                                right_answer_bottom_right_quarter_c2 += 1

                                if res.get("complexity") == 3:
                                    if not (
                                        "error" in ans.get("answer", "")
                                        or "unable to determine" in ans.get("answer_converted", "")
                                    ):
                                        wrong_answer_count_c3 += 1
                                        # discretizzazione in 4 parti
                                        if bbox_center[1] < top:
                                            if bbox_center[0] < left:
                                                wrong_answer_top_left_quarter_c3 += 1
                                            else:
                                                wrong_answer_top_right_quarter_c3 += 1
                                        else:
                                            if bbox_center[0] < left:
                                                wrong_answer_bottom_left_quarter_c3 += 1
                                            else:
                                                wrong_answer_bottom_right_quarter_c3 += 1
                                    else:
                                        right_answer_count_c3 += 1
                                        if bbox_center[1] < top:
                                            if bbox_center[0] < left:
                                                right_answer_top_left_quarter_c3 += 1
                                            else:
                                                right_answer_top_right_quarter_c3 += 1
                                        else:
                                            if bbox_center[0] < left:
                                                right_answer_bottom_left_quarter_c3 += 1
                                            else:
                                                right_answer_bottom_right_quarter_c3 += 1

        # print("right_answer_count", right_answer_count)
        # print("right_answer_count_c1", right_answer_count_c1)
        # print("right_answer_count_c2", right_answer_count_c2)
        # print("right_answer_count_c3", right_answer_count_c3)
        # print("right_answer_top_left_quarter", right_answer_top_left_quarter)
        # print("right_answer_top_right_quarter", right_answer_top_right_quarter)
        # print("right_answer_bottom_left_quarter", right_answer_bottom_left_quarter)
        # print("right_answer_bottom_right_quarter", right_answer_bottom_right_quarter)
        # print("right_answer_top_left_quarter_c1", right_answer_top_left_quarter_c1)
        # print("right_answer_top_right_quarter_c1", right_answer_top_right_quarter_c1)
        # print("right_answer_bottom_left_quarter_c1", right_answer_bottom_left_quarter_c1)
        # print("right_answer_bottom_right_quarter_c1", right_answer_bottom_right_quarter_c1)
        # print("right_answer_top_left_quarter_c2", right_answer_top_left_quarter_c2)
        # print("right_answer_top_right_quarter_c2", right_answer_top_right_quarter_c2)
        # print("right_answer_bottom_left_quarter_c2", right_answer_bottom_left_quarter_c2)
        # print("right_answer_bottom_right_quarter_c2", right_answer_bottom_right_quarter_c2)
        # print("right_answer_top_left_quarter_c3", right_answer_top_left_quarter_c3)
        # print("right_answer_top_right_quarter_c3", right_answer_top_right_quarter_c3)
        # print("right_answer_bottom_left_quarter_c3", right_answer_bottom_left_quarter_c3)
        # print("right_answer_bottom_right_quarter_c3", right_answer_bottom_right_quarter_c3)

        # print("wrong_answer_count", wrong_answer_count)
        # print("wrong_answer_count_c1", wrong_answer_count_c1)
        # print("wrong_answer_count_c2", wrong_answer_count_c2)
        # print("wrong_answer_count_c3", wrong_answer_count_c3)
        # print("wrong_answer_top_left_quarter", wrong_answer_top_left_quarter)
        # print("wrong_answer_top_right_quarter", wrong_answer_top_right_quarter)
        # print("wrong_answer_bottom_left_quarter", wrong_answer_bottom_left_quarter)
        # print("wrong_answer_bottom_right_quarter", wrong_answer_bottom_right_quarter)
        # print("wrong_answer_top_left_quarter_c1", wrong_answer_top_left_quarter_c1)
        # print("wrong_answer_top_right_quarter_c1", wrong_answer_top_right_quarter_c1)
        # print("wrong_answer_bottom_left_quarter_c1", wrong_answer_bottom_left_quarter_c1)
        # print("wrong_answer_bottom_right_quarter_c1", wrong_answer_bottom_right_quarter_c1)
        # print("wrong_answer_top_left_quarter_c2", wrong_answer_top_left_quarter_c2)
        # print("wrong_answer_top_right_quarter_c2", wrong_answer_top_right_quarter_c2)
        # print("wrong_answer_bottom_left_quarter_c2", wrong_answer_bottom_left_quarter_c2)
        # print("wrong_answer_bottom_right_quarter_c2", wrong_answer_bottom_right_quarter_c2)
        # print("wrong_answer_top_left_quarter_c3", wrong_answer_top_left_quarter_c3)
        # print("wrong_answer_top_right_quarter_c3", wrong_answer_top_right_quarter_c3)
        # print("wrong_answer_bottom_left_quarter_c3", wrong_answer_bottom_left_quarter_c3)
        # print("wrong_answer_bottom_right_quarter_c3", wrong_answer_bottom_right_quarter_c3)

        ratio_right_answer_count = right_answer_count / (right_answer_count + wrong_answer_count)
        ratio_right_answer_count_c1 = right_answer_count_c1 / (right_answer_count_c1 + wrong_answer_count_c1)
        ratio_right_answer_count_c2 = right_answer_count_c2 / (right_answer_count_c2 + wrong_answer_count_c2)
        ratio_right_answer_count_c3 = right_answer_count_c3 / (right_answer_count_c3 + wrong_answer_count_c3)
        ratio_right_answer_top_left_quarter = right_answer_count_top_left_quarter / (right_answer_count_top_left_quarter + wrong_answer_count_top_left_quarter)
        ratio_right_answer_top_right_quarter = right_answer_count_top_right_quarter / (right_answer_count_top_right_quarter + wrong_answer_count_top_right_quarter)
        ratio_right_answer_bottom_left_quarter = right_answer_count_bottom_left_quarter / (right_answer_count_bottom_left_quarter + wrong_answer_count_bottom_left_quarter)
        ratio_right_answer_bottom_right_quarter = right_answer_count_bottom_right_quarter / (right_answer_count_bottom_right_quarter + wrong_answer_count_bottom_right_quarter)
        ratio_right_answer_top_left_quarter_c1 = right_answer_top_left_quarter_c1 / (right_answer_top_left_quarter_c1 + wrong_answer_top_left_quarter_c1)
        ratio_right_answer_top_right_quarter_c1 = right_answer_top_right_quarter_c1 / (right_answer_top_right_quarter_c1 + wrong_answer_top_right_quarter_c1)
        ratio_right_answer_bottom_left_quarter_c1 = right_answer_bottom_left_quarter_c1 / (right_answer_bottom_left_quarter_c1 + wrong_answer_bottom_left_quarter_c1)
        ratio_right_answer_bottom_right_quarter_c1 = right_answer_bottom_right_quarter_c1 / (right_answer_bottom_right_quarter_c1 + wrong_answer_bottom_right_quarter_c1)
        ratio_right_answer_top_left_quarter_c2 = right_answer_top_left_quarter_c2 / (right_answer_top_left_quarter_c2 + wrong_answer_top_left_quarter_c2)
        ratio_right_answer_top_right_quarter_c2 = right_answer_top_right_quarter_c2 / (right_answer_top_right_quarter_c2 + wrong_answer_top_right_quarter_c2)
        ratio_right_answer_bottom_left_quarter_c2 = right_answer_bottom_left_quarter_c2 / (right_answer_bottom_left_quarter_c2 + wrong_answer_bottom_left_quarter_c2)
        ratio_right_answer_bottom_right_quarter_c2 = right_answer_bottom_right_quarter_c2 / (right_answer_bottom_right_quarter_c2 + wrong_answer_bottom_right_quarter_c2)
        ratio_right_answer_top_left_quarter_c3 = right_answer_top_left_quarter_c3 / (right_answer_top_left_quarter_c3 + wrong_answer_top_left_quarter_c3)
        ratio_right_answer_top_right_quarter_c3 = right_answer_top_right_quarter_c3 / (right_answer_top_right_quarter_c3 + wrong_answer_top_right_quarter_c3)
        ratio_right_answer_bottom_left_quarter_c3 = right_answer_bottom_left_quarter_c3 / (right_answer_bottom_left_quarter_c3 + wrong_answer_bottom_left_quarter_c3)
        ratio_right_answer_bottom_right_quarter_c3 = right_answer_bottom_right_quarter_c3 / (right_answer_bottom_right_quarter_c3 + wrong_answer_bottom_right_quarter_c3)

        ratio_wrong_answer_count = wrong_answer_count / (right_answer_count + wrong_answer_count)
        ratio_wrong_answer_count_c1 = wrong_answer_count_c1 / (right_answer_count_c1 + wrong_answer_count_c1)
        ratio_wrong_answer_count_c2 = wrong_answer_count_c2 / (right_answer_count_c2 + wrong_answer_count_c2)
        ratio_wrong_answer_count_c3 = wrong_answer_count_c3 / (right_answer_count_c3 + wrong_answer_count_c3)
        ratio_wrong_answer_top_left_quarter = wrong_answer_count_top_left_quarter / (right_answer_count_top_left_quarter + wrong_answer_count_top_left_quarter)
        ratio_wrong_answer_top_right_quarter = wrong_answer_count_top_right_quarter / (right_answer_count_top_right_quarter + wrong_answer_count_top_right_quarter)
        ratio_wrong_answer_bottom_left_quarter = wrong_answer_count_bottom_left_quarter / (right_answer_count_bottom_left_quarter + wrong_answer_count_bottom_left_quarter)
        ratio_wrong_answer_bottom_right_quarter = wrong_answer_count_bottom_right_quarter / (right_answer_count_bottom_right_quarter + wrong_answer_count_bottom_right_quarter)
        ratio_wrong_answer_top_left_quarter_c1 = wrong_answer_top_left_quarter_c1 / (right_answer_top_left_quarter_c1 + wrong_answer_top_left_quarter_c1)
        ratio_wrong_answer_top_right_quarter_c1 = wrong_answer_top_right_quarter_c1 / (right_answer_top_right_quarter_c1 + wrong_answer_top_right_quarter_c1)
        ratio_wrong_answer_bottom_left_quarter_c1 = wrong_answer_bottom_left_quarter_c1 / (right_answer_bottom_left_quarter_c1 + wrong_answer_bottom_left_quarter_c1)
        ratio_wrong_answer_bottom_right_quarter_c1 = wrong_answer_bottom_right_quarter_c1 / (right_answer_bottom_right_quarter_c1 + wrong_answer_bottom_right_quarter_c1)
        ratio_wrong_answer_top_left_quarter_c2 = wrong_answer_top_left_quarter_c2 / (right_answer_top_left_quarter_c2 + wrong_answer_top_left_quarter_c2)
        ratio_wrong_answer_top_right_quarter_c2 = wrong_answer_top_right_quarter_c2 / (right_answer_top_right_quarter_c2 + wrong_answer_top_right_quarter_c2)
        ratio_wrong_answer_bottom_left_quarter_c2 = wrong_answer_bottom_left_quarter_c2 / (right_answer_bottom_left_quarter_c2 + wrong_answer_bottom_left_quarter_c2)
        ratio_wrong_answer_bottom_right_quarter_c2 = wrong_answer_bottom_right_quarter_c2 / (right_answer_bottom_right_quarter_c2 + wrong_answer_bottom_right_quarter_c2)
        ratio_wrong_answer_top_left_quarter_c3 = wrong_answer_top_left_quarter_c3 / (right_answer_top_left_quarter_c3 + wrong_answer_top_left_quarter_c3)
        ratio_wrong_answer_top_right_quarter_c3 = wrong_answer_top_right_quarter_c3 / (right_answer_top_right_quarter_c3 + wrong_answer_top_right_quarter_c3)
        ratio_wrong_answer_bottom_left_quarter_c3 = wrong_answer_bottom_left_quarter_c3 / (right_answer_bottom_left_quarter_c3 + wrong_answer_bottom_left_quarter_c3)
        ratio_wrong_answer_bottom_right_quarter_c3 = wrong_answer_bottom_right_quarter_c3 / (right_answer_bottom_right_quarter_c3 + wrong_answer_bottom_right_quarter_c3)

        # print("ratio answer_count (right/wrong)", ratio_right_answer_count, ratio_wrong_answer_count)
        # print("ratio answer_count_c1 (right/wrong)", ratio_right_answer_count_c1, ratio_wrong_answer_count_c1)
        # print("ratio answer_count_c2 (right/wrong)", ratio_right_answer_count_c2, ratio_wrong_answer_count_c2)
        # print("ratio answer_count_c3 (right/wrong)", ratio_right_answer_count_c3, ratio_wrong_answer_count_c3)
        # print("ratio answer_top_left_quarter (right/wrong)", ratio_right_answer_top_left_quarter, ratio_wrong_answer_top_left_quarter)
        # print("ratio answer_top_right_quarter (right/wrong)", ratio_right_answer_top_right_quarter, ratio_wrong_answer_top_right_quarter)
        # print("ratio answer_bottom_left_quarter (right/wrong)", ratio_right_answer_bottom_left_quarter, ratio_wrong_answer_bottom_left_quarter)
        # print("ratio answer_bottom_right_quarter (right/wrong)", ratio_right_answer_bottom_right_quarter, ratio_wrong_answer_bottom_right_quarter)
        # print("ratio answer_top_left_quarter_c1 (right/wrong)", ratio_right_answer_top_left_quarter_c1, ratio_wrong_answer_top_left_quarter_c1)
        # print("ratio answer_top_right_quarter_c1 (right/wrong)", ratio_right_answer_top_right_quarter_c1, ratio_wrong_answer_top_right_quarter_c1)
        # print("ratio answer_bottom_left_quarter_c1 (right/wrong)", ratio_right_answer_bottom_left_quarter_c1, ratio_wrong_answer_bottom_left_quarter_c1)
        # print("ratio answer_bottom_right_quarter_c1 (right/wrong)", ratio_right_answer_bottom_right_quarter_c1, ratio_wrong_answer_bottom_right_quarter_c1)
        # print("ratio answer_top_left_quarter_c2 (right/wrong)", ratio_right_answer_top_left_quarter_c2, ratio_wrong_answer_top_left_quarter_c2)
        # print("ratio answer_top_right_quarter_c2 (right/wrong)", ratio_right_answer_top_right_quarter_c2, ratio_wrong_answer_top_right_quarter_c2)
        # print("ratio answer_bottom_left_quarter_c2 (right/wrong)", ratio_right_answer_bottom_left_quarter_c2, ratio_wrong_answer_bottom_left_quarter_c2)
        # print("ratio answer_bottom_right_quarter_c2 (right/wrong)", ratio_right_answer_bottom_right_quarter_c2, ratio_wrong_answer_bottom_right_quarter_c2)
        # print("ratio answer_top_left_quarter_c3 (right/wrong)", ratio_right_answer_top_left_quarter_c3, ratio_wrong_answer_top_left_quarter_c3)
        # print("ratio answer_top_right_quarter_c3 (right/wrong)", ratio_right_answer_top_right_quarter_c3, ratio_wrong_answer_top_right_quarter_c3)
        # print("ratio answer_bottom_left_quarter_c3 (right/wrong)", ratio_right_answer_bottom_left_quarter_c3, ratio_wrong_answer_bottom_left_quarter_c3)
        # print("ratio answer_bottom_right_quarter_c3 (right/wrong)", ratio_right_answer_bottom_right_quarter_c3, ratio_wrong_answer_bottom_right_quarter_c3)


        return (
            ratio_right_answer_count, ratio_wrong_answer_count,
            ratio_right_answer_count_c1, ratio_wrong_answer_count_c1,
            ratio_right_answer_count_c2, ratio_wrong_answer_count_c2,
            ratio_right_answer_count_c3, ratio_wrong_answer_count_c3,
            ratio_right_answer_top_left_quarter, ratio_wrong_answer_top_left_quarter,
            ratio_right_answer_top_right_quarter, ratio_wrong_answer_top_right_quarter,
            ratio_right_answer_bottom_left_quarter, ratio_wrong_answer_bottom_left_quarter,
            ratio_right_answer_bottom_right_quarter, ratio_wrong_answer_bottom_right_quarter,
            ratio_right_answer_top_left_quarter_c1, ratio_wrong_answer_top_left_quarter_c1,
            ratio_right_answer_top_right_quarter_c1, ratio_wrong_answer_top_right_quarter_c1,
            ratio_right_answer_bottom_left_quarter_c1, ratio_wrong_answer_bottom_left_quarter_c1,
            ratio_right_answer_bottom_right_quarter_c1, ratio_wrong_answer_bottom_right_quarter_c1,
            ratio_right_answer_top_left_quarter_c2, ratio_wrong_answer_top_left_quarter_c2,
            ratio_right_answer_top_right_quarter_c2, ratio_wrong_answer_top_right_quarter_c2,
            ratio_right_answer_bottom_left_quarter_c2, ratio_wrong_answer_bottom_left_quarter_c2,
            ratio_right_answer_bottom_right_quarter_c2, ratio_wrong_answer_bottom_right_quarter_c2,
            ratio_right_answer_top_left_quarter_c3, ratio_wrong_answer_top_left_quarter_c3,
            ratio_right_answer_top_right_quarter_c3, ratio_wrong_answer_top_right_quarter_c3,
            ratio_right_answer_bottom_left_quarter_c3, ratio_wrong_answer_bottom_left_quarter_c3,
            ratio_right_answer_bottom_right_quarter_c3, ratio_wrong_answer_bottom_right_quarter_c3
        )

    def CEBBOX_UTD(
        self, images_path
    ):  # corrupted entity bbox analysis --> misuro quante volte il modello risponde guardando la posizione della bbox dell'entità corrotta
        print("CEBBOX_UTD")
        correct_answer_count_c1 = 0
        correct_answer_count_c2 = 0
        correct_answer_count_c3 = 0

        # discretizzazione in 3 parti
        correct_answer_count_top = 0
        correct_answer_count_center = 0
        correct_answer_count_bottom = 0
        correct_answer_top_c1 = 0
        correct_answer_bottom_c1 = 0
        correct_answer_center_c1 = 0
        correct_answer_top_c2 = 0
        correct_answer_bottom_c2 = 0
        correct_answer_center_c2 = 0
        correct_answer_top_c3 = 0
        correct_answer_bottom_c3 = 0
        correct_answer_center_c3 = 0
        # discretizzazione in 6 parti
        correct_answer_count_top_left = 0
        correct_answer_count_top_right = 0
        correct_answer_count_bottom_left = 0
        correct_answer_count_bottom_right = 0
        correct_answer_count_center_left = 0
        correct_answer_count_center_right = 0
        correct_answer_top_left_c1 = 0
        correct_answer_top_right_c1 = 0
        correct_answer_bottom_left_c1 = 0
        correct_answer_bottom_right_c1 = 0
        correct_answer_center_left_c1 = 0
        correct_answer_center_right_c1 = 0
        correct_answer_top_left_c2 = 0
        correct_answer_top_right_c2 = 0
        correct_answer_bottom_left_c2 = 0
        correct_answer_bottom_right_c2 = 0
        correct_answer_center_left_c2 = 0
        correct_answer_center_right_c2 = 0
        correct_answer_top_left_c3 = 0
        correct_answer_top_right_c3 = 0
        correct_answer_bottom_left_c3 = 0
        correct_answer_bottom_right_c3 = 0
        correct_answer_center_left_c3 = 0
        correct_answer_center_right_c3 = 0
        # discretizzazione in 4 parti
        correct_answer_count_top_left_quarter = 0
        correct_answer_count_top_right_quarter = 0
        correct_answer_count_bottom_left_quarter = 0
        correct_answer_count_bottom_right_quarter = 0
        correct_answer_top_left_quarter_c1 = 0
        correct_answer_top_right_quarter_c1 = 0
        correct_answer_bottom_left_quarter_c1 = 0
        correct_answer_bottom_right_quarter_c1 = 0
        correct_answer_top_left_quarter_c2 = 0
        correct_answer_top_right_quarter_c2 = 0
        correct_answer_bottom_left_quarter_c2 = 0
        correct_answer_bottom_right_quarter_c2 = 0
        correct_answer_top_left_quarter_c3 = 0
        correct_answer_top_right_quarter_c3 = 0
        correct_answer_bottom_left_quarter_c3 = 0
        correct_answer_bottom_right_quarter_c3 = 0

        for res in self.results:
            vqa_result = res["verification_result"]["vqa_results"][0]
            vqa_results_ans = vqa_result.get("answers", vqa_result.get("answer", []))
            for ans in vqa_results_ans:
                if (
                    "error" not in ans.get("answer", "")
                    and "unable to determine" in ans.get("answer_converted").lower()
                ):
                    pages = ans.get("pages", [])
                    corrupted_entities = res.get("corrupted_entities", [])
                    for corrupted_entity in corrupted_entities:
                        corrupted_entity_pageId = corrupted_entity.get("page_id", "")
                        for page in pages:
                            ans_page_id = page.split("/")[-1]
                            if ans_page_id == corrupted_entity_pageId:
                                corrupted_entity_bbox = corrupted_entity.get("bbox", [])
                                bbox_center = (
                                    (
                                        corrupted_entity_bbox[0]
                                        + corrupted_entity_bbox[2]
                                    )
                                    / 2,
                                    (
                                        corrupted_entity_bbox[1]
                                        + corrupted_entity_bbox[3]
                                    )
                                    / 2,
                                )
                                image_path = images_path + "/" + ans_page_id
                                x_size, y_size = Image.open(image_path).size
                                # discretizzazione in 3 parti
                                if res.get("complexity") == 1:
                                    correct_answer_count_c1 += 1
                                    top = y_size / 3
                                    center = top * 2
                                    if bbox_center[1] < top:
                                        correct_answer_top_c1 += 1
                                    elif bbox_center[1] < center:
                                        correct_answer_center_c1 += 1
                                    else:
                                        correct_answer_bottom_c1 += 1

                                    # discretizzazione in 6 parti
                                    left = x_size / 2
                                    if bbox_center[1] < top:
                                        if bbox_center[0] < left:
                                            correct_answer_top_left_c1 += 1
                                        else:
                                            correct_answer_top_right_c1 += 1
                                    elif bbox_center[1] < center:
                                        if bbox_center[0] < left:
                                            correct_answer_center_left_c1 += 1
                                        else:
                                            correct_answer_center_right_c1 += 1
                                    else:
                                        if bbox_center[0] < left:
                                            correct_answer_bottom_left_c1 += 1
                                        else:
                                            correct_answer_bottom_right_c1 += 1

                                    # discretizzazione in 4 parti
                                    top = y_size / 2
                                    if bbox_center[1] < top:
                                        if bbox_center[0] < left:
                                            correct_answer_top_left_quarter_c1 += 1
                                        else:
                                            correct_answer_top_right_quarter_c1 += 1
                                    else:
                                        if bbox_center[0] < left:
                                            correct_answer_bottom_left_quarter_c1 += 1
                                        else:
                                            correct_answer_bottom_right_quarter_c1 += 1

                                if res.get("complexity") == 2:
                                    top = y_size / 3
                                    center = top * 2
                                    correct_answer_count_c2 += 1
                                    if bbox_center[1] < top:
                                        correct_answer_top_c2 += 1
                                    elif bbox_center[1] < center:
                                        correct_answer_center_c2 += 1
                                    else:
                                        correct_answer_bottom_c2 += 1

                                    # discretizzazione in 6 parti
                                    left = x_size / 2
                                    if bbox_center[1] < top:
                                        if bbox_center[0] < left:
                                            correct_answer_top_left_c2 += 1
                                        else:
                                            correct_answer_top_right_c2 += 1
                                    elif bbox_center[1] < center:
                                        if bbox_center[0] < left:
                                            correct_answer_center_left_c2 += 1
                                        else:
                                            correct_answer_center_right_c2 += 1
                                    else:
                                        if bbox_center[0] < left:
                                            correct_answer_bottom_left_c2 += 1
                                        else:
                                            correct_answer_bottom_right_c2 += 1

                                    # discretizzazione in 4 parti
                                    top = y_size / 2
                                    if bbox_center[1] < top:
                                        if bbox_center[0] < left:
                                            correct_answer_top_left_quarter_c2 += 1
                                        else:
                                            correct_answer_top_right_quarter_c2 += 1
                                    else:
                                        if bbox_center[0] < left:
                                            correct_answer_bottom_left_quarter_c2 += 1
                                        else:
                                            correct_answer_bottom_right_quarter_c2 += 1

                                if res.get("complexity") == 3:
                                    correct_answer_count_c3 += 1
                                    top = y_size / 3
                                    center = top * 2
                                    if bbox_center[1] < top:
                                        correct_answer_top_c3 += 1
                                    elif bbox_center[1] < center:
                                        correct_answer_center_c3 += 1
                                    else:
                                        correct_answer_bottom_c3 += 1

                                    # discretizzazione in 6 parti
                                    left = x_size / 2
                                    if bbox_center[1] < top:
                                        if bbox_center[0] < left:
                                            correct_answer_top_left_c3 += 1
                                        else:
                                            correct_answer_top_right_c3 += 1
                                    elif bbox_center[1] < center:
                                        if bbox_center[0] < left:
                                            correct_answer_center_left_c3 += 1
                                        else:
                                            correct_answer_center_right_c3 += 1
                                    else:
                                        if bbox_center[0] < left:
                                            correct_answer_bottom_left_c3 += 1
                                        else:
                                            correct_answer_bottom_right_c3 += 1

                                    # discretizzazione in 4 parti
                                    top = y_size / 2
                                    if bbox_center[1] < top:
                                        if bbox_center[0] < left:
                                            correct_answer_top_left_quarter_c3 += 1
                                        else:
                                            correct_answer_top_right_quarter_c3 += 1
                                    else:
                                        if bbox_center[0] < left:
                                            correct_answer_bottom_left_quarter_c3 += 1
                                        else:
                                            correct_answer_bottom_right_quarter_c3 += 1

        correct_answer_count = (
            correct_answer_count_c1 + correct_answer_count_c2 + correct_answer_count_c3
        )
        correct_answer_count_top = (
            correct_answer_top_c1 + correct_answer_top_c2 + correct_answer_top_c3
        )
        correct_answer_count_center = (
            correct_answer_center_c1
            + correct_answer_center_c2
            + correct_answer_center_c3
        )
        correct_answer_count_bottom = (
            correct_answer_bottom_c1
            + correct_answer_bottom_c2
            + correct_answer_bottom_c3
        )
        correct_answer_count_top_left = (
            correct_answer_top_left_c1
            + correct_answer_top_left_c2
            + correct_answer_top_left_c3
        )
        correct_answer_count_top_right = (
            correct_answer_top_right_c1
            + correct_answer_top_right_c2
            + correct_answer_top_right_c3
        )
        correct_answer_count_bottom_left = (
            correct_answer_bottom_left_c1
            + correct_answer_bottom_left_c2
            + correct_answer_bottom_left_c3
        )
        correct_answer_count_bottom_right = (
            correct_answer_bottom_right_c1
            + correct_answer_bottom_right_c2
            + correct_answer_bottom_right_c3
        )
        correct_answer_count_center_left = (
            correct_answer_center_left_c1
            + correct_answer_center_left_c2
            + correct_answer_center_left_c3
        )
        correct_answer_count_center_right = (
            correct_answer_center_right_c1
            + correct_answer_center_right_c2
            + correct_answer_center_right_c3
        )
        correct_answer_count_top_left_quarter = (
            correct_answer_top_left_quarter_c1
            + correct_answer_top_left_quarter_c2
            + correct_answer_top_left_quarter_c3
        )
        correct_answer_count_top_right_quarter = (
            correct_answer_top_right_quarter_c1
            + correct_answer_top_right_quarter_c2
            + correct_answer_top_right_quarter_c3
        )
        correct_answer_count_bottom_left_quarter = (
            correct_answer_bottom_left_quarter_c1
            + correct_answer_bottom_left_quarter_c2
            + correct_answer_bottom_left_quarter_c3
        )
        correct_answer_count_bottom_right_quarter = (
            correct_answer_bottom_right_quarter_c1
            + correct_answer_bottom_right_quarter_c2
            + correct_answer_bottom_right_quarter_c3
        )

        return (
            correct_answer_count,
            correct_answer_count_top,
            correct_answer_count_center,
            correct_answer_count_bottom,
            correct_answer_count_top_left,
            correct_answer_count_top_right,
            correct_answer_count_bottom_left,
            correct_answer_count_bottom_right,
            correct_answer_count_center_left,
            correct_answer_count_center_right,
            correct_answer_count_top_left_quarter,
            correct_answer_count_top_right_quarter,
            correct_answer_count_bottom_left_quarter,
            correct_answer_count_bottom_right_quarter,
            correct_answer_count_c1,
            correct_answer_count_c2,
            correct_answer_count_c3,
            correct_answer_top_c1,
            correct_answer_bottom_c1,
            correct_answer_center_c1,
            correct_answer_top_c2,
            correct_answer_bottom_c2,
            correct_answer_center_c2,
            correct_answer_top_c3,
            correct_answer_bottom_c3,
            correct_answer_center_c3,
            correct_answer_top_left_c1,
            correct_answer_top_right_c1,
            correct_answer_bottom_left_c1,
            correct_answer_bottom_right_c1,
            correct_answer_center_left_c1,
            correct_answer_center_right_c1,
            correct_answer_top_left_c2,
            correct_answer_top_right_c2,
            correct_answer_bottom_left_c2,
            correct_answer_bottom_right_c2,
            correct_answer_center_left_c2,
            correct_answer_center_right_c2,
            correct_answer_top_left_c3,
            correct_answer_top_right_c3,
            correct_answer_bottom_left_c3,
            correct_answer_bottom_right_c3,
            correct_answer_center_left_c3,
            correct_answer_center_right_c3,
            correct_answer_top_left_quarter_c1,
            correct_answer_top_right_quarter_c1,
            correct_answer_bottom_left_quarter_c1,
            correct_answer_bottom_right_quarter_c1,
            correct_answer_top_left_quarter_c2,
            correct_answer_top_right_quarter_c2,
            correct_answer_bottom_left_quarter_c2,
            correct_answer_bottom_right_quarter_c2,
            correct_answer_top_left_quarter_c3,
            correct_answer_top_right_quarter_c3,
            correct_answer_bottom_left_quarter_c3,
            correct_answer_bottom_right_quarter_c3,
        )

    

    def UTD_LAYOUT(self):
        correct_answer_count_c1 = 0
        correct_answer_count_c2 = 0
        correct_answer_count_c3 = 0
        correct_answer_count = 0
        dict_layout_type = {}
        dict_layout_type_c1 = {}
        dict_layout_type_c2 = {}
        dict_layout_type_c3 = {}

        for layout in LAYOUT_TYPES:
            dict_layout_type[layout] = 0
            dict_layout_type_c1[layout] = 0
            dict_layout_type_c2[layout] = 0
            dict_layout_type_c3[layout] = 0

        for res in self.results:
            vqa_result = res["verification_result"]["vqa_results"][0]
            vqa_results_ans = vqa_result.get("answers", vqa_result.get("answer", []))
            for ans in vqa_results_ans:
                if (
                    "error" not in ans.get("answer", "")
                    and ans.get("answer_converted").lower() == "unable to determine"
                ):
                    pages = ans.get("pages", [])
                    corrupted_entities = res.get("corrupted_entities", [])
                    for corrupted_entity in corrupted_entities:
                        corrupted_entity_pageId = corrupted_entity.get("page_id", "")
                        for page in pages:
                            ans_page_id = page.split("/")[-1]
                            if ans_page_id == corrupted_entity_pageId:
                                corrupted_entity_layout = corrupted_entity.get(
                                    "objectType"
                                )
                                if res.get("complexity") == 1:
                                    correct_answer_count_c1 += 1
                                    dict_layout_type_c1[corrupted_entity_layout] += 1
                                elif res.get("complexity") == 2:
                                    correct_answer_count_c2 += 1
                                    dict_layout_type_c2[corrupted_entity_layout] += 1
                                elif res.get("complexity") == 3:
                                    correct_answer_count_c3 += 1
                                    dict_layout_type_c3[corrupted_entity_layout] += 1

        correct_answer_count = (
            correct_answer_count_c1 + correct_answer_count_c2 + correct_answer_count_c3
        )
        for layout in LAYOUT_TYPES:
            dict_layout_type[layout] = (
                dict_layout_type_c1[layout]
                + dict_layout_type_c2[layout]
                + dict_layout_type_c3[layout]
            )
        # Convert dictionaries to lists of values
        if correct_answer_count > 0:
            layout_values = [
                dict_layout_type[layout] / correct_answer_count
                for layout in LAYOUT_TYPES
            ]
        else:
            layout_values = [0] * len(LAYOUT_TYPES)
        if correct_answer_count_c1 > 0:
            layout_values_c1 = [
                dict_layout_type_c1[layout] / correct_answer_count_c1
                for layout in LAYOUT_TYPES
            ]
        else:
            layout_values_c1 = [0] * len(LAYOUT_TYPES)
        if correct_answer_count_c2 > 0:
            layout_values_c2 = [
                dict_layout_type_c2[layout] / correct_answer_count_c2
                for layout in LAYOUT_TYPES
            ]
        else:
            layout_values_c2 = [0] * len(LAYOUT_TYPES)
        if correct_answer_count_c3 > 0:
            layout_values_c3 = [
                dict_layout_type_c3[layout] / correct_answer_count_c3
                for layout in LAYOUT_TYPES
            ]
        else:
            layout_values_c3 = [0] * len(LAYOUT_TYPES)

        # Return counts followed by layout values for each complexity
        return [
            correct_answer_count,
            *layout_values,
            correct_answer_count_c1,
            *layout_values_c1,
            correct_answer_count_c2,
            *layout_values_c2,
            correct_answer_count_c3,
            *layout_values_c3,
        ]
    
    def DEWP(self):
        correct_answer_count_c1 = 0
        correct_answer_count_c2 = 0
        correct_answer_count_c3 = 0
        correct_answer_count = 0
        wrong_answer_count_c1 = 0
        wrong_answer_count_c2 = 0
        wrong_answer_count_c3 = 0
        wrong_answer_count = 0
        dict_layout_type = {}
        dict_layout_type_c1 = {}
        dict_layout_type_c2 = {}
        dict_layout_type_c3 = {}
        dict_layout_type_wrong = {}
        dict_layout_type_wrong_c1 = {}
        dict_layout_type_wrong_c2 = {}
        dict_layout_type_wrong_c3 = {}

        for layout in LAYOUT_TYPES:
            dict_layout_type[layout] = 0
            dict_layout_type_c1[layout] = 0
            dict_layout_type_c2[layout] = 0
            dict_layout_type_c3[layout] = 0
            dict_layout_type_wrong[layout] = 0
            dict_layout_type_wrong_c1[layout] = 0
            dict_layout_type_wrong_c2[layout] = 0
            dict_layout_type_wrong_c3[layout] = 0

        for res in self.results:
            vqa_result = res["verification_result"]["vqa_results"][0]
            vqa_results_ans = vqa_result.get("answers", vqa_result.get("answer", []))
            for ans in vqa_results_ans:
                    pages = ans.get("pages", [])
                    corrupted_entities = res.get("corrupted_entities", [])
                    for corrupted_entity in corrupted_entities:
                        corrupted_entity_pageId = corrupted_entity.get("page_id", "")
                        for page in pages:
                            ans_page_id = page.split("/")[-1]
                            if ans_page_id == corrupted_entity_pageId:
                                corrupted_entity_layout = corrupted_entity.get("objectType")
                                if (
                                    "error" not in ans.get("answer", "")
                                    and ans.get("answer_converted").lower() == "unable to determine"
                                ):
                                    correct_answer_count += 1
                                    dict_layout_type[corrupted_entity_layout] = dict_layout_type.get(corrupted_entity_layout, 0) + 1
                                    if res.get("complexity") == 1:
                                        correct_answer_count_c1 += 1
                                        dict_layout_type_c1[corrupted_entity_layout] = dict_layout_type_c1.get(corrupted_entity_layout, 0) + 1
                                    elif res.get("complexity") == 2:
                                        correct_answer_count_c2 += 1
                                        dict_layout_type_c2[corrupted_entity_layout] = dict_layout_type_c2.get(corrupted_entity_layout, 0) + 1
                                    elif res.get("complexity") == 3:
                                        correct_answer_count_c3 += 1
                                        dict_layout_type_c3[corrupted_entity_layout] = dict_layout_type_c3.get(corrupted_entity_layout, 0) + 1
                                else:
                                    wrong_answer_count += 1
                                    dict_layout_type_wrong[corrupted_entity_layout] = dict_layout_type_wrong.get(corrupted_entity_layout, 0) + 1
                                    if res.get("complexity") == 1:
                                        wrong_answer_count_c1 += 1
                                        dict_layout_type_wrong_c1[corrupted_entity_layout] = dict_layout_type_wrong_c1.get(corrupted_entity_layout, 0) + 1
                                    elif res.get("complexity") == 2:
                                        wrong_answer_count_c2 += 1
                                        dict_layout_type_wrong_c2[corrupted_entity_layout] = dict_layout_type_wrong_c2.get(corrupted_entity_layout, 0) + 1
                                    elif res.get("complexity") == 3:
                                        wrong_answer_count_c3 += 1
                                        dict_layout_type_wrong_c3[corrupted_entity_layout] = dict_layout_type_wrong_c3.get(corrupted_entity_layout, 0) + 1


        # Convert dictionaries to ratio. A ratio is defined as doc_layout_type[el]/(doc_layout_type[el] + doc_layout_type_wrong[el])
        layout_values = {}
        layout_values_c1 = {}
        layout_values_c2 = {}
        layout_values_c3 = {}
        for layout in dict_layout_type.keys():
            # check if value is not 0
            if dict_layout_type[layout] + dict_layout_type_wrong[layout] != 0:
                layout_values[layout] = dict_layout_type[layout] / (dict_layout_type[layout] + dict_layout_type_wrong[layout])
            else:
                layout_values[layout] = 0

            if dict_layout_type_c1[layout] + dict_layout_type_wrong_c1[layout] != 0:
                layout_values_c1[layout] = dict_layout_type_c1[layout] / (dict_layout_type_c1[layout] + dict_layout_type_wrong_c1[layout])
            else:
                layout_values_c1[layout] = 0

            if dict_layout_type_c2[layout] + dict_layout_type_wrong_c2[layout] != 0:    
                layout_values_c2[layout] = dict_layout_type_c2[layout] / (dict_layout_type_c2[layout] + dict_layout_type_wrong_c2[layout])
            else:
                layout_values_c2[layout] = 0

            if dict_layout_type_c3[layout] + dict_layout_type_wrong_c3[layout] != 0:
                layout_values_c3[layout] = dict_layout_type_c3[layout] / (dict_layout_type_c3[layout] + dict_layout_type_wrong_c3[layout])
            else:
                layout_values_c3[layout] = 0
        
        layout_values = [layout_values[layout] for layout in LAYOUT_TYPES]
        layout_values_c1 = [layout_values_c1[layout] for layout in LAYOUT_TYPES]
        layout_values_c2 = [layout_values_c2[layout] for layout in LAYOUT_TYPES]
        layout_values_c3 = [layout_values_c3[layout] for layout in LAYOUT_TYPES]

        layout_values_wrong = {}
        layout_values_wrong_c1 = {}
        layout_values_wrong_c2 = {}
        layout_values_wrong_c3 = {}
        for layout in dict_layout_type_wrong.keys():
            # check if value is not 0
            if dict_layout_type[layout] + dict_layout_type_wrong[layout] != 0:
                layout_values_wrong[layout] = dict_layout_type_wrong[layout] / (dict_layout_type[layout] + dict_layout_type_wrong[layout])
            else:
                layout_values_wrong[layout] = 0

            if dict_layout_type_c1[layout] + dict_layout_type_wrong_c1[layout] != 0:
                layout_values_wrong_c1[layout] = dict_layout_type_wrong_c1[layout] / (dict_layout_type_c1[layout] + dict_layout_type_wrong_c1[layout])
            else:
                layout_values_wrong_c1[layout] = 0

            if dict_layout_type_c2[layout] + dict_layout_type_wrong_c2[layout] != 0:
                layout_values_wrong_c2[layout] = dict_layout_type_wrong_c2[layout] / (dict_layout_type_c2[layout] + dict_layout_type_wrong_c2[layout])
            else:
                layout_values_wrong_c2[layout] = 0

            if dict_layout_type_c3[layout] + dict_layout_type_wrong_c3[layout] != 0:
                layout_values_wrong_c3[layout] = dict_layout_type_wrong_c3[layout] / (dict_layout_type_c3[layout] + dict_layout_type_wrong_c3[layout])
            else:
                layout_values_wrong_c3[layout] = 0

        layout_values_wrong = [layout_values_wrong[layout] for layout in LAYOUT_TYPES]
        layout_values_wrong_c1 = [layout_values_wrong_c1[layout] for layout in LAYOUT_TYPES]
        layout_values_wrong_c2 = [layout_values_wrong_c2[layout] for layout in LAYOUT_TYPES]
        layout_values_wrong_c3 = [layout_values_wrong_c3[layout] for layout in LAYOUT_TYPES]

        return [
            layout_values,
            # layout_values_wrong,
            layout_values_c1,
            # layout_values_wrong_c1,
            layout_values_c2,
            # layout_values_wrong_c2,
            layout_values_c3,
            # layout_values_wrong_c3,
        ]


def generate_analysis_report(dataset, images_path):
    # Group results by window size (e.g., "w=1" and "w=2")
    print("Initializing Entity Verifier...")
    entity_verifier = EntityIdentifier(ENTITY_TYPES)
    print("Entity Verifier initialized")

    report_data_by_window = {}
    combined_wrong_answer_correlations = []  # Combined correlation data from all models
    base_path = Path(__file__).parent
    print(f"Base path: {base_path}")

    result_files = []
    dataset_path = base_path / f"{dataset}"

    # Only process if dataset_path is a directory
    if not dataset_path.is_dir():
        print(f"Warning: {dataset_path} is not a directory")
        return

    for folder in dataset_path.iterdir():
        # Skip if folder is not a directory
        if not folder.is_dir():
            continue

        if "results" not in folder.name:
            continue

        print("")
        print("#" * 100)
        print(f"Processing folder {folder}")

        # create path folder/results if not exists
        folder_results = folder / "results"
        os.makedirs(folder_results, exist_ok=True)

        dict_CEPAR = {}
        dict_CEPAR_C1 = {}
        dict_CEPAR_C2 = {}
        dict_CEPAR_C3 = {}
        dict_CEPAR_freq = {}
        dict_CEPAR_Counter = {}
        dict_CEPAR_PLT = {}
        dict_CEPAR_PLT_C1 = {}
        dict_CEPAR_PLT_C2 = {}
        dict_CEPAR_PLT_C3 = {}
        dict_CEPAR_PET = {}
        dict_CEPAR_PET_C1 = {}
        dict_CEPAR_PET_C2 = {}
        dict_CEPAR_PET_C3 = {}

        dict_ANSL = {}
        dict_OPAR = {}

        dict_QUR = {}

        dict_UR = {}

        dict_QEWR = {}
        dict_QEWR_RATIO = {}

        dict_QEPR = {}

        dict_AEMR_ALMR_HR = {}
        dict_AEMR_ALMR_HR_C1 = {}
        dict_AEMR_ALMR_HR_C2 = {}
        dict_AEMR_ALMR_HR_C3 = {}

        dict_CEBBOX = {}
        dict_CEBBOX_C1 = {}
        dict_CEBBOX_C2 = {}
        dict_CEBBOX_C3 = {}

        dict_CEBBOX_UTD = {}
        dict_CEBBOX_UTD_C1 = {}
        dict_CEBBOX_UTD_C2 = {}
        dict_CEBBOX_UTD_C3 = {}

        dict_UTD_LAYOUT = {}
        dict_UTD_LAYOUT_C1 = {}
        dict_UTD_LAYOUT_C2 = {}
        dict_UTD_LAYOUT_C3 = {}

        dict_LWP = {}
        dict_LWP_C1 = {}
        dict_LWP_C2 = {}
        dict_LWP_C3 = {}        

        dict_DEWP = {}
        dict_DEWP_C1 = {}
        dict_DEWP_C2 = {}
        dict_DEWP_C3 = {}
        dict_DEWP_WRONG = {}
        dict_DEWP_WRONG_C1 = {}
        dict_DEWP_WRONG_C2 = {}
        dict_DEWP_WRONG_C3 = {}

        processed_models = []
        folder_augmented = folder / "augmented"
        for result_file in folder_augmented.iterdir():
            print("-" * 100)

            file_path = base_path / result_file
            if not file_path.exists():
                print(f"Warning: File {result_file} not found, skipping...")
                continue

            try:

                model_name = result_file.stem.split("/")[-1].split("_")[0]
                print(f"Model name: {model_name}")
                processed_models.append(model_name)

                with open(file_path, "r") as f:
                    data = json.load(f)
                    results = data.get("corrupted_questions", [])
                    if not results:
                        print(f"Warning: No corrupted questions found in {result_file}")
                        continue

                    print(f"Processing {result_file}")
                    print(f"Found {len(results)} questions")

                # model_name = result_file.split("_")[0]
                analyzer = VQAAnalyzer(
                    results,
                    entity_verifier,
                    dataset,
                    debug=False,
                    images_path=images_path,
                )
                metrics = analyzer.calculate_metrics()

                print(f"Metrics")

                for key, value in metrics.items():
                    if key == "CEPAR":
                        print(f"Processing CEPAR")

                        (
                            ans_page_corr_ent_page,
                            tot,
                            ans_page_corr_ent_page_complexity_1,
                            ans_page_corr_ent_page_complexity_2,
                            ans_page_corr_ent_page_complexity_3,
                            tot_complexity_1,
                            tot_complexity_2,
                            tot_complexity_3,
                            percentage_layout_type,
                            percentage_entity_type,
                            percentage_layout_type_complexity_1,
                            percentage_layout_type_complexity_2,
                            percentage_layout_type_complexity_3,
                            percentage_entity_type_complexity_1,
                            percentage_entity_type_complexity_2,
                            percentage_entity_type_complexity_3,
                            freq,
                            COUNTER,
                        ) = value

                        v1 = 0
                        if tot_complexity_1 != 0:
                            v1 = ans_page_corr_ent_page_complexity_1 / tot_complexity_1
                        v2 = 0
                        if tot_complexity_2 != 0:
                            v2 = ans_page_corr_ent_page_complexity_2 / tot_complexity_2
                        v3 = 0
                        if tot_complexity_3 != 0:
                            v3 = ans_page_corr_ent_page_complexity_3 / tot_complexity_3

                        dict_CEPAR[model_name] = [
                            ans_page_corr_ent_page,
                            tot,
                            ans_page_corr_ent_page / tot,
                            ans_page_corr_ent_page_complexity_1,
                            tot_complexity_1,
                            ans_page_corr_ent_page_complexity_1 / tot,
                            v1,
                            ans_page_corr_ent_page_complexity_2,
                            tot_complexity_2,
                            ans_page_corr_ent_page_complexity_2 / tot,
                            v2,
                            ans_page_corr_ent_page_complexity_3,
                            tot_complexity_3,
                            ans_page_corr_ent_page_complexity_3 / tot,
                            v3,
                        ]

                        list_entity = []
                        list_entity_C1 = []
                        list_entity_C2 = []
                        list_entity_C3 = []
                        list_PET = []
                        list_PET_C1 = []
                        list_PET_C2 = []
                        list_PET_C3 = []
                        for entity in ENTITY_TYPES:
                            list_PET.append(percentage_entity_type.get(entity, 0))
                            list_PET_C1.append(
                                percentage_entity_type_complexity_1.get(entity, 0)
                            )
                            list_PET_C2.append(
                                percentage_entity_type_complexity_2.get(entity, 0)
                            )
                            list_PET_C3.append(
                                percentage_entity_type_complexity_3.get(entity, 0)
                            )
                        dict_CEPAR_PET[model_name] = list_PET
                        dict_CEPAR_PET_C1[model_name] = list_PET_C1
                        dict_CEPAR_PET_C2[model_name] = list_PET_C2
                        dict_CEPAR_PET_C3[model_name] = list_PET_C3

                        list_layout = []
                        list_layout_C1 = []
                        list_layout_C2 = []
                        list_layout_C3 = []
                        list_PLT = []
                        list_PLT_C1 = []
                        list_PLT_C2 = []
                        list_PLT_C3 = []
                        for layout in LAYOUT_TYPES:
                            list_PLT.append(percentage_layout_type.get(layout, 0))
                            list_PLT_C1.append(
                                percentage_layout_type_complexity_1.get(layout, 0)
                            )
                            list_PLT_C2.append(
                                percentage_layout_type_complexity_2.get(layout, 0)
                            )
                            list_PLT_C3.append(
                                percentage_layout_type_complexity_3.get(layout, 0)
                            )
                        dict_CEPAR_PLT[model_name] = list_PLT
                        dict_CEPAR_PLT_C1[model_name] = list_PLT_C1
                        dict_CEPAR_PLT_C2[model_name] = list_PLT_C2
                        dict_CEPAR_PLT_C3[model_name] = list_PLT_C3

                        dict_CEPAR_freq[model_name] = freq
                        dict_CEPAR_Counter[model_name] = COUNTER

                    if key == "OPAR_ANSL":
                        print(f"Processing OPAR_ANSL")
                        (
                            answer_same_page_original,
                            answer_same_page_same_text_original,
                            tot,
                            answer_same_page_original_complexity_1,
                            answer_same_page_original_complexity_2,
                            answer_same_page_original_complexity_3,
                            answer_same_page_same_text_original_complexity_1,
                            answer_same_page_same_text_original_complexity_2,
                            answer_same_page_same_text_original_complexity_3,
                            anls,
                            anls_complexity_1,
                            anls_complexity_2,
                            anls_complexity_3,
                        ) = value

                        dict_OPAR[model_name] = [
                            answer_same_page_original,
                            answer_same_page_same_text_original,
                            tot,
                            answer_same_page_original / tot,
                            answer_same_page_same_text_original / tot,
                            answer_same_page_original_complexity_1,
                            answer_same_page_same_text_original_complexity_1,
                            answer_same_page_original_complexity_1 / tot,
                            answer_same_page_same_text_original_complexity_1 / tot,
                            answer_same_page_original_complexity_2,
                            answer_same_page_same_text_original_complexity_2,
                            answer_same_page_original_complexity_2 / tot,
                            answer_same_page_same_text_original_complexity_2 / tot,
                            answer_same_page_original_complexity_3,
                            answer_same_page_same_text_original_complexity_3,
                            answer_same_page_original_complexity_3 / tot,
                            answer_same_page_same_text_original_complexity_3 / tot,
                        ]

                        list_anls = []
                        if len(anls) > 0:
                            list_anls.append(np.mean(anls))
                        else:
                            list_anls.append(0)
                        if len(anls_complexity_1) > 0:
                            list_anls.append(np.mean(anls_complexity_1))
                        else:
                            list_anls.append(0)
                        if len(anls_complexity_2) > 0:
                            list_anls.append(np.mean(anls_complexity_2))
                        else:
                            list_anls.append(0)
                        if len(anls_complexity_3) > 0:
                            list_anls.append(np.mean(anls_complexity_3))
                        else:
                            list_anls.append(0)

                        dict_ANSL[model_name] = list_anls

                    if key == "QUR":
                        print(f"Processing QUR")
                        (
                            correct_unable_total_corrupted,
                            correct_unable_complexity_1_total_corrupted,
                            correct_unable_complexity_2_total_corrupted,
                            correct_unable_complexity_3_total_corrupted,
                        ) = value
                        dict_QUR[model_name] = [
                            correct_unable_total_corrupted,
                            correct_unable_complexity_1_total_corrupted,
                            correct_unable_complexity_2_total_corrupted,
                            correct_unable_complexity_3_total_corrupted,
                        ]

                    if key == "UR":
                        print(f"Processing UR")
                        (
                            tot_unable_count_total_answers,
                            tot_unable_count_complexity_1_total_answers_complexity_1,
                            tot_unable_count_complexity_2_total_answers_complexity_2,
                            tot_unable_count_complexity_3_total_answers_complexity_3,
                        ) = value
                        dict_UR[model_name] = [
                            tot_unable_count_total_answers,
                            tot_unable_count_complexity_1_total_answers_complexity_1,
                            tot_unable_count_complexity_2_total_answers_complexity_2,
                            tot_unable_count_complexity_3_total_answers_complexity_3,
                        ]

                    if key == "QEWR":
                        print(f"Processing QEWR")
                        (
                            qur,
                            qur_complexity_1,
                            qur_complexity_2,
                            qur_complexity_3,
                            mean_qur_ratio,
                            mean_qur_ratio_complexity_1,
                            mean_qur_ratio_complexity_2,
                            mean_qur_ratio_complexity_3,
                            TOT,
                        ) = value
                        dict_QEWR[model_name] = [
                            qur,
                            TOT,
                            qur / TOT,
                            qur_complexity_1,
                            qur_complexity_1 / TOT,
                            qur_complexity_2,
                            qur_complexity_2 / TOT,
                            qur_complexity_3,
                            qur_complexity_3 / TOT,
                        ]
                        dict_QEWR_RATIO[model_name] = [
                            mean_qur_ratio,
                            mean_qur_ratio_complexity_1,
                            mean_qur_ratio_complexity_2,
                            mean_qur_ratio_complexity_3,
                        ]

                    if key == "AEMR_ALMR_HR":
                        print(f"Processing AEMR_ALMR_HR")
                        (
                            total_processed_answers,
                            match_entity,
                            match_layout,
                            match_entity_layout,
                            hallucination_count,
                            total_processed_answers_complexity_1,
                            match_entity_complexity_1,
                            match_layout_complexity_1,
                            match_entity_layout_complexity_1,
                            hallucination_count_complexity_1,
                            total_processed_answers_complexity_2,
                            match_entity_complexity_2,
                            match_layout_complexity_2,
                            match_entity_layout_complexity_2,
                            hallucination_count_complexity_2,
                            total_processed_answers_complexity_3,
                            match_entity_complexity_3,
                            match_layout_complexity_3,
                            match_entity_layout_complexity_3,
                            hallucination_count_complexity_3,
                        ) = value

                        # Overall metrics
                        dict_AEMR_ALMR_HR[model_name] = [
                            match_entity,
                            match_layout,
                            match_entity_layout,
                            hallucination_count,
                            total_processed_answers,
                            (
                                match_entity / total_processed_answers
                                if total_processed_answers > 0
                                else 0
                            ),
                            (
                                match_layout / total_processed_answers
                                if total_processed_answers > 0
                                else 0
                            ),
                            (
                                match_entity_layout / total_processed_answers
                                if total_processed_answers > 0
                                else 0
                            ),
                            (
                                hallucination_count / total_processed_answers
                                if total_processed_answers > 0
                                else 0
                            ),
                        ]

                        # Complexity 1 metrics
                        dict_AEMR_ALMR_HR_C1[model_name] = [
                            match_entity_complexity_1,
                            match_layout_complexity_1,
                            match_entity_layout_complexity_1,
                            hallucination_count_complexity_1,
                            total_processed_answers_complexity_1,
                            (
                                match_entity_complexity_1
                                / total_processed_answers_complexity_1
                                if total_processed_answers_complexity_1 > 0
                                else 0
                            ),
                            (
                                match_layout_complexity_1
                                / total_processed_answers_complexity_1
                                if total_processed_answers_complexity_1 > 0
                                else 0
                            ),
                            (
                                match_entity_layout_complexity_1
                                / total_processed_answers_complexity_1
                                if total_processed_answers_complexity_1 > 0
                                else 0
                            ),
                            (
                                hallucination_count_complexity_1
                                / total_processed_answers_complexity_1
                                if total_processed_answers_complexity_1 > 0
                                else 0
                            ),
                        ]

                        # Complexity 2 metrics
                        dict_AEMR_ALMR_HR_C2[model_name] = [
                            match_entity_complexity_2,
                            match_layout_complexity_2,
                            match_entity_layout_complexity_2,
                            hallucination_count_complexity_2,
                            total_processed_answers_complexity_2,
                            (
                                match_entity_complexity_2
                                / total_processed_answers_complexity_2
                                if total_processed_answers_complexity_2 > 0
                                else 0
                            ),
                            (
                                match_layout_complexity_2
                                / total_processed_answers_complexity_2
                                if total_processed_answers_complexity_2 > 0
                                else 0
                            ),
                            (
                                match_entity_layout_complexity_2
                                / total_processed_answers_complexity_2
                                if total_processed_answers_complexity_2 > 0
                                else 0
                            ),
                            (
                                hallucination_count_complexity_2
                                / total_processed_answers_complexity_2
                                if total_processed_answers_complexity_2 > 0
                                else 0
                            ),
                        ]

                        # Complexity 3 metrics
                        dict_AEMR_ALMR_HR_C3[model_name] = [
                            match_entity_complexity_3,
                            match_layout_complexity_3,
                            match_entity_layout_complexity_3,
                            hallucination_count_complexity_3,
                            total_processed_answers_complexity_3,
                            (
                                match_entity_complexity_3
                                / total_processed_answers_complexity_3
                                if total_processed_answers_complexity_3 > 0
                                else 0
                            ),
                            (
                                match_layout_complexity_3
                                / total_processed_answers_complexity_3
                                if total_processed_answers_complexity_3 > 0
                                else 0
                            ),
                            (
                                match_entity_layout_complexity_3
                                / total_processed_answers_complexity_3
                                if total_processed_answers_complexity_3 > 0
                                else 0
                            ),
                            (
                                hallucination_count_complexity_3
                                / total_processed_answers_complexity_3
                                if total_processed_answers_complexity_3 > 0
                                else 0
                            ),
                        ]

                    if key == "QEPR":
                        print(f"Processing QEPR")
                        COUNTER, CLOSEST, freq = value
                        key_names = ["UE", "CE", "SAME"]
                        val = [COUNTER]
                        for kn in key_names:
                            val.append(freq.get(kn, 0))
                            val.append(freq.get(kn, 0) / COUNTER)
                        dict_QEPR[model_name] = val

                    if key == "CEBBOX":
                        print(f"Processing CEBBOX")
                        (
                            wrong_answer_count,
                            wrong_answer_count_top,
                            wrong_answer_count_center,
                            wrong_answer_count_bottom,
                            wrong_answer_count_top_left,
                            wrong_answer_count_top_right,
                            wrong_answer_count_bottom_left,
                            wrong_answer_count_bottom_right,
                            wrong_answer_count_center_left,
                            wrong_answer_count_center_right,
                            wrong_answer_count_top_left_quarter,
                            wrong_answer_count_top_right_quarter,
                            wrong_answer_count_bottom_left_quarter,
                            wrong_answer_count_bottom_right_quarter,
                            wrong_answer_count_c1,
                            wrong_answer_count_c2,
                            wrong_answer_count_c3,
                            wrong_answer_top_c1,
                            wrong_answer_bottom_c1,
                            wrong_answer_center_c1,
                            wrong_answer_top_c2,
                            wrong_answer_bottom_c2,
                            wrong_answer_center_c2,
                            wrong_answer_top_c3,
                            wrong_answer_bottom_c3,
                            wrong_answer_center_c3,
                            wrong_answer_top_left_c1,
                            wrong_answer_top_right_c1,
                            wrong_answer_bottom_left_c1,
                            wrong_answer_bottom_right_c1,
                            wrong_answer_center_left_c1,
                            wrong_answer_center_right_c1,
                            wrong_answer_top_left_c2,
                            wrong_answer_top_right_c2,
                            wrong_answer_bottom_left_c2,
                            wrong_answer_bottom_right_c2,
                            wrong_answer_center_left_c2,
                            wrong_answer_center_right_c2,
                            wrong_answer_top_left_c3,
                            wrong_answer_top_right_c3,
                            wrong_answer_bottom_left_c3,
                            wrong_answer_bottom_right_c3,
                            wrong_answer_center_left_c3,
                            wrong_answer_center_right_c3,
                            wrong_answer_top_left_quarter_c1,
                            wrong_answer_top_right_quarter_c1,
                            wrong_answer_bottom_left_quarter_c1,
                            wrong_answer_bottom_right_quarter_c1,
                            wrong_answer_top_left_quarter_c2,
                            wrong_answer_top_right_quarter_c2,
                            wrong_answer_bottom_left_quarter_c2,
                            wrong_answer_bottom_right_quarter_c2,
                            wrong_answer_top_left_quarter_c3,
                            wrong_answer_top_right_quarter_c3,
                            wrong_answer_bottom_left_quarter_c3,
                            wrong_answer_bottom_right_quarter_c3,
                        ) = value
                        dict_CEBBOX[model_name] = [
                            wrong_answer_count,
                            wrong_answer_count_top / wrong_answer_count,
                            wrong_answer_count_center / wrong_answer_count,
                            wrong_answer_count_bottom / wrong_answer_count,
                            wrong_answer_count_top_left / wrong_answer_count,
                            wrong_answer_count_top_right / wrong_answer_count,
                            wrong_answer_count_bottom_left / wrong_answer_count,
                            wrong_answer_count_bottom_right / wrong_answer_count,
                            wrong_answer_count_center_left / wrong_answer_count,
                            wrong_answer_count_center_right / wrong_answer_count,
                            wrong_answer_count_top_left_quarter / wrong_answer_count,
                            wrong_answer_count_top_right_quarter / wrong_answer_count,
                            wrong_answer_count_bottom_left_quarter / wrong_answer_count,
                            wrong_answer_count_bottom_right_quarter
                            / wrong_answer_count,
                        ]
                        dict_CEBBOX_C1[model_name] = [
                            wrong_answer_count_c1,
                            wrong_answer_top_c1 / wrong_answer_count_c1,
                            wrong_answer_center_c1 / wrong_answer_count_c1,
                            wrong_answer_bottom_c1 / wrong_answer_count_c1,
                            wrong_answer_top_left_c1 / wrong_answer_count_c1,
                            wrong_answer_top_right_c1 / wrong_answer_count_c1,
                            wrong_answer_bottom_left_c1 / wrong_answer_count_c1,
                            wrong_answer_bottom_right_c1 / wrong_answer_count_c1,
                            wrong_answer_center_left_c1 / wrong_answer_count_c1,
                            wrong_answer_center_right_c1 / wrong_answer_count_c1,
                            wrong_answer_top_left_quarter_c1 / wrong_answer_count_c1,
                            wrong_answer_top_right_quarter_c1 / wrong_answer_count_c1,
                            wrong_answer_bottom_left_quarter_c1 / wrong_answer_count_c1,
                            wrong_answer_bottom_right_quarter_c1
                            / wrong_answer_count_c1,
                        ]
                        dict_CEBBOX_C2[model_name] = [
                            wrong_answer_count_c2,
                            wrong_answer_top_c2 / wrong_answer_count_c2,
                            wrong_answer_center_c2 / wrong_answer_count_c2,
                            wrong_answer_bottom_c2 / wrong_answer_count_c2,
                            wrong_answer_top_left_c2 / wrong_answer_count_c2,
                            wrong_answer_top_right_c2 / wrong_answer_count_c2,
                            wrong_answer_bottom_left_c2 / wrong_answer_count_c2,
                            wrong_answer_bottom_right_c2 / wrong_answer_count_c2,
                            wrong_answer_center_left_c2 / wrong_answer_count_c2,
                            wrong_answer_center_right_c2 / wrong_answer_count_c2,
                            wrong_answer_top_left_quarter_c2 / wrong_answer_count_c2,
                            wrong_answer_top_right_quarter_c2 / wrong_answer_count_c2,
                            wrong_answer_bottom_left_quarter_c2 / wrong_answer_count_c2,
                            wrong_answer_bottom_right_quarter_c2
                            / wrong_answer_count_c2,
                        ]
                        dict_CEBBOX_C3[model_name] = [
                            wrong_answer_count_c3,
                            wrong_answer_top_c3 / wrong_answer_count_c3,
                            wrong_answer_center_c3 / wrong_answer_count_c3,
                            wrong_answer_bottom_c3 / wrong_answer_count_c3,
                            wrong_answer_top_left_c3 / wrong_answer_count_c3,
                            wrong_answer_top_right_c3 / wrong_answer_count_c3,
                            wrong_answer_bottom_left_c3 / wrong_answer_count_c3,
                            wrong_answer_bottom_right_c3 / wrong_answer_count_c3,
                            wrong_answer_center_left_c3 / wrong_answer_count_c3,
                            wrong_answer_center_right_c3 / wrong_answer_count_c3,
                            wrong_answer_top_left_quarter_c3 / wrong_answer_count_c3,
                            wrong_answer_top_right_quarter_c3 / wrong_answer_count_c3,
                            wrong_answer_bottom_left_quarter_c3 / wrong_answer_count_c3,
                            wrong_answer_bottom_right_quarter_c3
                            / wrong_answer_count_c3,
                        ]

                    if key == "CEBBOX_UTD":
                        print(f"Processing CEBBOX_UTD")
                        (
                            correct_answer_count,
                            correct_answer_count_top,
                            correct_answer_count_center,
                            correct_answer_count_bottom,
                            correct_answer_count_top_left,
                            correct_answer_count_top_right,
                            correct_answer_count_bottom_left,
                            correct_answer_count_bottom_right,
                            correct_answer_count_center_left,
                            correct_answer_count_center_right,
                            correct_answer_count_top_left_quarter,
                            correct_answer_count_top_right_quarter,
                            correct_answer_count_bottom_left_quarter,
                            correct_answer_count_bottom_right_quarter,
                            correct_answer_count_c1,
                            correct_answer_count_c2,
                            correct_answer_count_c3,
                            correct_answer_top_c1,
                            correct_answer_bottom_c1,
                            correct_answer_center_c1,
                            correct_answer_top_c2,
                            correct_answer_bottom_c2,
                            correct_answer_center_c2,
                            correct_answer_top_c3,
                            correct_answer_bottom_c3,
                            correct_answer_center_c3,
                            correct_answer_top_left_c1,
                            correct_answer_top_right_c1,
                            correct_answer_bottom_left_c1,
                            correct_answer_bottom_right_c1,
                            correct_answer_center_left_c1,
                            correct_answer_center_right_c1,
                            correct_answer_top_left_c2,
                            correct_answer_top_right_c2,
                            correct_answer_bottom_left_c2,
                            correct_answer_bottom_right_c2,
                            correct_answer_center_left_c2,
                            correct_answer_center_right_c2,
                            correct_answer_top_left_c3,
                            correct_answer_top_right_c3,
                            correct_answer_bottom_left_c3,
                            correct_answer_bottom_right_c3,
                            correct_answer_center_left_c3,
                            correct_answer_center_right_c3,
                            correct_answer_top_left_quarter_c1,
                            correct_answer_top_right_quarter_c1,
                            correct_answer_bottom_left_quarter_c1,
                            correct_answer_bottom_right_quarter_c1,
                            correct_answer_top_left_quarter_c2,
                            correct_answer_top_right_quarter_c2,
                            correct_answer_bottom_left_quarter_c2,
                            correct_answer_bottom_right_quarter_c2,
                            correct_answer_top_left_quarter_c3,
                            correct_answer_top_right_quarter_c3,
                            correct_answer_bottom_left_quarter_c3,
                            correct_answer_bottom_right_quarter_c3,
                        ) = value
                        dict_CEBBOX_UTD[model_name] = [
                            correct_answer_count,
                            correct_answer_count_top / correct_answer_count,
                            correct_answer_count_center / correct_answer_count,
                            correct_answer_count_bottom / correct_answer_count,
                            correct_answer_count_top_left / correct_answer_count,
                            correct_answer_count_top_right / correct_answer_count,
                            correct_answer_count_bottom_left / correct_answer_count,
                            correct_answer_count_bottom_right / correct_answer_count,
                            correct_answer_count_center_left / correct_answer_count,
                            correct_answer_count_center_right / correct_answer_count,
                            correct_answer_count_top_left_quarter
                            / correct_answer_count,
                            correct_answer_count_top_right_quarter
                            / correct_answer_count,
                            correct_answer_count_bottom_left_quarter
                            / correct_answer_count,
                            correct_answer_count_bottom_right_quarter
                            / correct_answer_count,
                        ]

                        dict_CEBBOX_UTD_C1[model_name] = [
                            correct_answer_count_c1,
                            correct_answer_top_c1 / correct_answer_count_c1,
                            correct_answer_center_c1 / correct_answer_count_c1,
                            correct_answer_bottom_c1 / correct_answer_count_c1,
                            correct_answer_top_left_c1 / correct_answer_count_c1,
                            correct_answer_top_right_c1 / correct_answer_count_c1,
                            correct_answer_bottom_left_c1 / correct_answer_count_c1,
                            correct_answer_bottom_right_c1 / correct_answer_count_c1,
                            correct_answer_center_left_c1 / correct_answer_count_c1,
                            correct_answer_center_right_c1 / correct_answer_count_c1,
                            correct_answer_top_left_quarter_c1
                            / correct_answer_count_c1,
                            correct_answer_top_right_quarter_c1
                            / correct_answer_count_c1,
                            correct_answer_bottom_left_quarter_c1
                            / correct_answer_count_c1,
                            correct_answer_bottom_right_quarter_c1
                            / correct_answer_count_c1,
                        ]
                        dict_CEBBOX_UTD_C2[model_name] = [
                            correct_answer_count_c2,
                            correct_answer_top_c2 / correct_answer_count_c2,
                            correct_answer_center_c2 / correct_answer_count_c2,
                            correct_answer_bottom_c2 / correct_answer_count_c2,
                            correct_answer_top_left_c2 / correct_answer_count_c2,
                            correct_answer_top_right_c2 / correct_answer_count_c2,
                            correct_answer_bottom_left_c2 / correct_answer_count_c2,
                            correct_answer_bottom_right_c2 / correct_answer_count_c2,
                            correct_answer_center_left_c2 / correct_answer_count_c2,
                            correct_answer_center_right_c2 / correct_answer_count_c2,
                            correct_answer_top_left_quarter_c2
                            / correct_answer_count_c2,
                            correct_answer_top_right_quarter_c2
                            / correct_answer_count_c2,
                            correct_answer_bottom_left_quarter_c2
                            / correct_answer_count_c2,
                            correct_answer_bottom_right_quarter_c2
                            / correct_answer_count_c2,
                        ]
                        dict_CEBBOX_UTD_C3[model_name] = [
                            correct_answer_count_c3,
                            correct_answer_top_c3 / correct_answer_count_c3,
                            correct_answer_center_c3 / correct_answer_count_c3,
                            correct_answer_bottom_c3 / correct_answer_count_c3,
                            correct_answer_top_left_c3 / correct_answer_count_c3,
                            correct_answer_top_right_c3 / correct_answer_count_c3,
                            correct_answer_bottom_left_c3 / correct_answer_count_c3,
                            correct_answer_bottom_right_c3 / correct_answer_count_c3,
                            correct_answer_center_left_c3 / correct_answer_count_c3,
                            correct_answer_center_right_c3 / correct_answer_count_c3,
                            correct_answer_top_left_quarter_c3
                            / correct_answer_count_c3,
                            correct_answer_top_right_quarter_c3
                            / correct_answer_count_c3,
                            correct_answer_bottom_left_quarter_c3
                            / correct_answer_count_c3,
                            correct_answer_bottom_right_quarter_c3
                            / correct_answer_count_c3,
                        ]

                    if key == "UTD_LAYOUT":
                        print(f"Processing UTD_LAYOUT")
                        values = value  # This is now a flat list
                        num_layouts = len(LAYOUT_TYPES)

                        # Extract values for each complexity level
                        dict_UTD_LAYOUT[model_name] = [
                            values[0],  # correct_answer_count
                            *values[1:num_layouts + 1],  # layout values
                        ]
                        dict_UTD_LAYOUT_C1[model_name] = [
                            values[num_layouts + 1],  # correct_answer_count_c1
                            *values[num_layouts + 2:2 * num_layouts + 2],  # layout values for C1
                        ]
                        dict_UTD_LAYOUT_C2[model_name] = [
                            values[2 * num_layouts + 2],  # correct_answer_count_c2
                            *values[2 * num_layouts + 3:3 * num_layouts + 3],  # layout values for C2
                        ]
                        dict_UTD_LAYOUT_C3[model_name] = [
                            values[3 * num_layouts + 3],  # correct_answer_count_c3
                            *values[3 * num_layouts + 4:4 * num_layouts + 4],  # layout values for C3
                        ]

                    if key=="LWP":
                        (
                            ratio_right_answer_count, ratio_wrong_answer_count,
                            ratio_right_answer_count_c1, ratio_wrong_answer_count_c1,
                            ratio_right_answer_count_c2, ratio_wrong_answer_count_c2,
                            ratio_right_answer_count_c3, ratio_wrong_answer_count_c3,
                            ratio_right_answer_top_left_quarter, ratio_wrong_answer_top_left_quarter,
                            ratio_right_answer_top_right_quarter, ratio_wrong_answer_top_right_quarter,
                            ratio_right_answer_bottom_left_quarter, ratio_wrong_answer_bottom_left_quarter,
                            ratio_right_answer_bottom_right_quarter, ratio_wrong_answer_bottom_right_quarter,
                            ratio_right_answer_top_left_quarter_c1, ratio_wrong_answer_top_left_quarter_c1,
                            ratio_right_answer_top_right_quarter_c1, ratio_wrong_answer_top_right_quarter_c1,
                            ratio_right_answer_bottom_left_quarter_c1, ratio_wrong_answer_bottom_left_quarter_c1,
                            ratio_right_answer_bottom_right_quarter_c1, ratio_wrong_answer_bottom_right_quarter_c1,
                            ratio_right_answer_top_left_quarter_c2, ratio_wrong_answer_top_left_quarter_c2,
                            ratio_right_answer_top_right_quarter_c2, ratio_wrong_answer_top_right_quarter_c2,
                            ratio_right_answer_bottom_left_quarter_c2, ratio_wrong_answer_bottom_left_quarter_c2,
                            ratio_right_answer_bottom_right_quarter_c2, ratio_wrong_answer_bottom_right_quarter_c2,
                            ratio_right_answer_top_left_quarter_c3, ratio_wrong_answer_top_left_quarter_c3,
                            ratio_right_answer_top_right_quarter_c3, ratio_wrong_answer_top_right_quarter_c3,
                            ratio_right_answer_bottom_left_quarter_c3, ratio_wrong_answer_bottom_left_quarter_c3,
                            ratio_right_answer_bottom_right_quarter_c3, ratio_wrong_answer_bottom_right_quarter_c3
                        ) = value

                        dict_LWP[model_name] = [
                            ratio_right_answer_count, ratio_wrong_answer_count,
                            ratio_right_answer_count_c1, ratio_wrong_answer_count_c1,
                            ratio_right_answer_count_c2, ratio_wrong_answer_count_c2,
                            ratio_right_answer_count_c3, ratio_wrong_answer_count_c3,
                            ratio_right_answer_top_left_quarter, ratio_wrong_answer_top_left_quarter,
                            ratio_right_answer_top_left_quarter_c1, ratio_wrong_answer_top_left_quarter_c1,
                            ratio_right_answer_top_left_quarter_c2, ratio_wrong_answer_top_left_quarter_c2,
                            ratio_right_answer_top_left_quarter_c3, ratio_wrong_answer_top_left_quarter_c3,
                            ratio_right_answer_top_right_quarter, ratio_wrong_answer_top_right_quarter,
                            ratio_right_answer_top_right_quarter_c1, ratio_wrong_answer_top_right_quarter_c1,
                            ratio_right_answer_top_right_quarter_c2, ratio_wrong_answer_top_right_quarter_c2,
                            ratio_right_answer_top_right_quarter_c3, ratio_wrong_answer_top_right_quarter_c3,
                            ratio_right_answer_bottom_left_quarter, ratio_wrong_answer_bottom_left_quarter,
                            ratio_right_answer_bottom_left_quarter_c1, ratio_wrong_answer_bottom_left_quarter_c1,
                            ratio_right_answer_bottom_left_quarter_c2, ratio_wrong_answer_bottom_left_quarter_c2,
                            ratio_right_answer_bottom_left_quarter_c3, ratio_wrong_answer_bottom_left_quarter_c3,
                            ratio_right_answer_bottom_right_quarter, ratio_wrong_answer_bottom_right_quarter,
                            ratio_right_answer_bottom_right_quarter_c1, ratio_wrong_answer_bottom_right_quarter_c1,
                            ratio_right_answer_bottom_right_quarter_c2, ratio_wrong_answer_bottom_right_quarter_c2,
                            ratio_right_answer_bottom_right_quarter_c3, ratio_wrong_answer_bottom_right_quarter_c3
                        ]
                    
                    if key == "DEWP":
                        [
                            layout_values,
                            layout_values_wrong,
                            layout_values_c1,
                            layout_values_wrong_c1,
                            layout_values_c2,
                            layout_values_wrong_c2,
                            layout_values_c3,
                            layout_values_wrong_c3,
                        ] = value

                        dict_DEWP[model_name] = layout_values
                        dict_DEWP_WRONG[model_name] = layout_values_wrong
                        dict_DEWP_C1[model_name] = layout_values_c1
                        dict_DEWP_WRONG_C1[model_name] = layout_values_wrong_c1
                        dict_DEWP_C2[model_name] = layout_values_c2
                        dict_DEWP_WRONG_C2[model_name] = layout_values_wrong_c2
                        dict_DEWP_C3[model_name] = layout_values_c3
                        dict_DEWP_WRONG_C3[model_name] = layout_values_wrong_c3



            except Exception as e:
                print(f"Error processing {result_file}: {e}")
                # continue

            # break
        print(f"Saving files")
        print(f"Processed models: {processed_models}")

        df_DEWP = pd.DataFrame(dict_DEWP)
        df_DEWP.index = LAYOUT_TYPES
        df_DEWP.to_csv(folder_results / "DEWP.csv")

        # df_DEWP_WRONG = pd.DataFrame(dict_DEWP_WRONG)
        # df_DEWP_WRONG.index = LAYOUT_TYPES
        # df_DEWP_WRONG.to_csv(folder_results / "DEWP_WRONG.csv")

        df_DEWP_C1 = pd.DataFrame(dict_DEWP_C1)
        df_DEWP_C1.index = LAYOUT_TYPES
        df_DEWP_C1.to_csv(folder_results / "DEWP_C1.csv")

        # df_DEWP_WRONG_C1 = pd.DataFrame(dict_DEWP_WRONG_C1)
        # df_DEWP_WRONG_C1.index = LAYOUT_TYPES
        # df_DEWP_WRONG_C1.to_csv(folder_results / "DEWP_WRONG_C1.csv")

        df_DEWP_C2 = pd.DataFrame(dict_DEWP_C2)
        df_DEWP_C2.index = LAYOUT_TYPES
        df_DEWP_C2.to_csv(folder_results / "DEWP_C2.csv")

        # df_DEWP_WRONG_C2 = pd.DataFrame(dict_DEWP_WRONG_C2)
        # df_DEWP_WRONG_C2.index = LAYOUT_TYPES
        # df_DEWP_WRONG_C2.to_csv(folder_results / "DEWP_WRONG_C2.csv")

        df_DEWP_C3 = pd.DataFrame(dict_DEWP_C3)
        df_DEWP_C3.index = LAYOUT_TYPES
        df_DEWP_C3.to_csv(folder_results / "DEWP_C3.csv")

        # df_DEWP_WRONG_C3 = pd.DataFrame(dict_DEWP_WRONG_C3)
        # df_DEWP_WRONG_C3.index = LAYOUT_TYPES
        # df_DEWP_WRONG_C3.to_csv(folder_results / "DEWP_WRONG_C3.csv")



        df_LWP = pd.DataFrame(dict_LWP)
        df_LWP.index = [
            "right_answer_count", "wrong_answer_count",
            "right_answer_count_c1", "wrong_answer_count_c1",
            "right_answer_count_c2", "wrong_answer_count_c2",
            "right_answer_count_c3", "wrong_answer_count_c3",
            "right_answer_top_left_quarter", "wrong_answer_top_left_quarter",
            "right_answer_top_left_quarter_c1", "wrong_answer_top_left_quarter_c1",
            "right_answer_top_left_quarter_c2", "wrong_answer_top_left_quarter_c2",
            "right_answer_top_left_quarter_c3", "wrong_answer_top_left_quarter_c3",
            "right_answer_top_right_quarter", "wrong_answer_top_right_quarter",
            "right_answer_top_right_quarter_c1", "wrong_answer_top_right_quarter_c1",
            "right_answer_top_right_quarter_c2", "wrong_answer_top_right_quarter_c2",
            "right_answer_top_right_quarter_c3", "wrong_answer_top_right_quarter_c3",
            "right_answer_bottom_left_quarter", "wrong_answer_bottom_left_quarter",
            "right_answer_bottom_left_quarter_c1", "wrong_answer_bottom_left_quarter_c1",
            "right_answer_bottom_left_quarter_c2", "wrong_answer_bottom_left_quarter_c2",
            "right_answer_bottom_left_quarter_c3", "wrong_answer_bottom_left_quarter_c3",
            "right_answer_bottom_right_quarter", "wrong_answer_bottom_right_quarter",
            "right_answer_bottom_right_quarter_c1", "wrong_answer_bottom_right_quarter_c1",
            "right_answer_bottom_right_quarter_c2", "wrong_answer_bottom_right_quarter_c2",
            "right_answer_bottom_right_quarter_c3", "wrong_answer_bottom_right_quarter_c3",
        ]
        # order columns following this order ['Qwen', 'InternVL', 'Phi', 'Molmo', 'Ovis', 'DocOwl']
        # df_LWP = df_LWP[['Qwen', 'InternVL', 'Phi', 'Molmo', 'Ovis', 'DocOwl']]
        df_LWP.to_csv(folder_results / "LWP_NEW.csv")

        # Save all dataframes to the folder/results directory
        df_CEPAR = pd.DataFrame(dict_CEPAR)
        df_CEPAR.index = [
            "CEPAR",
            "TOT",
            "Percentage",
            "CEPAR_C1",
            "TOT_C1",
            "Percengage_C1_TOT",
            "Percentage_C1",
            "CEPAR_C2",
            "TOT_C2",
            "Percengage_C2_TOT",
            "Percentage_C2",
            "CEPAR_C3",
            "TOT_C3",
            "Percengage_C3_TOT",
            "Percentage_C3",
        ]
        df_CEPAR.to_csv(folder_results / "CEPAR.csv")

        df_CEPAR_PET = pd.DataFrame(dict_CEPAR_PET)
        df_CEPAR_PET.index = ENTITY_TYPES
        df_CEPAR_PET.to_csv(folder_results / "CEPAR_PET.csv")

        df_CEPAR_PET_C1 = pd.DataFrame(dict_CEPAR_PET_C1)
        df_CEPAR_PET_C1.index = ENTITY_TYPES
        df_CEPAR_PET_C1.to_csv(folder_results / "CEPAR_PET_C1.csv")

        df_CEPAR_PET_C2 = pd.DataFrame(dict_CEPAR_PET_C2)
        df_CEPAR_PET_C2.index = ENTITY_TYPES
        df_CEPAR_PET_C2.to_csv(folder_results / "CEPAR_PET_C2.csv")

        df_CEPAR_PET_C3 = pd.DataFrame(dict_CEPAR_PET_C3)
        df_CEPAR_PET_C3.index = ENTITY_TYPES
        df_CEPAR_PET_C3.to_csv(folder_results / "CEPAR_PET_C3.csv")

        df_CEPAR_PLT = pd.DataFrame(dict_CEPAR_PLT)
        df_CEPAR_PLT.index = LAYOUT_TYPES
        df_CEPAR_PLT.to_csv(folder_results / "CEPAR_PLT.csv")

        df_CEPAR_PLT_C1 = pd.DataFrame(dict_CEPAR_PLT_C1)
        df_CEPAR_PLT_C1.index = LAYOUT_TYPES
        df_CEPAR_PLT_C1.to_csv(folder_results / "CEPAR_PLT_C1.csv")

        df_CEPAR_PLT_C2 = pd.DataFrame(dict_CEPAR_PLT_C2)
        df_CEPAR_PLT_C2.index = LAYOUT_TYPES
        df_CEPAR_PLT_C2.to_csv(folder_results / "CEPAR_PLT_C2.csv")

        df_CEPAR_PLT_C3 = pd.DataFrame(dict_CEPAR_PLT_C3)
        df_CEPAR_PLT_C3.index = LAYOUT_TYPES
        df_CEPAR_PLT_C3.to_csv(folder_results / "CEPAR_PLT_C3.csv")

        df_OPAR = pd.DataFrame(dict_OPAR)
        df_OPAR.index = [
            "OPAR",
            "OPAR_TXT",
            "TOT",
            "Percentage",
            "Percentage_TXT",
            "OPAR_C1",
            "OPAR_C1_TOT",
            "Percentage_C1_TOT",
            "Percentage_C1",
            "OPAR_C2",
            "OPAR_C2_TOT",
            "Percentage_C2_TOT",
            "Percentage_C2",
            "OPAR_C3",
            "OPAR_C3_TOT",
            "Percentage_C3_TOT",
            "Percentage_C3",
        ]
        df_OPAR.to_csv(folder_results / "OPAR.csv")

        df_ANSL = pd.DataFrame(dict_ANSL)
        df_ANSL.index = ["ANSL", "ANSL_C1", "ANSL_C2", "ANSL_C3"]
        df_ANSL.to_csv(folder_results / "ANSL.csv")

        df_QUR = pd.DataFrame(dict_QUR)
        df_QUR.index = ["QUR", "QUR_C1", "QUR_C2", "QUR_C3"]
        df_QUR.to_csv(folder_results / "QUR.csv")

        df_UR = pd.DataFrame(dict_UR)
        df_UR.index = ["UR", "UR_C1", "UR_C2", "UR_C3"]
        df_UR.to_csv(folder_results / "UR.csv")

        df_QEWR = pd.DataFrame(dict_QEWR)
        df_QEWR.index = [
            "QEWR",
            "TOT",
            "Percentage",
            "QEWR_C1",
            "Percentage_C1",
            "QEWR_C2",
            "Percentage_C2",
            "QEWR_C3",
            "Percentage_C3",
        ]
        df_QEWR.to_csv(folder_results / "QEWR.csv")

        df_QEWR_RATIO = pd.DataFrame(dict_QEWR_RATIO)
        df_QEWR_RATIO.index = [
            "QEWR_RATIO",
            "QEWR_RATIO_C1",
            "QEWR_RATIO_C2",
            "QEWR_RATIO_C3",
        ]
        df_QEWR_RATIO.to_csv(folder_results / "QEWR_RATIO.csv")

        df_QEPR = pd.DataFrame(dict_QEPR)
        df_QEPR.index = [
            "TOT",
            "UE",
            "Percentage_UE",
            "CE",
            "Percentage_CE",
            "SAME",
            "Percentage_SAME",
        ]
        df_QEPR.to_csv(folder_results / "QEPR.csv")

        metrics_index = [
            "Match Entity Count",
            "Match Layout Count",
            "Match Entity-Layout Count",
            "Hallucination Count",
            "Total Processed",
            "Match Entity Rate",
            "Match Layout Rate",
            "Match Entity-Layout Rate",
            "Hallucination Rate",
        ]

        df_AEMR_ALMR_HR = pd.DataFrame(dict_AEMR_ALMR_HR)
        df_AEMR_ALMR_HR.index = metrics_index
        df_AEMR_ALMR_HR.to_csv(folder_results / "AEMR_ALMR_HR.csv")

        df_AEMR_ALMR_HR_C1 = pd.DataFrame(dict_AEMR_ALMR_HR_C1)
        df_AEMR_ALMR_HR_C1.index = metrics_index
        df_AEMR_ALMR_HR_C1.to_csv(folder_results / "AEMR_ALMR_HR_C1.csv")

        df_AEMR_ALMR_HR_C2 = pd.DataFrame(dict_AEMR_ALMR_HR_C2)
        df_AEMR_ALMR_HR_C2.index = metrics_index
        df_AEMR_ALMR_HR_C2.to_csv(folder_results / "AEMR_ALMR_HR_C2.csv")

        df_AEMR_ALMR_HR_C3 = pd.DataFrame(dict_AEMR_ALMR_HR_C3)
        df_AEMR_ALMR_HR_C3.index = metrics_index
        df_AEMR_ALMR_HR_C3.to_csv(folder_results / "AEMR_ALMR_HR_C3.csv") 


        df_CEBBOX = pd.DataFrame(dict_CEBBOX)
        df_CEBBOX.index = [
            "CEBBOX",
            "CEBBOX_TOP",
            "CEBBOX_CENTER",
            "CEBBOX_BOTTOM",
            "CEBBOX_TOP_LEFT",
            "CEBBOX_TOP_RIGHT",
            "CEBBOX_BOTTOM_LEFT",
            "CEBBOX_BOTTOM_RIGHT",
            "CEBBOX_CENTER_LEFT",
            "CEBBOX_CENTER_RIGHT",
            "CEBBOX_TOP_LEFT_QUARTER",
            "CEBBOX_TOP_RIGHT_QUARTER",
            "CEBBOX_BOTTOM_LEFT_QUARTER",
            "CEBBOX_BOTTOM_RIGHT_QUARTER",
        ]
        df_CEBBOX_C1 = pd.DataFrame(dict_CEBBOX_C1)
        df_CEBBOX_C1.index = [
            "CEBBOX",
            "CEBBOX_TOP",
            "CEBBOX_CENTER",
            "CEBBOX_BOTTOM",
            "CEBBOX_TOP_LEFT",
            "CEBBOX_TOP_RIGHT",
            "CEBBOX_BOTTOM_LEFT",
            "CEBBOX_BOTTOM_RIGHT",
            "CEBBOX_CENTER_LEFT",
            "CEBBOX_CENTER_RIGHT",
            "CEBBOX_TOP_LEFT_QUARTER",
            "CEBBOX_TOP_RIGHT_QUARTER",
            "CEBBOX_BOTTOM_LEFT_QUARTER",
            "CEBBOX_BOTTOM_RIGHT_QUARTER",
        ]

        df_CEBBOX_C2 = pd.DataFrame(dict_CEBBOX_C2)
        df_CEBBOX_C2.index = [
            "CEBBOX",
            "CEBBOX_TOP",
            "CEBBOX_CENTER",
            "CEBBOX_BOTTOM",
            "CEBBOX_TOP_LEFT",
            "CEBBOX_TOP_RIGHT",
            "CEBBOX_BOTTOM_LEFT",
            "CEBBOX_BOTTOM_RIGHT",
            "CEBBOX_CENTER_LEFT",
            "CEBBOX_CENTER_RIGHT",
            "CEBBOX_TOP_LEFT_QUARTER",
            "CEBBOX_TOP_RIGHT_QUARTER",
            "CEBBOX_BOTTOM_LEFT_QUARTER",
            "CEBBOX_BOTTOM_RIGHT_QUARTER",
        ]

        df_CEBBOX_C3 = pd.DataFrame(dict_CEBBOX_C3)
        df_CEBBOX_C3.index = [
            "CEBBOX",
            "CEBBOX_TOP",
            "CEBBOX_CENTER",
            "CEBBOX_BOTTOM",
            "CEBBOX_TOP_LEFT",
            "CEBBOX_TOP_RIGHT",
            "CEBBOX_BOTTOM_LEFT",
            "CEBBOX_BOTTOM_RIGHT",
            "CEBBOX_CENTER_LEFT",
            "CEBBOX_CENTER_RIGHT",
            "CEBBOX_TOP_LEFT_QUARTER",
            "CEBBOX_TOP_RIGHT_QUARTER",
            "CEBBOX_BOTTOM_LEFT_QUARTER",
            "CEBBOX_BOTTOM_RIGHT_QUARTER",
        ]

        df_CEBBOX.to_csv(folder_results / "CEBBOX.csv")
        df_CEBBOX_C1.to_csv(folder_results / "CEBBOX_C1.csv")
        df_CEBBOX_C2.to_csv(folder_results / "CEBBOX_C2.csv")
        df_CEBBOX_C3.to_csv(folder_results / "CEBBOX_C3.csv")

        df_CEBBOX_UTD = pd.DataFrame(dict_CEBBOX_UTD)
        df_CEBBOX_UTD.index = [
            "CEBBOX",
            "CEBBOX_TOP",
            "CEBBOX_CENTER",
            "CEBBOX_BOTTOM",
            "CEBBOX_TOP_LEFT",
            "CEBBOX_TOP_RIGHT",
            "CEBBOX_BOTTOM_LEFT",
            "CEBBOX_BOTTOM_RIGHT",
            "CEBBOX_CENTER_LEFT",
            "CEBBOX_CENTER_RIGHT",
            "CEBBOX_TOP_LEFT_QUARTER",
            "CEBBOX_TOP_RIGHT_QUARTER",
            "CEBBOX_BOTTOM_LEFT_QUARTER",
            "CEBBOX_BOTTOM_RIGHT_QUARTER",
        ]

        df_CEBBOX_UTD_C1 = pd.DataFrame(dict_CEBBOX_UTD_C1)
        df_CEBBOX_UTD_C1.index = [
            "CEBBOX",
            "CEBBOX_TOP",
            "CEBBOX_CENTER",
            "CEBBOX_BOTTOM",
            "CEBBOX_TOP_LEFT",
            "CEBBOX_TOP_RIGHT",
            "CEBBOX_BOTTOM_LEFT",
            "CEBBOX_BOTTOM_RIGHT",
            "CEBBOX_CENTER_LEFT",
            "CEBBOX_CENTER_RIGHT",
            "CEBBOX_TOP_LEFT_QUARTER",
            "CEBBOX_TOP_RIGHT_QUARTER",
            "CEBBOX_BOTTOM_LEFT_QUARTER",
            "CEBBOX_BOTTOM_RIGHT_QUARTER",
        ]

        df_CEBBOX_UTD_C2 = pd.DataFrame(dict_CEBBOX_UTD_C2)
        df_CEBBOX_UTD_C2.index = [
            "CEBBOX",
            "CEBBOX_TOP",
            "CEBBOX_CENTER",
            "CEBBOX_BOTTOM",
            "CEBBOX_TOP_LEFT",
            "CEBBOX_TOP_RIGHT",
            "CEBBOX_BOTTOM_LEFT",
            "CEBBOX_BOTTOM_RIGHT",
            "CEBBOX_CENTER_LEFT",
            "CEBBOX_CENTER_RIGHT",
            "CEBBOX_TOP_LEFT_QUARTER",
            "CEBBOX_TOP_RIGHT_QUARTER",
            "CEBBOX_BOTTOM_LEFT_QUARTER",
            "CEBBOX_BOTTOM_RIGHT_QUARTER",
        ]

        df_CEBBOX_UTD_C3 = pd.DataFrame(dict_CEBBOX_UTD_C3)
        df_CEBBOX_UTD_C3.index = [
            "CEBBOX",
            "CEBBOX_TOP",
            "CEBBOX_CENTER",
            "CEBBOX_BOTTOM",
            "CEBBOX_TOP_LEFT",
            "CEBBOX_TOP_RIGHT",
            "CEBBOX_BOTTOM_LEFT",
            "CEBBOX_BOTTOM_RIGHT",
            "CEBBOX_CENTER_LEFT",
            "CEBBOX_CENTER_RIGHT",
            "CEBBOX_TOP_LEFT_QUARTER",
            "CEBBOX_TOP_RIGHT_QUARTER",
            "CEBBOX_BOTTOM_LEFT_QUARTER",
            "CEBBOX_BOTTOM_RIGHT_QUARTER",
        ]

        df_CEBBOX_UTD.to_csv(folder_results / "CEBBOX_UTD.csv")
        df_CEBBOX_UTD_C1.to_csv(folder_results / "CEBBOX_UTD_C1.csv")
        df_CEBBOX_UTD_C2.to_csv(folder_results / "CEBBOX_UTD_C2.csv")
        df_CEBBOX_UTD_C3.to_csv(folder_results / "CEBBOX_UTD_C3.csv")

        df_UTD_LAYOUT = pd.DataFrame(dict_UTD_LAYOUT)
        df_UTD_LAYOUT.index = [
            "Total Count",
            *[f"Layout_{layout}" for layout in LAYOUT_TYPES],
        ]

        df_UTD_LAYOUT_C1 = pd.DataFrame(dict_UTD_LAYOUT_C1)
        df_UTD_LAYOUT_C1.index = [
            "Total Count",
            *[f"Layout_{layout}" for layout in LAYOUT_TYPES],
        ]

        df_UTD_LAYOUT_C2 = pd.DataFrame(dict_UTD_LAYOUT_C2)
        df_UTD_LAYOUT_C2.index = [
            "Total Count",
            *[f"Layout_{layout}" for layout in LAYOUT_TYPES],
        ]

        df_UTD_LAYOUT_C3 = pd.DataFrame(dict_UTD_LAYOUT_C3)
        df_UTD_LAYOUT_C3.index = [
            "Total Count",
            *[f"Layout_{layout}" for layout in LAYOUT_TYPES],
        ]

        df_UTD_LAYOUT.to_csv(folder_results / "UTD_LAYOUT.csv")
        df_UTD_LAYOUT_C1.to_csv(folder_results / "UTD_LAYOUT_C1.csv")
        df_UTD_LAYOUT_C2.to_csv(folder_results / "UTD_LAYOUT_C2.csv")
        df_UTD_LAYOUT_C3.to_csv(folder_results / "UTD_LAYOUT_C3.csv")


if __name__ == "__main__":
    generate_analysis_report(
        dataset="DUDE_NEW_METRICS", images_path="/Users/david/Desktop/ACL/DUMP SERVERS/CORR_RHODES/DUDE_train-val-test_binaries/DUDE_train-val-test_binaries/images/train"
        # dataset="MPDocVQA_NEW_METRICS", images_path="/Users/david/Desktop/ACL/DUMP SERVERS/CORR_RHODES/images/images"
    )
