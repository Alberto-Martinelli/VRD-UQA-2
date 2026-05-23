

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from collections import Counter
import math
import re  # Added for window size extraction
# from anls_star import anls_score
# from gliner import GLiNER
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

MACRO_ENTITY_TYPES = [
    "NUMERIC",
    "TEMPORAL",
    "ENTITY",
    "LOCATION",
    "STRUCTURE"
]

PAGE_LAYOUT = [
    "TOP_LEFT",
    "TOP_RIGHT",
    "BOTTOM_LEFT",
    "BOTTOM_RIGHT",
]

MACRO_ENTITY_MAPPER={
    "numerical_value_number": "NUMERIC",
    "measure_unit": "NUMERIC",
    "price_number_information": "NUMERIC",
    "price_numerical_value": "NUMERIC",
    "percentage": "NUMERIC",
    "temperature": "NUMERIC",
    "currency": "NUMERIC",
    "date_information": "TEMPORAL",
    "date_numerical_value": "TEMPORAL",
    "time_information": "TEMPORAL",
    "time_numerical_value": "TEMPORAL",
    "year_number_information": "TEMPORAL",
    "year_numerical_value": "TEMPORAL",
    "person_name": "ENTITY",
    "company_name": "ENTITY",
    "event": "ENTITY",
    "product": "ENTITY",
    "food": "ENTITY",
    "chemical_element": "ENTITY",
    "job_title_name": "ENTITY",
    "job_title_information": "ENTITY",
    "animal": "ENTITY",
    "plant": "ENTITY",
    "movie": "ENTITY",
    "book": "ENTITY",
    "transport_means": "ENTITY",
    "country": "LOCATION",
    "city": "LOCATION",
    "street": "LOCATION",
    "spatial_information": "LOCATION",
    "continent": "LOCATION",
    "postal_code_information": "LOCATION",
    "postal_code_numerical_value": "LOCATION",
    "document_position_information": "STRUCTURE",
    "page_number_information": "STRUCTURE",
    "page_number_numerical_value": "STRUCTURE",
    "document_element_type": "STRUCTURE",
    "document_element_information": "STRUCTURE",
    "document_structure_information": "STRUCTURE",
}

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

MACRO_LAYOUT_TYPES = [
    "text",
    "vre"
]

MAPPER_LAYOUT_TYPES = {
    "title": "text",
    "plain text": "text",
    "abandon": "text",
    "figure": "vre",
    "figure_caption": "text",
    "table": "vre",
    "table_caption": "text",
    "table_footnote": "text",
    "isolate_formula": "vre",
    "formula_caption": "text",
}

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
        self.debug = debug
        self.entity_identifier = entity_verifier
        self.dataset = dataset
        self.images_path = images_path
        # Pre-filter once so every metric method iterates only valid corrupted results.
        self.valid_results = [
            r for r in results
            if r["is_corrupted"]
            and "verification_result" in r
            and "vqa_results" in r["verification_result"]
            and len(r["verification_result"]["vqa_results"]) > 0
        ]

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _get_answers(res):
        """Return the list of answer dicts for a valid result."""
        vqa_result = res["verification_result"]["vqa_results"][0]
        return vqa_result.get("answers", vqa_result.get("answer", []))

    @staticmethod
    def _unique_entities(corrupted_entities):
        """Return corrupted_entities deduplicated by text."""
        seen = []
        unique = []
        for entity in corrupted_entities:
            if entity["text"] not in seen:
                seen.append(entity["text"])
                unique.append(entity)
        return unique

    @staticmethod
    def _count_by_complexity(complexity, c1, c2, c3, amount=1):
        """Increment the counter matching this result's complexity level."""
        if complexity == 1:
            c1 += amount
        elif complexity == 2:
            c2 += amount
        elif complexity == 3:
            c3 += amount
        return c1, c2, c3

    @staticmethod
    def _normalize_sliced(hit, counter, hit_c1, counter_c1, hit_c2, counter_c2, hit_c3, counter_c3):
        """Divide hit counts by their counters for each key and each complexity level.
        Returns four dicts: (total, c1, c2, c3)."""
        def _div(h, c):
            return {k: h[k] / c[k] if c[k] != 0 else 0 for k in h}
        return _div(hit, counter), _div(hit_c1, counter_c1), _div(hit_c2, counter_c2), _div(hit_c3, counter_c3)

    @staticmethod
    def _bbox_quadrant(bbox, avg_x, avg_y):
        """Return the PAGE_LAYOUT quadrant label for a bounding box."""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        if cx < avg_x / 2 and cy < avg_y / 2:
            return "TOP_LEFT"
        elif cx < avg_x / 2:
            return "BOTTOM_LEFT"
        elif cy < avg_y / 2:
            return "TOP_RIGHT"
        else:
            return "BOTTOM_RIGHT"

    @staticmethod
    def _page_dimensions(pages):
        """Return (avg_x, avg_y) pixel size across a list of page image paths."""
        avg_x = avg_y = 0
        for page in pages:
            if "data1" in page:
                page = page.replace("data1", "data2")
            x, y = Image.open(page).size
            avg_x += x
            avg_y += y
        avg_x /= len(pages)
        avg_y /= len(pages)
        return avg_x, avg_y

    @staticmethod
    def _is_unable(ans):
        return ans.get("answer_converted", "").lower() == "unable to determine"

    # ------------------------------------------------------------------ metrics

    def calculate_metrics(self):
        metrics = {
            "QUR": self.QUR(),
            "QUR_DE": self.QUR_DE(),
            "QUR_NLPE": self.QUR_NLPE(),
            "QUR_QP": self.QUR_QP(),
            "QUR_PL": self.QUR_PL(),
            "QUR_DED": self.QUR_DED(),
            "UR": self.UR(),
            "UR_DE": self.UR_DE(),
            "UR_NLPE": self.UR_NLPE(),
            "UR_PAGE": self.UR_PAGE(),
            "UR_PAGE_QP": self.UR_PAGE_QP(),
            "UR_PAGE_DE": self.UR_PAGE_DE(),
            "UR_PAGE_DED": self.UR_PAGE_DED(),
        }
        return metrics

    def QUR(self):
        # QUR = fraction of corrupted questions where ALL pages answered "unable to determine"
        correct = correct_c1 = correct_c2 = correct_c3 = 0
        total = total_c1 = total_c2 = total_c3 = 0

        for res in self.valid_results:
            all_answers = self._get_answers(res)
            complexity = res["complexity"]

            total += 1
            total_c1, total_c2, total_c3 = self._count_by_complexity(complexity, total_c1, total_c2, total_c3)

            unable = sum(1 for a in all_answers if self._is_unable(a))
            n = len(all_answers)

            if n > 0 and unable == n:
                correct += 1
                correct_c1, correct_c2, correct_c3 = self._count_by_complexity(
                    complexity, correct_c1, correct_c2, correct_c3
                )

        if self.debug:
            print(f"Total corrupted: {total} (c1={total_c1} c2={total_c2} c3={total_c3})")
            print(f"Correct unable: {correct} ({correct/total*100:.2f}%)")

        weighted_unable = 0  # placeholder, not used downstream
        return [
            correct / total,
            correct_c1 / total_c1 if total_c1 else 0,
            correct_c2 / total_c2 if total_c2 else 0,
            correct_c3 / total_c3 if total_c3 else 0,
            weighted_unable,
        ]


    def QUR_DE(self):
        # QUR sliced by document element (layout) type of the corrupted entity
        hit    = {el: 0 for el in LAYOUT_TYPES}
        hit_c1 = {el: 0 for el in LAYOUT_TYPES}
        hit_c2 = {el: 0 for el in LAYOUT_TYPES}
        hit_c3 = {el: 0 for el in LAYOUT_TYPES}
        cnt    = {el: 0 for el in LAYOUT_TYPES}
        cnt_c1 = {el: 0 for el in LAYOUT_TYPES}
        cnt_c2 = {el: 0 for el in LAYOUT_TYPES}
        cnt_c3 = {el: 0 for el in LAYOUT_TYPES}

        for res in self.valid_results:
            all_answers = self._get_answers(res)
            complexity = res["complexity"]
            unique_ents = self._unique_entities(res["corrupted_entities"])

            unable = sum(1 for a in all_answers if self._is_unable(a))
            n = len(all_answers)
            all_unable = n > 0 and unable == n

            for el in unique_ents:
                t = el["objectType"]
                cnt[t] += 1
                if complexity == 1: cnt_c1[t] += 1
                elif complexity == 2: cnt_c2[t] += 1
                elif complexity == 3: cnt_c3[t] += 1
                if all_unable:
                    hit[t] += 1
                    if complexity == 1: hit_c1[t] += 1
                    elif complexity == 2: hit_c2[t] += 1
                    elif complexity == 3: hit_c3[t] += 1

        return list(self._normalize_sliced(hit, cnt, hit_c1, cnt_c1, hit_c2, cnt_c2, hit_c3, cnt_c3))

    
    def QUR_QP(self):
        # QUR sliced by page quadrant (bounding-box position) of the corrupted entity
        hit    = {el: 0 for el in PAGE_LAYOUT}
        hit_c1 = {el: 0 for el in PAGE_LAYOUT}
        hit_c2 = {el: 0 for el in PAGE_LAYOUT}
        hit_c3 = {el: 0 for el in PAGE_LAYOUT}
        cnt    = {el: 0 for el in PAGE_LAYOUT}
        cnt_c1 = {el: 0 for el in PAGE_LAYOUT}
        cnt_c2 = {el: 0 for el in PAGE_LAYOUT}
        cnt_c3 = {el: 0 for el in PAGE_LAYOUT}

        for res in self.valid_results:
            all_answers = self._get_answers(res)
            complexity = res["complexity"]

            unique_pages = list(set(p for a in all_answers for p in a.get("pages", [])))
            avg_x, avg_y = self._page_dimensions(unique_pages)

            unable = sum(1 for a in all_answers if self._is_unable(a))
            n = len(all_answers)
            all_unable = n > 0 and unable == n

            for el in res["corrupted_entities"]:
                q = self._bbox_quadrant(el.get("bbox", []), avg_x, avg_y)
                cnt[q] += 1
                if complexity == 1: cnt_c1[q] += 1
                elif complexity == 2: cnt_c2[q] += 1
                elif complexity == 3: cnt_c3[q] += 1
                if all_unable:
                    hit[q] += 1
                    if complexity == 1: hit_c1[q] += 1
                    elif complexity == 2: hit_c2[q] += 1
                    elif complexity == 3: hit_c3[q] += 1

        return list(self._normalize_sliced(hit, cnt, hit_c1, cnt_c1, hit_c2, cnt_c2, hit_c3, cnt_c3))
    

    def QUR_NLPE(self):
        # QUR sliced by NLP entity type (NUMERIC, TEMPORAL, ENTITY, LOCATION, STRUCTURE)
        hit    = {el: 0 for el in MACRO_ENTITY_TYPES}
        hit_c1 = {el: 0 for el in MACRO_ENTITY_TYPES}
        hit_c2 = {el: 0 for el in MACRO_ENTITY_TYPES}
        hit_c3 = {el: 0 for el in MACRO_ENTITY_TYPES}
        cnt    = {el: 0 for el in MACRO_ENTITY_TYPES}
        cnt_c1 = {el: 0 for el in MACRO_ENTITY_TYPES}
        cnt_c2 = {el: 0 for el in MACRO_ENTITY_TYPES}
        cnt_c3 = {el: 0 for el in MACRO_ENTITY_TYPES}

        for res in self.valid_results:
            all_answers = self._get_answers(res)
            complexity = res["complexity"]
            entity_types = res["entity_type"]

            unable = sum(1 for a in all_answers if self._is_unable(a))
            n = len(all_answers)
            all_unable = n > 0 and unable == n

            for et in entity_types:
                macro = MACRO_ENTITY_MAPPER[et]
                cnt[macro] += 1
                if complexity == 1: cnt_c1[macro] += 1
                elif complexity == 2: cnt_c2[macro] += 1
                elif complexity == 3: cnt_c3[macro] += 1
                if all_unable:
                    hit[macro] += 1
                    if complexity == 1: hit_c1[macro] += 1
                    elif complexity == 2: hit_c2[macro] += 1
                    elif complexity == 3: hit_c3[macro] += 1

        return list(self._normalize_sliced(hit, cnt, hit_c1, cnt_c1, hit_c2, cnt_c2, hit_c3, cnt_c3))


    def QUR_PL(self):
        # QUR sliced by page-length (number of unique pages seen across all answers).
        # The page-count axis is dynamic, so dicts are keyed by num_pages rather than a fixed enum.
        # list_len is returned alongside the metric dicts so callers can use it as a DataFrame index.
        hit    = {}
        hit_c1 = {}
        hit_c2 = {}
        hit_c3 = {}
        cnt    = {}
        cnt_c1 = {}
        cnt_c2 = {}
        cnt_c3 = {}

        for res in self.valid_results:
            all_answers = self._get_answers(res)
            complexity = res["complexity"]

            num_pages = len(set(p for a in all_answers for p in a.get("pages", [])))

            # Initialise buckets on first encounter of this page count
            for d in (cnt, cnt_c1, cnt_c2, cnt_c3, hit, hit_c1, hit_c2, hit_c3):
                d.setdefault(num_pages, 0)

            cnt[num_pages] += 1
            if complexity == 1: cnt_c1[num_pages] += 1
            elif complexity == 2: cnt_c2[num_pages] += 1
            elif complexity == 3: cnt_c3[num_pages] += 1

            unable = sum(1 for a in all_answers if self._is_unable(a))
            n = len(all_answers)
            all_unable = n > 0 and unable == n

            if all_unable:
                hit[num_pages] += 1
                if complexity == 1: hit_c1[num_pages] += 1
                elif complexity == 2: hit_c2[num_pages] += 1
                elif complexity == 3: hit_c3[num_pages] += 1

        list_len = sorted(cnt.keys())
        res, res_c1, res_c2, res_c3 = self._normalize_sliced(
            hit, cnt, hit_c1, cnt_c1, hit_c2, cnt_c2, hit_c3, cnt_c3
        )
        return [res, res_c1, res_c2, res_c3, list_len]


    def QUR_DED(self):
        # QUR sliced by document element density: fraction of layout objects that are visual
        # (figures/tables/formulas). Buckets: <15%, 15–25%, >25%.
        DED_BUCKETS = ["<15", "15-25", ">25"]
        hit    = {k: 0 for k in DED_BUCKETS}
        hit_c1 = {k: 0 for k in DED_BUCKETS}
        hit_c2 = {k: 0 for k in DED_BUCKETS}
        hit_c3 = {k: 0 for k in DED_BUCKETS}
        cnt    = {k: 0 for k in DED_BUCKETS}
        cnt_c1 = {k: 0 for k in DED_BUCKETS}
        cnt_c2 = {k: 0 for k in DED_BUCKETS}
        cnt_c3 = {k: 0 for k in DED_BUCKETS}

        for res in self.valid_results:
            all_answers = self._get_answers(res)
            complexity = res["complexity"]

            doc_dist = {el: 0 for el in MACRO_LAYOUT_TYPES}
            for page, info in res["layout_analysis"]["pages"].items():
                for obj in info["layout_analysis"].values():
                    doc_dist[MAPPER_LAYOUT_TYPES[obj["ObjectType"]]] += 1

            text_count, vre_count = doc_dist["text"], doc_dist["vre"]
            if text_count == 0:
                text_count = 1  # avoid divide-by-zero when doc is purely visual
            vre_ratio = vre_count / (vre_count + text_count)
            if vre_ratio < 0.15:
                key = "<15"
            elif vre_ratio < 0.25:
                key = "15-25"
            else:
                key = ">25"

            cnt[key] += 1
            if complexity == 1: cnt_c1[key] += 1
            elif complexity == 2: cnt_c2[key] += 1
            elif complexity == 3: cnt_c3[key] += 1

            unable = sum(1 for a in all_answers if self._is_unable(a))
            n = len(all_answers)
            all_unable = n > 0 and unable == n

            if all_unable:
                hit[key] += 1
                if complexity == 1: hit_c1[key] += 1
                elif complexity == 2: hit_c2[key] += 1
                elif complexity == 3: hit_c3[key] += 1

        return list(self._normalize_sliced(hit, cnt, hit_c1, cnt_c1, hit_c2, cnt_c2, hit_c3, cnt_c3))


    def UR(self):
        # UR = fraction of individual page answers that are "unable to determine"
        total = total_c1 = total_c2 = total_c3 = 0
        unable = unable_c1 = unable_c2 = unable_c3 = 0

        for res in self.valid_results:
            all_answers = self._get_answers(res)
            complexity = res["complexity"]

            for ans in all_answers:
                total += 1
                total_c1, total_c2, total_c3 = self._count_by_complexity(complexity, total_c1, total_c2, total_c3)
                if self._is_unable(ans):
                    unable += 1
                    unable_c1, unable_c2, unable_c3 = self._count_by_complexity(complexity, unable_c1, unable_c2, unable_c3)

        if total == 0:
            return [0, 0, 0, 0]
        return [
            unable / total,
            unable_c1 / total_c1 if total_c1 else 0,
            unable_c2 / total_c2 if total_c2 else 0,
            unable_c3 / total_c3 if total_c3 else 0,
        ]


    def UR_DE(self):
        # UR sliced by document element type of the corrupted entity, measured per answer
        hit    = {el: 0 for el in LAYOUT_TYPES}
        hit_c1 = {el: 0 for el in LAYOUT_TYPES}
        hit_c2 = {el: 0 for el in LAYOUT_TYPES}
        hit_c3 = {el: 0 for el in LAYOUT_TYPES}
        cnt    = {el: 0 for el in LAYOUT_TYPES}
        cnt_c1 = {el: 0 for el in LAYOUT_TYPES}
        cnt_c2 = {el: 0 for el in LAYOUT_TYPES}
        cnt_c3 = {el: 0 for el in LAYOUT_TYPES}

        for res in self.valid_results:
            all_answers = self._get_answers(res)
            complexity = res["complexity"]
            unique_ents = self._unique_entities(res["corrupted_entities"])

            for ans in all_answers:
                is_unable = self._is_unable(ans)
                for el in unique_ents:
                    t = el["objectType"]
                    cnt[t] += 1
                    if complexity == 1: cnt_c1[t] += 1
                    elif complexity == 2: cnt_c2[t] += 1
                    elif complexity == 3: cnt_c3[t] += 1
                    if is_unable:
                        hit[t] += 1
                        if complexity == 1: hit_c1[t] += 1
                        elif complexity == 2: hit_c2[t] += 1
                        elif complexity == 3: hit_c3[t] += 1

        return list(self._normalize_sliced(hit, cnt, hit_c1, cnt_c1, hit_c2, cnt_c2, hit_c3, cnt_c3))


    def UR_PAGE(self):
        # UR split by whether the answer's page contains the corrupted entity (inpage) or not (outpage).
        in_hit  = in_hit_c1  = in_hit_c2  = in_hit_c3  = 0
        in_cnt  = in_cnt_c1  = in_cnt_c2  = in_cnt_c3  = 0
        out_hit = out_hit_c1 = out_hit_c2 = out_hit_c3 = 0
        out_cnt = out_cnt_c1 = out_cnt_c2 = out_cnt_c3 = 0

        for res in self.valid_results:
            all_answers = self._get_answers(res)
            complexity = res["complexity"]
            patch_entities = res["patch_entities"]

            seen = [e["text"] for e in self._unique_entities(res["corrupted_entities"])]

            for ans in all_answers:
                pages_id = [p.split("/")[-1] for p in ans.get("pages", [])]

                # An answer is "inpage" when at least one of its pages contains a corrupted entity
                on_corrupted_page = any(
                    entity["text"] in seen
                    for pID, p in patch_entities.items() if pID in pages_id
                    for obj in p.values()
                    for entity in obj["entities"]
                )
                is_unable = self._is_unable(ans)

                if on_corrupted_page:
                    in_cnt += 1
                    in_cnt_c1, in_cnt_c2, in_cnt_c3 = self._count_by_complexity(complexity, in_cnt_c1, in_cnt_c2, in_cnt_c3)
                    if is_unable:
                        in_hit += 1
                        in_hit_c1, in_hit_c2, in_hit_c3 = self._count_by_complexity(complexity, in_hit_c1, in_hit_c2, in_hit_c3)
                else:
                    out_cnt += 1
                    out_cnt_c1, out_cnt_c2, out_cnt_c3 = self._count_by_complexity(complexity, out_cnt_c1, out_cnt_c2, out_cnt_c3)
                    if is_unable:
                        out_hit += 1
                        out_hit_c1, out_hit_c2, out_hit_c3 = self._count_by_complexity(complexity, out_hit_c1, out_hit_c2, out_hit_c3)

        def _safe_div(h, c): return h / c if c else 0

        return [
            _safe_div(in_hit,  in_cnt),
            _safe_div(in_hit_c1,  in_cnt_c1),
            _safe_div(in_hit_c2,  in_cnt_c2),
            _safe_div(in_hit_c3,  in_cnt_c3),
            _safe_div(out_hit, out_cnt),
            _safe_div(out_hit_c1, out_cnt_c1),
            _safe_div(out_hit_c2, out_cnt_c2),
            _safe_div(out_hit_c3, out_cnt_c3),
        ]

    
    def UR_NLPE(self):
        # UR sliced by NLP entity type, measured per answer
        hit    = {el: 0 for el in MACRO_ENTITY_TYPES}
        hit_c1 = {el: 0 for el in MACRO_ENTITY_TYPES}
        hit_c2 = {el: 0 for el in MACRO_ENTITY_TYPES}
        hit_c3 = {el: 0 for el in MACRO_ENTITY_TYPES}
        cnt    = {el: 0 for el in MACRO_ENTITY_TYPES}
        cnt_c1 = {el: 0 for el in MACRO_ENTITY_TYPES}
        cnt_c2 = {el: 0 for el in MACRO_ENTITY_TYPES}
        cnt_c3 = {el: 0 for el in MACRO_ENTITY_TYPES}

        for res in self.valid_results:
            all_answers = self._get_answers(res)
            complexity = res["complexity"]
            entity_types = res["entity_type"]

            for ans in all_answers:
                is_unable = self._is_unable(ans)
                for et in entity_types:
                    macro = MACRO_ENTITY_MAPPER[et]
                    cnt[macro] += 1
                    if complexity == 1: cnt_c1[macro] += 1
                    elif complexity == 2: cnt_c2[macro] += 1
                    elif complexity == 3: cnt_c3[macro] += 1
                    if is_unable:
                        hit[macro] += 1
                        if complexity == 1: hit_c1[macro] += 1
                        elif complexity == 2: hit_c2[macro] += 1
                        elif complexity == 3: hit_c3[macro] += 1

        return list(self._normalize_sliced(hit, cnt, hit_c1, cnt_c1, hit_c2, cnt_c2, hit_c3, cnt_c3))


    def UR_PAGE_DE(self):
        # UR restricted to inpage answers, sliced by the layout type of the element
        # that contains the corrupted entity on that page
        hit    = {el: 0 for el in LAYOUT_TYPES}
        hit_c1 = {el: 0 for el in LAYOUT_TYPES}
        hit_c2 = {el: 0 for el in LAYOUT_TYPES}
        hit_c3 = {el: 0 for el in LAYOUT_TYPES}
        cnt    = {el: 0 for el in LAYOUT_TYPES}
        cnt_c1 = {el: 0 for el in LAYOUT_TYPES}
        cnt_c2 = {el: 0 for el in LAYOUT_TYPES}
        cnt_c3 = {el: 0 for el in LAYOUT_TYPES}

        for res in self.valid_results:
            all_answers = self._get_answers(res)
            complexity = res["complexity"]
            patch_entities = res["patch_entities"]
            seen = [e["text"] for e in self._unique_entities(res["corrupted_entities"])]

            for ans in all_answers:
                pages_id = [p.split("/")[-1] for p in ans.get("pages", [])]

                # Collect the layout types of objects on inpage pages that contain corrupted entities
                de_types = list(set(
                    obj["type"]
                    for pID, p in patch_entities.items() if pID in pages_id
                    for obj in p.values()
                    for entity in obj["entities"] if entity["text"] in seen
                ))

                if de_types:
                    is_unable = self._is_unable(ans)
                    for t in de_types:
                        cnt[t] += 1
                        if complexity == 1: cnt_c1[t] += 1
                        elif complexity == 2: cnt_c2[t] += 1
                        elif complexity == 3: cnt_c3[t] += 1
                        if is_unable:
                            hit[t] += 1
                            if complexity == 1: hit_c1[t] += 1
                            elif complexity == 2: hit_c2[t] += 1
                            elif complexity == 3: hit_c3[t] += 1

        return list(self._normalize_sliced(hit, cnt, hit_c1, cnt_c1, hit_c2, cnt_c2, hit_c3, cnt_c3))


    def UR_PAGE_QP(self):
        # UR restricted to inpage answers, sliced by the quadrant of the corrupted entity's bbox
        hit    = {el: 0 for el in PAGE_LAYOUT}
        hit_c1 = {el: 0 for el in PAGE_LAYOUT}
        hit_c2 = {el: 0 for el in PAGE_LAYOUT}
        hit_c3 = {el: 0 for el in PAGE_LAYOUT}
        cnt    = {el: 0 for el in PAGE_LAYOUT}
        cnt_c1 = {el: 0 for el in PAGE_LAYOUT}
        cnt_c2 = {el: 0 for el in PAGE_LAYOUT}
        cnt_c3 = {el: 0 for el in PAGE_LAYOUT}

        for res in self.valid_results:
            all_answers = self._get_answers(res)
            complexity = res["complexity"]
            patch_entities = res["patch_entities"]
            seen = [e["text"] for e in self._unique_entities(res["corrupted_entities"])]

            unique_pages = list(set(p for a in all_answers for p in a.get("pages", [])))
            avg_x, avg_y = self._page_dimensions(unique_pages)

            for ans in all_answers:
                pages_id = [p.split("/")[-1] for p in ans.get("pages", [])]

                # Objects on inpage pages that contain corrupted entities
                inpage_objs = [
                    obj
                    for pID, p in patch_entities.items() if pID in pages_id
                    for obj in p.values()
                    if any(e["text"] in seen for e in obj["entities"])
                ]

                if inpage_objs:
                    is_unable = self._is_unable(ans)
                    for obj in inpage_objs:
                        q = self._bbox_quadrant(obj.get("bbox", []), avg_x, avg_y)
                        cnt[q] += 1
                        if complexity == 1: cnt_c1[q] += 1
                        elif complexity == 2: cnt_c2[q] += 1
                        elif complexity == 3: cnt_c3[q] += 1
                        if is_unable:
                            hit[q] += 1
                            if complexity == 1: hit_c1[q] += 1
                            elif complexity == 2: hit_c2[q] += 1
                            elif complexity == 3: hit_c3[q] += 1

        return list(self._normalize_sliced(hit, cnt, hit_c1, cnt_c1, hit_c2, cnt_c2, hit_c3, cnt_c3))


    def UR_PAGE_DED(self):
        # UR per answer, sliced by number of visual elements (figures/tables/formulas) on that answer's pages.
        # Buckets: 0, 1, >1 visual elements.
        DED_BUCKETS = ["0", "1", ">1"]
        hit    = {k: 0 for k in DED_BUCKETS}
        hit_c1 = {k: 0 for k in DED_BUCKETS}
        hit_c2 = {k: 0 for k in DED_BUCKETS}
        hit_c3 = {k: 0 for k in DED_BUCKETS}
        cnt    = {k: 0 for k in DED_BUCKETS}
        cnt_c1 = {k: 0 for k in DED_BUCKETS}
        cnt_c2 = {k: 0 for k in DED_BUCKETS}
        cnt_c3 = {k: 0 for k in DED_BUCKETS}

        for res in self.valid_results:
            all_answers = self._get_answers(res)
            complexity = res["complexity"]
            layout_doc = res["layout_analysis"]["pages"]

            for ans in all_answers:
                pages_id = [p.split("/")[-1] for p in ans.get("pages", [])]

                doc_dist = {el: 0 for el in MACRO_LAYOUT_TYPES}
                for page, info in layout_doc.items():
                    if page in pages_id:
                        for obj in info["layout_analysis"].values():
                            doc_dist[MAPPER_LAYOUT_TYPES[obj["ObjectType"]]] += 1

                vre = doc_dist["vre"]
                key = "0" if vre == 0 else ("1" if vre == 1 else ">1")

                cnt[key] += 1
                if complexity == 1: cnt_c1[key] += 1
                elif complexity == 2: cnt_c2[key] += 1
                elif complexity == 3: cnt_c3[key] += 1
                if self._is_unable(ans):
                    hit[key] += 1
                    if complexity == 1: hit_c1[key] += 1
                    elif complexity == 2: hit_c2[key] += 1
                    elif complexity == 3: hit_c3[key] += 1

        return list(self._normalize_sliced(hit, cnt, hit_c1, cnt_c1, hit_c2, cnt_c2, hit_c3, cnt_c3))


def save_metric(folder, name, data, index, complexity_data=None):
    """Save one metric to CSV. If complexity_data is provided (list of 3 dicts),
    also saves _complexity_1/2/3 variants with the same index."""
    df = pd.DataFrame(data)
    df.index = index
    df.to_csv(folder / f"{name}.csv")
    if complexity_data:
        for i, c_data in enumerate(complexity_data, start=1):
            df_c = pd.DataFrame(c_data)
            df_c.index = index
            df_c.to_csv(folder / f"{name}_complexity_{i}.csv")


def _process_model_file(result_file, entity_verifier, dataset, images_path):
    """Load one augmented JSON, run all metrics, and return (model_name, metrics, list_len).
    Returns None if the file should be skipped."""
    model_name = result_file.stem.split("/")[-1].split("_")[0]
    print(f"Model name: {model_name}")

    with open(result_file, "r") as f:
        data = json.load(f)

    # Extract base_image_dir from the input file if present, otherwise fall back to images_path argument
    images_path = data.get("base_image_dir", images_path)

    results = data.get("corrupted_questions", [])
    if not results:
        print(f"Warning: No corrupted questions found in {result_file}")
        return None

    print(f"Processing {result_file}")
    print(f"Found {len(results)} questions")

    analyzer = VQAAnalyzer(results, entity_verifier, dataset, debug=False, images_path=images_path)
    metrics = analyzer.calculate_metrics()

    # QUR_PL returns the page-count axis alongside the metric dicts; extract it here
    # so the caller can use it as the DataFrame index when saving.
    *qur_pl_dicts, list_len = metrics["QUR_PL"]
    metrics["QUR_PL"] = qur_pl_dicts

    return model_name, metrics, list_len


def _collect_metrics_into_accum(accum, model_name, metrics):
    """Merge one model's metrics into the folder-level accumulator."""
    v, v1, v2, v3, w = metrics["QUR"]
    accum["QUR"]["total"][model_name] = [v, v1, v2, v3, w]

    # Metrics that return (base, c1, c2, c3) dicts sliced by a categorical dimension
    for name in ["QUR_DE", "QUR_NLPE", "QUR_QP", "QUR_DED", "QUR_PL",
                 "UR_DE", "UR_NLPE", "UR_PAGE_DE", "UR_PAGE_QP", "UR_PAGE_DED"]:
        base, c1, c2, c3 = metrics[name]
        accum[name]["total"][model_name] = base.values()
        accum[name]["c1"][model_name] = c1.values()
        accum[name]["c2"][model_name] = c2.values()
        accum[name]["c3"][model_name] = c3.values()

    v, v1, v2, v3 = metrics["UR"]
    accum["UR"]["total"][model_name] = [v, v1, v2, v3]

    # UR_PAGE splits answers into those on the page containing the corrupted entity (inpage)
    # and those on other pages (outpage), so it gets two separate accumulator entries.
    inpage, ip_c1, ip_c2, ip_c3, outpage, op_c1, op_c2, op_c3 = metrics["UR_PAGE"]
    accum["UR_PAGE_inpage"]["total"][model_name] = [inpage, ip_c1, ip_c2, ip_c3]
    accum["UR_PAGE_outpage"]["total"][model_name] = [outpage, op_c1, op_c2, op_c3]


def generate_analysis_report(dataset, images_path):
    entity_verifier = None  # EntityIdentifier(ENTITY_TYPES) — disabled, no external NER needed
    base_path = Path(__file__).parent.parent.parent / "VQA_analysis" / "models" / "results"
    dataset_path = base_path / dataset
    print(f"Base path: {base_path}")

    if not dataset_path.is_dir():
        print(f"ERROR: {dataset_path} is not a directory")
        return

    # Search one level deep to handle both flat (dataset/results_w2/) and
    # nested (dataset/LLM/results_w2/) folder layouts.
    candidate_parents = [dataset_path] + [p for p in dataset_path.iterdir() if p.is_dir()]
    result_folders = [
        p for parent in candidate_parents
        for p in parent.iterdir()
        if p.is_dir() and "results" in p.name
    ]

    for folder in result_folders:
        print(f"\n{'#' * 100}")
        print(f"Processing folder {folder}")

        folder_results = folder / "results"
        os.makedirs(folder_results, exist_ok=True)

        # accum[metric_name][total|c1|c2|c3][model_name] = values
        accum = {
            name: {"total": {}, "c1": {}, "c2": {}, "c3": {}}
            for name in [
                "QUR_DE", "QUR_NLPE", "QUR_QP", "QUR_PL", "QUR_DED",
                "UR_DE", "UR_NLPE", "UR_PAGE_DE", "UR_PAGE_QP", "UR_PAGE_DED",
            ]
        }
        # Flat metrics (no categorical breakdown, only total + complexity slices)
        accum["QUR"] = {"total": {}}
        accum["UR"] = {"total": {}}
        accum["UR_PAGE_inpage"] = {"total": {}}
        accum["UR_PAGE_outpage"] = {"total": {}}
        list_len = []  # page-count axis for QUR_PL; set by the last model processed

        processed_models = []
        for result_file in (folder / "augmented").iterdir():
            print("-" * 100)
            try:
                result = _process_model_file(result_file, entity_verifier, dataset, images_path)
                if result is None:
                    continue
                model_name, metrics, list_len = result
                processed_models.append(model_name)
                _collect_metrics_into_accum(accum, model_name, metrics)
            except Exception as e:
                print(f"Error processing {result_file}: {e}")

        print(f"Saving files")
        print(f"Processed models: {processed_models}")

        a = accum
        save_metric(folder_results, "QUR",           a["QUR"]["total"],           ["QUR", "QUR_C1", "QUR_C2", "QUR_C3", "QUR_weighted"])
        save_metric(folder_results, "QUR_DE",         a["QUR_DE"]["total"],         LAYOUT_TYPES,       [a["QUR_DE"]["c1"],       a["QUR_DE"]["c2"],       a["QUR_DE"]["c3"]])
        save_metric(folder_results, "QUR_NLPE",       a["QUR_NLPE"]["total"],       MACRO_ENTITY_TYPES, [a["QUR_NLPE"]["c1"],     a["QUR_NLPE"]["c2"],     a["QUR_NLPE"]["c3"]])
        save_metric(folder_results, "QUR_QP",         a["QUR_QP"]["total"],         PAGE_LAYOUT,        [a["QUR_QP"]["c1"],       a["QUR_QP"]["c2"],       a["QUR_QP"]["c3"]])
        save_metric(folder_results, "QUR_PL",         a["QUR_PL"]["total"],         list_len,           [a["QUR_PL"]["c1"],       a["QUR_PL"]["c2"],       a["QUR_PL"]["c3"]])
        save_metric(folder_results, "QUR_DED",        a["QUR_DED"]["total"],        ["<15", "15-25", ">25"], [a["QUR_DED"]["c1"], a["QUR_DED"]["c2"],      a["QUR_DED"]["c3"]])
        save_metric(folder_results, "UR",             a["UR"]["total"],             ["UR", "UR_C1", "UR_C2", "UR_C3"])
        save_metric(folder_results, "UR_DE",          a["UR_DE"]["total"],          LAYOUT_TYPES,       [a["UR_DE"]["c1"],        a["UR_DE"]["c2"],        a["UR_DE"]["c3"]])
        save_metric(folder_results, "UR_PAGE_inpage", a["UR_PAGE_inpage"]["total"], ["UR_inpage",  "UR_inpage_C1",  "UR_inpage_C2",  "UR_inpage_C3"])
        save_metric(folder_results, "UR_PAGE_outpage",a["UR_PAGE_outpage"]["total"],["UR_outpage", "UR_outpage_C1", "UR_outpage_C2", "UR_outpage_C3"])
        save_metric(folder_results, "UR_PAGE_DE",     a["UR_PAGE_DE"]["total"],     LAYOUT_TYPES,       [a["UR_PAGE_DE"]["c1"],   a["UR_PAGE_DE"]["c2"],   a["UR_PAGE_DE"]["c3"]])
        save_metric(folder_results, "UR_NLPE",        a["UR_NLPE"]["total"],        MACRO_ENTITY_TYPES, [a["UR_NLPE"]["c1"],      a["UR_NLPE"]["c2"],      a["UR_NLPE"]["c3"]])
        save_metric(folder_results, "UR_PAGE_QP",     a["UR_PAGE_QP"]["total"],     PAGE_LAYOUT,        [a["UR_PAGE_QP"]["c1"],   a["UR_PAGE_QP"]["c2"],   a["UR_PAGE_QP"]["c3"]])
        save_metric(folder_results, "UR_PAGE_DED",    a["UR_PAGE_DED"]["total"],    ["0", "1", ">1"],   [a["UR_PAGE_DED"]["c1"],  a["UR_PAGE_DED"]["c2"],  a["UR_PAGE_DED"]["c3"]])

        print(f"Files saved in {folder_results}")
        print("-" * 100)

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate VQA analysis report")
    parser.add_argument("--dataset", type=str, default="MPDocVQA", help="Dataset name (must match the folder under models/results/)")
    parser.add_argument("--images_path", type=str, default=None, help="Path to the images directory (optional fallback)")
    args = parser.parse_args()
    generate_analysis_report(dataset=args.dataset, images_path=args.images_path)
