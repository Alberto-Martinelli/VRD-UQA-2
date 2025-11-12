

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
        print("QUR")
        correct_unable = 0
        correct_unable_complexity_1 = 0
        correct_unable_complexity_2 = 0
        correct_unable_complexity_3 = 0
        total_corrupted = 0
        total_corrupted_complexity_1 = 0
        total_corrupted_complexity_2 = 0
        total_corrupted_complexity_3 = 0

        for res in self.results:
            if (
                res["is_corrupted"]
                and "verification_result" in res
                and "vqa_results" in res["verification_result"]
                and len(res["verification_result"]["vqa_results"]) > 0
            ):
                
                # Get all answers for this question
                vqa_result = res["verification_result"]["vqa_results"][0]
                all_answers = vqa_result.get("answers", vqa_result.get("answer", []))
                # print(len(all_answers))

                complexity = res["complexity"]

                total_corrupted += 1
                if complexity == 1:
                    total_corrupted_complexity_1 += 1
                if complexity == 2:
                    total_corrupted_complexity_2 += 1
                if complexity == 3: 
                    total_corrupted_complexity_3 += 1

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

                # If majority of answers are "unable to determine", count this as correct
                if tot_ans > 0 and unable_count / tot_ans == 1:
                    correct_unable += 1

                if (tot_ans_complexity_1 > 0 and unable_count_complexity_1 / tot_ans_complexity_1 == 1):
                    correct_unable_complexity_1 += 1

                if (tot_ans_complexity_2 > 0 and unable_count_complexity_2 / tot_ans_complexity_2 == 1):
                    correct_unable_complexity_2 += 1

                if (tot_ans_complexity_3 > 0 and unable_count_complexity_3 / tot_ans_complexity_3 == 1):
                    correct_unable_complexity_3 += 1

        if self.debug:
            print(f"Total corrupted questions: {total_corrupted}")
            print(total_corrupted, total_corrupted_complexity_1, total_corrupted_complexity_2, total_corrupted_complexity_3)
            print(f"Correct unable to determine: {correct_unable} ({(correct_unable/total_corrupted)*100:.2f}%)")
            print(f"Correct unable to determine (Complexity 1): {correct_unable_complexity_1} ({(correct_unable_complexity_1/total_corrupted)*100:.2f}%)")
            print(f"Correct unable to determine (Complexity 2): {correct_unable_complexity_2} ({(correct_unable_complexity_2/total_corrupted)*100:.2f}%)")
            print(f"Correct unable to determine (Complexity 3): {correct_unable_complexity_3} ({(correct_unable_complexity_3/total_corrupted)*100:.2f}%)")
        
        weighted_unable = 0 #(total_corrupted_complexity_1 * 1.0) / total_corrupted
        return [
            correct_unable / total_corrupted,
            correct_unable_complexity_1 / total_corrupted_complexity_1,
            correct_unable_complexity_2 / total_corrupted_complexity_2,
            correct_unable_complexity_3 / total_corrupted_complexity_3,
            weighted_unable
        ]


    def QUR_DE(self):
        print("QUR_DE")
        correct_unable = 0
        correct_unable_complexity_1 = 0
        correct_unable_complexity_2 = 0
        correct_unable_complexity_3 = 0
        total_corrupted = 0
        total_corrupted_complexity_1 = 0
        total_corrupted_complexity_2 = 0
        total_corrupted_complexity_3 = 0

        layout_dict={el:0 for el in LAYOUT_TYPES}
        layout_dict_complexity_1={el:0 for el in LAYOUT_TYPES}
        layout_dict_complexity_2={el:0 for el in LAYOUT_TYPES}
        layout_dict_complexity_3={el:0 for el in LAYOUT_TYPES}

        counter_layout = {el:0 for el in LAYOUT_TYPES}
        counter_layout_complexity_1 = {el:0 for el in LAYOUT_TYPES}
        counter_layout_complexity_2 = {el:0 for el in LAYOUT_TYPES}
        counter_layout_complexity_3 = {el:0 for el in LAYOUT_TYPES}

        for res in self.results:
            if (
                res["is_corrupted"]
                and "verification_result" in res
                and "vqa_results" in res["verification_result"]
                and len(res["verification_result"]["vqa_results"]) > 0
            ):
                
                # Get all answers for this question
                vqa_result = res["verification_result"]["vqa_results"][0]
                all_answers = vqa_result.get("answers", vqa_result.get("answer", []))
                # print(len(all_answers))

                complexity = res["complexity"]

                total_corrupted += 1
                if complexity == 1:
                    total_corrupted_complexity_1 += 1
                if complexity == 2:
                    total_corrupted_complexity_2 += 1
                if complexity == 3: 
                    total_corrupted_complexity_3 += 1

                corrupted_entities = res["corrupted_entities"]
                unique_corrupted_entities = []

                seen=[]
                for entity in corrupted_entities:
                    if entity["text"] not in seen:
                        seen.append(entity["text"])
                        unique_corrupted_entities.append(entity)
                
                for el in unique_corrupted_entities:
                    counter_layout[el["objectType"]]+=1
                    if complexity == 1:
                        counter_layout_complexity_1[el["objectType"]]+=1
                    if complexity == 2:
                        counter_layout_complexity_2[el["objectType"]]+=1
                    if complexity == 3:
                        counter_layout_complexity_3[el["objectType"]]+=1


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

                # If majority of answers are "unable to determine", count this as correct
                if tot_ans > 0 and unable_count / tot_ans == 1:
                    correct_unable += 1
                    for el in unique_corrupted_entities:
                        layout_dict[el["objectType"]]+=1

                if (tot_ans_complexity_1 > 0 and unable_count_complexity_1 / tot_ans_complexity_1 == 1):
                    correct_unable_complexity_1 += 1
                    for el in unique_corrupted_entities:
                        layout_dict_complexity_1[el["objectType"]]+=1

                if (tot_ans_complexity_2 > 0 and unable_count_complexity_2 / tot_ans_complexity_2 == 1):
                    correct_unable_complexity_2 += 1
                    for el in unique_corrupted_entities:
                        layout_dict_complexity_2[el["objectType"]]+=1

                if (tot_ans_complexity_3 > 0 and unable_count_complexity_3 / tot_ans_complexity_3 == 1):
                    correct_unable_complexity_3 += 1
                    for el in unique_corrupted_entities:
                        layout_dict_complexity_3[el["objectType"]]+=1


        # normalize the layout_dict wrt total_corrupted
        res_layout = {}
        res_layout_complexity_1 = {}
        res_layout_complexity_2 = {}
        res_layout_complexity_3 = {}
        for el in layout_dict:
            res_layout[el] = layout_dict[el] / counter_layout[el] if counter_layout[el] != 0 else 0
            res_layout_complexity_1[el] = layout_dict_complexity_1[el] / counter_layout_complexity_1[el] if counter_layout_complexity_1[el] != 0 else 0
            res_layout_complexity_2[el] = layout_dict_complexity_2[el] / counter_layout_complexity_2[el] if counter_layout_complexity_2[el] != 0 else 0
            res_layout_complexity_3[el] = layout_dict_complexity_3[el] / counter_layout_complexity_3[el] if counter_layout_complexity_3[el] != 0 else 0

        if self.debug:
            print(f"Total corrupted questions: {total_corrupted, total_corrupted_complexity_1, total_corrupted_complexity_2, total_corrupted_complexity_3}")
            print(F"TOT\tC1\tC2\tC3")
            for k in LAYOUT_TYPES:
                print(f"{layout_dict[k]:.2f}\t{layout_dict_complexity_1[k]:.2f}\t{layout_dict_complexity_2[k]:.2f}\t{layout_dict_complexity_3[k]:.2f}")
        

        return [
            res_layout,
            res_layout_complexity_1,    
            res_layout_complexity_2,
            res_layout_complexity_3,
        ]

    
    def QUR_QP(self):
        print("QUR_QP")
        correct_unable = 0
        correct_unable_complexity_1 = 0
        correct_unable_complexity_2 = 0
        correct_unable_complexity_3 = 0
        total_corrupted = 0
        total_corrupted_complexity_1 = 0
        total_corrupted_complexity_2 = 0
        total_corrupted_complexity_3 = 0

        pos_dict={el:0 for el in PAGE_LAYOUT}
        pos_dict_complexity_1={el:0 for el in PAGE_LAYOUT}
        pos_dict_complexity_2={el:0 for el in PAGE_LAYOUT}
        pos_dict_complexity_3={el:0 for el in PAGE_LAYOUT}

        counter_pos = {el:0 for el in PAGE_LAYOUT}
        counter_pos_complexity_1 = {el:0 for el in PAGE_LAYOUT}
        counter_pos_complexity_2 = {el:0 for el in PAGE_LAYOUT}
        counter_pos_complexity_3 = {el:0 for el in PAGE_LAYOUT}

        for res in self.results:
            if (
                res["is_corrupted"]
                and "verification_result" in res
                and "vqa_results" in res["verification_result"]
                and len(res["verification_result"]["vqa_results"]) > 0
            ):
                
                # Get all answers for this question
                vqa_result = res["verification_result"]["vqa_results"][0]
                all_answers = vqa_result.get("answers", vqa_result.get("answer", []))
                pages=[]
                for ans in all_answers:
                    pages.extend(ans.get("pages", []))
                unique_pages = list(set(pages))
                average_x_coord = 0
                average_y_coord = 0
                for page in unique_pages:
                    #/data2/dnapolitano/VQA/data/DUDE_train-val-test_binaries/images
                    if "data1" in page:
                        page=page.replace("data1","data2")
                    x_size, y_size = Image.open(page).size
                    average_x_coord += x_size
                    average_y_coord += y_size
                average_x_coord /= len(unique_pages)
                average_y_coord /= len(unique_pages)

                complexity = res["complexity"]

                total_corrupted += 1
                if complexity == 1:
                    total_corrupted_complexity_1 += 1
                if complexity == 2:
                    total_corrupted_complexity_2 += 1
                if complexity == 3: 
                    total_corrupted_complexity_3 += 1

                corrupted_entities = res["corrupted_entities"]
                # print("Corrupted Entities",corrupted_entities)
                # unique_corrupted_entities = []
                
                for el in corrupted_entities:
                    corrupted_entity_bbox = el.get("bbox", [])
                    bbox_center = [
                        (corrupted_entity_bbox[0] + corrupted_entity_bbox[2]) / 2,
                        (corrupted_entity_bbox[1] + corrupted_entity_bbox[3]) / 2
                    ]
                    if bbox_center[0] < average_x_coord / 2 and bbox_center[1] < average_y_coord / 2:
                        counter_pos["TOP_LEFT"]+=1
                        if complexity == 1:
                            counter_pos_complexity_1["TOP_LEFT"]+=1
                        if complexity == 2:
                            counter_pos_complexity_2["TOP_LEFT"]+=1
                        if complexity == 3:
                            counter_pos_complexity_3["TOP_LEFT"]+=1
                    elif bbox_center[0] < average_x_coord / 2 and bbox_center[1] >= average_y_coord / 2:
                        counter_pos["BOTTOM_LEFT"]+=1
                        if complexity == 1:
                            counter_pos_complexity_1["BOTTOM_LEFT"]+=1
                        if complexity == 2:
                            counter_pos_complexity_2["BOTTOM_LEFT"]+=1
                        if complexity == 3:
                            counter_pos_complexity_3["BOTTOM_LEFT"]+=1
                    elif bbox_center[0] >= average_x_coord / 2 and bbox_center[1] < average_y_coord / 2:
                        counter_pos["TOP_RIGHT"]+=1
                        if complexity == 1:
                            counter_pos_complexity_1["TOP_RIGHT"]+=1
                        if complexity == 2:
                            counter_pos_complexity_2["TOP_RIGHT"]+=1
                        if complexity == 3:
                            counter_pos_complexity_3["TOP_RIGHT"]+=1
                    elif bbox_center[0] >= average_x_coord / 2 and bbox_center[1] >= average_y_coord / 2:
                        counter_pos["BOTTOM_RIGHT"]+=1
                        if complexity == 1:
                            counter_pos_complexity_1["BOTTOM_RIGHT"]+=1
                        if complexity == 2:
                            counter_pos_complexity_2["BOTTOM_RIGHT"]+=1
                        if complexity == 3:
                            counter_pos_complexity_3["BOTTOM_RIGHT"]+=1


                                   

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

                # If majority of answers are "unable to determine", count this as correct
                if tot_ans > 0 and unable_count / tot_ans == 1:
                    correct_unable += 1
                    for el in corrupted_entities:
                        corrupted_entity_bbox = el.get("bbox", [])
                        bbox_center = [
                            (corrupted_entity_bbox[0] + corrupted_entity_bbox[2]) / 2,
                            (corrupted_entity_bbox[1] + corrupted_entity_bbox[3]) / 2
                        ]
                        if bbox_center[0] < average_x_coord / 2 and bbox_center[1] < average_y_coord / 2:
                            pos_dict["TOP_LEFT"]+=1
                        elif bbox_center[0] < average_x_coord / 2 and bbox_center[1] >= average_y_coord / 2:
                            pos_dict["BOTTOM_LEFT"]+=1
                        elif bbox_center[0] >= average_x_coord / 2 and bbox_center[1] < average_y_coord / 2:
                            pos_dict["TOP_RIGHT"]+=1
                        elif bbox_center[0] >= average_x_coord / 2 and bbox_center[1] >= average_y_coord / 2:
                            pos_dict["BOTTOM_RIGHT"]+=1

                if (tot_ans_complexity_1 > 0 and unable_count_complexity_1 / tot_ans_complexity_1 == 1):
                    correct_unable_complexity_1 += 1
                    for el in corrupted_entities:
                        corrupted_entity_bbox = el.get("bbox", [])
                        bbox_center = [
                            (corrupted_entity_bbox[0] + corrupted_entity_bbox[2]) / 2,
                            (corrupted_entity_bbox[1] + corrupted_entity_bbox[3]) / 2
                        ]
                        if bbox_center[0] < average_x_coord / 2 and bbox_center[1] < average_y_coord / 2:
                            pos_dict_complexity_1["TOP_LEFT"]+=1
                        elif bbox_center[0] < average_x_coord / 2 and bbox_center[1] >= average_y_coord / 2:
                            pos_dict_complexity_1["BOTTOM_LEFT"]+=1
                        elif bbox_center[0] >= average_x_coord / 2 and bbox_center[1] < average_y_coord / 2:
                            pos_dict_complexity_1["TOP_RIGHT"]+=1
                        elif bbox_center[0] >= average_x_coord / 2 and bbox_center[1] >= average_y_coord / 2:
                            pos_dict_complexity_1["BOTTOM_RIGHT"]+=1

                if (tot_ans_complexity_2 > 0 and unable_count_complexity_2 / tot_ans_complexity_2 == 1):
                    correct_unable_complexity_2 += 1
                    for el in corrupted_entities:
                        corrupted_entity_bbox = el.get("bbox", [])
                        bbox_center = [
                            (corrupted_entity_bbox[0] + corrupted_entity_bbox[2]) / 2,
                            (corrupted_entity_bbox[1] + corrupted_entity_bbox[3]) / 2
                        ]
                        if bbox_center[0] < average_x_coord / 2 and bbox_center[1] < average_y_coord / 2:
                            pos_dict_complexity_2["TOP_LEFT"]+=1
                        elif bbox_center[0] < average_x_coord / 2 and bbox_center[1] >= average_y_coord / 2:
                            pos_dict_complexity_2["BOTTOM_LEFT"]+=1
                        elif bbox_center[0] >= average_x_coord / 2 and bbox_center[1] < average_y_coord / 2:
                            pos_dict_complexity_2["TOP_RIGHT"]+=1
                        elif bbox_center[0] >= average_x_coord / 2 and bbox_center[1] >= average_y_coord / 2:
                            pos_dict_complexity_2["BOTTOM_RIGHT"]+=1

                if (tot_ans_complexity_3 > 0 and unable_count_complexity_3 / tot_ans_complexity_3 == 1):
                    correct_unable_complexity_3 += 1
                    for el in corrupted_entities:
                        corrupted_entity_bbox = el.get("bbox", [])
                        bbox_center = [
                            (corrupted_entity_bbox[0] + corrupted_entity_bbox[2]) / 2,
                            (corrupted_entity_bbox[1] + corrupted_entity_bbox[3]) / 2
                        ]
                        if bbox_center[0] < average_x_coord / 2 and bbox_center[1] < average_y_coord / 2:
                            pos_dict_complexity_3["TOP_LEFT"]+=1
                        elif bbox_center[0] < average_x_coord / 2 and bbox_center[1] >= average_y_coord / 2:
                            pos_dict_complexity_3["BOTTOM_LEFT"]+=1
                        elif bbox_center[0] >= average_x_coord / 2 and bbox_center[1] < average_y_coord / 2:
                            pos_dict_complexity_3["TOP_RIGHT"]+=1
                        elif bbox_center[0] >= average_x_coord / 2 and bbox_center[1] >= average_y_coord / 2:
                            pos_dict_complexity_3["BOTTOM_RIGHT"]+=1


        res_pos = {}
        res_pos_complexity_1 = {}
        res_pos_complexity_2 = {}
        res_pos_complexity_3 = {}
        # print("Entity dict",entity_dict)
        # print("Entity counter",counter_entity)
        for el in pos_dict:
            # print("Final",el)
            res_pos[el] = pos_dict[el] / counter_pos[el] if counter_pos[el] != 0 else 0
            res_pos_complexity_1[el] = pos_dict_complexity_1[el] / counter_pos_complexity_1[el] if counter_pos_complexity_1[el] != 0 else 0
            res_pos_complexity_2[el] = pos_dict_complexity_2[el] / counter_pos_complexity_2[el] if counter_pos_complexity_2[el] != 0 else 0
            res_pos_complexity_3[el] = pos_dict_complexity_3[el] / counter_pos_complexity_3[el] if counter_pos_complexity_3[el] != 0 else 0

        if self.debug:
        # if True:
            print(f"Total corrupted questions: {total_corrupted, total_corrupted_complexity_1, total_corrupted_complexity_2, total_corrupted_complexity_3}")
            print(F"TOT\tC1\tC2\tC3")
            for k in MACRO_ENTITY_TYPES:
                print(f"{res_pos[k]:.2f}\t{res_pos_complexity_1[k]:.2f}\t{res_pos_complexity_2[k]:.2f}\t{res_pos_complexity_3[k]:.2f}")
        

        return [
            res_pos,
            res_pos_complexity_1,    
            res_pos_complexity_2,
            res_pos_complexity_3,
        ]
    

    def QUR_NLPE(self):
        print("QUR_NLPE")
        correct_unable = 0
        correct_unable_complexity_1 = 0
        correct_unable_complexity_2 = 0
        correct_unable_complexity_3 = 0
        total_corrupted = 0
        total_corrupted_complexity_1 = 0
        total_corrupted_complexity_2 = 0
        total_corrupted_complexity_3 = 0

        entity_dict={el:0 for el in MACRO_ENTITY_TYPES}
        entity_dict_complexity_1={el:0 for el in MACRO_ENTITY_TYPES}
        entity_dict_complexity_2={el:0 for el in MACRO_ENTITY_TYPES}
        entity_dict_complexity_3={el:0 for el in MACRO_ENTITY_TYPES}

        counter_entity = {el:0 for el in MACRO_ENTITY_TYPES}
        counter_entity_complexity_1 = {el:0 for el in MACRO_ENTITY_TYPES}
        counter_entity_complexity_2 = {el:0 for el in MACRO_ENTITY_TYPES}
        counter_entity_complexity_3 = {el:0 for el in MACRO_ENTITY_TYPES}

        for res in self.results:
            if (
                res["is_corrupted"]
                and "verification_result" in res
                and "vqa_results" in res["verification_result"]
                and len(res["verification_result"]["vqa_results"]) > 0
            ):
                
                # Get all answers for this question
                vqa_result = res["verification_result"]["vqa_results"][0]
                all_answers = vqa_result.get("answers", vqa_result.get("answer", []))
                # print(len(all_answers))

                complexity = res["complexity"]

                total_corrupted += 1
                if complexity == 1:
                    total_corrupted_complexity_1 += 1
                if complexity == 2:
                    total_corrupted_complexity_2 += 1
                if complexity == 3: 
                    total_corrupted_complexity_3 += 1

                corrupted_entities = res["entity_type"]
                # print("Corrupted Entities",corrupted_entities)
                # unique_corrupted_entities = []
                
                for el in corrupted_entities:
                    # print(el,MACRO_ENTITY_MAPPER[el])
                    counter_entity[MACRO_ENTITY_MAPPER[el]]+=1
                    if complexity == 1:
                        counter_entity_complexity_1[MACRO_ENTITY_MAPPER[el]]+=1
                    if complexity == 2:
                        counter_entity_complexity_2[MACRO_ENTITY_MAPPER[el]]+=1
                    if complexity == 3:
                        counter_entity_complexity_3[MACRO_ENTITY_MAPPER[el]]+=1
                                   

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

                # If majority of answers are "unable to determine", count this as correct
                if tot_ans > 0 and unable_count / tot_ans == 1:
                    correct_unable += 1
                    for el in corrupted_entities:
                        entity_dict[MACRO_ENTITY_MAPPER[el]]+=1

                if (tot_ans_complexity_1 > 0 and unable_count_complexity_1 / tot_ans_complexity_1 == 1):
                    correct_unable_complexity_1 += 1
                    for el in corrupted_entities:
                        entity_dict_complexity_1[MACRO_ENTITY_MAPPER[el]]+=1

                if (tot_ans_complexity_2 > 0 and unable_count_complexity_2 / tot_ans_complexity_2 == 1):
                    correct_unable_complexity_2 += 1
                    for el in corrupted_entities:
                        entity_dict_complexity_2[MACRO_ENTITY_MAPPER[el]]+=1

                if (tot_ans_complexity_3 > 0 and unable_count_complexity_3 / tot_ans_complexity_3 == 1):
                    correct_unable_complexity_3 += 1
                    for el in corrupted_entities:
                        entity_dict_complexity_3[MACRO_ENTITY_MAPPER[el]]+=1


        res_entity = {}
        res_entity_complexity_1 = {}
        res_entity_complexity_2 = {}
        res_entity_complexity_3 = {}
        # print("Entity dict",entity_dict)
        # print("Entity counter",counter_entity)
        for el in entity_dict:
            # print("Final",el)
            res_entity[el] = entity_dict[el] / counter_entity[el] if counter_entity[el] != 0 else 0
            res_entity_complexity_1[el] = entity_dict_complexity_1[el] / counter_entity_complexity_1[el] if counter_entity_complexity_1[el] != 0 else 0
            res_entity_complexity_2[el] = entity_dict_complexity_2[el] / counter_entity_complexity_2[el] if counter_entity_complexity_2[el] != 0 else 0
            res_entity_complexity_3[el] = entity_dict_complexity_3[el] / counter_entity_complexity_3[el] if counter_entity_complexity_3[el] != 0 else 0

        if self.debug:
        # if True:
            print(f"Total corrupted questions: {total_corrupted, total_corrupted_complexity_1, total_corrupted_complexity_2, total_corrupted_complexity_3}")
            print(F"TOT\tC1\tC2\tC3")
            for k in MACRO_ENTITY_TYPES:
                print(f"{res_entity[k]:.2f}\t{res_entity_complexity_1[k]:.2f}\t{res_entity_complexity_2[k]:.2f}\t{res_entity_complexity_3[k]:.2f}")
        

        return [
            res_entity,
            res_entity_complexity_1,    
            res_entity_complexity_2,
            res_entity_complexity_3,
        ]    


    def QUR_PL(self):
        print("QUR_PL")
        correct_unable = 0
        correct_unable_complexity_1 = 0
        correct_unable_complexity_2 = 0
        correct_unable_complexity_3 = 0
        total_corrupted = 0
        total_corrupted_complexity_1 = 0
        total_corrupted_complexity_2 = 0
        total_corrupted_complexity_3 = 0

        len_dict={}
        len_dict_complexity_1={}
        len_dict_complexity_2={}
        len_dict_complexity_3={}

        counter_len = {}
        counter_len_complexity_1 = {}
        counter_len_complexity_2 = {}
        counter_len_complexity_3 = {}

        for res in self.results:
            if (
                res["is_corrupted"]
                and "verification_result" in res
                and "vqa_results" in res["verification_result"]
                and len(res["verification_result"]["vqa_results"]) > 0
            ):
                
                # Get all answers for this question
                vqa_result = res["verification_result"]["vqa_results"][0]
                all_answers = vqa_result.get("answers", vqa_result.get("answer", []))
                pages=[]
                for ans in all_answers:
                    pages.extend(ans.get("pages", []))
                unique_pages = list(set(pages))
                num_pages = len(unique_pages)

                complexity = res["complexity"]
                
                if num_pages not in counter_len:
                    counter_len[num_pages] = 0
                    counter_len_complexity_1[num_pages] = 0
                    counter_len_complexity_2[num_pages] = 0
                    counter_len_complexity_3[num_pages] = 0
                counter_len[num_pages] += 1
                if complexity == 1:
                    counter_len_complexity_1[num_pages] += 1
                if complexity == 2:
                    counter_len_complexity_2[num_pages] += 1
                if complexity == 3:
                    counter_len_complexity_3[num_pages] += 1



                total_corrupted += 1
                if complexity == 1:
                    total_corrupted_complexity_1 += 1
                if complexity == 2:
                    total_corrupted_complexity_2 += 1
                if complexity == 3: 
                    total_corrupted_complexity_3 += 1
                                   

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

                # If majority of answers are "unable to determine", count this as correct
                if tot_ans > 0 and unable_count / tot_ans == 1:
                    correct_unable += 1
                    len_dict[num_pages] = len_dict.get(num_pages, 0) + 1

                if (tot_ans_complexity_1 > 0 and unable_count_complexity_1 / tot_ans_complexity_1 == 1):
                    correct_unable_complexity_1 += 1
                    len_dict_complexity_1[num_pages] = len_dict_complexity_1.get(num_pages, 0) + 1

                if (tot_ans_complexity_2 > 0 and unable_count_complexity_2 / tot_ans_complexity_2 == 1):
                    correct_unable_complexity_2 += 1
                    len_dict_complexity_2[num_pages] = len_dict_complexity_2.get(num_pages, 0) + 1
                    
                if (tot_ans_complexity_3 > 0 and unable_count_complexity_3 / tot_ans_complexity_3 == 1):
                    correct_unable_complexity_3 += 1
                    len_dict_complexity_3[num_pages] = len_dict_complexity_3.get(num_pages, 0) + 1
                        

        res_len = {}
        res_len_complexity_1 = {}
        res_len_complexity_2 = {}
        res_len_complexity_3 = {}
        # print("Len dict",len_dict)
        # print("Len counter",counter_len)
        list_len = []
        for el in counter_len:
            list_len.append(el)
            if el not in len_dict:
                res_len[el] = 0
            else:
                res_len[el] = len_dict[el] / counter_len[el] 
            if el not in len_dict_complexity_1:
                res_len_complexity_1[el] = 0
            else:
                res_len_complexity_1[el] = len_dict_complexity_1[el] / counter_len_complexity_1[el] if counter_len_complexity_1[el] != 0 else 0
            if el not in len_dict_complexity_2:
                res_len_complexity_2[el] = 0
            else:
                res_len_complexity_2[el] = len_dict_complexity_2[el] / counter_len_complexity_2[el] if counter_len_complexity_2[el] != 0 else 0
            if el not in len_dict_complexity_3:
                res_len_complexity_3[el] = 0
            else:
                res_len_complexity_3[el] = len_dict_complexity_3[el] / counter_len_complexity_3[el] if counter_len_complexity_3[el] != 0 else 0
        # print("Res dict",res_len)

        if self.debug:
        # if True:
            print(f"Total corrupted questions: {total_corrupted, total_corrupted_complexity_1, total_corrupted_complexity_2, total_corrupted_complexity_3}")
            print(F"TOT\tC1\tC2\tC3")
            for k in MACRO_ENTITY_TYPES:
                print(f"{res_len[k]:.2f}\t{res_len_complexity_1[k]:.2f}\t{res_len_complexity_2[k]:.2f}\t{res_len_complexity_3[k]:.2f}")
        

        return [
            res_len,
            res_len_complexity_1,    
            res_len_complexity_2,
            res_len_complexity_3,
            list_len
        ]


    def QUR_DED(self):
        print("QUR_DED")
        correct_unable = 0
        correct_unable_complexity_1 = 0
        correct_unable_complexity_2 = 0
        correct_unable_complexity_3 = 0
        total_corrupted = 0
        total_corrupted_complexity_1 = 0
        total_corrupted_complexity_2 = 0
        total_corrupted_complexity_3 = 0

        layout_dict={"<15":0, "15-25":0, ">25":0}
        # print(layout_dict)
        layout_dict_complexity_1={"<15":0, "15-25":0, ">25":0}
        layout_dict_complexity_2={"<15":0, "15-25":0, ">25":0}
        layout_dict_complexity_3={"<15":0, "15-25":0, ">25":0}

        counter_layout = {"<15":0, "15-25":0, ">25":0}
        counter_layout_complexity_1 = {"<15":0, "15-25":0, ">25":0}
        counter_layout_complexity_2 = {"<15":0, "15-25":0, ">25":0}
        counter_layout_complexity_3 = {"<15":0, "15-25":0, ">25":0}

        for res in self.results:
            if (
                res["is_corrupted"]
                and "verification_result" in res
                and "vqa_results" in res["verification_result"]
                and len(res["verification_result"]["vqa_results"]) > 0
            ):
                complexity = res["complexity"]
                # Get all answers for this question
                vqa_result = res["verification_result"]["vqa_results"][0]
                all_answers = vqa_result.get("answers", vqa_result.get("answer", []))
                
                layout_doc = res["layout_analysis"]["pages"]

                doc_dist = {el:0 for el in MACRO_LAYOUT_TYPES}
                for page, info in layout_doc.items():
                    layout_page=info["layout_analysis"]
                    for objID, obj in layout_page.items():
                        doc_dist[MAPPER_LAYOUT_TYPES[obj["ObjectType"]]]+=1

                v1,v2=doc_dist.values()
                if v1==0:
                    v1=1
                if v2/(v2+v1) < 0.15:
                    key="<15"
                    counter_layout["<15"]+=1
                    if complexity == 1:
                        counter_layout_complexity_1["<15"]+=1
                    if complexity == 2:
                        counter_layout_complexity_2["<15"]+=1
                    if complexity == 3:
                        counter_layout_complexity_3["<15"]+=1
                elif v2/(v1+v2) < 0.25:
                    key="15-25"
                    counter_layout["15-25"]+=1
                    if complexity == 1:
                        counter_layout_complexity_1["15-25"]+=1
                    if complexity == 2:
                        counter_layout_complexity_2["15-25"]+=1
                    if complexity == 3:
                        counter_layout_complexity_3["15-25"]+=1
                else:
                    key=">25"
                    counter_layout[">25"]+=1
                    if complexity == 1:
                        counter_layout_complexity_1[">25"]+=1
                    if complexity == 2:
                        counter_layout_complexity_2[">25"]+=1
                    if complexity == 3:
                        counter_layout_complexity_3[">25"]+=1

                total_corrupted += 1
                if complexity == 1:
                    total_corrupted_complexity_1 += 1
                if complexity == 2:
                    total_corrupted_complexity_2 += 1
                if complexity == 3: 
                    total_corrupted_complexity_3 += 1

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

                # If majority of answers are "unable to determine", count this as correct
                if tot_ans > 0 and unable_count / tot_ans == 1:
                    correct_unable += 1
                    layout_dict[key] = layout_dict.get(key, 0) + 1

                if (tot_ans_complexity_1 > 0 and unable_count_complexity_1 / tot_ans_complexity_1 == 1):
                    correct_unable_complexity_1 += 1
                    layout_dict_complexity_1[key] = layout_dict_complexity_1.get(key, 0) + 1

                if (tot_ans_complexity_2 > 0 and unable_count_complexity_2 / tot_ans_complexity_2 == 1):
                    correct_unable_complexity_2 += 1
                    layout_dict_complexity_2[key] = layout_dict_complexity_2.get(key, 0) + 1
                    
                if (tot_ans_complexity_3 > 0 and unable_count_complexity_3 / tot_ans_complexity_3 == 1):
                    correct_unable_complexity_3 += 1
                    layout_dict_complexity_3[key] = layout_dict_complexity_3.get(key, 0) + 1
                        

        res_lay = {}
        res_lay_complexity_1 = {}
        res_lay_complexity_2 = {}
        res_lay_complexity_3 = {}
        # print(len(counter_layout))
        # print("COUNTER LAYOUT")
        # for k,v in counter_layout.items():
        #     print(k,v)
        # list_len = []
        for el in counter_layout:
            if el not in layout_dict:
                res_lay[el] = 0
            else:
                res_lay[el] = layout_dict[el] / counter_layout[el] if counter_layout[el] != 0 else 0
            
            if el not in layout_dict_complexity_1:
                res_lay_complexity_1[el] = 0
            else:
                res_lay_complexity_1[el] = layout_dict_complexity_1[el] / counter_layout_complexity_1[el] if counter_layout_complexity_1[el] != 0 else 0
            
            if el not in layout_dict_complexity_2:
                res_lay_complexity_2[el] = 0
            else:
                res_lay_complexity_2[el] = layout_dict_complexity_2[el] / counter_layout_complexity_2[el] if counter_layout_complexity_2[el] != 0 else 0
            
            if el not in layout_dict_complexity_3:
                res_lay_complexity_3[el] = 0
            else:
                res_lay_complexity_3[el] = layout_dict_complexity_3[el] / counter_layout_complexity_3[el] if counter_layout_complexity_3[el] != 0 else 0
        # print("Res dict",res_lay)

        if self.debug:
        # if True:
            print(f"Total corrupted questions: {total_corrupted, total_corrupted_complexity_1, total_corrupted_complexity_2, total_corrupted_complexity_3}")
            print(F"TOT\tC1\tC2\tC3")
            for k in counter_layout:
                print(f"{res_lay[k]:.2f}\t{res_lay_complexity_1[k]:.2f}\t{res_lay_complexity_2[k]:.2f}\t{res_lay_complexity_3[k]:.2f}")
        

        return [
            res_lay,
            res_lay_complexity_1,    
            res_lay_complexity_2,
            res_lay_complexity_3,
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
            print(f"Unable to determine/errors: {tot_unable_count} ({(tot_unable_count/total_answers)*100:.2f}%)")
            print(f"Total answers (Complexity 1): {total_answers_complexity_1}")
            print(f"Unable to determine/errors (Complexity 1): {tot_unable_count_complexity_1} ({(tot_unable_count_complexity_1/total_answers_complexity_1)*100:.2f}%)")
            print(f"Total answers (Complexity 2): {total_answers_complexity_2}")
            print(f"Unable to determine/errors (Complexity 2): {tot_unable_count_complexity_2} ({(tot_unable_count_complexity_2/total_answers_complexity_2)*100:.2f}%)")
            print(f"Total answers (Complexity 3): {total_answers_complexity_3}")
            print(f"Unable to determine/errors (Complexity 3): {tot_unable_count_complexity_3} ({(tot_unable_count_complexity_3/total_answers_complexity_3)*100:.2f}%)")

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


    def UR_DE(self):
        print("UR_DE")
        total_answers = 0
        total_answers_complexity_1 = 0
        total_answers_complexity_2 = 0
        total_answers_complexity_3 = 0
        tot_unable_count = 0
        tot_unable_count_complexity_1 = 0
        tot_unable_count_complexity_2 = 0
        tot_unable_count_complexity_3 = 0

        layout_dict={el:0 for el in LAYOUT_TYPES}
        layout_dict_complexity_1={el:0 for el in LAYOUT_TYPES}
        layout_dict_complexity_2={el:0 for el in LAYOUT_TYPES}
        layout_dict_complexity_3={el:0 for el in LAYOUT_TYPES}

        counter_layout = {el:0 for el in LAYOUT_TYPES}
        counter_layout_complexity_1 = {el:0 for el in LAYOUT_TYPES}
        counter_layout_complexity_2 = {el:0 for el in LAYOUT_TYPES}
        counter_layout_complexity_3 = {el:0 for el in LAYOUT_TYPES}

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

                corrupted_entities = res["corrupted_entities"]
                unique_corrupted_entities = []

                seen=[]
                for entity in corrupted_entities:
                    if entity["text"] not in seen:
                        seen.append(entity["text"])
                        unique_corrupted_entities.append(entity)

                for ans in all_answers:
                    if ans.get("answer_converted", "").lower() == "unable to determine":
                        for el in unique_corrupted_entities:
                            layout_dict[el["objectType"]]+=1
                        if complexity == 1:
                            for el in unique_corrupted_entities:
                                layout_dict_complexity_1[el["objectType"]]+=1
                        if complexity == 2:
                            for el in unique_corrupted_entities:
                                layout_dict_complexity_2[el["objectType"]]+=1
                        if complexity == 3:
                            for el in unique_corrupted_entities:
                                layout_dict_complexity_3[el["objectType"]]+=1
                            
                    for el in unique_corrupted_entities:
                        counter_layout[el["objectType"]]+=1
                    if complexity == 1:
                        for el in unique_corrupted_entities:
                            counter_layout_complexity_1[el["objectType"]]+=1
                    if complexity == 2:
                        for el in unique_corrupted_entities:
                            counter_layout_complexity_2[el["objectType"]]+=1
                    if complexity == 3:
                        for el in unique_corrupted_entities:
                            counter_layout_complexity_3[el["objectType"]]+=1
        res_layout = {}
        res_layout_complexity_1 = {}
        res_layout_complexity_2 = {}
        res_layout_complexity_3 = {}
        for el in layout_dict:
            res_layout[el] = layout_dict[el] / counter_layout[el] if counter_layout[el] != 0 else 0
            res_layout_complexity_1[el] = layout_dict_complexity_1[el] / counter_layout_complexity_1[el] if counter_layout_complexity_1[el] != 0 else 0
            res_layout_complexity_2[el] = layout_dict_complexity_2[el] / counter_layout_complexity_2[el] if counter_layout_complexity_2[el] != 0 else 0
            res_layout_complexity_3[el] = layout_dict_complexity_3[el] / counter_layout_complexity_3[el] if counter_layout_complexity_3[el] != 0 else 0 
        
        return [
            res_layout,
            res_layout_complexity_1,
            res_layout_complexity_2,
            res_layout_complexity_3,
        ]
    

    def UR_PAGE(self):
        print("UR_PAGE")

        total_inpage = 0 
        total_inpage_complexity_1 = 0
        total_inpage_complexity_2 = 0
        total_inpage_complexity_3 = 0
        total_inpage_counter = 0
        total_inpage_counter_complexity_1 = 0
        total_inpage_counter_complexity_2 = 0
        total_inpage_counter_complexity_3 = 0

        total_outpage = 0
        total_outpage_complexity_1 = 0
        total_outpage_complexity_2 = 0
        total_outpage_complexity_3 = 0
        total_outpage_counter = 0
        total_outpage_counter_complexity_1 = 0
        total_outpage_counter_complexity_2 = 0
        total_outpage_counter_complexity_3 = 0

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

                corrupted_entities = res["corrupted_entities"]
                unique_corrupted_entities = []
                seen=[]
                for entity in corrupted_entities:
                    if entity["text"] not in seen:
                        seen.append(entity["text"])
                        unique_corrupted_entities.append(entity)

                patch_entities = res["patch_entities"]

                for ans in all_answers:
                    pages_path = ans.get("pages", [])
                    pages_id = []
                    for page in pages_path:
                        pages_id.append(page.split("/")[-1])

                    contains_corrupted_entities = False
                    for pID,p in patch_entities.items():
                        if pID in pages_id:
                            for objID, obj in p.items():
                                obj_entities = obj["entities"]
                                for entity in obj_entities:
                                    if entity["text"] in seen:
                                        contains_corrupted_entities = True
                                        break
                            if contains_corrupted_entities:
                                break
                    if contains_corrupted_entities:
                        total_inpage_counter += 1
                        if complexity == 1:
                            total_inpage_counter_complexity_1 += 1
                        if complexity == 2:
                            total_inpage_counter_complexity_2 += 1
                        if complexity == 3:
                            total_inpage_counter_complexity_3 += 1
                        if ans.get("answer_converted", "").lower() == "unable to determine":
                            total_inpage += 1
                            if complexity == 1:
                                total_inpage_complexity_1 += 1
                            if complexity == 2:
                                total_inpage_complexity_2 += 1
                            if complexity == 3:
                                total_inpage_complexity_3 += 1

                        # EXTEND ON DOCUMENT ELEMENT AND LAYOUT POSITION
                    else:
                        total_outpage_counter += 1
                        if complexity == 1:
                            total_outpage_counter_complexity_1 += 1
                        if complexity == 2:
                            total_outpage_counter_complexity_2 += 1
                        if complexity == 3:
                            total_outpage_counter_complexity_3 += 1
                        if ans.get("answer_converted", "").lower() == "unable to determine":
                            total_outpage += 1
                            if complexity == 1:
                                total_outpage_complexity_1 += 1
                            if complexity == 2:
                                total_outpage_complexity_2 += 1
                            if complexity == 3:
                                total_outpage_complexity_3 += 1
        

        if total_inpage_counter == 0:
            res_inpage = 0
            res_inpage_complexity_1 = 0
            res_inpage_complexity_2 = 0
            res_inpage_complexity_3 = 0
        else:
            res_inpage = total_inpage / total_inpage_counter
            res_inpage_complexity_1 = total_inpage_complexity_1 / total_inpage_counter_complexity_1 if total_inpage_counter_complexity_1 != 0 else 0
            res_inpage_complexity_2 = total_inpage_complexity_2 / total_inpage_counter_complexity_2 if total_inpage_counter_complexity_2 != 0 else 0
            res_inpage_complexity_3 = total_inpage_complexity_3 / total_inpage_counter_complexity_3 if total_inpage_counter_complexity_3 != 0 else 0
        if total_outpage_counter == 0:
            res_outpage = 0
            res_outpage_complexity_1 = 0
            res_outpage_complexity_2 = 0
            res_outpage_complexity_3 = 0
        else:
            res_outpage = total_outpage / total_outpage_counter
            res_outpage_complexity_1 = total_outpage_complexity_1 / total_outpage_counter_complexity_1 if total_outpage_counter_complexity_1 != 0 else 0
            res_outpage_complexity_2 = total_outpage_complexity_2 / total_outpage_counter_complexity_2 if total_outpage_counter_complexity_2 != 0 else 0
            res_outpage_complexity_3 = total_outpage_complexity_3 / total_outpage_counter_complexity_3 if total_outpage_counter_complexity_3 != 0 else 0
        if self.debug:
            print(f"Total inpage counter: {total_inpage_counter}, {total_inpage_counter_complexity_1}, {total_inpage_counter_complexity_2}, {total_inpage_counter_complexity_3}")
            print(f"Total inpage: {total_inpage}, {total_inpage_complexity_1}, {total_inpage_complexity_2}, {total_inpage_complexity_3}")
            print(f"Total outpage counter: {total_outpage_counter}, {total_outpage_counter_complexity_1}, {total_outpage_counter_complexity_2}, {total_outpage_counter_complexity_3}")
            print(f"Total outpage: {total_outpage}, {total_outpage_complexity_1}, {total_outpage_complexity_2}, {total_outpage_complexity_3}")
        
        return [
            res_inpage,
            res_inpage_complexity_1,
            res_inpage_complexity_2,
            res_inpage_complexity_3,
            res_outpage,
            res_outpage_complexity_1,
            res_outpage_complexity_2,
            res_outpage_complexity_3,
        ]

    
    def UR_NLPE(self):
        print("UR_NLPE")
        correct_unable = 0
        correct_unable_complexity_1 = 0
        correct_unable_complexity_2 = 0
        correct_unable_complexity_3 = 0
        total_corrupted = 0
        total_corrupted_complexity_1 = 0
        total_corrupted_complexity_2 = 0
        total_corrupted_complexity_3 = 0

        entity_dict={el:0 for el in MACRO_ENTITY_TYPES}
        entity_dict_complexity_1={el:0 for el in MACRO_ENTITY_TYPES}
        entity_dict_complexity_2={el:0 for el in MACRO_ENTITY_TYPES}
        entity_dict_complexity_3={el:0 for el in MACRO_ENTITY_TYPES}

        counter_entity = {el:0 for el in MACRO_ENTITY_TYPES}
        counter_entity_complexity_1 = {el:0 for el in MACRO_ENTITY_TYPES}
        counter_entity_complexity_2 = {el:0 for el in MACRO_ENTITY_TYPES}
        counter_entity_complexity_3 = {el:0 for el in MACRO_ENTITY_TYPES}

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
                corrupted_entities = res["entity_type"]    

                for ans in all_answers:
                    if ans.get("answer_converted", "").lower() == "unable to determine":
                        # tot_unable_count += 1
                        for el in corrupted_entities:
                            entity_dict[MACRO_ENTITY_MAPPER[el]]+=1
                            if complexity == 1:
                                entity_dict_complexity_1[MACRO_ENTITY_MAPPER[el]]+=1
                            if complexity == 2:
                                entity_dict_complexity_2[MACRO_ENTITY_MAPPER[el]]+=1
                            if complexity == 3: 
                                entity_dict_complexity_3[MACRO_ENTITY_MAPPER[el]]+=1

                    # total_answers += 1
                    for el in corrupted_entities:
                        counter_entity[MACRO_ENTITY_MAPPER[el]]+=1
                        if complexity == 1:
                            counter_entity_complexity_1[MACRO_ENTITY_MAPPER[el]]+=1
                        if complexity == 2:
                            counter_entity_complexity_2[MACRO_ENTITY_MAPPER[el]]+=1
                        if complexity == 3:
                            counter_entity_complexity_3[MACRO_ENTITY_MAPPER[el]]+=1

        res_entity = {}
        res_entity_complexity_1 = {}
        res_entity_complexity_2 = {}
        res_entity_complexity_3 = {}
        # print("Entity dict",entity_dict)
        # print("Entity counter",counter_entity)
        for el in entity_dict:
            # print("Final",el)
            res_entity[el] = entity_dict[el] / counter_entity[el] if counter_entity[el] != 0 else 0
            res_entity_complexity_1[el] = entity_dict_complexity_1[el] / counter_entity_complexity_1[el] if counter_entity_complexity_1[el] != 0 else 0
            res_entity_complexity_2[el] = entity_dict_complexity_2[el] / counter_entity_complexity_2[el] if counter_entity_complexity_2[el] != 0 else 0
            res_entity_complexity_3[el] = entity_dict_complexity_3[el] / counter_entity_complexity_3[el] if counter_entity_complexity_3[el] != 0 else 0

        if self.debug:
        # if True:
            print(f"Total corrupted questions: {total_corrupted, total_corrupted_complexity_1, total_corrupted_complexity_2, total_corrupted_complexity_3}")
            print(F"TOT\tC1\tC2\tC3")
            for k in MACRO_ENTITY_TYPES:
                print(f"{res_entity[k]:.2f}\t{res_entity_complexity_1[k]:.2f}\t{res_entity_complexity_2[k]:.2f}\t{res_entity_complexity_3[k]:.2f}")
        

        return [
            res_entity,
            res_entity_complexity_1,    
            res_entity_complexity_2,
            res_entity_complexity_3,
        ]    


    def UR_PAGE_DE(self):
        print("UR_PAGE_DE")

        total_inpage = 0 
        total_inpage_complexity_1 = 0
        total_inpage_complexity_2 = 0
        total_inpage_complexity_3 = 0
        total_inpage_counter = 0
        total_inpage_counter_complexity_1 = 0
        total_inpage_counter_complexity_2 = 0
        total_inpage_counter_complexity_3 = 0

        total_outpage = 0
        total_outpage_complexity_1 = 0
        total_outpage_complexity_2 = 0
        total_outpage_complexity_3 = 0
        total_outpage_counter = 0
        total_outpage_counter_complexity_1 = 0
        total_outpage_counter_complexity_2 = 0
        total_outpage_counter_complexity_3 = 0

        layout_dict_inpage = {el:0 for el in LAYOUT_TYPES}
        layout_dict_complexity_1_inpage = {el:0 for el in LAYOUT_TYPES}
        layout_dict_complexity_2_inpage = {el:0 for el in LAYOUT_TYPES}
        layout_dict_complexity_3_inpage = {el:0 for el in LAYOUT_TYPES}
        counter_layout_inpage = {el:0 for el in LAYOUT_TYPES}
        counter_layout_complexity_1_inpage = {el:0 for el in LAYOUT_TYPES}
        counter_layout_complexity_2_inpage = {el:0 for el in LAYOUT_TYPES}
        counter_layout_complexity_3_inpage = {el:0 for el in LAYOUT_TYPES}

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

                corrupted_entities = res["corrupted_entities"]
                unique_corrupted_entities = []
                seen=[]
                for entity in corrupted_entities:
                    if entity["text"] not in seen:
                        seen.append(entity["text"])
                        unique_corrupted_entities.append(entity)

                patch_entities = res["patch_entities"]

                for ans in all_answers:
                    pages_path = ans.get("pages", [])
                    pages_id = []
                    for page in pages_path:
                        pages_id.append(page.split("/")[-1])

                    contains_corrupted_entities = False
                    doc_elements_pages = []
                    for pID,p in patch_entities.items():
                        if pID in pages_id:
                            for objID, obj in p.items():
                                obj_entities = obj["entities"]
                                for entity in obj_entities:
                                    if entity["text"] in seen:
                                        contains_corrupted_entities = True
                                        doc_elements_pages.append(obj["type"])
                                        # break
                    unique_doc_elements_pages = list(set(doc_elements_pages))

                    if contains_corrupted_entities:
                        for el in unique_doc_elements_pages:
                            counter_layout_inpage[el]+=1
                            if complexity == 1:
                                counter_layout_complexity_1_inpage[el]+=1
                            if complexity == 2:
                                counter_layout_complexity_2_inpage[el]+=1
                            if complexity == 3:
                                counter_layout_complexity_3_inpage[el]+=1

                        if ans.get("answer_converted", "").lower() == "unable to determine":
                            for el in unique_doc_elements_pages:
                                layout_dict_inpage[el]+=1
                                if complexity == 1:
                                    layout_dict_complexity_1_inpage[el]+=1
                                if complexity == 2:
                                    layout_dict_complexity_2_inpage[el]+=1  
                                if complexity == 3:
                                    layout_dict_complexity_3_inpage[el]+=1
        

        # normalize the layout_dict wrt total_corrupted
        res_layout = {}
        res_layout_complexity_1 = {}
        res_layout_complexity_2 = {}
        res_layout_complexity_3 = {}
        for el in layout_dict_inpage:
            res_layout[el] = layout_dict_inpage[el] / counter_layout_inpage[el] if counter_layout_inpage[el] != 0 else 0
            res_layout_complexity_1[el] = layout_dict_complexity_1_inpage[el] / counter_layout_complexity_1_inpage[el] if counter_layout_complexity_1_inpage[el] != 0 else 0
            res_layout_complexity_2[el] = layout_dict_complexity_2_inpage[el] / counter_layout_complexity_2_inpage[el] if counter_layout_complexity_2_inpage[el] != 0 else 0
            res_layout_complexity_3[el] = layout_dict_complexity_3_inpage[el] / counter_layout_complexity_3_inpage[el] if counter_layout_complexity_3_inpage[el] != 0 else 0

        if self.debug:
            print(f"Total corrupted questions: {total_corrupted, total_corrupted_complexity_1, total_corrupted_complexity_2, total_corrupted_complexity_3}")
            print(F"TOT\tC1\tC2\tC3")
            for k in LAYOUT_TYPES:
                print(f"{layout_dict_inpage[k]:.2f}\t{layout_dict_complexity_1_inpage[k]:.2f}\t{layout_dict_complexity_2_inpage[k]:.2f}\t{layout_dict_complexity_3_inpage[k]:.2f}")
        

        return [
            res_layout,
            res_layout_complexity_1,    
            res_layout_complexity_2,
            res_layout_complexity_3,
        ]


    def UR_PAGE_QP(self):
        print("UR_PAGE_QP")

        total_inpage = 0 
        total_inpage_complexity_1 = 0
        total_inpage_complexity_2 = 0
        total_inpage_complexity_3 = 0
        total_inpage_counter = 0
        total_inpage_counter_complexity_1 = 0
        total_inpage_counter_complexity_2 = 0
        total_inpage_counter_complexity_3 = 0

        total_outpage = 0
        total_outpage_complexity_1 = 0
        total_outpage_complexity_2 = 0
        total_outpage_complexity_3 = 0
        total_outpage_counter = 0
        total_outpage_counter_complexity_1 = 0
        total_outpage_counter_complexity_2 = 0
        total_outpage_counter_complexity_3 = 0

        pos_dict={el:0 for el in PAGE_LAYOUT}
        pos_dict_complexity_1={el:0 for el in PAGE_LAYOUT}
        pos_dict_complexity_2={el:0 for el in PAGE_LAYOUT}
        pos_dict_complexity_3={el:0 for el in PAGE_LAYOUT}

        counter_pos = {el:0 for el in PAGE_LAYOUT}
        counter_pos_complexity_1 = {el:0 for el in PAGE_LAYOUT}
        counter_pos_complexity_2 = {el:0 for el in PAGE_LAYOUT}
        counter_pos_complexity_3 = {el:0 for el in PAGE_LAYOUT}

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

                corrupted_entities = res["corrupted_entities"]
                unique_corrupted_entities = []
                seen=[]
                for entity in corrupted_entities:
                    if entity["text"] not in seen:
                        seen.append(entity["text"])
                        unique_corrupted_entities.append(entity)

                patch_entities = res["patch_entities"]


                pages=[]
                for ans in all_answers:
                    pages.extend(ans.get("pages", []))
                unique_pages = list(set(pages))
                average_x_coord = 0
                average_y_coord = 0
                for page in unique_pages:
                    #/data2/dnapolitano/VQA/data/DUDE_train-val-test_binaries/images
                    if "data1" in page:
                        page=page.replace("data1","data2")
                    x_size, y_size = Image.open(page).size
                    average_x_coord += x_size
                    average_y_coord += y_size
                average_x_coord /= len(unique_pages)
                average_y_coord /= len(unique_pages)

                for ans in all_answers:
                    pages_path = ans.get("pages", [])
                    pages_id = []
                    for page in pages_path:
                        pages_id.append(page.split("/")[-1])

                    contains_corrupted_entities = False
                    doc_elements_pages = []
                    for pID, p in patch_entities.items():
                        if pID in pages_id:
                            for objID, obj in p.items():
                                obj_entities = obj["entities"]
                                for entity in obj_entities:
                                    if entity["text"] in seen:
                                        contains_corrupted_entities = True
                                        doc_elements_pages.append(obj)

                    if contains_corrupted_entities:
                        for el in doc_elements_pages:
                            corrupted_entity_bbox = el.get("bbox", [])
                            bbox_center = [
                                (corrupted_entity_bbox[0] + corrupted_entity_bbox[2]) / 2,
                                (corrupted_entity_bbox[1] + corrupted_entity_bbox[3]) / 2
                            ]
                            if bbox_center[0] < average_x_coord / 2 and bbox_center[1] < average_y_coord / 2:
                                counter_pos["TOP_LEFT"]+=1
                                if complexity == 1:
                                    counter_pos_complexity_1["TOP_LEFT"]+=1
                                if complexity == 2:
                                    counter_pos_complexity_2["TOP_LEFT"]+=1
                                if complexity == 3:
                                    counter_pos_complexity_3["TOP_LEFT"]+=1
                            elif bbox_center[0] < average_x_coord / 2 and bbox_center[1] >= average_y_coord / 2:
                                counter_pos["BOTTOM_LEFT"]+=1
                                if complexity == 1:
                                    counter_pos_complexity_1["BOTTOM_LEFT"]+=1
                                if complexity == 2:
                                    counter_pos_complexity_2["BOTTOM_LEFT"]+=1
                                if complexity == 3:
                                    counter_pos_complexity_3["BOTTOM_LEFT"]+=1
                            elif bbox_center[0] >= average_x_coord / 2 and bbox_center[1] < average_y_coord / 2:
                                counter_pos["TOP_RIGHT"]+=1
                                if complexity == 1:
                                    counter_pos_complexity_1["TOP_RIGHT"]+=1
                                if complexity == 2:
                                    counter_pos_complexity_2["TOP_RIGHT"]+=1
                                if complexity == 3:
                                    counter_pos_complexity_3["TOP_RIGHT"]+=1
                            elif bbox_center[0] >= average_x_coord / 2 and bbox_center[1] >= average_y_coord / 2:
                                counter_pos["BOTTOM_RIGHT"]+=1
                                if complexity == 1:
                                    counter_pos_complexity_1["BOTTOM_RIGHT"]+=1
                                if complexity == 2:
                                    counter_pos_complexity_2["BOTTOM_RIGHT"]+=1
                                if complexity == 3:
                                    counter_pos_complexity_3["BOTTOM_RIGHT"]+=1

                        if ans.get("answer_converted", "").lower() == "unable to determine":
                            for el in doc_elements_pages:
                                corrupted_entity_bbox = el.get("bbox", [])
                                bbox_center = [
                                    (corrupted_entity_bbox[0] + corrupted_entity_bbox[2]) / 2,
                                    (corrupted_entity_bbox[1] + corrupted_entity_bbox[3]) / 2
                                ]
                                if bbox_center[0] < average_x_coord / 2 and bbox_center[1] < average_y_coord / 2:
                                    pos_dict["TOP_LEFT"]+=1
                                    if complexity == 1:
                                        pos_dict_complexity_1["TOP_LEFT"]+=1
                                    if complexity == 2:
                                        pos_dict_complexity_2["TOP_LEFT"]+=1
                                    if complexity == 3:
                                        pos_dict_complexity_3["TOP_LEFT"]+=1
                                elif bbox_center[0] < average_x_coord / 2 and bbox_center[1] >= average_y_coord / 2:
                                    pos_dict["BOTTOM_LEFT"]+=1
                                    if complexity == 1:
                                        pos_dict_complexity_1["BOTTOM_LEFT"]+=1
                                    if complexity == 2:
                                        pos_dict_complexity_2["BOTTOM_LEFT"]+=1
                                    if complexity == 3:
                                        pos_dict_complexity_3["BOTTOM_LEFT"]+=1
                                elif bbox_center[0] >= average_x_coord / 2 and bbox_center[1] < average_y_coord / 2:
                                    pos_dict["TOP_RIGHT"]+=1
                                    if complexity == 1:
                                        pos_dict_complexity_1["TOP_RIGHT"]+=1
                                    if complexity == 2:
                                        pos_dict_complexity_2["TOP_RIGHT"]+=1
                                    if complexity == 3:
                                        pos_dict_complexity_3["TOP_RIGHT"]+=1
                                elif bbox_center[0] >= average_x_coord / 2 and bbox_center[1] >= average_y_coord / 2:
                                    pos_dict["BOTTOM_RIGHT"]+=1
                                    if complexity == 1:
                                        pos_dict_complexity_1["BOTTOM_RIGHT"]+=1
                                    if complexity == 2:
                                        pos_dict_complexity_2["BOTTOM_RIGHT"]+=1
                                    if complexity == 3:
                                        pos_dict_complexity_3["BOTTOM_RIGHT"]+=1
        

        res_pos = {}
        res_pos_complexity_1 = {}
        res_pos_complexity_2 = {}
        res_pos_complexity_3 = {}
        # print("Entity dict",entity_dict)
        # print("Entity counter",counter_entity)
        for el in pos_dict:
            # print("Final",el)
            res_pos[el] = pos_dict[el] / counter_pos[el] if counter_pos[el] != 0 else 0
            res_pos_complexity_1[el] = pos_dict_complexity_1[el] / counter_pos_complexity_1[el] if counter_pos_complexity_1[el] != 0 else 0
            res_pos_complexity_2[el] = pos_dict_complexity_2[el] / counter_pos_complexity_2[el] if counter_pos_complexity_2[el] != 0 else 0
            res_pos_complexity_3[el] = pos_dict_complexity_3[el] / counter_pos_complexity_3[el] if counter_pos_complexity_3[el] != 0 else 0

        if self.debug:
        # if True:
            print(f"Total corrupted questions: {total_corrupted, total_corrupted_complexity_1, total_corrupted_complexity_2, total_corrupted_complexity_3}")
            print(F"TOT\tC1\tC2\tC3")
            for k in MACRO_ENTITY_TYPES:
                print(f"{res_pos[k]:.2f}\t{res_pos_complexity_1[k]:.2f}\t{res_pos_complexity_2[k]:.2f}\t{res_pos_complexity_3[k]:.2f}")
        

        return [
            res_pos,
            res_pos_complexity_1,    
            res_pos_complexity_2,
            res_pos_complexity_3,
        ]


    def UR_PAGE_DED(self):
        print("UR_PAGE_DED")

        total_inpage = 0 
        total_inpage_complexity_1 = 0
        total_inpage_complexity_2 = 0
        total_inpage_complexity_3 = 0
        total_inpage_counter = 0
        total_inpage_counter_complexity_1 = 0
        total_inpage_counter_complexity_2 = 0
        total_inpage_counter_complexity_3 = 0

        total_outpage = 0
        total_outpage_complexity_1 = 0
        total_outpage_complexity_2 = 0
        total_outpage_complexity_3 = 0
        total_outpage_counter = 0
        total_outpage_counter_complexity_1 = 0
        total_outpage_counter_complexity_2 = 0
        total_outpage_counter_complexity_3 = 0

        layout_dict={"0":0, "1":0, ">1":0}
        layout_dict_complexity_1={"0":0, "1":0, ">1":0}
        layout_dict_complexity_2={"0":0, "1":0, ">1":0}
        layout_dict_complexity_3={"0":0, "1":0, ">1":0}

        counter_layout = {"0":0, "1":0, ">1":0}
        counter_layout_complexity_1 = {"0":0, "1":0, ">1":0}
        counter_layout_complexity_2 = {"0":0, "1":0, ">1":0}
        counter_layout_complexity_3 = {"0":0, "1":0, ">1":0}

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

                corrupted_entities = res["corrupted_entities"]
                unique_corrupted_entities = []
                seen=[]
                for entity in corrupted_entities:
                    if entity["text"] not in seen:
                        seen.append(entity["text"])
                        unique_corrupted_entities.append(entity)

                patch_entities = res["patch_entities"]
                layout_doc = res["layout_analysis"]["pages"]

                for ans in all_answers:
                    pages_path = ans.get("pages", [])
                    pages_id = []
                    for page in pages_path:
                        pages_id.append(page.split("/")[-1])

                    doc_dist = {el:0 for el in MACRO_LAYOUT_TYPES}
                    for page, info in layout_doc.items():
                        if page in pages_id:
                            layout_page=info["layout_analysis"]
                            for objID, obj in layout_page.items():
                                doc_dist[MAPPER_LAYOUT_TYPES[obj["ObjectType"]]]+=1
                    
                    if doc_dist["vre"]==0:
                        counter_layout["0"]+=1
                        if complexity == 1:
                            counter_layout_complexity_1["0"]+=1
                        if complexity == 2:
                            counter_layout_complexity_2["0"]+=1
                        if complexity == 3:
                            counter_layout_complexity_3["0"]+=1
                    elif doc_dist["vre"]==1:
                        counter_layout["1"]+=1
                        if complexity == 1:
                            counter_layout_complexity_1["1"]+=1
                        if complexity == 2:
                            counter_layout_complexity_2["1"]+=1
                        if complexity == 3:
                            counter_layout_complexity_3["1"]+=1
                    else:
                        counter_layout[">1"]+=1
                        if complexity == 1:
                            counter_layout_complexity_1[">1"]+=1
                        if complexity == 2:
                            counter_layout_complexity_2[">1"]+=1
                        if complexity == 3:
                            counter_layout_complexity_3[">1"]+=1


                    if ans.get("answer_converted", "").lower() == "unable to determine":
                        if doc_dist["vre"]==0:
                            layout_dict["0"]+=1
                            if complexity == 1:
                                layout_dict_complexity_1["0"]+=1
                            if complexity == 2:
                                layout_dict_complexity_2["0"]+=1
                            if complexity == 3:
                                layout_dict_complexity_3["0"]+=1
                        elif doc_dist["vre"]==1:
                            layout_dict["1"]+=1
                            if complexity == 1:
                                layout_dict_complexity_1["1"]+=1
                            if complexity == 2:
                                layout_dict_complexity_2["1"]+=1
                            if complexity == 3:
                                layout_dict_complexity_3["1"]+=1
                        else:
                            layout_dict[">1"]+=1
                            if complexity == 1:
                                layout_dict_complexity_1[">1"]+=1
                            if complexity == 2:
                                layout_dict_complexity_2[">1"]+=1
                            if complexity == 3:
                                layout_dict_complexity_3[">1"]+=1
        

        # normalize the layout_dict wrt total_corrupted
        res_layout = {}
        res_layout_complexity_1 = {}
        res_layout_complexity_2 = {}
        res_layout_complexity_3 = {}
        for el in layout_dict:  
            res_layout[el] = layout_dict[el] / counter_layout[el] if counter_layout[el] != 0 else 0
            res_layout_complexity_1[el] = layout_dict_complexity_1[el] / counter_layout_complexity_1[el] if counter_layout_complexity_1[el] != 0 else 0
            res_layout_complexity_2[el] = layout_dict_complexity_2[el] / counter_layout_complexity_2[el] if counter_layout_complexity_2[el] != 0 else 0
            res_layout_complexity_3[el] = layout_dict_complexity_3[el] / counter_layout_complexity_3[el] if counter_layout_complexity_3[el] != 0 else 0
        if self.debug:
            print(f"Total corrupted questions: {total_corrupted, total_corrupted_complexity_1, total_corrupted_complexity_2, total_corrupted_complexity_3}")
            print(F"TOT\tC1\tC2\tC3")
            for k in MACRO_ENTITY_TYPES:
                print(f"{layout_dict[k]:.2f}\t{layout_dict_complexity_1[k]:.2f}\t{layout_dict_complexity_2[k]:.2f}\t{layout_dict_complexity_3[k]:.2f}")
        
        return [
            res_layout,
            res_layout_complexity_1,
            res_layout_complexity_2,    
            res_layout_complexity_3,
        ]
    

def generate_analysis_report(dataset, images_path):
    # Group results by window size (e.g., "w=1" and "w=2")
    print("Initializing Entity Verifier...")
    entity_verifier = None #EntityIdentifier(ENTITY_TYPES)
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

        # if folder.name != "results_w1":
        #     print(f"Skipping folder {folder.name}")
        #     continue

        print("")
        print("#" * 100)
        print(f"Processing folder {folder}")

        # create path folder/results if not exists
        folder_results = folder / "results"
        os.makedirs(folder_results, exist_ok=True)


        dict_QUR = {}
        dict_QUR_DE = {}
        dict_QUR_DE_complexity_1 = {}
        dict_QUR_DE_complexity_2 = {}
        dict_QUR_DE_complexity_3 = {}
        dict_QUR_NLPE = {}
        dict_QUR_NLPE_complexity_1 = {}
        dict_QUR_NLPE_complexity_2 = {}
        dict_QUR_NLPE_complexity_3 = {}
        dict_QUR_QP = {}
        dict_QUR_QP_complexity_1 = {}
        dict_QUR_QP_complexity_2 = {}
        dict_QUR_QP_complexity_3 = {}
        dict_QUR_PL = {}
        dict_QUR_PL_complexity_1 = {}
        dict_QUR_PL_complexity_2 = {}
        dict_QUR_PL_complexity_3 = {}
        dict_QUR_DED = {}
        dict_QUR_DED_complexity_1 = {}
        dict_QUR_DED_complexity_2 = {}
        dict_QUR_DED_complexity_3 = {}

        dict_UR = {}
        dict_UR_DE = {}
        dict_UR_DE_complexity_1 = {}
        dict_UR_DE_complexity_2 = {}
        dict_UR_DE_complexity_3 = {}
        dict_UR_PAGE_outpage = {}
        dict_UR_PAGE_inpage = {}
        dict_UR_PAGE_DE = {}
        dict_UR_PAGE_DE_complexity_1 = {}
        dict_UR_PAGE_DE_complexity_2 = {}
        dict_UR_PAGE_DE_complexity_3 = {}
        dict_UR_NLPE = {}
        dict_UR_NLPE_complexity_1 = {}
        dict_UR_NLPE_complexity_2 = {}
        dict_UR_NLPE_complexity_3 = {}
        dict_UR_PAGE_QP = {}
        dict_UR_PAGE_QP_complexity_1 = {}
        dict_UR_PAGE_QP_complexity_2 = {}
        dict_UR_PAGE_QP_complexity_3 = {}
        dict_UR_PAGE_DED = {}
        dict_UR_PAGE_DED_complexity_1 = {}
        dict_UR_PAGE_DED_complexity_2 = {}
        dict_UR_PAGE_DED_complexity_3 = {}


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

                print(f"- Metrics -")

                for key, value in metrics.items():
                    

                    if key == "QUR":
                        print(f"Processing QUR")
                        (
                            correct_unable_total_corrupted,
                            correct_unable_complexity_1_total_corrupted,
                            correct_unable_complexity_2_total_corrupted,
                            correct_unable_complexity_3_total_corrupted,
                            weight_unable
                        ) = value
                        dict_QUR[model_name] = [
                            correct_unable_total_corrupted,
                            correct_unable_complexity_1_total_corrupted,
                            correct_unable_complexity_2_total_corrupted,
                            correct_unable_complexity_3_total_corrupted,
                            weight_unable,
                        ]

                    if key == "QUR_DE":
                        print(f"Processing QUR_DE")
                        (
                            layout_dict,
                            layout_dict_complexity_1,
                            layout_dict_complexity_2,
                            layout_dict_complexity_3,
                        ) = value
                        dict_QUR_DE[model_name] = layout_dict.values()
                        dict_QUR_DE_complexity_1[model_name] = layout_dict_complexity_1.values()
                        dict_QUR_DE_complexity_2[model_name] = layout_dict_complexity_2.values()
                        dict_QUR_DE_complexity_3[model_name] = layout_dict_complexity_3.values()

                    if key == "QUR_NLPE":
                        print(f"Processing QUR_NLPE")
                        (
                            entity_dict,
                            entity_dict_complexity_1,
                            entity_dict_complexity_2,
                            entity_dict_complexity_3,
                        ) = value
                        dict_QUR_NLPE[model_name] = entity_dict.values()
                        dict_QUR_NLPE_complexity_1[model_name] = entity_dict_complexity_1.values()
                        dict_QUR_NLPE_complexity_2[model_name] = entity_dict_complexity_2.values()
                        dict_QUR_NLPE_complexity_3[model_name] = entity_dict_complexity_3.values()

                    if key == "QUR_QP":
                        print(f"Processing QUR_QP")
                        (
                            pos_dict,
                            pos_dict_complexity_1,
                            pos_dict_complexity_2,
                            pos_dict_complexity_3,
                        ) = value
                        dict_QUR_QP[model_name] = pos_dict.values()
                        dict_QUR_QP_complexity_1[model_name] = pos_dict_complexity_1.values()
                        dict_QUR_QP_complexity_2[model_name] = pos_dict_complexity_2.values()
                        dict_QUR_QP_complexity_3[model_name] = pos_dict_complexity_3.values()

                    if key == "QUR_PL":
                        print(f"Processing QUR_PL")
                        (
                            len_dict,
                            len_dict_complexity_1,
                            len_dict_complexity_2,
                            len_dict_complexity_3,
                            list_len
                        ) = value
                        dict_QUR_PL[model_name] = len_dict.values()
                        dict_QUR_PL_complexity_1[model_name] = len_dict_complexity_1.values()
                        dict_QUR_PL_complexity_2[model_name] = len_dict_complexity_2.values()
                        dict_QUR_PL_complexity_3[model_name] = len_dict_complexity_3.values()

                    if key == "QUR_DED":
                        print(f"Processing QUR_DED")
                        (
                            ded_dict,
                            ded_dict_complexity_1,
                            ded_dict_complexity_2,
                            ded_dict_complexity_3,
                        ) = value
                        dict_QUR_DED[model_name] = ded_dict.values()
                        dict_QUR_DED_complexity_1[model_name] = ded_dict_complexity_1.values()
                        dict_QUR_DED_complexity_2[model_name] = ded_dict_complexity_2.values()
                        dict_QUR_DED_complexity_3[model_name] = ded_dict_complexity_3.values()
                        
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

                    if key == "UR_DE":
                        print(f"Processing UR_DE")
                        (
                            layout_dict,
                            layout_dict_complexity_1,
                            layout_dict_complexity_2,
                            layout_dict_complexity_3,
                        ) = value
                        dict_UR_DE[model_name] = layout_dict.values()
                        dict_UR_DE_complexity_1[model_name] = layout_dict_complexity_1.values()
                        dict_UR_DE_complexity_2[model_name] = layout_dict_complexity_2.values()
                        dict_UR_DE_complexity_3[model_name] = layout_dict_complexity_3.values()
                    
                    if key == "UR_PAGE":
                        print(f"Processing UR_PAGE")
                        (
                            inpage,
                            inpage_complexity_1,
                            inpage_complexity_2,
                            inpage_complexity_3,
                            outpage,
                            outpage_complexity_1,   
                            outpage_complexity_2,
                            outpage_complexity_3,
                        ) = value
                        dict_UR_PAGE_inpage[model_name] = [
                            inpage,
                            inpage_complexity_1,
                            inpage_complexity_2,
                            inpage_complexity_3,
                        ]
                        dict_UR_PAGE_outpage[model_name] = [
                            outpage,
                            outpage_complexity_1,
                            outpage_complexity_2,
                            outpage_complexity_3,
                        ] 
                    
                    if key == "UR_NLPE":
                        print(f"Processing UR_NLPE")
                        (
                            entity_dict,
                            entity_dict_complexity_1,
                            entity_dict_complexity_2,
                            entity_dict_complexity_3,
                        ) = value
                        dict_UR_NLPE[model_name] = entity_dict.values()
                        dict_UR_NLPE_complexity_1[model_name] = entity_dict_complexity_1.values()
                        dict_UR_NLPE_complexity_2[model_name] = entity_dict_complexity_2.values()
                        dict_UR_NLPE_complexity_3[model_name] = entity_dict_complexity_3.values()
                    
                    if key == "UR_PAGE_DE":
                        print(f"Processing UR_PAGE_DE")
                        (
                            layout_dict,
                            layout_dict_complexity_1,
                            layout_dict_complexity_2,
                            layout_dict_complexity_3,
                        ) = value
                        dict_UR_PAGE_DE[model_name] = layout_dict.values()
                        dict_UR_PAGE_DE_complexity_1[model_name] = layout_dict_complexity_1.values()
                        dict_UR_PAGE_DE_complexity_2[model_name] = layout_dict_complexity_2.values()
                        dict_UR_PAGE_DE_complexity_3[model_name] = layout_dict_complexity_3.values()

                    if key == "UR_PAGE_QP":
                        print(f"Processing UR_PAGE_QP")
                        (
                            pos_dict,
                            pos_dict_complexity_1,
                            pos_dict_complexity_2,
                            pos_dict_complexity_3,
                        ) = value
                        dict_UR_PAGE_QP[model_name] = pos_dict.values()
                        dict_UR_PAGE_QP_complexity_1[model_name] = pos_dict_complexity_1.values()
                        dict_UR_PAGE_QP_complexity_2[model_name] = pos_dict_complexity_2.values()
                        dict_UR_PAGE_QP_complexity_3[model_name] = pos_dict_complexity_3.values()

                    if key == "UR_PAGE_DED":
                        print(f"Processing UR_PAGE_DED")
                        (
                            ded_dict,
                            ded_dict_complexity_1,
                            ded_dict_complexity_2,
                            ded_dict_complexity_3,
                        ) = value
                        dict_UR_PAGE_DED[model_name] = ded_dict.values()
                        dict_UR_PAGE_DED_complexity_1[model_name] = ded_dict_complexity_1.values()
                        dict_UR_PAGE_DED_complexity_2[model_name] = ded_dict_complexity_2.values()
                        dict_UR_PAGE_DED_complexity_3[model_name] = ded_dict_complexity_3.values()

            except Exception as e:
                print(f"Error processing {result_file}: {e}")
                # continue

            # break

        print(f"Saving files")
        print(f"Processed models: {processed_models}")

        df_QUR = pd.DataFrame(dict_QUR)
        df_QUR.index = ["QUR", "QUR_C1", "QUR_C2", "QUR_C3", "QUR_weighted"]
        df_QUR.to_csv(folder_results / "QUR.csv")

        df_QUR_DE = pd.DataFrame(dict_QUR_DE)
        df_QUR_DE.index = LAYOUT_TYPES
        df_QUR_DE.to_csv(folder_results / "QUR_DE.csv")
        df_QUR_DE_complexity_1 = pd.DataFrame(dict_QUR_DE_complexity_1)
        df_QUR_DE_complexity_1.index = LAYOUT_TYPES
        df_QUR_DE_complexity_1.to_csv(folder_results / "QUR_DE_complexity_1.csv")
        df_QUR_DE_complexity_2 = pd.DataFrame(dict_QUR_DE_complexity_2)
        df_QUR_DE_complexity_2.index = LAYOUT_TYPES
        df_QUR_DE_complexity_2.to_csv(folder_results / "QUR_DE_complexity_2.csv")
        df_QUR_DE_complexity_3 = pd.DataFrame(dict_QUR_DE_complexity_3)
        df_QUR_DE_complexity_3.index = LAYOUT_TYPES
        df_QUR_DE_complexity_3.to_csv(folder_results / "QUR_DE_complexity_3.csv")

        df_QUR_NLPE = pd.DataFrame(dict_QUR_NLPE)
        df_QUR_NLPE.index = MACRO_ENTITY_TYPES
        df_QUR_NLPE.to_csv(folder_results / "QUR_NLPE.csv")
        df_QUR_NLPE_complexity_1 = pd.DataFrame(dict_QUR_NLPE_complexity_1)
        df_QUR_NLPE_complexity_1.index = MACRO_ENTITY_TYPES
        df_QUR_NLPE_complexity_1.to_csv(folder_results / "QUR_NLPE_complexity_1.csv")
        df_QUR_NLPE_complexity_2 = pd.DataFrame(dict_QUR_NLPE_complexity_2)
        df_QUR_NLPE_complexity_2.index = MACRO_ENTITY_TYPES
        df_QUR_NLPE_complexity_2.to_csv(folder_results / "QUR_NLPE_complexity_2.csv")
        df_QUR_NLPE_complexity_3 = pd.DataFrame(dict_QUR_NLPE_complexity_3)
        df_QUR_NLPE_complexity_3.index = MACRO_ENTITY_TYPES
        df_QUR_NLPE_complexity_3.to_csv(folder_results / "QUR_NLPE_complexity_3.csv")

        df_QUR_QP = pd.DataFrame(dict_QUR_QP)
        df_QUR_QP.index = PAGE_LAYOUT
        df_QUR_QP.to_csv(folder_results / "QUR_QP.csv")
        df_QUR_QP_complexity_1 = pd.DataFrame(dict_QUR_QP_complexity_1)
        df_QUR_QP_complexity_1.index = PAGE_LAYOUT
        df_QUR_QP_complexity_1.to_csv(folder_results / "QUR_QP_complexity_1.csv")
        df_QUR_QP_complexity_2 = pd.DataFrame(dict_QUR_QP_complexity_2)
        df_QUR_QP_complexity_2.index = PAGE_LAYOUT
        df_QUR_QP_complexity_2.to_csv(folder_results / "QUR_QP_complexity_2.csv")
        df_QUR_QP_complexity_3 = pd.DataFrame(dict_QUR_QP_complexity_3)
        df_QUR_QP_complexity_3.index = PAGE_LAYOUT
        df_QUR_QP_complexity_3.to_csv(folder_results / "QUR_QP_complexity_3.csv")

        df_QUR_PL = pd.DataFrame(dict_QUR_PL)
        df_QUR_PL.index = list_len
        df_QUR_PL.to_csv(folder_results / "QUR_PL.csv")
        df_QUR_PL_complexity_1 = pd.DataFrame(dict_QUR_PL_complexity_1)
        df_QUR_PL_complexity_1.index = list_len
        df_QUR_PL_complexity_1.to_csv(folder_results / "QUR_PL_complexity_1.csv")
        df_QUR_PL_complexity_2 = pd.DataFrame(dict_QUR_PL_complexity_2)
        df_QUR_PL_complexity_2.index = list_len
        df_QUR_PL_complexity_2.to_csv(folder_results / "QUR_PL_complexity_2.csv")
        df_QUR_PL_complexity_3 = pd.DataFrame(dict_QUR_PL_complexity_3)
        df_QUR_PL_complexity_3.index = list_len
        df_QUR_PL_complexity_3.to_csv(folder_results / "QUR_PL_complexity_3.csv")

        df_QUR_DED = pd.DataFrame(dict_QUR_DED)
        df_QUR_DED.index = ["<15", "15-25", ">25"]
        df_QUR_DED.to_csv(folder_results / "QUR_DED.csv")
        df_QUR_DED_complexity_1 = pd.DataFrame(dict_QUR_DED_complexity_1)
        df_QUR_DED_complexity_1.index = ["<15", "15-25", ">25"]
        df_QUR_DED_complexity_1.to_csv(folder_results / "QUR_DED_complexity_1.csv")
        df_QUR_DED_complexity_2 = pd.DataFrame(dict_QUR_DED_complexity_2)
        df_QUR_DED_complexity_2.index = ["<15", "15-25", ">25"]
        df_QUR_DED_complexity_2.to_csv(folder_results / "QUR_DED_complexity_2.csv")
        df_QUR_DED_complexity_3 = pd.DataFrame(dict_QUR_DED_complexity_3)
        df_QUR_DED_complexity_3.index = ["<15", "15-25", ">25"]
        df_QUR_DED_complexity_3.to_csv(folder_results / "QUR_DED_complexity_3.csv")

        df_UR = pd.DataFrame(dict_UR)
        df_UR.index = ["UR", "UR_C1", "UR_C2", "UR_C3"]
        df_UR.to_csv(folder_results / "UR.csv")

        df_UR_DE = pd.DataFrame(dict_UR_DE)
        df_UR_DE.index = LAYOUT_TYPES
        df_UR_DE.to_csv(folder_results / "UR_DE.csv")
        df_UR_DE_complexity_1 = pd.DataFrame(dict_UR_DE_complexity_1)
        df_UR_DE_complexity_1.index = LAYOUT_TYPES
        df_UR_DE_complexity_1.to_csv(folder_results / "UR_DE_complexity_1.csv")
        df_UR_DE_complexity_2 = pd.DataFrame(dict_UR_DE_complexity_2)
        df_UR_DE_complexity_2.index = LAYOUT_TYPES
        df_UR_DE_complexity_2.to_csv(folder_results / "UR_DE_complexity_2.csv")
        df_UR_DE_complexity_3 = pd.DataFrame(dict_UR_DE_complexity_3)
        df_UR_DE_complexity_3.index = LAYOUT_TYPES
        df_UR_DE_complexity_3.to_csv(folder_results / "UR_DE_complexity_3.csv")

        df_UR_PAGE_inpage = pd.DataFrame(dict_UR_PAGE_inpage)
        df_UR_PAGE_inpage.index = ["UR_inpage", "UR_inpage_C1", "UR_inpage_C2", "UR_inpage_C3"]
        df_UR_PAGE_inpage.to_csv(folder_results / "UR_PAGE_inpage.csv")
        df_UR_PAGE_outpage = pd.DataFrame(dict_UR_PAGE_outpage)
        df_UR_PAGE_outpage.index = ["UR_outpage", "UR_outpage_C1", "UR_outpage_C2", "UR_outpage_C3"]
        df_UR_PAGE_outpage.to_csv(folder_results / "UR_PAGE_outpage.csv")

        df_UR_PAGE_DE = pd.DataFrame(dict_UR_PAGE_DE)
        df_UR_PAGE_DE.index = LAYOUT_TYPES
        df_UR_PAGE_DE.to_csv(folder_results / "UR_PAGE_DE.csv")
        df_UR_PAGE_DE_complexity_1 = pd.DataFrame(dict_UR_PAGE_DE_complexity_1)
        df_UR_PAGE_DE_complexity_1.index = LAYOUT_TYPES
        df_UR_PAGE_DE_complexity_1.to_csv(folder_results / "UR_PAGE_DE_complexity_1.csv")
        df_UR_PAGE_DE_complexity_2 = pd.DataFrame(dict_UR_PAGE_DE_complexity_2) 
        df_UR_PAGE_DE_complexity_2.index = LAYOUT_TYPES
        df_UR_PAGE_DE_complexity_2.to_csv(folder_results / "UR_PAGE_DE_complexity_2.csv")
        df_UR_PAGE_DE_complexity_3 = pd.DataFrame(dict_UR_PAGE_DE_complexity_3)
        df_UR_PAGE_DE_complexity_3.index = LAYOUT_TYPES
        df_UR_PAGE_DE_complexity_3.to_csv(folder_results / "UR_PAGE_DE_complexity_3.csv")
        df_UR_NLPE = pd.DataFrame(dict_UR_NLPE)
        df_UR_NLPE.index = MACRO_ENTITY_TYPES
        df_UR_NLPE.to_csv(folder_results / "UR_NLPE.csv")
        df_UR_NLPE_complexity_1 = pd.DataFrame(dict_UR_NLPE_complexity_1)
        df_UR_NLPE_complexity_1.index = MACRO_ENTITY_TYPES
        df_UR_NLPE_complexity_1.to_csv(folder_results / "UR_NLPE_complexity_1.csv")
        df_UR_NLPE_complexity_2 = pd.DataFrame(dict_UR_NLPE_complexity_2)
        df_UR_NLPE_complexity_2.index = MACRO_ENTITY_TYPES
        df_UR_NLPE_complexity_2.to_csv(folder_results / "UR_NLPE_complexity_2.csv")
        df_UR_NLPE_complexity_3 = pd.DataFrame(dict_UR_NLPE_complexity_3)
        df_UR_NLPE_complexity_3.index = MACRO_ENTITY_TYPES
        df_UR_NLPE_complexity_3.to_csv(folder_results / "UR_NLPE_complexity_3.csv")
        df_UR_PAGE_QP = pd.DataFrame(dict_UR_PAGE_QP)
        df_UR_PAGE_QP.index = PAGE_LAYOUT
        df_UR_PAGE_QP.to_csv(folder_results / "UR_PAGE_QP.csv")
        df_UR_PAGE_QP_complexity_1 = pd.DataFrame(dict_UR_PAGE_QP_complexity_1)
        df_UR_PAGE_QP_complexity_1.index = PAGE_LAYOUT
        df_UR_PAGE_QP_complexity_1.to_csv(folder_results / "UR_PAGE_QP_complexity_1.csv")
        df_UR_PAGE_QP_complexity_2 = pd.DataFrame(dict_UR_PAGE_QP_complexity_2)
        df_UR_PAGE_QP_complexity_2.index = PAGE_LAYOUT
        df_UR_PAGE_QP_complexity_2.to_csv(folder_results / "UR_PAGE_QP_complexity_2.csv")
        df_UR_PAGE_QP_complexity_3 = pd.DataFrame(dict_UR_PAGE_QP_complexity_3)
        df_UR_PAGE_QP_complexity_3.index = PAGE_LAYOUT
        df_UR_PAGE_QP_complexity_3.to_csv(folder_results / "UR_PAGE_QP_complexity_3.csv")

        df_UR_PAGE_DED = pd.DataFrame(dict_UR_PAGE_DED)
        df_UR_PAGE_DED.index = ["0", "1", ">1"]
        df_UR_PAGE_DED.to_csv(folder_results / "UR_PAGE_DED.csv")
        df_UR_PAGE_DED_complexity_1 = pd.DataFrame(dict_UR_PAGE_DED_complexity_1)
        df_UR_PAGE_DED_complexity_1.index = ["0", "1", ">1"]
        df_UR_PAGE_DED_complexity_1.to_csv(folder_results / "UR_PAGE_DED_complexity_1.csv")
        df_UR_PAGE_DED_complexity_2 = pd.DataFrame(dict_UR_PAGE_DED_complexity_2)
        df_UR_PAGE_DED_complexity_2.index = ["0", "1", ">1"]
        df_UR_PAGE_DED_complexity_2.to_csv(folder_results / "UR_PAGE_DED_complexity_2.csv")
        df_UR_PAGE_DED_complexity_3 = pd.DataFrame(dict_UR_PAGE_DED_complexity_3)
        df_UR_PAGE_DED_complexity_3.index = ["0", "1", ">1"]
        df_UR_PAGE_DED_complexity_3.to_csv(folder_results / "UR_PAGE_DED_complexity_3.csv")
        
        
        print(f"Files saved in {folder_results}")
        print("-" * 100)

        # break

        


if __name__ == "__main__":
    generate_analysis_report(
        # dataset="DUDE_MM", 
        # images_path="/data2/dnapolitano/VQA/data/DUDE_train-val-test_binaries/images/train"
        dataset="MPDocVQA_MM", 
        images_path="/data2/dnapolitano/VQA/data/mpdocvqa/images"
    )
