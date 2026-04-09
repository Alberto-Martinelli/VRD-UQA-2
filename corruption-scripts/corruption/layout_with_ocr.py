import supervision as sv
from qwen_vl_utils import process_vision_info
import tqdm
import json
import logging
import os
from pathlib import Path
import random
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
import torch.serialization
import torch
from PIL import Image
import numpy as np
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForVision2Seq,
    AutoModelForImageTextToText,
    AutoProcessor,
)
import io
from tqdm.auto import tqdm

# Store the original torch.load function
original_torch_load = torch.load


class DocumentAnalyzer:
    def __init__(self, model_config=None, patch_saving_dir=None, layout_saving_dir=None):
        """
        Sets up 3 models:
        1) DocLayout-YOLO-DocStructBench (output: bboxes around each element + element type)
        2) GOT-OCR2_0 (Extracts text)
        3) Qwen2-VL-2B-Instruct (Used to describe tables, figures, charts)
        """
        self.model_config = model_config or {
            # Default configuration if none provided
            "model_name": "Qwen/Qwen2-VL-2B-Instruct",
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 720 * 28 * 28,
        }
        self.patch_saving_dir = patch_saving_dir
        self.layout_saving_dir = layout_saving_dir
        self.device, self.device_type = self._detect_device()

        self.model = None    # DocLayout-YOLO
        self.ocr_model = None    # GOT-OCR2_0
        self.qwen_model = None    # Qwen2-VL-2B-Instruct
        self.tokenizer = None
        self.processor = None

        # # Define color palette for visualization
        # self.class_colors = [
        #     sv.Color(255, 0, 0),  # Red
        #     sv.Color(0, 255, 0),  # Green
        #     sv.Color(0, 0, 255),  # Blue
        #     sv.Color(255, 255, 0),  # Yellow
        #     sv.Color(255, 0, 255),  # Magenta
        #     sv.Color(0, 255, 255),  # Cyan
        #     sv.Color(128, 0, 128),  # Purple
        #     sv.Color(128, 128, 0),  # Olive
        #     sv.Color(128, 128, 128),  # Gray
        #     sv.Color(0, 128, 128),  # Teal
        #     sv.Color(128, 0, 0),  # Maroon
        # ]

        # Define class names mapping
        self.class_names = {
            0: "title",
            1: "plain text",
            2: "abandon",
            3: "figure",
            4: "figure_caption",
            5: "table",
            6: "table_caption",
            7: "table_footnote",
            8: "isolate_formula",
            9: "formula_caption",
        }

    # ----------- Model Loading -----------
    def load_models(self):
        """Load all models."""
        self._load_layout_model()
        self._load_ocr_model()
        self._load_qwen_model()

    def _detect_device(self):
        """ Dynamically determine the best available hardware """
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        # elif torch.backends.mps.is_available():
        #     return torch.device("mps"), "mps"
        return torch.device("cpu"), "cpu"

    def _load_layout_model(self):
        """Load DocLayout-YOLO for element detection."""
        filepath = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        )

        # Override torch.load with a modified version
        def custom_load(f, *args, **kwargs):
            kwargs.pop("weights_only", None)
            return original_torch_load(f, *args, **kwargs)

        torch.load = custom_load
        torch.load = custom_load
        self.model = YOLOv10(filepath)

        # Restore original torch.load
        torch.load = original_torch_load

    def _load_ocr_model(self):
        """Load GOT-OCR2_0 for text extraction."""
        # Initialize OCR model for CUDA
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ucaslcl/GOT-OCR2_0", trust_remote_code=True
        )
        self.ocr_model = AutoModel.from_pretrained(
            "ucaslcl/GOT-OCR2_0",
            trust_remote_code=True,
            use_safetensors=True,
            pad_token_id=self.tokenizer.eos_token_id,
            device_map=self.device_type,
        )
        self.ocr_model = self.ocr_model.eval().to(self.device)

    def _load_qwen_model(self):
        """Load Qwen2-VL-2B-Instruct for table/figure description."""
        # Initialize Qwen2-VL model with config parameters
        # Force float16 for Mac MPS compatibility, otherwise use auto
        model_dtype = torch.float16 if self.device_type == "mps" else "auto"
        # self.qwen_model = AutoModelForVision2Seq.from_pretrained(
        #     self.model_config["model_name"], 
        #     torch_dtype=model_dtype, 
        #     device_map="auto"
        # )
        # Note: We removed device_map="auto" to prevent the fatal disk-offloading crash
        self.qwen_model = AutoModelForImageTextToText.from_pretrained(
            self.model_config["model_name"], 
            torch_dtype=model_dtype
        ).to(self.device)

        # Set processor with specific pixel limits from config
        min_pixels = self.model_config.get("min_pixels", 256 * 28 * 28)
        max_pixels = self.model_config.get("max_pixels", 720 * 28 * 28)
        self.processor = AutoProcessor.from_pretrained(
            self.model_config["model_name"], min_pixels=min_pixels, max_pixels=max_pixels
        )

    # ----------- Filtering Methods -----------
    def box_area(self, box):
        return max(0, (box[2] - box[0])) * max(0, (box[3] - box[1]))

    def intersection_area(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        return interW * interH

    def is_mostly_inside(self, boxA, boxB, threshold=0.8):
        interA = self.intersection_area(boxA, boxB)
        areaB = self.box_area(boxB)
        if areaB == 0:
            return False
        return (interA / areaB) >= threshold

    def filter_boxes(self, det_res, threshold):
        if len(det_res[0].boxes) == 0:
            return det_res

        boxes = det_res[0].boxes.xyxy
        box_data = []
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            box_data.append((idx, (x1, y1, x2, y2), area))

        box_data.sort(key=lambda x: x[2], reverse=True)
        covered_boxes = set()

        for i, (idx1, box1, _) in enumerate(box_data):
            if idx1 in covered_boxes:
                continue
            for j, (idx2, box2, _) in enumerate(box_data[i + 1 :], i + 1):
                if idx2 in covered_boxes:
                    continue
                if self.is_mostly_inside(box1, box2, threshold=threshold):
                    covered_boxes.add(idx2)

        keep_mask = torch.tensor(
            [i not in covered_boxes for i in range(len(boxes))], dtype=torch.bool
        )
        det_res[0].boxes = det_res[0].boxes[keep_mask]
        return det_res

    # ----------- Visual Content Analysis -----------
    def analyze_visual_content(self, image_path, content_type="image", prompt=None):
        if prompt is None:
            prompt = f"Describe this {content_type} in detail."

        try:
            # Prepare messages in the required format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            # Generate output
            with torch.inference_mode():
                generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=1000)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                return output_text[0] if output_text else "Error: No output generated"

        except Exception as e:
            logging.error(f"Error in analyze_visual_content: {str(e)}")
            return f"Error analyzing {content_type}: {str(e)}"

    def _extract_text(self, patch_path, class_name):
        """Extract text from a patch using the appropriate model.
        - Tables/figures → Qwen2-VL (visual analysis), with GOT-OCR fallback
        - Everything else → GOT-OCR directly
        """

        # Enhanced error handling for visual content analysis
        if class_name in ["table", "figure"]:
            prompt = self._get_visual_prompt(class_name)
            try:
                result = self.analyze_visual_content(
                    patch_path,
                    content_type=class_name,
                    prompt=prompt,
                )

                if not result.startswith("Error"):
                    return result
                logging.warning(f"Visual analysis failed for {class_name}, falling back to OCR")
            except Exception as e:
                logging.error(f"Error in visual analysis for {class_name}: {e}")
        
        # Fallback or default: GOT-OCR
        return self._run_ocr(patch_path)

    def _get_visual_prompt(self, class_name):
        """Return the appropriate prompt for visual content analysis."""
        prompts = {
            "table": (
                "Analyze this table and provide a clear description of its content, "
                "including: 1) what information it contains, 2) how many rows and columns, "
                "3) key data points or trends. Be specific but concise."
            ),
            "figure": (
                "Describe this image in detail, including: 1) what it shows, "
                "2) key visual elements, 3) any text or numbers visible, "
                "4) the overall context or purpose. Be specific but concise."
            ),
        }
        return prompts.get(class_name)

    def _run_ocr(self, patch_path):
        """Run GOT-OCR on a patch image."""
        with self._autocast_context():
            with torch.no_grad():
                return self.ocr_model.chat(self.tokenizer, patch_path, ocr_type="ocr")
    
    def _autocast_context(self):
        """Return the appropriate autocast context for the current device."""
        import contextlib
        # Use autocast for the specific hardware, or null context for CPU
        if self.device_type in ["cuda", "mps"]:
            return torch.autocast(device_type=self.device_type)
        return contextlib.nullcontext()

    def _clear_cache(self):
        """Clear GPU cache if applicable."""
        if self.device_type == "cuda":
            torch.cuda.empty_cache()
        elif self.device_type == "mps":
            torch.mps.empty_cache()

    def crop_and_ocr(self, image_path, boxes, classes):
        """
        Step 3) For each detected document: crop the region, save the path to disk
                + process tables and figures with analyze_visual_content (Qwen) and other elements with GOT-OCR 2.0
        Step 4) Save JSON with all the results
        """
        # Load the full image
        image = Image.open(image_path)
        results = {}
        os.makedirs(self.patch_saving_dir, exist_ok=True)

        # Get page ID from image path
        page_id = os.path.splitext(os.path.basename(image_path))[0]

        for idx, (box, cls) in enumerate(zip(boxes, classes)):
            try:
                x1, y1, x2, y2 = map(int, box)
                class_id = cls.item()
                class_name = self.class_names.get(class_id, "unknown")

                # Crop and save patch
                cropped = image.crop((x1, y1, x2, y2))
                patch_filename = f"{page_id}_obj{idx}.jpg"
                patch_path = os.path.join(self.patch_saving_dir, patch_filename)
                cropped.save(patch_path)

                # Extract text using the appropriate model
                ocr_result = self._extract_text(patch_path, class_name, prompt)

                results[f"object{idx}"] = {
                    "BBOX": [x1, y1, x2, y2],
                    "ObjectType": f"{class_name}",
                    "ObjectTypeID": f"{int(class_id)}",
                    "OCR": ocr_result,
                    "PatchPath": patch_path,
                }
                print(f"Successfully processed box {idx} ({class_name})")

            except Exception as e:
                print(f"Error processing box {idx}: {str(e)}")
                continue

            # Clear CUDA cache periodically
            if idx % 10 == 0:
                self._clear_cache()
        # Save results
        os.makedirs(self.layout_saving_dir, exist_ok=True)
        json_path = f"{self.layout_saving_dir}/{page_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        return results

    def analyze_pages_for_question(self, question_data):
        """
        Processes each document page for a question
        Step 1) Run DocLayout-YOLO
        Step 2) Filter overlapping boxes
        Call crop_and_ocr() for each page
        """
        results = {"pages": {}}

        # Extract page information
        documents = question_data.get("document", [])
        if not isinstance(documents, list):
            documents = [documents]

        for document in documents:
            try:
                if not document or not os.path.exists(document):
                    logging.warning(f"Document path not found: {document}")
                    continue

                # Check if document has been already analyzed
                doc_path = Path(document)
                doc_name = doc_path.name # "sslg0227_p0.jpg"
                json_path = Path(self.layout_saving_dir) / f"{doc_path.stem}.json" 
                if os.path.exists(json_path):
                    # Load existing analysis
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            page_results = json.load(f)
                        logging.info(f"Loading existing analysis for {document}")
                        results["pages"][doc_name] = {
                            "layout_analysis": page_results,
                            "image_path": document,
                        }
                        continue
                    except Exception as e:
                        logging.warning(
                            f"Error loading existing analysis for {document}: {e}"
                        )
                        # If loading fails, proceed with new analysis

                logging.info(f"Processing document path {document}")

                # Predict layout
                det_res = self.model.predict(document, conf=0.1)

                # Filter overlapping boxes
                det_res = self.filter_boxes(det_res, threshold=0.6)

                # Process layout elements if detected
                if len(det_res[0].boxes) > 0:
                    # Perform OCR and layout analysis
                    page_results = self.crop_and_ocr(
                        document, det_res[0].boxes.xyxy, det_res[0].boxes.cls
                    )

                    # Store results for this page
                    results["pages"][doc_name] = {
                        "layout_analysis": page_results,
                        "image_path": document,
                    }

                # Clear cache after each page based on device
                self._clear_cache()

            except Exception as e:
                logging.error(f"Error in analyze_pages_for_question: {str(e)}")
                logging.error(f"Question data: {question_data}")

        return results

    def process_dataset_questions(self, df, augmented_dataset_path):
        """
        Process dataset questions and save results incrementally.

        Args:
            df: Input dataframe with questions
            augmented_dataset_path: Path to save the results
        """
        # Load existing results if file exists
        processed_data = {}
        if os.path.exists(augmented_dataset_path):
            try:
                with open(augmented_dataset_path, "r", encoding="utf-8") as f:
                    processed_data = json.load(f)
            except json.JSONDecodeError:
                logging.warning(
                    f"Could not load existing results from {augmented_dataset_path}. Starting fresh."
                )

        results = []
        # Iterate through questions
        for index in tqdm(range(len(df)), desc="Analyzing document layouts"):
            row = df.iloc[index]
            question_data = row.to_dict()

            # Use questionId as the unique identifier
            question_key = str(question_data.get("questionId"))
            if not question_key:
                logging.error(f"No questionId found for row {index}")
                continue

            # Skip if already processed
            if question_key in processed_data:
                logging.info(f"Skipping already processed question {question_key}")
                results.append(processed_data[question_key]["layout_analysis"])
                continue

            # Process new question
            layout_results = self.analyze_pages_for_question(question_data)
            results.append(layout_results)

            # Store both the question data and layout analysis
            processed_data[question_key] = {
                "question_data": question_data,
                "layout_analysis": layout_results,
            }

            # Save incrementally
            try:
                with open(augmented_dataset_path, "w", encoding="utf-8") as f:
                    json.dump(processed_data, f, indent=4, ensure_ascii=False)
            except Exception as e:
                logging.error(f"Error saving results to {augmented_dataset_path}: {str(e)}")

        # Add layout analysis results to the dataframe
        df["layout_analysis"] = results

        return df
