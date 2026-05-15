import base64
import os
import pandas as pd
import torch
import PIL.Image
import google.generativeai as genai
import json
import argparse
import random
import time
import torchvision.transforms as transforms
from collections import deque
from datetime import datetime
from transformers import AutoTokenizer, AutoModel

GEMINI_MODEL = "gemini-2.5-flash"
LOCAL_MODEL = "OpenGVLab/InternVL3_5-8B"


class AnswerabilityVerifier:
    def __init__(self, config_path=None):
        # Load configuration
        if config_path is None:
            config_path = "corruption-scripts/config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"Loaded configuration from {config_path}")

        verification_config = config.get("verification", {})
        
        self.model_name = verification_config.get("model_name", " ")
        self.verification_percentage = verification_config.get(
            "verification_percentage", 100
        )

        self.provider = verification_config.get("provider", "local")
        if self.provider == 'gemini':
            api_key = verification_config.get("api_key")
            genai.configure(api_key=api_key)

            # Add rate limiting properties
            self.api_calls = deque()  # Store timestamps of API calls
            self.max_calls_per_minute = 15
            self.call_window = 60  # seconds

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            f"AnswerabilityVerifier using device: {self.device} with provider: {self.provider}"
        )
        print(f"Will verify {self.verification_percentage}% of questions")

        if self.provider == 'local':
            self._init_local_model(verification_config)

        # Get input and output file paths from config
        self.input_file = verification_config.get("verification_input_file")
        self.output_file = verification_config.get("verification_output_file")

        # Get log file path from config
    
    def _init_local_model(self, verification_config):
        model_name = LOCAL_MODEL
        cache_dir = os.environ.get("HF_HOME", "/data1/hf_cache/models")
        self.local_input_size = 448
        self.local_max_num = 12
        self.local_max_tokens = 512

        print(f"Loading local model {model_name}...")
        self.local_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False, cache_dir=cache_dir
        )
        self.local_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_flash_attn=True,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        ).eval()

        self.local_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((self.local_input_size, self.local_input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("Local model loaded successfully")

    def _load_image_local(self, image_path):
        image = PIL.Image.open(image_path).convert("RGB")
        tiles = self._dynamic_preprocess(image, image_size=self.local_input_size, use_thumbnail=True, max_num=self.local_max_num)
        pixel_values = torch.stack([self.local_transform(t) for t in tiles])
        return pixel_values.to(torch.bfloat16)

    def _dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = sorted(set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        ), key=lambda x: x[0] * x[1])

        best_ratio = (1, 1)
        best_diff = float("inf")
        area = orig_width * orig_height
        for ratio in target_ratios:
            diff = abs(aspect_ratio - ratio[0] / ratio[1])
            if diff < best_diff or (diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]):
                best_diff = diff
                best_ratio = ratio

        tw, th = image_size * best_ratio[0], image_size * best_ratio[1]
        resized = image.resize((tw, th))
        tiles = []
        for i in range(best_ratio[0] * best_ratio[1]):
            col = i % best_ratio[0]
            row = i // best_ratio[0]
            box = (col * image_size, row * image_size, (col + 1) * image_size, (row + 1) * image_size)
            tiles.append(resized.crop(box))
        if use_thumbnail and len(tiles) != 1:
            tiles.append(image.resize((image_size, image_size)))
        return tiles

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _check_rate_limit(self):
        """Ensures we don't exceed 15 API calls per minute"""
        now = time.time()

        # Remove timestamps older than our window
        while self.api_calls and self.api_calls[0] < now - self.call_window:
            self.api_calls.popleft()

        # If we've hit our limit, sleep until we can make another call
        if len(self.api_calls) >= self.max_calls_per_minute:
            sleep_time = self.api_calls[0] + self.call_window - now
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Add current timestamp to our queue
        self.api_calls.append(now)

    def verify_answerability(self, question, image_path, original_entities=None, corrupted_entities=None, ocr_text=""):
        """
        Verify if a question is answerable based on the image content.
        Args:
            question (str): The question to verify
            image_path (str): Path to the image
            original_entities (list, optional): List of original entities
            corrupted_entities (list, optional): List of corrupted entities
            ocr_text (str, optional): OCR text from the image
        """

        # Add rate limiting for Gemini calls
        if self.provider == "gemini":
            self._check_rate_limit()
            try:
                image = PIL.Image.open(image_path)
                model = genai.GenerativeModel(model_name=self.model_name)

                # Create entities string if entities are provided
                entities_string = ""
                if original_entities and corrupted_entities:
                    for orig, corr in zip(original_entities, corrupted_entities):
                        entities_string += f"{orig} --> {corr}\n"

                entities_section = ""
                if entities_string:
                    entities_section = f"In addition here we provide the original entities found in the question and the corrupted ones in order to allow you to place special focus on the corrupted ones. The entities are reported with the format: ORIGINAL --> CORRUPTED:\n{entities_string}"

                prompt = (
                    "You are an expert in Visual Question Answering on Document images. "
                    "We are working on a project to verify the answerability of questions based on the information provided in a given image. "
                    "In detail we have taken questions from a multipage VQA dataset and we have corrupted the questions based on the entities found in the whole document associated to the question. "
                    "Now, given the corrupted question and each image of the document, we want to verify if the question is answerable based solely on the information provided in the given image. "
                    "Your task is to help us to determine if the following corrupted question is answerable based solely on the information provided in the given image. "
                    "The question answer must be explicitly stated in the image. "
                    f"In order to have a better document understanding, we extracted the following OCR text from the document:\n{ocr_text}\n\n"
                    f"{entities_section}\n\n"
                    "Respond with a structured response in JSON format with the following fields:\n"
                    "{\n"
                    '    "verification_result": "true if the question is answerable based solely on the information provided in the given image, or \'false\' if it\'s not answerable",\n'
                    '    "question_answer": "The answer to the question or only the words \'not found\' if the answer is not explicitly stated in the image"\n'
                    "}\n"
                    "Return only the JSON response. Without any other text or explanation.\n"
                    f"\nQuestion: {question}"
                )

                response = model.generate_content([prompt, image])
                try:
                    # Clean the response text by removing markdown code blocks
                    clean_response = response.text.strip()
                    if clean_response.startswith("```"):
                        clean_response = clean_response.split("```")[1]
                    if clean_response.startswith("json"):
                        clean_response = clean_response[4:]
                    clean_response = clean_response.strip()

                    json_response = json.loads(clean_response)
                    response = json_response.get("verification_result", "false").lower()

                    # Store the full response for verification_result
                    self.last_response = json_response

                    

                except json.JSONDecodeError:
                    response = "false"
                    self.last_response = {
                        "verification_result": "False",
                        "question_answer": "Error parsing response",
                    }

            except Exception:
                return False

        elif self.provider == "local":
            try:
                entities_string = ""
                if original_entities and corrupted_entities:
                    for orig, corr in zip(original_entities, corrupted_entities):
                        entities_string += f"{orig} --> {corr}\n"

                entities_section = ""
                if entities_string:
                    entities_section = (
                        "In addition here we provide the original entities found in the question "
                        "and the corrupted ones in order to allow you to place special focus on the corrupted ones. "
                        f"The entities are reported with the format: ORIGINAL --> CORRUPTED:\n{entities_string}"
                    )

                prompt = (
                    "You are an expert in Visual Question Answering on Document images. "
                    "We are working on a project to verify the answerability of questions based on the information provided in a given image. "
                    "In detail we have taken questions from a multipage VQA dataset and we have corrupted the questions based on the entities found in the whole document associated to the question. "
                    "Now, given the corrupted question and each image of the document, we want to verify if the question is answerable based solely on the information provided in the given image. "
                    "Your task is to help us to determine if the following corrupted question is answerable based solely on the information provided in the given image. "
                    "The question answer must be explicitly stated in the image. "
                    f"In order to have a better document understanding, we extracted the following OCR text from the document:\n{ocr_text}\n\n"
                    f"{entities_section}\n\n"
                    "Respond with a structured response in JSON format with the following fields:\n"
                    "{\n"
                    '    "verification_result": "true if the question is answerable based solely on the information provided in the given image, or \'false\' if it\'s not answerable",\n'
                    '    "question_answer": "The answer to the question or only the words \'not found\' if the answer is not explicitly stated in the image"\n'
                    "}\n"
                    "Return only the JSON response. Without any other text or explanation.\n"
                    f"\nQuestion: {question}"
                )

                pixel_values = self._load_image_local(image_path).to(self.device)
                query = f"<image>\n{prompt}"
                generation_config = dict(max_new_tokens=self.local_max_tokens, do_sample=False)
                raw_response, _ = self.local_model.chat(
                    self.local_tokenizer, pixel_values, query, generation_config,
                    history=None, return_history=True,
                )

                clean_response = raw_response.strip()
                if clean_response.startswith("```"):
                    clean_response = clean_response.split("```")[1]
                if clean_response.startswith("json"):
                    clean_response = clean_response[4:]
                clean_response = clean_response.strip()

                try:
                    json_response = json.loads(clean_response)
                    response = json_response.get("verification_result", "false").lower()
                    self.last_response = json_response
                except json.JSONDecodeError:
                    response = "false"
                    self.last_response = {
                        "verification_result": "False",
                        "question_answer": "Error parsing response",
                    }

            except Exception:
                return False

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        return response == "true"

    def get_sorted_ocr_text(self, layout_analysis):
        """Extract and sort OCR text based on y-position from layout analysis"""
        objects = []
        for obj_info in layout_analysis.values():
            if isinstance(obj_info, dict) and "BBOX" in obj_info:
                y_pos = obj_info["BBOX"][1]  # Get y-position from BBOX
                ocr_text = obj_info.get("OCR", "").strip()
                if ocr_text:
                    objects.append((y_pos, ocr_text))

        # Sort by y-position and join texts
        sorted_objects = sorted(objects, key=lambda x: x[0])
        return "\n".join(text for _, text in sorted_objects)

    def get_relevant_pages(self, item):
        """Determine relevant pages to check based on corrupted entities"""
        # Get all available pages and sort them alphabetically
        # This ensures pages like "sslg0227_p0.jpg", "sslg0227_p1.jpg" are in correct order
        all_pages = sorted(
            list(item.get("layout_analysis", {}).get("pages", {}).keys())
        )
        if not all_pages:
            return []

        # Get indices of pages with corrupted entities
        relevant_indices = set()
        for entity in item.get("corrupted_entities", []):
            page_id = entity.get("page_id")
            if page_id and page_id in all_pages:
                idx = all_pages.index(page_id)
                # Add the page index and its adjacent indices
                relevant_indices.update([idx - 1, idx, idx + 1])

        # Filter valid indices (remove negative or out of bounds indices)
        valid_indices = {idx for idx in relevant_indices if 0 <= idx < len(all_pages)}

        # Get the final list of pages using the valid indices
        final_pages = [all_pages[idx] for idx in valid_indices]

        return final_pages

    def verify_questions_from_file(self):
        """
        Verify answerability for all questions in the input JSON file
        """
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        # Load questions from JSON
        with open(self.input_file, "r") as f:
            data = json.load(f)

        # Calculate how many questions to verify
        total_questions = len(data["corrupted_questions"])
        num_to_verify = int(total_questions * self.verification_percentage / 100)

        # Randomly select questions to verify
        question_IDs_to_verify = random.sample(range(total_questions), num_to_verify)

        # Create new list for verified questions
        verified_questions = []

        # Process selected questions
        for current_idx, question_id in enumerate(question_IDs_to_verify, 1):
            item = data["corrupted_questions"][question_id]
            question = item["corrupted_question"]

            # Get relevant pages to check
            relevant_pages = self.get_relevant_pages(item)
        
            # Get image paths only for relevant pages
            image_paths = []
            for page_id, page_info in item["layout_analysis"]["pages"].items():
                if page_id not in relevant_pages:
                    continue
                image_path = page_info["image_path"]
                if not os.path.exists(image_path):
                    continue
                # Get OCR text for this page
                layout_analysis = page_info.get("layout_analysis", {})
                ocr_text = self.get_sorted_ocr_text(layout_analysis)
                image_paths.append((image_path, ocr_text))

            if not image_paths:
                continue

            # Verify if the question is answerable
            is_answerable = False
            answerable_result = None

            for image_path, ocr_text in image_paths:
                if self.verify_answerability(question, image_path, ocr_text=ocr_text):
                    is_answerable = True
                    # Store the successful verification result
                    answerable_result = {
                        "verification_result": getattr(self, "last_response", {}).get(
                            "verification_result", "True"
                        ),
                        "question_answer": getattr(self, "last_response", {}).get(
                            "question_answer", "Answer found"
                        ),
                        "image_path": image_path,
                    }
                    break  # Stop checking other images once we find an answerable one

            # Update verification result with the appropriate information
            if is_answerable:
                item["verification_result"] = answerable_result
            else:
                # If no answerable result found, store the last verification attempt
                item["verification_result"] = {
                    "verification_result": "False",
                    "question_answer": "Not found in any relevant page",
                    "image_path": image_paths[-1][0] if image_paths else None,
                }

            verified_questions.append(item)

        # Create output data structure with only verified questions
        output_data = {"corrupted_questions": verified_questions}

        # Save results to the configured output file
        with open(self.output_file, "w") as f:
            json.dump(output_data, f, indent=2)

    def __del__(self):
        """Destructor to ensure log file is closed"""
        if hasattr(self, "log_file"):
            self.log_file.close()


def main():
    parser = argparse.ArgumentParser(description='Run question corruption.')
    parser.add_argument('--config', type=str, help='Path to the configuration file', default="code/corruption-scripts/config.json")
    args = parser.parse_args()

    try:
        # Initialize verifier with default config
        verifier = AnswerabilityVerifier(config_path=args.config)

        # Process all questions in the input file
        verifier.verify_questions_from_file()

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
