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
from collections import deque
from datetime import datetime


class AnswerabilityVerifier:
    def __init__(self, config_path=None):
        # Load configuration
        if config_path is None:
            config_path = "code/corruption-scripts/config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        verification_config = config.get("verification", {})
        provider = verification_config.get("provider", "openai")
        api_key = verification_config.get("api_key")
        self.model_name = verification_config.get("model_name", " ")
        self.verification_percentage = verification_config.get(
            "verification_percentage", 100
        )

        genai.configure(api_key=api_key)

        self.default_provider = provider
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            f"AnswerabilityVerifier using device: {self.device} with provider: {provider}"
        )
        print(f"Will verify {self.verification_percentage}% of questions")

        # Get input and output file paths from config
        self.input_file = verification_config.get("verification_input_file")
        self.output_file = verification_config.get("verification_output_file")

        # Get log file path from config

        # Add rate limiting properties
        self.api_calls = deque()  # Store timestamps of API calls
        self.max_calls_per_minute = 15
        self.call_window = 60  # seconds

    

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

    def verify_answerability(self, question, image_path, original_entities=None, corrupted_entities=None, ocr_text="", provider=None):
        """
        Verify if a question is answerable based on the image content.
        Args:
            question (str): The question to verify
            image_path (str): Path to the image
            original_entities (list, optional): List of original entities
            corrupted_entities (list, optional): List of corrupted entities
            ocr_text (str, optional): OCR text from the image
            provider (str, optional): The provider to use ('openai' or 'gemini')
        """
        # Use default provider if none specified
        provider = provider or self.default_provider

        # Add rate limiting for Gemini calls
        if provider == "gemini" or (
            provider is None and self.default_provider == "gemini"
        ):
            self._check_rate_limit()

       
        if provider == "gemini":
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

            except Exception as e:
                return False

        else:
            raise ValueError(f"Unsupported provider: {provider}")

        result_message = "Answerable" if response == "true" else "Not Answerable"
        return response == "true"

    def verify_unanswerable(self, question, image_paths):
        for image_path in image_paths:
            if not self.verify_answerability(question, image_path):
                return False  # Question is answerable for at least one image
        return True  # Question is unanswerable for all images

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
        questions_to_verify = random.sample(range(total_questions), num_to_verify)


        # Create new list for verified questions
        verified_questions = []

        # Process selected questions
        for current_idx, question_idx in enumerate(questions_to_verify, 1):
            item = data["corrupted_questions"][question_idx]
            question = item["corrupted_question"]



            # Get relevant pages to check
            relevant_pages = self.get_relevant_pages(item)
            

            # Get image paths only for relevant pages
            image_paths = []
            for page_id, page_info in item["layout_analysis"]["pages"].items():
                if page_id in relevant_pages:
                    image_path = page_info["image_path"]
                    if os.path.exists(image_path):
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
