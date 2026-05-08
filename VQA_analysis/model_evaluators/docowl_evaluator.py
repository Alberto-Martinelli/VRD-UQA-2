import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import datetime
import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torchvision.transforms as transforms
from difflib import SequenceMatcher
from tqdm.auto import tqdm
import argparse

class DocOwlEvaluator:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)
        
        # Get DocOwl-specific configuration - now nested under "llm"
        self.model_config = self.config["open_source_models"]["llm"]["docowl"]
        self.sampling_percentage = self.config.get("sampling_percentage", 100)
        self.unable_to_respond_aware = self.config.get("unable_to_respond_aware", True)
        self.initialize_model()

    def initialize_model(self):
        print("Initializing DocOwl model...")
        print("Model configuration:", self.model_config)
        model_name = self.model_config["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir="/data1/hf_cache/models",
        )
        self.model.init_processor(
            tokenizer=self.tokenizer,
            basic_image_size=self.model_config.get("input_size", 504),
            crop_anchors=self.model_config.get("crop_anchors", "grid_12"),
        )
        self.max_tokens = self.model_config.get("max_tokens", 1024)
        self.input_size = self.model_config.get("input_size", 448)

        # Initialize image transform
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
            ]
        )
        print("DocOwl model initialized successfully")

    def _cleanup_model(self):
        if hasattr(self, "model"):
            del self.model
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    def _create_prompt(self, question, ocr_text=None):
        unable_to_respond_line = "- If uncertain, return 'Unable to determine'\n- If you can't find the answer, return 'Unable to determine'" if self.unable_to_respond_aware else ""
        if ocr_text:
            return (
                f"You are an AI assistant specialized in analyzing document images and text. "
                f"Your task is to answer questions about the document image content precisely.\n\n"
                f"For this question, you have the following OCR text:\n{ocr_text}\n\n"
                f"Guidelines:\n"
                f"- Provide concise, focused answers (single word or short phrase preferred)\n"
                f"- Base your answer on both the image and the provided OCR text\n"
                f"{unable_to_respond_line}\n"
                f"Question: {question}\n"
            )
        return (
            f"You are an AI assistant specialized in analyzing document images. "
            f"Your task is to answer questions about the document image content precisely.\n\n"
            f"Guidelines:\n"
            f"- Provide concise, focused answers (single word or short phrase preferred)\n"
            f"- Base your answer solely on what you see in the image\n"
            f"{unable_to_respond_line}\n"
            f"Question: {question}"
        )

    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)

    def get_sorted_ocr_text(self, layout_analysis):
        """Extract and sort OCR text by bounding box position"""
        ocr_items = []
        for obj in layout_analysis.values():
            if isinstance(obj, dict) and "OCR" in obj and "BBOX" in obj:
                bbox = obj["BBOX"]
                ocr_items.append((bbox[1], bbox[0], obj["OCR"]))  # y, x, text

        # Sort by y coordinate first, then x coordinate
        ocr_items.sort()
        return "\n".join(item[2] for item in ocr_items)

    def generate_answer(self, question, image_paths, ocr_text=None):
        try:
            window_size = self.model_config.get("batch_size", 1)
            if window_size > 1:
                stride = self.model_config.get("stride", window_size // 2)  # Default stride is half the window size
            else:
                stride = 1
            total_images = len(image_paths)

            # Calculate total number of windows
            total_windows = max(1, (total_images - window_size + stride) // stride)

            # print(
            #     f"\nProcessing {total_images} images with window size {window_size}, "
            #     f"stride {stride} ({total_windows} window{'s' if total_windows > 1 else ''})"
            # )
            all_responses = []

            # Process images using moving window
            for window_idx in range(total_windows):
                start_idx = window_idx * stride
                end_idx = min(start_idx + window_size, total_images)
                window_paths = image_paths[start_idx:end_idx]
                
                # Handle last window to ensure all images are processed
                if window_idx == total_windows - 1 and end_idx < total_images:
                    window_paths = image_paths[-window_size:]


                # print(f"\nBatch {batch_idx + 1}/{total_batches}")
                # print(f"Processing images {start_idx + 1}-{end_idx} of {total_images}")
                # print(f"Images in this batch: {len(batch_paths)}")
                # for idx, path in enumerate(batch_paths, 1):
                #     print(f"  {idx}. {os.path.basename(path)}")

                # print(f"\nBatch {window_idx + 1}/{total_windows}")
                # print(f"Processing images {start_idx + 1}-{end_idx} of {total_images}")
                # print(f"Images in this batch: {len(window_paths)}")
                batch_paths = window_paths
                # for idx, path in enumerate(batch_paths, 1):
                #     print(f"  {idx}. {os.path.basename(path)}")

                # Get OCR text for this batch if available
                batch_ocr = None
                if ocr_text:
                    batch_ocr = []
                    for page_idx, path in enumerate(batch_paths, start_idx):
                        page_num = page_idx + 1
                        page_ocr = ocr_text.get(path, "")  # Get OCR text for specific page
                        if page_ocr:
                            batch_ocr.append(f"Page {page_num}:\n{page_ocr}")
                    batch_ocr = "\n\n".join(batch_ocr) if batch_ocr else None

                try:
                    # print("=== Loading Images ===")
                    # Format query with OCR if available
                    # query = question
                    # if batch_ocr:
                    #     query = f"{question}\nHere is the OCR text from the document:\n{batch_ocr}"
                    question_prompt = self._create_prompt(question, batch_ocr)

                    # Create messages format as per official implementation
                    messages = [
                        {
                            "role": "USER",
                            "content": "<|image|>" * len(batch_paths) + question_prompt,
                        }
                    ]

                    # Generate response using chat method
                    response = self.model.chat(
                        messages=messages, images=[el for el in batch_paths], tokenizer=self.tokenizer
                    )

                    all_responses.append(
                        {
                            "pages": batch_paths,
                            "answer": response,
                        }
                    )

                except Exception as e:
                    print(f"Error in batch {window_idx + 1}: {str(e)}")
                    all_responses.append(
                        {
                            "pages": batch_paths,
                            "answer": "Error in processing batch",
                            "error": str(e),
                        }
                    )

            return {
                "answer": all_responses,
                "query": question,
                "image_paths": image_paths,
                "analysis_type": f"window_size_{window_size}",
            }

        except Exception as e:
            print(f"Error in generate_answer: {str(e)}")
            print(f"Full error: {traceback.format_exc()}")
            return {
                "answer": "Unable to determine",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def _save_results(self, data):
        # Construct base path
        base_path = f"/VQA_analysis/models/results/{self.config["dataset"]}/LLM"
        
        # Get window size from config
        window_size = self.model_config.get("batch_size", 1)
        
        # Create processing type folder name
        processing_folder = f"results_w{window_size}"
        
        # Add OCR and UNABLE flags if enabled
        if self.config["ocr_enabled"]:
            processing_folder += "_OCR"
        if not self.config["unable_to_respond_aware"]:
            processing_folder += "_UNABLE"
        
        # Create output filename with model name
        output_filename = f"{self.model_config['name']}_vqa_analysis_results.json"
        
        # Combine paths
        output_dir = os.path.join(base_path, processing_folder, "original")
        
        # Create directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Full path for the output file
        output_file = os.path.join(output_dir, output_filename)
        
        try:
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Results successfully saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")

    def evaluate(self):
        try:
            print("\nStarting DocOwl evaluation...")

            # Load input data - check if it's a previous output file
            with open(self.config["input_file"]) as f:
                data = json.load(f)
                if "corrupted_questions" not in data:
                    print("Input file appears to be a previous output. Using existing structure.")
                else:
                    print(f"Successfully loaded input file: {self.config['input_file']}")

            # Sample questions from the appropriate structure
            questions = data.get("corrupted_questions", data)
            total_questions = len(questions)
            num_samples = int(total_questions * (self.sampling_percentage / 100))

            if self.sampling_percentage < 100:
                questions_to_process = random.sample(questions, num_samples)
                print(f"Sampled {num_samples} questions ({self.sampling_percentage}%) for evaluation")
            else:
                questions_to_process = questions
                print("Processing 100% of questions (no sampling)")

            processed_count = 0
            success_count = 0
            error_count = 0

            for item in tqdm(questions_to_process):
                try:
                    processed_count += 1
                    # print(f"\nProcessing question {processed_count}/{len(questions_to_process)}")

                    if "verification_result" not in item:
                        item["verification_result"] = {}
                    if "vqa_results" not in item["verification_result"]:
                        item["verification_result"]["vqa_results"] = []

                    question = item.get("corrupted_question", item.get("question"))
                    pages = item.get("layout_analysis", {}).get("pages", item.get("pages", {}))

                    image_paths = []
                    for page_id in pages:
                        image_filename = os.path.basename(page_id)
                        image_path = os.path.join(self.config["images_base_path"], image_filename)
                        image_paths.append(image_path)

                    # print(f"Question: {question[:100]}...")
                    # print(f"Number of pages to analyze: {len(image_paths)}")

                    # Get OCR text if enabled
                    ocr_text = None
                    if self.config.get("ocr_enabled", False):
                        # print("Extracting OCR text...")
                        ocr_text = {}
                        for page_id in pages:
                            # Navigate through the nested structure correctly
                            page_layout = pages[page_id]["layout_analysis"]
                            page_ocr = self.get_sorted_ocr_text(page_layout)
                            if page_ocr:
                                # Use the full path as the key since that's what we use in generate_answer
                                image_filename = os.path.basename(page_id)
                                image_path = os.path.join(self.config["images_base_path"], image_filename)
                                ocr_text[image_path] = page_ocr
                                # print(f"Extracted OCR text for page: {image_filename}")
                            else:
                                print(f"No OCR text found for page: {image_filename}")

                    result = self.generate_answer(question, image_paths, ocr_text)

                    vqa_result = {
                        "model_type": "docowl",
                        "model_config": {
                            "batch_size": self.model_config.get("batch_size", 1),
                            "max_tokens": self.max_tokens
                        },
                        "ocr_enabled": bool(ocr_text),
                        "question": question,
                        "answer": result.get("answer", "Unable to determine"),
                        "image_paths": result.get("image_paths", []),
                        "timestamp": datetime.datetime.now().isoformat(),
                        "analysis_type": result.get("analysis_type", "unknown")
                    }

                    if "error" in result:
                        vqa_result["error"] = result["error"]
                        vqa_result["traceback"] = result.get("traceback", "")
                        error_count += 1
                    else:
                        success_count += 1

                    item["verification_result"]["vqa_results"].append(vqa_result)

                    # Save intermediate results
                    # if processed_count % 10 == 0:
                    #     self._save_results(data)
                    #     print("Intermediate results saved")

                except Exception as e:
                    print(f"Error processing question: {str(e)}")
                    print(f"Full error: {traceback.format_exc()}")
                    error_count += 1

            # Final statistics
            print(f"\nProcessing completed:")
            print(f"Total questions processed: {processed_count}")
            print(f"Successful generations: {success_count}")
            print(f"Errors encountered: {error_count}")
            if processed_count > 0:
                print(f"Success rate: {(success_count/processed_count)*100:.2f}%")

            # Save final results
            self._save_results(data)
            output_file = self.model_config["name"]+"_"+self.config["output_file"]
            if self.config["ocr_enabled"] and not self.config["unable_to_respond_aware"]:
                output_file = output_file.replace(".json", "_OCR_UNABLE.json")
            elif self.config["ocr_enabled"]:
                output_file = output_file.replace(".json", "_OCR.json")
            elif not self.config["unable_to_respond_aware"]:
                output_file = output_file.replace(".json", "_UNABLE.json")
            print(f"Final results saved to {output_file}")

        except Exception as e:
            print(f"Critical error in evaluate: {str(e)}")
            print(f"Full error: {traceback.format_exc()}")
        finally:
            self._cleanup_model()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    evaluator = DocOwlEvaluator(args.config_path)
    evaluator.evaluate()

if __name__ == "__main__":
    main() 