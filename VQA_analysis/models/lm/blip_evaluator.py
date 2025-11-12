import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import datetime
import traceback
import re
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
from tqdm.auto import tqdm


class BlipVQAEvaluator:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Get BLIP-specific configuration from the config file (key "blip" under "lm")
        self.model_config = self.config["open_source_models"]["lm"]["blip"]
        self.sampling_percentage = self.config.get("sampling_percentage", 100)
        self.unable_to_respond_aware = self.config.get("unable_to_respond_aware", True)
        self.initialize_model()

    def _create_prompt(self, question):
        # return f"Question answering. Can you answer to this question: {question}? Yes or No"
        return f"Can you answer to this question: {question}? Yes or No"

    def initialize_model(self):
        print("Initializing BLIP model...")
        model_name = self.model_config["model_name"]  # e.g. "Salesforce/blip-vqa-base"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        print("BLIP model initialized successfully")

    def _cleanup_model(self):
        if hasattr(self, "model"):
            del self.model
            torch.cuda.empty_cache()
            import gc

            gc.collect()

    def generate_answer(self, question, image_paths):
        try:
            batch_size = self.model_config.get("batch_size", 1)
            total_images = len(image_paths)
            total_batches = (total_images + batch_size - 1) // batch_size

            # print(
            #     f"\nProcessing {total_images} images with batch size {batch_size} ({total_batches} batch{'es' if total_batches > 1 else ''})"
            # )
            all_responses = []
            prompt = self._create_prompt(question)
            # print(prompt)

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_images)
                batch_paths = image_paths[start_idx:end_idx]

                # print(f"\nBatch {batch_idx + 1}/{total_batches}")
                # print(f"Processing images {start_idx + 1}-{end_idx} of {total_images}")

                responses = []
                for image_path in batch_paths:
                    pil_image = Image.open(image_path).convert("RGB")
                    inputs = self.processor(pil_image, prompt, return_tensors="pt").to(
                        self.device
                    )

                    with torch.no_grad():
                        outputs = self.model.generate(**inputs)

                    answer = self.processor.decode(
                        outputs[0], skip_special_tokens=True
                    ).strip()

                    responses.append(answer)

                all_responses.append(
                    {
                        "pages": batch_paths,
                        "answer": responses[0] if len(responses) == 1 else responses,
                    }
                )

            return {
                "answer": all_responses,
                "query": question,
                "image_paths": image_paths,
                "analysis_type": f"batch_size_{batch_size}",
            }

        except Exception as e:
            print(f"\nCritical error: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return {
                "answer": "Error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def _save_results(self, data):
        # Construct base path
        base_path = f"/VQA_analysis/models/results/{self.config['dataset']}/LM"
        
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
            # print("\nStarting BLIP evaluation...")

            # Load input data
            with open(self.config["input_file"]) as f:
                data = json.load(f)
                print(f"Successfully loaded input file: {self.config['input_file']}")

            total_questions = len(data["corrupted_questions"])
            num_samples = int(total_questions * (self.sampling_percentage / 100))

            if self.sampling_percentage < 100:
                sampled_questions = random.sample(
                    data["corrupted_questions"], num_samples
                )
                data["corrupted_questions"] = sampled_questions
                print(
                    f"Sampled {num_samples} questions ({self.sampling_percentage}%) for evaluation"
                )
            else:
                print("Processing 100% of questions (no sampling)")

            processed_count = 0
            success_count = 0
            error_count = 0

            for item in tqdm(data["corrupted_questions"]):
                try:
                    processed_count += 1
                    # print(
                    #     f"\nProcessing question {processed_count}/{len(data['corrupted_questions'])}"
                    # )

                    if "verification_result" not in item:
                        item["verification_result"] = {}
                    if "vqa_results" not in item["verification_result"]:
                        item["verification_result"]["vqa_results"] = []

                    question = item["corrupted_question"]
                    pages = item["layout_analysis"]["pages"]

                    image_paths = []
                    for page_id in pages:
                        image_filename = os.path.basename(page_id)
                        image_path = os.path.join(
                            self.config["images_base_path"], image_filename
                        )
                        image_paths.append(image_path)

                    # print(f"Question: {question[:100]}...")
                    # print(f"Number of pages to analyze: {len(image_paths)}")

                    # Get OCR text if enabled
                    

                    # print("Generating answer...")
                    result = self.generate_answer(question, image_paths)
                    # print(f"Answer received: {result.get('answer', 'No answer')}")

                    # Create the VQA result structure
                    vqa_result = {
                        "model_type": "blip",
                        "model_config": {
                            "batch_size": self.model_config.get("batch_size", 1)
                        },
                        "ocr_enabled": False,
                        "question": question,
                        "answer": result.get("answer", "Unable to determine"),
                        "image_paths": image_paths,
                        "analysis_type": result.get("analysis_type", ""),
                        "timestamp": datetime.datetime.now().isoformat(),
                    }

                    if "error" in result:
                        vqa_result["error"] = result["error"]
                        vqa_result["traceback"] = result.get("traceback", "")
                        error_count += 1
                    else:
                        success_count += 1

                    item["verification_result"]["vqa_results"].append(vqa_result)

                    # Save intermediate results every 10 questions
                    # if processed_count % 10 == 0:
                    #     self._save_results(data)
                    #     print("Intermediate results saved")

                except Exception as e:
                    print(f"Error processing question: {str(e)}")
                    print(f"Full error: {traceback.format_exc()}")
                    error_count += 1

            print(f"\nProcessing completed:")
            print(f"Total questions processed: {processed_count}")
            print(f"Successful generations: {success_count}")
            print(f"Errors encountered: {error_count}")
            if processed_count > 0:
                print(f"Success rate: {(success_count/processed_count)*100:.2f}%")

            self._save_results(data)
            output_file = self.model_config['name']+'_'+self.config["output_file"]
            print(f"Final results saved to {output_file}")

        except Exception as e:
            print(f"Critical error in evaluate: {str(e)}")
            print(f"Full error: {traceback.format_exc()}")
        finally:
            self._cleanup_model()


def main():
    config_path = "../config.json"  # Update this path if needed
    evaluator = BlipVQAEvaluator(config_path)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
