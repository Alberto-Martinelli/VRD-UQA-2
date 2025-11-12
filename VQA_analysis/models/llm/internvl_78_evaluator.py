import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import datetime
import traceback
import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as transforms
from difflib import SequenceMatcher
from tqdm.auto import tqdm


class InternVL78Evaluator:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)

        # Get InternVL-specific configuration
        self.model_config = self.config["open_source_models"]["internvl378"]
        self.sampling_percentage = self.config.get("sampling_percentage", 100)
        self.unable_to_respond_aware = self.config.get("unable_to_respond_aware", True)
        self.initialize_model()

    def initialize_model(self):
        print("Initializing InternVL model...")
        print("Model configuration:", self.model_config)
        model_name = self.model_config["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_flash_attn=True,
            low_cpu_mem_usage=True,
            # cache_dir="/data1/hf_cache/models"
        ).eval()
        self.max_tokens = self.model_config.get("max_tokens", 1024)
        self.input_size = self.model_config.get("input_size", 448)

        # set pad_token_id to eos_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print("Pad token ID set to EOS token ID")

        # Initialize image transform with correct normalization
        self.transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda img: img.convert("RGB") if img.mode != "RGB" else img
                ),
                transforms.Resize(
                    (self.input_size, self.input_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet mean
                    std=[0.229, 0.224, 0.225],  # ImageNet std
                ),
            ]
        )
        print("InternVL model initialized successfully")

    def _cleanup_model(self):
        if hasattr(self, "model"):
            del self.model
            torch.cuda.empty_cache()
            import gc

            gc.collect()

    def load_image(self, image_path):
        """Load and preprocess image according to InternVL requirements"""
        image = Image.open(image_path).convert("RGB")
        # Use dynamic preprocessing as shown in the docs
        images = self.dynamic_preprocess(
            image,
            image_size=self.input_size,
            use_thumbnail=True,
            max_num=self.model_config.get("max_num", 12),
        )
        pixel_values = [self.transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values.to(torch.bfloat16)  # Convert to bfloat16

    def dynamic_preprocess(
        self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
    ):
        """Implementation of dynamic preprocessing from official docs"""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find closest aspect ratio
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []

        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)

        return processed_images

    def find_closest_aspect_ratio(
        self, aspect_ratio, target_ratios, width, height, image_size
    ):
        """Helper function for dynamic preprocessing"""
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height

        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

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

    def _create_prompt(self, question, ocr_text=None):
        """Create a consistent prompt format for the model"""
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

                batch_paths = window_paths
                try:
                    # print("=== Loading Images ===")
                    pixel_values_list = []
                    for path in batch_paths:
                        try:
                            pixel_values = self.load_image(path)
                            # print(
                            #     f"Image loaded successfully, shape: {pixel_values.shape}"
                            # )
                            pixel_values_list.append(pixel_values)
                        except Exception as e:
                            print(f"Error loading image: {str(e)}")
                            raise

                    # print("=== Processing Images ===")
                    pixel_values = torch.cat(pixel_values_list, dim=0)
                    # print(f"Combined tensor shape: {pixel_values.shape}")

                    # Move tensor to GPU and ensure bfloat16
                    pixel_values = pixel_values.to(device='cuda', dtype=torch.bfloat16)

                    # print("=== Creating Query ===")
                    # Get OCR text for this batch if available
                    # Get OCR text if enabled
                    batch_ocr = None
                    if ocr_text:
                        batch_ocr = []
                        for page_idx, path in enumerate(batch_paths, start_idx):
                            page_num = page_idx + 1
                            page_ocr = ocr_text.get(path, "")  # Get OCR text for specific page
                            if page_ocr:
                                batch_ocr.append(f"Page {page_num}:\n{page_ocr}")
                        batch_ocr = "\n\n".join(batch_ocr) if batch_ocr else None

                    # Create image prefix for each image in batch
                    image_prefix = "".join(
                        [f"Image-{i+1}: <image>\n" for i in range(len(batch_paths))]
                    )
                    # Use the _create_prompt function here
                    prompt = self._create_prompt(question, batch_ocr)
                    query = f"{image_prefix}{prompt}"
                    # print(f"Final query: {query}")

                    try:
                        # print("\n=== Starting Generation ===")
                        generation_config = dict(
                            max_new_tokens=self.max_tokens,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                        )
                        # print("Generation config:", generation_config)

                        # print("Calling model.chat...")
                        response, history = self.model.chat(
                            self.tokenizer,
                            pixel_values,
                            query,
                            generation_config,
                            history=None,
                            return_history=True,
                        )
                        # print(
                        #     f"Generated response for batch {window_idx + 1}: {response}"
                        # )

                        all_responses.append({"pages": batch_paths, "answer": response})

                    except Exception as e:
                        print(f"Error in model generation: {str(e)}")
                        print(f"Full generation error: {traceback.format_exc()}")
                        all_responses.append(
                            {
                                "pages": batch_paths,
                                "answer": "Error in InternVL model",
                                "error": str(e),
                            }
                        )

                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    print(f"Full error: {traceback.format_exc()}")
                    all_responses.append(
                        {
                            "pages": batch_paths,
                            "answer": "Unable to determine",
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
        output_file = self.model_config["name"]+"_"+self.config["output_file"]
        if self.config["ocr_enabled"] and not self.config["unable_to_respond_aware"]:
            output_file = output_file.replace(".json", "_OCR_UNABLE.json")
        elif self.config["ocr_enabled"]:
            output_file = output_file.replace(".json", "_OCR.json")
        elif not self.config["unable_to_respond_aware"]:
            output_file = output_file.replace(".json", "_UNABLE.json")
        try:
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Results successfully saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")

    def evaluate(self):
        try:
            print("\nStarting InternVL evaluation...")

            # Load input data
            with open(self.config["input_file"]) as f:
                data = json.load(f)
                print(f"Successfully loaded input file: {self.config['input_file']}")

            # Sample questions
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
                torch.cuda.empty_cache()
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

                    # Generate answer
                    # print("Generating answer...")
                    result = self.generate_answer(question, image_paths, ocr_text)
                    # print(f"Answer received: {result.get('answer', 'No answer')}")

                    # Create VQA result
                    vqa_result = {
                        "model_type": "internvl",
                        "ocr_enabled": bool(ocr_text),
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
    evaluator = InternVL78Evaluator("../config.json")
    evaluator.evaluate()


if __name__ == "__main__":
    main()
