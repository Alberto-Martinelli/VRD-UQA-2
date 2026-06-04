import gc
import json
import os
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import datetime
import traceback
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from difflib import SequenceMatcher
from qwen_vl_utils import process_vision_info
from tqdm.auto import tqdm


class QwenVQAEvaluator:
    def __init__(self, config_path, finetuned, answerable=False):
        with open(config_path) as f:
            self.config = json.load(f)

        # Get Qwen-specific configuration - now nested under "llm"
        self.finetuned = finetuned
        self.answerable = answerable
        if self.finetuned:
            self.model_config = self.config["open_source_models"]["qwen2.5_finetuned"]
        else:
            self.model_config = self.config["open_source_models"]["qwen2.5"]

        self.sampling_percentage = self.config.get("sampling_percentage", 100)
        self.unable_to_respond_aware = self.config.get("unable_to_respond_aware", True)

        # Extract base_image_dir from the input JSON file if present, else fallback to images_base_path
        input_file = self.config.get("input_file")
        self.images_base_path = self.config.get("images_base_path")
        if input_file and os.path.exists(input_file):
            try:
                with open(input_file) as f_in:
                    in_data = json.load(f_in)
                    if "base_image_dir" in in_data:
                        self.images_base_path = in_data["base_image_dir"]
                        print(f"Extracted images_base_path from input file: {self.images_base_path}")
            except Exception as e:
                print(f"Warning: could not parse base_image_dir from input file {input_file}: {e}")

        self.initialize_model()

    def _create_prompt(self, question, ocr_text=None):
        unable_to_respond_line = (
            "- If uncertain, return 'Unable to determine'\n- If you can't find the answer, return 'Unable to determine'"
            if self.unable_to_respond_aware
            else ""
        )

        if ocr_text:
            return (
                f"You are an AI assistant specialized in analyzing document images and text. "
                f"Your task is to answer questions about the document image content precisely.\n\n"
                f"For this question, you have the following OCR text:\n{ocr_text}\n\n"
                f"Guidelines:\n"
                f"- Provide concise, focused answers (single word or short phrase preferred)\n"
                f"- Base your answer on both the image and the provided OCR text\n"
                f"{unable_to_respond_line}\n"
                f"Question: {question}"
            )
        return (
            f"You are an AI assistant specialized in analyzing document images. "
            f"Your task is to answer questions about the document image content precisely.\n\n"
            f"Guidelines:\n"
            f"- Provide concise, focused answers (single word or short phrase preferred)\n"
            f"- Base your answer solely on what you see in the image\n"
            + (f"{unable_to_respond_line}\n" if self.unable_to_respond_aware else "")
            + f"Question: {question}"
        )

    def initialize_model(self):
        if self.config.get("mock", False):
            print("Mock mode enabled — skipping model initialization")
            self.max_tokens = self.model_config.get("max_tokens", 1024)
            return

        print("Initializing Qwen model...")
        print("Model configuration:", self.model_config)
        model_name = self.model_config["model_name"]

        # Initialize processor with pixel constraints if provided
        processor_kwargs = {}
        if "min_pixels" in self.model_config and "max_pixels" in self.model_config:
            processor_kwargs.update(
                {
                    "min_pixels": self.model_config["min_pixels"],
                    "max_pixels": self.model_config["max_pixels"],
                }
            )

        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)

        # Initialize model with flash attention if enabled
        model_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
        if self.model_config.get("use_flash_attention", False):
            model_kwargs.update(
                {
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": "flash_attention_2",
                }
            )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, **model_kwargs
        )

        # Dynamically attach LoRA adapter if path is supplied
        adapter_path = self.model_config.get("adapter_path")
        if adapter_path:
            resolved_adapter_path = os.path.abspath(os.path.expandvars(os.path.expanduser(adapter_path)))
            print(f"Loading PEFT/LoRA adapter from: {resolved_adapter_path}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, resolved_adapter_path)

        self.model = self.model.eval()

        self.max_tokens = self.model_config.get("max_tokens", 1024)
        print("Qwen model initialized successfully")

    def _cleanup_model(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

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

    def get_ocr_text(self, pages):
        # print("Extracting OCR text...")
        ocr_text = {}
        for page_id in pages:
            # Navigate through the nested structure correctly
            page_layout = pages[page_id]["layout_analysis"]
            page_ocr = self.get_sorted_ocr_text(page_layout)
            if page_ocr:
                # Use the full path as the key since that's what we use in generate_answer
                image_filename = os.path.basename(page_id)
                image_path = os.path.join(
                    self.images_base_path, image_filename
                )
                ocr_text[image_path] = page_ocr
                # print(f"Extracted OCR text for page: {image_filename}")
            else:
                print(f"No OCR text found for page: {image_filename}")
        return ocr_text

    def generate_answer(self, question, image_paths, ocr_text=None, few_shot_turns=None):
        """Generates model responses using a robust sliding-window image context and optional few-shot demonstrations."""
        if self.config.get("mock", False):
            window_size = self.model_config.get("batch_size", 1)
            windows = [image_paths[i:i + window_size] for i in range(0, len(image_paths), window_size)]
            return {
                "answer": [{"pages": w, "answer": "Mock answer"} for w in windows],
                "query": question,
                "image_paths": image_paths,
                "analysis_type": f"window_size_{window_size}_mock",
            }

        try:
            # Warn about any image paths that don't exist before attempting inference
            missing = [p for p in image_paths if not os.path.exists(p)]
            if missing:
                print(f"WARNING: {len(missing)} image path(s) not found on disk:")
                for p in missing:
                    print(f"  [MISSING] {p}")

            window_size = self.model_config.get("batch_size", 1)
            if window_size > 1:
                stride = self.model_config.get("stride", window_size // 2)
            else:
                stride = 1
            total_images = len(image_paths)

            # 1. Pre-build window slices to prevent any indexing bugs or skipped images
            windows = []
            start_idx = 0
            while start_idx < total_images:
                end_idx = min(start_idx + window_size, total_images)
                
                # Slide the last window backward to keep full window size if preferred
                if end_idx == total_images and (end_idx - start_idx) < window_size and total_images >= window_size:
                    window = image_paths[-window_size:]
                else:
                    window = image_paths[start_idx:end_idx]
                
                if window not in windows:  # Avoid duplicate evaluations
                    windows.append(window)
                if end_idx == total_images:
                    break
                start_idx += stride

            all_responses = []

            # 2. Process each sliding window batch
            for window in windows:
                # Format OCR text for this batch if available
                batch_ocr = None
                if ocr_text:
                    ocr_lines = []
                    for path in window:
                        page_num = image_paths.index(path) + 1  # 1-based page number
                        page_ocr = ocr_text.get(path, "")
                        if page_ocr:
                            ocr_lines.append(f"Page {page_num}:\n{page_ocr}")
                    batch_ocr = "\n\n".join(ocr_lines) if ocr_lines else None

                # Generate model prompt and format user messages
                question_prompt = self._create_prompt(question, batch_ocr)
                
                # Build conversational history
                messages = []
                if few_shot_turns:
                    messages.extend(few_shot_turns)
                
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            *[
                                {"type": "image", "image": f"file://{path}"}
                                for path in window
                            ],
                            {"type": "text", "text": question_prompt},
                        ],
                    }
                )

                # Prepare inputs for Qwen
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
                inputs = inputs.to("cuda")

                # Perform inference
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=self.max_tokens
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                all_responses.append(
                    {
                        "pages": window,
                        "answer": response,
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
                "answer": "Unable to determine: error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def _save_results(self, data):
        # Construct base path
        base_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "results", self.config["dataset"], "LLM"
        )

        # Get window size from config
        window_size = self.model_config.get("batch_size", 1)

        # Create processing type folder name
        processing_folder = f"results_w{window_size}"

        # Add OCR and UNABLE flags if enabled
        if self.config["ocr_enabled"]:
            processing_folder += "_OCR"
        if not self.config["unable_to_respond_aware"]:
            processing_folder += "_UNABLE"

        # Add unique folder tag if few-shot is enabled
        few_shot_config = self.config.get("few_shot", {})
        if few_shot_config.get("enabled", False):
            n_shots = few_shot_config.get("n_shots", 2)
            shot_type = few_shot_config.get("shot_type", "mixed")
            processing_folder += f"_fewshot_{shot_type}_{n_shots}"

        # Add adapter suffix if PEFT adapter is loaded
        adapter_path = self.model_config.get("adapter_path")
        if adapter_path:
            adapter_name = os.path.basename(adapter_path.rstrip("/"))
            processing_folder += f"_{adapter_name}"

        # Answerable pass goes to its own folder so it never overwrites corrupted results
        if self.answerable:
            processing_folder += "_answerable"

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

    def _select_few_shot_examples(self, dataset_pool, current_item):
        """Selects N distinct few-shot demonstrations based on configured shot_type."""
        few_shot_config = self.config.get("few_shot", {})
        if not few_shot_config.get("enabled", False):
            return []

        n_shots = few_shot_config.get("n_shots", 2)
        shot_type = few_shot_config.get("shot_type", "mixed")

        # Exclude the current item to prevent data leakage
        pool = [item for item in dataset_pool if item != current_item]
        if not pool:
            return []

        # Filter out candidates whose image files are missing on disk
        valid_pool = []
        for item in pool:
            pages = item.get("layout_analysis", {}).get("pages", {})
            missing = [
                p_id for p_id in pages
                if not os.path.exists(os.path.join(self.images_base_path, os.path.basename(p_id)))
            ]
            if missing:
                print(f"WARNING: few-shot candidate skipped — missing images: {missing}")
            else:
                valid_pool.append(item)
        pool = valid_pool
        if not pool:
            print("WARNING: no few-shot candidates have valid image paths; skipping few-shot.")
            return []

        if shot_type == "answerable":
            selected = random.sample(pool, min(n_shots, len(pool)))
            return [{"type": "answerable", "item": s} for s in selected]

        elif shot_type == "unanswerable":
            selected = random.sample(pool, min(n_shots, len(pool)))
            return [{"type": "unanswerable", "item": s} for s in selected]

        elif shot_type == "mixed":
            n_ans = n_shots // 2
            n_unans = n_shots - n_ans

            # Sample without replacement across both roles to avoid using the same item twice
            sampled = random.sample(pool, min(n_shots, len(pool)))
            ans_selected = sampled[:n_ans]
            unans_selected = sampled[n_ans:n_ans + n_unans]

            shots = [{"type": "answerable", "item": s} for s in ans_selected]
            shots.extend([{"type": "unanswerable", "item": s} for s in unans_selected])
            random.shuffle(shots)
            return shots

        return []

    def _build_few_shot_turns(self, shots):
        """Constructs conversational message turns for few-shot visual prompts."""
        window_size = self.model_config.get("batch_size", 1)
        turns = []
        for shot in shots:
            item = shot["item"]
            is_ans = shot["type"] == "answerable"

            # 1. Select a window of batch_size pages anchored on the answer page
            #    (original_answer_locations holds the answer page for both answerable
            #    and unanswerable shots — for the latter it's the pre-corruption page).
            all_pages = list(item["layout_analysis"]["pages"].keys())
            anchor = 0
            if item.get("original_answer_locations"):
                answer_page = os.path.basename(
                    item["original_answer_locations"][0]["page_id"]
                )
                all_basenames = [os.path.basename(p) for p in all_pages]
                if answer_page in all_basenames:
                    anchor = all_basenames.index(answer_page)

            start = max(0, min(anchor - window_size // 2, len(all_pages) - window_size))
            window_pages = all_pages[start:start + window_size]

            image_paths = [
                os.path.join(self.images_base_path, os.path.basename(p_id))
                for p_id in window_pages
            ]

            # 2. Get OCR if enabled and format into a string matching generate_answer's batch_ocr
            ocr_text = None
            if self.config.get("ocr_enabled", False):
                window_pages_dict = {p: item["layout_analysis"]["pages"][p] for p in window_pages}
                ocr_dict = self.get_ocr_text(window_pages_dict)
                ocr_lines = []
                for i, path in enumerate(image_paths):
                    page_ocr = ocr_dict.get(path, "")
                    if page_ocr:
                        ocr_lines.append(f"Page {i + 1}:\n{page_ocr}")
                ocr_text = "\n\n".join(ocr_lines) if ocr_lines else None

            # 3. Determine prompt and correct answer
            if is_ans:
                q_text = item["original_question"]
                ans_text = item["original_answer_locations"][0]["answer"]
            else:
                q_text = item["corrupted_question"]
                ans_text = "Unable to determine"

            prompt = self._create_prompt(q_text, ocr_text)

            # User Turn
            turns.append({
                "role": "user",
                "content": [
                    *[{"type": "image", "image": f"file://{path}"} for path in image_paths],
                    {"type": "text", "text": prompt}
                ]
            })

            # Assistant Turn
            turns.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ans_text}
                ]
            })

        return turns

    def _process_single_question(self, item, dataset_pool):
        """Processes a single visual question item and appends its evaluation results."""
        if "verification_result" not in item:
            item["verification_result"] = {}
        if "vqa_results" not in item["verification_result"]:
            item["verification_result"]["vqa_results"] = []

        question = item["original_question"] if self.answerable else item["corrupted_question"]
        pages = item["layout_analysis"]["pages"]

        # Resolve raw page identifiers to absolute image paths
        image_paths = [
            os.path.join(self.images_base_path, os.path.basename(page_id))
            for page_id in pages
        ]

        # Retrieve OCR text if enabled
        ocr_text = self.get_ocr_text(pages) if self.config.get("ocr_enabled", False) else None

        # Build few-shot turns if enabled
        few_shot_turns = None
        few_shot_config = self.config.get("few_shot", {})
        if few_shot_config.get("enabled", False):
            shots = self._select_few_shot_examples(dataset_pool, item)
            few_shot_turns = self._build_few_shot_turns(shots)

        # Generate model response
        result = self.generate_answer(question, image_paths, ocr_text, few_shot_turns=few_shot_turns)

        # Create structured VQA result
        vqa_result = {
            "model_type": "qwen",
            "model_config": {
                "batch_size": self.model_config.get("batch_size", 1),
                "max_tokens": self.max_tokens,
                "use_flash_attention": self.model_config.get("use_flash_attention", False),
                "adapter_path": self.model_config.get("adapter_path", None),
            },
            "ocr_enabled": bool(ocr_text),
            "few_shot_config": self.config.get("few_shot", {"enabled": False}),
            "question": question,
            "answer": result.get("answer", "Unable to determine"),
            "image_paths": image_paths,
            "analysis_type": result.get("analysis_type", ""),
            "timestamp": datetime.datetime.now().isoformat(),
        }

        # Check and record any generation errors
        if "error" in result:
            vqa_result["error"] = result["error"]
            vqa_result["traceback"] = result.get("traceback", "")
            success = False
        else:
            success = True

        item["verification_result"]["vqa_results"].append(vqa_result)
        return success

    def evaluate(self):
        """Orchestrates the loading, processing, and evaluation of the dataset."""
        try:
            print("\nStarting Qwen evaluation...")

            # Load input data
            with open(self.config["input_file"]) as f:
                data = json.load(f)
                print(f"Successfully loaded input file: {self.config['input_file']}")

            # Sample questions
            total_questions = len(data["corrupted_questions"])
            num_samples = int(total_questions * (self.sampling_percentage / 100))

            if self.sampling_percentage < 100:
                sampled_questions = random.sample(data["corrupted_questions"], num_samples)
                data["corrupted_questions"] = sampled_questions
                print(f"Sampled {num_samples} questions ({self.sampling_percentage}%) for evaluation")
            else:
                print("Processing 100% of questions (no sampling)")

            processed_count = 0
            success_count = 0
            error_count = 0

            # Iterate through the sampled/full questions
            for item in tqdm(data["corrupted_questions"]):
                try:
                    processed_count += 1
                    success = self._process_single_question(item, data["corrupted_questions"])
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    print(f"Error processing question: {str(e)}")
                    print(f"Full error: {traceback.format_exc()}")
                    error_count += 1

            # Log final statistics
            print(f"\nProcessing completed:")
            print(f"Total questions processed: {processed_count}")
            print(f"Successful generations: {success_count}")
            print(f"Errors encountered: {error_count}")
            if processed_count > 0:
                print(f"Success rate: {(success_count / processed_count) * 100:.2f}%")

            # Save results (unified path log handled inside _save_results)
            self._save_results(data)

        except Exception as e:
            print(f"Critical error in evaluate: {str(e)}")
            print(f"Full error: {traceback.format_exc()}")
        finally:
            self._cleanup_model()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--answerable", action="store_true")
    args = parser.parse_args()
    config_path = args.config_path
    finetuned = args.finetuned

    evaluator = QwenVQAEvaluator(config_path, finetuned, answerable=args.answerable)
    print("Starting QWEN 2.5 evaluator")
    evaluator.evaluate()


if __name__ == "__main__":
    main()
