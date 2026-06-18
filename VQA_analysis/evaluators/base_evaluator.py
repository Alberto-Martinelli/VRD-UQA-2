import gc
import json
import os
import random
import datetime
import traceback
import torch
from tqdm.auto import tqdm

from config import run_layout as rl


class BaseVQAEvaluator:
    """Model-agnostic VQA evaluation. Subclasses implement _load_model + _generate."""

    MODEL_KEY: str = ""            # config key under open_source_models
    FINETUNED_MODEL_KEY = None     # set only where a fine-tuned variant exists
    MODEL_TYPE: str = "base"       # written to vqa_result["model_type"]
    MODEL_LEAF_PREFIX: str = ""     # "" for Qwen (back-compat); model key otherwise

    def __init__(self, config, finetuned, questions="both"):
        # config may be a path (loaded here) or an already-built dict (passed by run_eval.py).
        if isinstance(config, (str, os.PathLike)):
            with open(config) as f:
                self.config = json.load(f)
        else:
            self.config = config

        self.finetuned = finetuned
        self.questions = questions
        self.seed = self.config.get("seed", 42)
        self._set_seed()

        if self.finetuned:
            if not self.FINETUNED_MODEL_KEY:
                raise ValueError(
                    f"--finetuned requested but {type(self).__name__} has no "
                    f"fine-tuned variant configured (FINETUNED_MODEL_KEY is None)."
                )
            self.model_config = self.config["open_source_models"][self.FINETUNED_MODEL_KEY]
        else:
            self.model_config = self.config["open_source_models"][self.MODEL_KEY]

        self.sampling_percentage = self.config.get("sampling_percentage", 100)
        self.unable_to_respond_aware = self.config.get("unable_to_respond_aware", True)

        # Extract base_image_dir from the input JSON if present, else images_base_path
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

        if self.config.get("mock", False):
            print("Mock mode enabled — skipping model initialization")
            self.max_tokens = self.model_config.get("max_tokens", 1024)
        else:
            self._load_model()

    # ---- abstract hooks (subclass implements) -------------------------------
    def _load_model(self):
        """Populate self.model, self.processor or self.tokenizer, self.max_tokens."""
        raise NotImplementedError

    def _generate(self, window_image_paths, question_prompt, few_shot_turns) -> str:
        """Run inference for one window of pages; return the decoded answer string.
        few_shot_turns is the neutral list from _build_few_shot_turns (may be None)."""
        raise NotImplementedError

    # ---- verbatim-moved methods from qwen2.5_evaluator.py ------------------

    def _set_seed(self):
        # Seeds stdlib random (drives sampling + few-shot selection in evaluate,
        # independent of the Transformers model-load RNG usage) plus the
        # torch/numpy states for completeness.
        random.seed(self.seed)
        try:
            import numpy as np
            np.random.seed(self.seed)
        except ImportError:
            pass
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

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
            image_filename = os.path.basename(page_id)
            if page_ocr:
                # Use the full path as the key since that's what we use in generate_answer
                image_path = os.path.join(
                    self.images_base_path, image_filename
                )
                ocr_text[image_path] = page_ocr
                # print(f"Extracted OCR text for page: {image_filename}")
            else:
                print(f"No OCR text found for page: {image_filename}")
        return ocr_text

    def _cleanup_model(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _save_results(self, data):
        n_items = len(data.get("corrupted_questions", []))
        run_id, slug, leaf = self._resolve_leaf(n_items)
        leaf.mkdir(parents=True, exist_ok=True)

        dataset = self.config["dataset"]
        split = self.config.get("split", "val")
        ocr_enabled = bool(self.config.get("ocr_enabled", False))
        window_size = self.model_config.get("batch_size", 1)

        predictions_path = leaf / "predictions.json"
        try:
            with open(predictions_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Predictions saved to {predictions_path}")
        except Exception as e:
            print(f"Error saving predictions: {str(e)}")
            return

        counts = {}
        if self.questions in ("both", "corrupted"):
            counts["corrupted"] = n_items
        if self.questions in ("both", "clean"):
            counts["clean"] = n_items

        manifest = {
            "run_id": run_id,
            "dataset": dataset,
            "config": slug,
            "split": split,
            "n": n_items,
            "seed": self.seed,
            "config_path": os.environ.get("VQA_CONFIG_PATH", ""),
            "input_file": self.config.get("input_file"),
            "model": self.model_config["model_name"],
            "model_name": self.model_config["name"],
            "adapter": self.model_config.get("adapter_path"),
            "ocr_enabled": ocr_enabled,
            "window_size": window_size,
            "few_shot": self.config.get("few_shot", {"enabled": False}),
            "questions": self.questions,
            "counts": counts,
            "min_pixels": self.model_config.get("min_pixels"),
            "max_pixels": self.model_config.get("max_pixels"),
            "git_commit": rl.git_commit(),
            "git_dirty": rl.git_dirty(),
            "created_at": rl.utc_now_iso(),
        }
        manifest["label"] = rl.human_label(manifest)
        rl.write_manifest(leaf / "manifest.json", manifest)
        print(f"Manifest saved to {leaf / 'manifest.json'}")

    def _get_fs_pool(self, dataset_pool: list) -> list:
        """Return the few-shot candidate pool, loading from pool_file once if configured."""
        pool_file = self.config.get("few_shot", {}).get("pool_file", "")
        if not pool_file:
            return list(dataset_pool)
        if not hasattr(self, "_fs_pool_cache"):
            if os.path.exists(pool_file):
                with open(pool_file) as f:
                    data = json.load(f)
                self._fs_pool_cache = data.get("corrupted_questions") or data.get("items", [])
                print(f"Few-shot pool loaded from {pool_file}: {len(self._fs_pool_cache)} items")
            else:
                print(f"WARNING: pool_file {pool_file!r} not found; falling back to eval pool.")
                self._fs_pool_cache = list(dataset_pool)
        return self._fs_pool_cache

    def _fs_specific_score(self, candidate: dict, current: dict) -> tuple:
        """Score a candidate for 'specific' selection: (complexity_match, entity_type_jaccard).
        Sorted descending so exact-complexity + highest entity overlap ranks first."""
        et_c = set(candidate.get("entity_type") or [])
        et_q = set(current.get("entity_type") or [])
        union = et_c | et_q
        jacc = len(et_c & et_q) / len(union) if union else 0.0
        same_cx = 1.0 if candidate.get("complexity") == current.get("complexity") else 0.0
        return (same_cx, jacc)

    def _select_few_shot_examples(self, dataset_pool, current_item):
        """Select N distinct few-shot demonstrations based on shot_type and selection strategy."""
        few_shot_config = self.config.get("few_shot", {})
        if not few_shot_config.get("enabled", False):
            return []

        n_shots = few_shot_config.get("n_shots", 2)
        shot_type = few_shot_config.get("shot_type", "mixed")
        selection = few_shot_config.get("selection", "random")

        # Resolve pool: use pool_file (train set) if configured, else fall back to eval pool
        pool = self._get_fs_pool(dataset_pool)
        if not few_shot_config.get("pool_file"):
            # Using the eval pool: exclude current item to prevent leakage
            pool = [item for item in pool if item != current_item]
        if not pool:
            return []

        # Drop candidates with missing images
        valid_pool = []
        for item in pool:
            pages = item.get("layout_analysis", {}).get("pages", {})
            item_base = item.get("images_base_path", self.images_base_path)
            missing = [
                p_id for p_id in pages
                if not os.path.exists(os.path.join(item_base, os.path.basename(p_id)))
            ]
            if missing:
                print(f"WARNING: few-shot candidate skipped — missing images: {missing}")
            else:
                valid_pool.append(item)
        pool = valid_pool
        if not pool:
            print("WARNING: no few-shot candidates have valid image paths; skipping few-shot.")
            return []

        # Pick n_shots candidates according to selection strategy
        n_select = min(n_shots, len(pool))
        if selection == "handpicked":
            # Fixed ordered pool — no ranking or filtering, always answerable
            sampled = pool[:n_select]
            return [{"type": "answerable", "item": s} for s in sampled]
        if selection == "specific":
            # Rank by (complexity_match, entity_type_jaccard); stable sort for determinism
            scored = sorted(pool, key=lambda c: self._fs_specific_score(c, current_item), reverse=True)
            sampled = scored[:n_select]
        else:
            sampled = random.sample(pool, n_select)

        # Assign answerable / unanswerable roles
        if shot_type == "answerable":
            return [{"type": "answerable", "item": s} for s in sampled]
        if shot_type == "unanswerable":
            return [{"type": "unanswerable", "item": s} for s in sampled]
        # mixed: first half answerable, second half unanswerable (n_shots//2 each)
        n_ans = n_shots // 2
        shots = [{"type": "answerable", "item": s} for s in sampled[:n_ans]]
        shots.extend([{"type": "unanswerable", "item": s} for s in sampled[n_ans:]])
        random.shuffle(shots)
        return shots

    QUESTION_SIDES = {
        "both": ["corrupted", "clean"],
        "corrupted": ["corrupted"],
        "clean": ["clean"],
    }

    def _process_single_question(self, item, dataset_pool):
        """Evaluate the requested question side(s) for one item, storing both
        answers in a single vqa_result (corrupted -> QUR/UR, clean -> FRR)."""
        if "verification_result" not in item:
            item["verification_result"] = {}
        if "vqa_results" not in item["verification_result"]:
            item["verification_result"]["vqa_results"] = []

        pages = item["layout_analysis"]["pages"]
        image_paths = [
            os.path.join(self.images_base_path, os.path.basename(page_id))
            for page_id in pages
        ]
        ocr_text = self.get_ocr_text(pages) if self.config.get("ocr_enabled", False) else None

        few_shot_turns = None
        few_shot_config = self.config.get("few_shot", {})
        if few_shot_config.get("enabled", False):
            shots = self._select_few_shot_examples(dataset_pool, item)
            few_shot_turns = self._build_few_shot_turns(shots)

        question_text = {
            "corrupted": item["corrupted_question"],
            "clean": item["original_question"],
        }

        vqa_result = {
            "model_type": self.MODEL_TYPE,
            "model_config": {
                "batch_size": self.model_config.get("batch_size", 1),
                "max_tokens": self.max_tokens,
                "use_flash_attention": self.model_config.get("use_flash_attention", False),
                "adapter_path": self.model_config.get("adapter_path", None),
            },
            "ocr_enabled": bool(ocr_text),
            "few_shot_config": self.config.get("few_shot", {"enabled": False}),
            "timestamp": datetime.datetime.now().isoformat(),
        }

        success = True
        for side in self.QUESTION_SIDES[self.questions]:
            result = self.generate_answer(
                question_text[side], image_paths, ocr_text, few_shot_turns=few_shot_turns
            )
            answer = result.get("answer", "Unable to determine")
            if not isinstance(answer, list):
                # Error path: generate_answer returns a flat string. Wrap it so the
                # answer schema stays uniform (list of {pages, answer}) for the
                # normalize + metrics steps, which iterate per-answer dicts.
                answer = [{"pages": image_paths, "answer": answer}]
            vqa_result[f"question_{side}"] = question_text[side]
            vqa_result[f"answer_{side}"] = answer
            vqa_result["analysis_type"] = result.get("analysis_type", "")
            if "error" in result:
                vqa_result[f"error_{side}"] = result["error"]
                vqa_result[f"traceback_{side}"] = result.get("traceback", "")
                success = False

        item["verification_result"]["vqa_results"].append(vqa_result)
        return success

    def evaluate(self):
        """Orchestrates the loading, processing, and evaluation of the dataset."""
        try:
            print("\nStarting evaluation...")

            # Load input data
            with open(self.config["input_file"]) as f:
                data = json.load(f)
                print(f"Successfully loaded input file: {self.config['input_file']}")

            # Resume: if this dataset/config already has predictions for the run,
            # skip it (idempotent re-runs; lets a timed-out job be resubmitted
            # with the same VQA_RUN_ID and only finish what's missing).
            _, slug, leaf = self._resolve_leaf(len(data["corrupted_questions"]))
            if (leaf / "predictions.json").exists():
                print(f"RESUME: {leaf / 'predictions.json'} already exists — "
                      f"skipping {self.config['dataset']}/{slug}.")
                return

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

    # ---- reshaped methods (neutral/model-agnostic versions) -----------------

    def generate_answer(self, question, image_paths, ocr_text=None, few_shot_turns=None):
        """Sliding-window image context + optional few-shot; per-window inference
        is delegated to the model-specific self._generate()."""
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
            missing = [p for p in image_paths if not os.path.exists(p)]
            if missing:
                print(f"WARNING: {len(missing)} image path(s) not found on disk:")
                for p in missing:
                    print(f"  [MISSING] {p}")

            window_size = self.model_config.get("batch_size", 1)
            stride = self.model_config.get("stride", window_size // 2) if window_size > 1 else 1
            total_images = len(image_paths)

            windows = []
            start_idx = 0
            while start_idx < total_images:
                end_idx = min(start_idx + window_size, total_images)
                if end_idx == total_images and (end_idx - start_idx) < window_size and total_images >= window_size:
                    window = image_paths[-window_size:]
                else:
                    window = image_paths[start_idx:end_idx]
                if window not in windows:
                    windows.append(window)
                if end_idx == total_images:
                    break
                start_idx += stride

            all_responses = []
            for window in windows:
                batch_ocr = None
                if ocr_text:
                    ocr_lines = []
                    for path in window:
                        page_num = image_paths.index(path) + 1
                        page_ocr = ocr_text.get(path, "")
                        if page_ocr:
                            ocr_lines.append(f"Page {page_num}:\n{page_ocr}")
                    batch_ocr = "\n\n".join(ocr_lines) if ocr_lines else None

                question_prompt = self._create_prompt(question, batch_ocr)
                response = self._generate(window, question_prompt, few_shot_turns)
                all_responses.append({"pages": window, "answer": response})

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

    def _build_few_shot_turns(self, shots):
        """Neutral conversational turns for few-shot. Each subclass's _generate
        converts these to its native message format.
        Shape: {"role":"user","image_paths":[...],"text":...} / {"role":"assistant","text":...}"""
        window_size = self.model_config.get("batch_size", 1)
        turns = []
        for shot in shots:
            item = shot["item"]
            is_ans = shot["type"] == "answerable"

            all_pages = list(item["layout_analysis"]["pages"].keys())
            anchor = 0
            if item.get("original_answer_locations"):
                answer_page = os.path.basename(item["original_answer_locations"][0]["page_id"])
                all_basenames = [os.path.basename(p) for p in all_pages]
                if answer_page in all_basenames:
                    anchor = all_basenames.index(answer_page)

            start = max(0, min(anchor - window_size // 2, len(all_pages) - window_size))
            window_pages = all_pages[start:start + window_size]
            item_base = item.get("images_base_path", self.images_base_path)
            image_paths = [
                os.path.join(item_base, os.path.basename(p_id))
                for p_id in window_pages
            ]

            ocr_text = None
            if self.config.get("ocr_enabled", False):
                window_pages_dict = {p: item["layout_analysis"]["pages"][p] for p in window_pages}
                # Handpicked demos store minimal page dicts (no layout_analysis inside each page);
                # skip OCR extraction for those to avoid KeyError.
                if all("layout_analysis" in v for v in window_pages_dict.values()):
                    ocr_dict = self.get_ocr_text(window_pages_dict)
                    ocr_lines = []
                    for i, path in enumerate(image_paths):
                        page_ocr = ocr_dict.get(path, "")
                        if page_ocr:
                            ocr_lines.append(f"Page {i + 1}:\n{page_ocr}")
                    ocr_text = "\n\n".join(ocr_lines) if ocr_lines else None

            if is_ans:
                q_text = item["original_question"]
                ans_text = item["original_answer_locations"][0]["answer"]
            else:
                q_text = item["corrupted_question"]
                ans_text = "Unable to determine"

            prompt = self._create_prompt(q_text, ocr_text)
            turns.append({"role": "user", "image_paths": image_paths, "text": prompt})
            turns.append({"role": "assistant", "text": ans_text})
        return turns

    def _resolve_leaf(self, n_items):
        few_shot_cfg = self.config.get("few_shot", {})
        few_shot_enabled = few_shot_cfg.get("enabled", False)
        mode = rl.derive_mode(self.finetuned, few_shot_enabled)
        slug = rl.build_slug(mode, bool(self.config.get("ocr_enabled", False)),
                             self.model_config.get("batch_size", 1),
                             few_shot=few_shot_cfg if few_shot_enabled else None)
        run_id = os.environ.get("VQA_RUN_ID") or rl.make_run_id(
            self.config.get("split", "val"), n_items
        )
        return run_id, slug, rl.leaf_dir(run_id, self.config["dataset"], slug, self.MODEL_LEAF_PREFIX)
