"""
Post-processes VQA result JSON files by normalising model answers that mean
"unable to answer" into a canonical "unable to determine" string.

Scans all 'original/' sub-folders under
the pipeline directory and writes converted files into a sibling 'converted/'
folder, skipping files that have already been processed.
"""
import json
import time
import os
from pathlib import Path
from tqdm import tqdm
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# import google.generativeai as genai
# from google import genai
# genai.configure(api_key="")

# model = genai.GenerativeModel(
#     model_name="gemini-2.5-flash",
# )
# max_tokens = 1024
# print("Gemini model initialized successfully")

classifier_components = None

def load_classifier_model():
    global classifier_components

    if classifier_components is not None:
        return classifier_components

    model_id = "Qwen/Qwen3-32B"
    max_new_tokens = 64
    temperature = 0.0
    enable_thinking = False

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    ).eval()

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    classifier_components = {
        "model_id": model_id,
        "tokenizer": tokenizer,
        "model": model,
        "generation_kwargs": generation_kwargs,
        "enable_thinking": enable_thinking,
    }
    return classifier_components

def classify_unanswerable_answer(answer):
    classifier = load_classifier_model()
    prompt = (
        "I'm performing an evaluation test on the ability of different models to answer VQA questions from document images. "
        "The model could return different answers to determine if the answer is 'unable to determine' or not. "
        "Your task is to  detect if the answer means that the model is unable to determine the answer or not. "
        "Examples of answers that mean that the model is unable to determine the answer: "
        "- Not available. "
        "- Not provided in document. "
        "- The image does not provide information to answer the question. "
        "- I cannot provide an answer based on the given text. "
        "- The document does not provide information "
        "If the answer means 'unable to determine', respond with 'unable to determine', otherwise return the original answer. "
        f"The answer is: {answer} "
        "Please respond only with the original answer or 'unable to determine' only."
    )

    messages = [{"role": "user", "content": prompt}]
    text = classifier["tokenizer"].apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=classifier["enable_thinking"],
    )
    inputs = classifier["tokenizer"]([text], return_tensors="pt").to(
        classifier["model"].device
    )

    with torch.inference_mode():
        generated_ids = classifier["model"].generate(
            **inputs, **classifier["generation_kwargs"]
        )

    generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
    result = classifier["tokenizer"].batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()
    return result

def label_vqa_answers(input_file, output_file):
    """
    Reads a JSON file with VQA results, evaluates each answer using the Gemini model,
    and appends a new field 'answer_converted' for each answer with the evaluation result.
    The updated JSON is saved to the output_file.
    """
    # Load JSON data from the input file
    with open(input_file, "r") as f:
        data = json.load(f)

    print(f"Processing file: {input_file}")
    # Process each corrupted question
    q_index = 0
    for question in tqdm(data.get("corrupted_questions", [])):
        verification_result = question.get("verification_result", {})
        vqa_results = verification_result.get("vqa_results", [])
        if not vqa_results:
            print(f"No vqa_results found in question {q_index}")
        for r_index, result in enumerate(vqa_results):
            answers = result.get("answers", result.get("answer", []))
            if not answers:
                print(f"No answers found for question {q_index}, result {r_index}")
            for a_index, answer_obj in enumerate(answers):
                original_answer = answer_obj.get("answer", "")
                # print(f"Processing question {q_index}, result {r_index}, answer {a_index}: {original_answer}")
                unable_phrases = [
                    "unable to determine",
                    "not answerable",
                    "not provided",
                    "not available",
                    "not in the image",
                    "not in the document",
                    "not found",
                    "not contain",
                    "not include",
                    "cannot determine",
                    "cannot answer",
                    "cannot provide",
                    "cannot find",
                    "cannot answer",
                    "i don ' t know",
                    "unknown",
                ]

                def is_numeric(text):
                    # Remove currency symbols, spaces, and commas
                    text = (
                        text.replace("$", "").replace("€", "").replace(",", "").strip()
                    )
                    # Split by spaces and join to handle cases like "$ 100"
                    text = "".join(text.split())
                    # Remove % if present and check if it's a number
                    text = text.rstrip("%")
                    text = text.rstrip(".")
                    try:
                        float(text)
                        return True
                    except ValueError:
                        return False

                # First check if any of the unable phrases appear in the answer
                if (
                    any(phrase in original_answer.lower() for phrase in unable_phrases)
                    or original_answer.lower() == ""
                ):
                    converted_answer = "unable to determine"
                # Then check if the answer is numeric
                elif is_numeric(original_answer):
                    converted_answer = original_answer
                # Finally, if none of the above, use Gemini to evaluate
                else:
                    converted_answer = classify_unanswerable_answer(original_answer)
                    time.sleep(0.5)

                # print(f"Original answer: {original_answer}")
                # print(f"Converted answer: {converted_answer}")
                answer_obj["answer_converted"] = converted_answer
        q_index += 1

    # Save the updated JSON data to the output file
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Processed file saved to {output_file}")

def process_all_folders():
    """
    Processes all JSON files in the results directory and its subdirectories.
    Skips files that have already been converted.
    """
    # Results are written by the evaluators under VQA_analysis/models/results/,
    # which is relative to the project root (two levels above this file).
    results_dir = Path(__file__).parent.parent.parent / "VQA_analysis" / "models" / "results"

    print(f"{'='*100}")
    print(f"Unable Converter — scanning for results under: {results_dir}")
    if not results_dir.exists():
        print(f"ERROR: Results directory does not exist: {results_dir}")
        return
    print(f"{'='*100}")

    total_processed = 0
    total_skipped = 0
    total_errors = 0

    # Walk through all subdirectories
    found_any = False
    for root, dirs, files in os.walk(results_dir):
        root_path = Path(root)

        # Only process if we're in an 'original' folder
        if root_path.name != "original":
            continue

        found_any = True
        json_files = [f for f in files if f.endswith(".json")]
        if not json_files:
            print(f"No JSON files found in: {root_path}")
            continue

        # Get parent directory (result_type folder)
        parent_dir = root_path.parent

        # Create 'converted' folder at the same level as 'original'
        converted_dir = parent_dir / "converted"
        converted_dir.mkdir(exist_ok=True)

        for json_file in json_files:
            input_path = root_path / json_file
            output_filename = json_file.replace(".json", "_converted.json")
            output_path = converted_dir / output_filename

            # Skip if the file has already been converted
            if output_path.exists():
                print(f"Skipping (already converted): {input_path.name}")
                total_skipped += 1
                continue

            print(f"\n{'-'*100}")
            print(f"Processing : {input_path}")
            print(f"Output     : {output_path}")
            try:
                label_vqa_answers(str(input_path), str(output_path))
                total_processed += 1
                print(f"Done       : {output_path.name}")
            except Exception as e:
                print(f"ERROR processing {input_path}: {str(e)}")
                total_errors += 1
                continue

    if not found_any:
        print(f"WARNING: No 'original/' folders found under {results_dir} — nothing to convert.")

    print(f"\n{'='*100}")
    print(f"Unable Converter complete — processed: {total_processed}, skipped: {total_skipped}, errors: {total_errors}")
    print(f"{'='*100}")

if __name__ == "__main__":
    process_all_folders()
