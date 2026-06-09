"""
Post-processes VQA result JSON files by normalising model answers that mean
"unable to answer" into a canonical "unable to determine" string.

Scans all 'original/' sub-folders under
the pipeline directory and writes converted files into a sibling 'converted/'
folder, skipping files that have already been processed.
"""
import json
import os
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.run_layout import EVAL_RUNS_DIR, run_dir

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

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens = 64
    temperature = 0.0

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

UNABLE_PHRASES = [
    "unable to determine", "not answerable", "not provided", "not available",
    "not in the image", "not in the document", "not found", "not contain",
    "not include", "cannot determine", "cannot answer", "cannot provide",
    "cannot find", "i don ' t know", "unknown",
]


def _is_numeric(text):
    text = text.replace("$", "").replace("€", "").replace(",", "").strip()
    text = "".join(text.split()).rstrip("%").rstrip(".")
    try:
        float(text)
        return True
    except ValueError:
        return False


def canonicalize_answer(answer, classify_fn=None):
    """Map a free-text answer to a canonical form. Rule-based first (cheap),
    falling back to the injected classifier only for ambiguous free text."""
    if classify_fn is None:
        classify_fn = classify_unanswerable_answer
    low = answer.lower()
    if any(p in low for p in UNABLE_PHRASES) or low == "":
        return "unable to determine"
    if _is_numeric(answer):
        return answer
    return classify_fn(answer)


def label_vqa_answers(input_file, output_file, classify_fn=None):
    with open(input_file, "r") as f:
        data = json.load(f)

    for question in tqdm(data.get("corrupted_questions", []), mininterval=30):
        for result in question.get("verification_result", {}).get("vqa_results", []):
            for side_key in ("answer_corrupted", "answer_clean", "answers", "answer"):
                answers = result.get(side_key)
                if not isinstance(answers, list):
                    continue
                for answer_obj in answers:
                    if not isinstance(answer_obj, dict):
                        continue
                    answer_obj["answer_converted"] = canonicalize_answer(
                        answer_obj.get("answer", ""), classify_fn
                    )

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Processed file saved to {output_file}")

def process_all_folders(run_id=None, use_model=True):
    root = run_dir(run_id) if run_id else (EVAL_RUNS_DIR / "latest")
    print(f"{'='*100}\nNORMALIZE — scanning leaves under: {root}\n{'='*100}")
    if not root.exists():
        print(f"ERROR: run directory does not exist: {root}")
        return

    classify_fn = (lambda a: a) if not use_model else classify_unanswerable_answer
    total_processed = total_skipped = total_errors = 0

    for manifest_path in root.rglob("manifest.json"):
        leaf = manifest_path.parent
        preds = leaf / "predictions.json"
        if not preds.exists():
            continue
        cache = leaf / "_cache"
        cache.mkdir(exist_ok=True)
        out = cache / "normalized.json"
        if out.exists():
            print(f"Skipping (already normalized): {leaf}")
            total_skipped += 1
            continue
        try:
            label_vqa_answers(str(preds), str(out), classify_fn=classify_fn)
            total_processed += 1
        except Exception as e:
            print(f"ERROR normalizing {preds}: {e}")
            total_errors += 1

    print(f"Normalize complete — processed: {total_processed}, skipped: {total_skipped}, errors: {total_errors}")


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=os.environ.get("VQA_RUN_ID"))
    parser.add_argument("--no-model", action="store_true", help="Skip LLM classifier (rule-based only); for tests/debug.")
    args = parser.parse_args()
    process_all_folders(run_id=args.run_id, use_model=not args.no_model)
