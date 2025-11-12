import google.generativeai as genai
import json
import time
import os
from pathlib import Path

# from google import genai
genai.configure(api_key="")

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
)
max_tokens = 1024
print("Gemini model initialized successfully")


def unable_to_determine_answer(answer):
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
    response = model.generate_content([prompt])
    result = response.text.strip()
    return result


def process_vqa_file(input_file, output_file):
    """
    Reads a JSON file with VQA results, evaluates each answer using the Gemini model,
    and appends a new field 'answer_converted' for each answer with the evaluation result.
    The updated JSON is saved to the output_file.
    """
    # Load JSON data from the input file
    with open(input_file, "r") as f:
        data = json.load(f)

    # Process each corrupted question
    for q_index, question in enumerate(data.get("corrupted_questions", [])):
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
                    converted_answer = unable_to_determine_answer(original_answer)
                    time.sleep(0.5)

                answer_obj["answer_converted"] = converted_answer

    # Save the updated JSON data to the output file
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Processed file saved to {output_file}")


def process_all_folders():
    """
    Processes all JSON files in the results directory and its subdirectories.
    Skips files that have already been converted.
    """
    script_dir = Path(__file__).parent

    # Walk through all subdirectories
    for root, dirs, files in os.walk(script_dir):
        root_path = Path(root)

        # Only process if we're in an 'original' folder
        if root_path.name == "original":
            # Get parent directory (result_type folder)
            parent_dir = root_path.parent

            # Create 'converted' folder at the same level as 'original'
            converted_dir = parent_dir / "converted"
            converted_dir.mkdir(exist_ok=True)

            # Process all JSON files in the original folder
            json_files = [f for f in files if f.endswith(".json")]
            for json_file in json_files:
                input_path = root_path / json_file
                output_filename = json_file.replace(".json", "_converted.json")
                output_path = converted_dir / output_filename

                # Skip if the file has already been converted
                if output_path.exists():
                    print(f"Skipping already converted file: {output_path}")
                    continue

                print(f"Processing: {input_path}")
                print(f"Saving to: {output_path}")

                try:
                    process_vqa_file(str(input_path), str(output_path))
                    print(f"Successfully processed: {input_path}")
                except Exception as e:
                    print(f"Error processing {input_path}: {str(e)}")
                    continue


if __name__ == "__main__":
    process_all_folders()
