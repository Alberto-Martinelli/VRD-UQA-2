import json
import argparse

def filter_false_verifications(data):
    """Filter questions where verification_result is false."""
    if "corrupted_questions" in data:
        total_questions = len(data["corrupted_questions"])
        filtered_questions = [
            q
            for q in data["corrupted_questions"]
            if str(q["verification_result"]["verification_result"]).lower() == "false"
        ]
        print(f"Total questions: {total_questions}")
        print(f"Questions with false verification: {len(filtered_questions)}")
        return {"corrupted_questions": filtered_questions}
    return data


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_file", type=str, required=True)
    args.add_argument("--output_file", type=str, required=True)
    args = args.parse_args()
    with open(args.input_file, "r") as f:
        data = json.load(f)
    filtered_data = filter_false_verifications(data)
    with open(args.output_file, "w") as f:
        json.dump(filtered_data, f)
