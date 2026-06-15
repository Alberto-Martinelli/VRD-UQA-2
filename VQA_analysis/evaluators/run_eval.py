"""Unified by-path eval launcher: model -> evaluator class factory + in-memory
config overrides. Replaces the bash `case` routing and the `python -c` config
mutation in run_vqa_analysis.sh.

Run by-path so it works under the phi4/gemma4 pinned venv (no editable install):
    python VQA_analysis/evaluators/run_eval.py --model qwen2.5 --dataset BDocs --split val_300
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path

from config.paths import REPO_ROOT

HERE = Path(__file__).resolve().parent

# CLI model key -> (evaluator module filename beside this file, class name).
EVALUATORS = {
    "qwen2.5":  ("qwen2.5_evaluator.py",  "QwenVQAEvaluator"),
    "phi4":     ("phi4_evaluator.py",     "Phi4VQAEvaluator"),
    "internvl": ("internvl_evaluator.py", "InternVLVQAEvaluator"),
    "gemma4":   ("gemma4_evaluator.py",   "Gemma4VQAEvaluator"),
}


def _load_evaluator_class(model):
    """Import ONLY the requested evaluator module. importlib handles the dot in
    'qwen2.5_evaluator.py', and loading just one module keeps the pinned venv
    from importing other models' incompatible deps."""
    module_file, class_name = EVALUATORS[model]
    mod_name = module_file[:-3].replace(".", "_")  # qwen2.5_evaluator -> qwen2_5_evaluator
    spec = importlib.util.spec_from_file_location(mod_name, HERE / module_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot locate evaluator module: {HERE / module_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _resolve_input_file(dataset, split):
    return str(
        REPO_ROOT / "data" / dataset / f"{dataset}_{split}"
        / f"{dataset}_unanswerable_corrupted_questions_just_false.json"
    )


def main():
    parser = argparse.ArgumentParser(description="Unified VQA evaluator launcher.")
    parser.add_argument("--model", required=True, choices=sorted(EVALUATORS))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="val_300")
    parser.add_argument("--config", default="VQA_analysis/config_fewshot.json")
    parser.add_argument("--input-file", default=None,
                        help="Explicit eval input; overrides the data/<dataset>/<split> derivation.")
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--questions", choices=["both", "corrupted", "clean"], default="both")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    config["dataset"] = args.dataset
    config["split"] = args.split.split("_")[0]  # strips size suffix: val_300 -> "val"
    config["input_file"] = args.input_file or _resolve_input_file(args.dataset, args.split)

    # Provenance for the run manifest (base_evaluator reads VQA_CONFIG_PATH).
    os.environ.setdefault("VQA_CONFIG_PATH", args.config)

    evaluator_cls = _load_evaluator_class(args.model)
    evaluator = evaluator_cls(config, args.finetuned, questions=args.questions)
    print(f"Running {args.model} | dataset={args.dataset} split={args.split} "
          f"finetuned={args.finetuned} questions={args.questions} seed={evaluator.seed}")
    evaluator.evaluate()


if __name__ == "__main__":
    main()
