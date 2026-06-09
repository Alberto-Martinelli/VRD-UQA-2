"""Step-2 enrich checks. Run: uv run python -m tests.test_metrics_enrich"""
import json
import tempfile
from pathlib import Path
from config.paths import REPO_ROOT
import importlib.util


def _load(rel_path, name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


STEP2 = _load("VQA_analysis/metrics/2_enrich_metadata.py", "step2")


def _minimal_normalized():
    return {"corrupted_questions": [{
        "complexity": 1,
        "entity_type": ["year_number_information"],
        "original_entity": [{"text": "2019"}],
        "corrupted_entities": [{"text": "2019"}],
        "question_entities": [{"text": "2019"}],
        "patch_entities": {},
        "verification_result": {"vqa_results": [{
            "answer_corrupted": [{"pages": ["p1"], "answer": "x", "answer_converted": "unable to determine"}],
        }]},
    }]}


def test_enrich_in_place_and_idempotent():
    tmp = Path(tempfile.mkdtemp())
    norm = tmp / "normalized.json"
    norm.write_text(json.dumps(_minimal_normalized()))

    STEP2.enrich_file(str(norm))
    with open(norm) as f:
        data = json.load(f)
    assert data.get("_enriched") is True
    # question_entities rebuilt with positions key
    assert "positions" in data["corrupted_questions"][0]["question_entities"][0]
    # answers untouched
    vr = data["corrupted_questions"][0]["verification_result"]["vqa_results"][0]
    assert vr["answer_corrupted"][0]["answer_converted"] == "unable to determine"

    # Idempotent: second run is a no-op (flag already set), file unchanged.
    marker = json.dumps(data, sort_keys=True)
    STEP2.enrich_file(str(norm))
    with open(norm) as f:
        assert json.dumps(json.load(f), sort_keys=True) == marker


if __name__ == "__main__":
    test_enrich_in_place_and_idempotent()
    print("OK: metrics step 2 enrich")
