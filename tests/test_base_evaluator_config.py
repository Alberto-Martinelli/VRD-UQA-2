"""Unit test: BaseVQAEvaluator.__init__ accepts a config dict OR a path.
Run: uv run python -m tests.test_base_evaluator_config"""
import importlib.util
import json
import tempfile
from pathlib import Path
from config.paths import REPO_ROOT


def _load_base():
    path = REPO_ROOT / "VQA_analysis" / "evaluators" / "base_evaluator.py"
    spec = importlib.util.spec_from_file_location("base_evaluator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dummy_cls(base):
    class Dummy(base.BaseVQAEvaluator):
        MODEL_KEY = "m"
    return Dummy


def _cfg():
    return {"open_source_models": {"m": {"max_tokens": 7}}, "seed": 13, "mock": True}


def test_init_accepts_dict():
    base = _load_base()
    e = _dummy_cls(base)(_cfg(), finetuned=False)
    assert e.config["seed"] == 13
    assert e.seed == 13
    assert e.max_tokens == 7


def test_init_still_accepts_path():
    base = _load_base()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cfg.json"
        p.write_text(json.dumps(_cfg()))
        e = _dummy_cls(base)(str(p), finetuned=False)
        assert e.seed == 13


if __name__ == "__main__":
    test_init_accepts_dict()
    test_init_still_accepts_path()
    print("OK: base_evaluator dict-or-path __init__")
