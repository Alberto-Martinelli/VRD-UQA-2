"""Tests for handpicked demo selection in base_evaluator.
Run: uv run python -m tests.test_few_shot_handpicked
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "VQA_analysis" / "evaluators"))

from base_evaluator import BaseVQAEvaluator  # noqa: E402

_MOCK_CONFIG = {
    "mock": True,
    "seed": 42,
    "sampling_percentage": 100,
    "ocr_enabled": False,
    "unable_to_respond_aware": True,
    "dataset": "BDocs",
    "input_file": "",
    "open_source_models": {
        "qwen2.5": {
            "model_name": "mock",
            "batch_size": 1,
            "max_tokens": 64,
            "name": "mock_qwen",
        }
    },
    "few_shot": {
        "enabled": True,
        "n_shots": 2,
        "shot_type": "answerable",
        "selection": "handpicked",
    },
}


class _StubEvaluator(BaseVQAEvaluator):
    MODEL_KEY = "qwen2.5"
    MODEL_TYPE = "qwen"
    MODEL_LEAF_PREFIX = ""

    def _load_model(self):
        pass

    def _generate(self, *a, **kw):
        return "mock"


def _make_handpicked_item(page_id, images_base_path=None):
    item = {
        "original_question": "What is X?",
        "original_answer_locations": [{"answer": "42", "page_id": page_id}],
        "layout_analysis": {"pages": {page_id: {}}},
    }
    if images_base_path is not None:
        item["images_base_path"] = images_base_path
    return item


# ---- RED tests (all should fail before implementation) ----------------------

def test_get_fs_pool_loads_items_key():
    """_get_fs_pool must load from the 'items' key in the pool file."""
    with tempfile.TemporaryDirectory() as tmp:
        # Write a fake image so the existence check passes
        img = Path(tmp) / "page_0.jpg"
        img.write_bytes(b"")

        pool_file = Path(tmp) / "handpicked.json"
        items = [
            _make_handpicked_item("page_0.jpg", tmp),
            _make_handpicked_item("page_1.jpg", tmp),
        ]
        pool_file.write_text(json.dumps({"description": "test", "items": items}))

        cfg = {**_MOCK_CONFIG, "few_shot": {**_MOCK_CONFIG["few_shot"], "pool_file": str(pool_file)}}
        ev = _StubEvaluator(cfg, finetuned=False)
        ev.images_base_path = tmp

        pool = ev._get_fs_pool([])
        assert len(pool) == 2, f"expected 2 items, got {len(pool)}"
        assert pool[0]["original_question"] == "What is X?"


def test_select_handpicked_returns_first_n_answerable():
    """selection='handpicked' must return the first n_shots items, all type='answerable'."""
    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "page_0.jpg"
        img.write_bytes(b"")

        items = [_make_handpicked_item("page_0.jpg", tmp)] * 4
        pool_file = Path(tmp) / "handpicked.json"
        pool_file.write_text(json.dumps({"items": items}))

        cfg = {**_MOCK_CONFIG, "few_shot": {**_MOCK_CONFIG["few_shot"], "n_shots": 2, "pool_file": str(pool_file)}}
        ev = _StubEvaluator(cfg, finetuned=False)
        ev.images_base_path = tmp

        shots = ev._select_few_shot_examples([], current_item={})
        assert len(shots) == 2, f"expected 2 shots, got {len(shots)}"
        assert all(s["type"] == "answerable" for s in shots), "all shots must be type='answerable'"


def test_handpicked_image_validation_uses_per_item_base_path():
    """Items with images_base_path must be validated against that path, not self.images_base_path."""
    with tempfile.TemporaryDirectory() as tmp:
        item_dir = Path(tmp) / "item_images"
        item_dir.mkdir()
        (item_dir / "page_0.jpg").write_bytes(b"")

        wrong_dir = Path(tmp) / "wrong_dir"
        wrong_dir.mkdir()
        # no image in wrong_dir

        items = [_make_handpicked_item("page_0.jpg", str(item_dir))]
        pool_file = Path(tmp) / "handpicked.json"
        pool_file.write_text(json.dumps({"items": items}))

        cfg = {**_MOCK_CONFIG, "few_shot": {**_MOCK_CONFIG["few_shot"], "n_shots": 1, "pool_file": str(pool_file)}}
        ev = _StubEvaluator(cfg, finetuned=False)
        ev.images_base_path = str(wrong_dir)  # item must NOT be skipped despite this

        shots = ev._select_few_shot_examples([], current_item={})
        assert len(shots) == 1, "item with valid per-item images_base_path must not be skipped"


def test_build_few_shot_turns_uses_per_item_base_path():
    """_build_few_shot_turns must resolve image_paths using item['images_base_path']."""
    with tempfile.TemporaryDirectory() as tmp:
        item_dir = Path(tmp) / "item_images"
        item_dir.mkdir()
        (item_dir / "page_0.jpg").write_bytes(b"")

        cfg = {**_MOCK_CONFIG}
        ev = _StubEvaluator(cfg, finetuned=False)
        ev.images_base_path = "/some/wrong/path"

        item = _make_handpicked_item("page_0.jpg", str(item_dir))
        shots = [{"type": "answerable", "item": item}]
        turns = ev._build_few_shot_turns(shots)

        user_turn = turns[0]
        assert user_turn["image_paths"] == [str(item_dir / "page_0.jpg")], (
            f"expected {item_dir / 'page_0.jpg'}, got {user_turn['image_paths']}"
        )


def test_build_few_shot_turns_skips_ocr_for_minimal_pages():
    """When ocr_enabled=True but demo pages have no layout_analysis, OCR must be skipped (not crash)."""
    with tempfile.TemporaryDirectory() as tmp:
        item_dir = Path(tmp) / "imgs"
        item_dir.mkdir()
        (item_dir / "page_0.jpg").write_bytes(b"")

        cfg = {
            **_MOCK_CONFIG,
            "ocr_enabled": True,  # the condition that triggered the crash
        }
        ev = _StubEvaluator(cfg, finetuned=False)
        ev.images_base_path = str(item_dir)

        # Minimal page dict — no "layout_analysis" key inside the page value (our handpicked format)
        item = _make_handpicked_item("page_0.jpg", str(item_dir))
        shots = [{"type": "answerable", "item": item}]

        # Must not raise KeyError: 'layout_analysis'
        turns = ev._build_few_shot_turns(shots)
        assert len(turns) == 2  # user + assistant turn
        # OCR text in the prompt should be absent (no layout_analysis to extract from)
        assert "OCR" not in turns[0]["text"]


if __name__ == "__main__":
    test_get_fs_pool_loads_items_key()
    test_select_handpicked_returns_first_n_answerable()
    test_handpicked_image_validation_uses_per_item_base_path()
    test_build_few_shot_turns_uses_per_item_base_path()
    test_build_few_shot_turns_skips_ocr_for_minimal_pages()
    print("OK: test_few_shot_handpicked")
