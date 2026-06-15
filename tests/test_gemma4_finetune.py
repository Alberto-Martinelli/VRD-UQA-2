"""CI-safe tests for the Gemma 4 fine-tune data path (no model/GPU needed).
Run: uv run python -m tests.test_gemma4_finetune

Exercises the pure-Python parts of finetuning/gemma4/finetune_gemma4.py — record
-> chat-message text building, answer-only label masking, and the collator — with a
MOCK processor. Real training runs on the A40 via scripts/slurm/run_finetune_gemma4.sh.
"""
import json
import tempfile
from pathlib import Path

import torch
from PIL import Image

from finetuning.gemma4.finetune_gemma4 import (
    VrdUqaGemmaDataset,
    build_collate_fn,
    _IGNORE_INDEX,
)


class _MockTokenizer:
    eos_token = "<eos>"
    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        # 1 id per whitespace token; deterministic, content-independent.
        ids = torch.tensor([[5 + i for i, _ in enumerate(text.split())]])
        r = type("R", (), {})()
        r.input_ids = ids
        return r


class _MockProcessor:
    """Records the last apply_chat_template / __call__ args for assertions."""

    def __init__(self):
        self.tokenizer = _MockTokenizer()
        self.last_messages = None
        self.last_images = None
        self.last_text = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.last_messages = messages
        parts = [c["text"] for m in messages for c in m["content"] if c["type"] == "text"]
        return "PROMPT " + " ".join(parts)

    def __call__(self, images=None, text=None, return_tensors=None):
        self.last_images = images
        self.last_text = text
        n = max(1, len(text.split()))

        class _R(dict):
            def __init__(self, d):
                super().__init__(d)
                self.input_ids = d["input_ids"]

        return _R({
            "input_ids": torch.arange(100, 100 + n).unsqueeze(0),
            "pixel_values": torch.zeros(1, 3, 48, 48),
        })


def _fixture(tmp, n=2):
    img_path = Path(tmp) / "p.jpg"
    Image.new("RGB", (32, 32), (255, 255, 255)).save(img_path)
    recs = [{
        "instruction": "GUIDE",
        "input": "<image>\nWhat is X?",
        "output": "ANS WORD",
        "images": [str(img_path)],
    } for _ in range(n)]
    j = Path(tmp) / "train.json"
    j.write_text(json.dumps(recs))
    return str(j)


def test_getitem_builds_message_and_masks_labels():
    tmp = tempfile.mkdtemp()
    proc = _MockProcessor()
    ds = VrdUqaGemmaDataset(proc, _fixture(tmp))
    item = ds[0]

    # message carries an image placeholder + combined instruction/question text
    content = proc.last_messages[0]["content"]
    assert {c["type"] for c in content} == {"image", "text"}
    text_part = [c["text"] for c in content if c["type"] == "text"][0]
    assert text_part == "GUIDE\nWhat is X?"      # <image> stripped, instruction prepended
    assert len(proc.last_images) == 1            # exactly one PIL image passed

    # labels: only the answer span is supervised; everything before is masked
    ans_text = ds.data[0]["output"] + proc.tokenizer.eos_token
    n_answer = proc.tokenizer(ans_text).input_ids.shape[1]
    assert item["input_ids"].shape == item["labels"].shape
    assert (item["labels"][0, :-n_answer] == _IGNORE_INDEX).all()
    assert (item["labels"][0, -n_answer:] != _IGNORE_INDEX).all()
    assert "pixel_values" in item                # image tensor carried through


def test_collate_pads_and_stacks():
    tmp = tempfile.mkdtemp()
    proc = _MockProcessor()
    ds = VrdUqaGemmaDataset(proc, _fixture(tmp, n=2))
    batch = [ds[0], ds[1]]
    collate = build_collate_fn(pad_token_id=0)
    out = collate(batch)
    assert out["input_ids"].shape[0] == 2
    assert out["labels"].shape == out["input_ids"].shape
    assert out["attention_mask"].shape == out["input_ids"].shape
    assert out["pixel_values"].shape[0] == 2     # image tensors stacked along dim 0


if __name__ == "__main__":
    test_getitem_builds_message_and_masks_labels()
    test_collate_pads_and_stacks()
    print("OK: gemma4 finetune data path")
