<div align="center">

# VRD-UQA

**Improving Unanswerable Question Detection on Visually Rich Documents: A Comparison of Prompting and Fine-Tuning Strategies for Visual LLMs**

</div>

---

## What is this?

VLLMs are good at document VQA — but do they know when a question _can't_ be answered?

**VRD-UQA** is a benchmark that stress-tests visual LLMs against **plausible but unanswerable questions** on multi-page documents (PDFs, forms, reports). We automatically corrupt valid questions by swapping entities, layout references, or document elements, then verify unanswerability with a VLLM-as-judge, and evaluate model robustness at scale.

This repo also includes a **fine-tuning extension** (LoRA on Qwen2.5-VL-7B) that trains models to explicitly detect unanswerable questions — reducing hallucination rates substantially.

---

## How it works

```
Document + Valid Question
        │
        ▼
  Corruption Engine ──── 3 types: Entity · Layout · Element
        │                3 levels: C1 (simple) · C2 · C3 (complex)
        ▼
  Answerability Verifier (InternVL3.5-8B / Gemini 2.5 Flash as judge)
        │
        ▼
  Evaluation Pipeline ── sliding-window multi-page inference
        │                OCR injection (GOT-OCR 2.0) as ICL signal
        ▼
  Metrics: QUR · FRR · Acc_D · Acc_P
```

---

## Datasets

|           |                                            MPDocVQA                                             |                                              DUDE                                               | BoundingDocs | SlideVQA |
| --------- | :---------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------: | :----------: | :------: |
| Full      |                      [link\*](https://rrc.cvc.uab.es/?ch=17&com=downloads)                      |                      [link\*](https://rrc.cvc.uab.es/?ch=23&com=downloads)                      |      —       |    —     |
| Reduced   | [link](https://drive.google.com/drive/folders/1-SZzvuMJarRDi4rTz6svkVP8MsWTCejO?usp=drive_link) | [link](https://drive.google.com/drive/folders/1URFqchC37AoGMkl0HQP22oAeqM-lV2ns?usp=drive_link) |      —       |    —     |
| Corrupted | [link](https://drive.google.com/drive/folders/1bMjgHAiBJTwDAZu589abNCaMTWKIOXtq?usp=drive_link) | [link](https://drive.google.com/drive/folders/11Yd9l1J-f0FB-E8S5ZTPrSse3Vjie_wl?usp=drive_link) |      —       |    —     |
| Verified  | [link](https://drive.google.com/drive/folders/1fcwycWWO2D9hRjrididVcSXoy6GyPac6?usp=drive_link) | [link](https://drive.google.com/drive/folders/12ltYWllJAoEIkJlbZegnWrrYSul9K6Oy?usp=drive_link) |      —       |    —     |

\* Original dataset repository. Full processed datasets will be released.

ZIP downloads: [MPDocVQA.zip](https://drive.google.com/file/d/1Qn4zG_nCnx0sebhTBHKHpFH41-OEsex2/view?usp=drive_link) · [DUDE.zip](https://drive.google.com/file/d/1JNIB-a1vvXjWDaDedX8JsdioOVAs1_03/view?usp=drive_link)

---

## Models evaluated

| Model             |   Size   |  License   |
| ----------------- | :------: | :--------: |
| Qwen2.5-VL        | 7B / 72B | Apache 2.0 |
| InternVL3         | 9B / 78B |    MIT     |
| Phi-4 Multimodal  |    5B    |    MIT     |
| Gemma 3 / Gemma 4 |   27B    |   Gemma    |
| Molmo             |    7B    | Apache 2.0 |
| Ovis 1.6          |    9B    | Apache 2.0 |
| Llama 3.2 Vision  |   11B    | Llama 3.2  |
| LLaVA 1.6         |   34B    | Apache 2.0 |

Verification judge: **Gemini 2.5 Flash** (paper) / **InternVL3.5-8B** (local runs).

---

## Fine-tuning extension

LoRA fine-tuning of Qwen2.5-VL-7B on a 50/50 balanced dataset (answerable / unanswerable), with binary output supervision (`Answerable.` / `Unanswerable.`). Trained via LLaMA-Factory on an NVIDIA A40 48GB.

Results show a significant reduction in hallucination on corrupted questions, with minimal drop on answerable ones.

---

## Quickstart

```bash
pip install -r requirements.txt
```

See [`example.ipynb`](example.ipynb) for end-to-end usage: corruption → verification → evaluation.

> Tested with `transformers==4.49.0.dev0`. Llama 3.2 and LLaVA 1.6 require [Ollama](https://ollama.com/).

---

## License

CC BY-NC 4.0 — see [LICENSE](LICENSE).

## Citation

_Coming soon._

## Contact

davide.napolitano@polito.it
