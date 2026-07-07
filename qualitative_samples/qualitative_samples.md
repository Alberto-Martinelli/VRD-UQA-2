# Qualitative Samples — Zero-Shot Hallucination vs Fine-Tuned Refusal
*All images verified on disk. ZS source: `zeroshot/augmented/` (Qwen) or `zeroshot_ocr/predictions.json` (Gemma4, Phi4). FT source: `finetuned_ocr/predictions.json`.*

---

## Sample 1 — Entity Corruption (Person Name) | Qwen2.5-VL | DUDE | C1

| Field | Value |
|-------|-------|
| **Model** | Qwen2.5-VL-7B |
| **Corruption type** | Entity — person name swap |
| **Dataset** | DUDE |
| **Complexity** | C1 |
| **Original question** | What was the recommendation given by CareFirst CEO **Chet Burrell**? |
| **Corrupted question** | What was the recommendation given by CareFirst CEO **Governor O'Malley**? |
| **Gold answer** | *the state continue making progress with our current system, rather than switch to a partnership with the FFM, at this time* |
| **Zero-shot answer** | *"The recommendation given by CareFirst CEO Chet Burrell was that the state continue making progress with their current system, rather than switch to a partnership with the Federal Exchange at this time."* |
| **Fine-tuned answer** | **Unable to determine** |
| **Document page** | `/mnt/beegfs/amartinelli/DUDE_images/8eef3fb74f87f6eeb784d8489a100e48_0.jpg` |

**Why it illustrates the thesis:** The zero-shot model ignores the corrupted entity (Governor O'Malley) and answers as if the question asked about Chet Burrell — the real CEO mentioned in the document. It produces the correct answer to the wrong question. The fine-tuned model correctly detects that "Governor O'Malley" is not the person whose recommendation is documented and refuses.

---

## Sample 2 — Element Corruption (Acronym) | Qwen2.5-VL | MPDocVQA | C1

| Field | Value |
|-------|-------|
| **Model** | Qwen2.5-VL-7B |
| **Corruption type** | Element — acronym replaced with unrelated term |
| **Dataset** | MPDocVQA |
| **Complexity** | C1 |
| **Original question** | What does **EC** stand for? |
| **Corrupted question** | What does **Data Points** stand for? |
| **Gold answer** | *Error Correct* |
| **Zero-shot answer** | *"Data Points stands for the specific details recorded in the table, such as the remote station identifier, start time, duration, number of pages, communication mode, and results of the communication session."* |
| **Fine-tuned answer** | **Unable to determine** |
| **Document page** | `/mnt/beegfs/amartinelli/MPDocVQA_images/fglc0003_p0.jpg` |

**Why it illustrates the thesis:** "Data Points" is not an acronym defined anywhere in the document. The zero-shot model fabricates a plausible-sounding definition by describing column headers in a table it sees. The fine-tuned model recognises that no expansion of "Data Points" is present in the document and refuses.

---

## Sample 3 — Layout Corruption (Spatial Reference) | Qwen2.5-VL | BoundingDocs | C1

| Field | Value |
|-------|-------|
| **Model** | Qwen2.5-VL-7B |
| **Corruption type** | Layout — spatial/positional label replaced |
| **Dataset** | BoundingDocs |
| **Complexity** | C1 |
| **Original question** | What is **Flight To**? |
| **Corrupted question** | What is the **purpose of Flight sections**? |
| **Gold answer** | *11/04/14* |
| **Zero-shot answer** | *"The purpose of the 'Flight Dates' section is to specify the time frame during which the advertisement will be aired. In this case, it indicates that the advertisement for Pam Bondi's campaign will be broadcast from October 30, 2014, to November 4, 2014."* |
| **Fine-tuned answer** | **Unable to determine** |
| **Document page** | `/mnt/beegfs/amartinelli/BDocs_images/WKMG-TV_14146759775631_p0.jpg` |

**Why it illustrates the thesis:** The corrupted question asks for the "purpose" of a section rather than the value of a specific labelled field. The zero-shot model produces a plausible narrative about what flight-date fields are for, completely fabricating an explanation. The fine-tuned model recognises that no "purpose" is stated in the document and refuses.

---

## Sample 4 — Element Corruption (Section Name, C2) | Qwen2.5-VL | DUDE | C2

| Field | Value |
|-------|-------|
| **Model** | Qwen2.5-VL-7B |
| **Corruption type** | Element — double entity swap (section name × 2) |
| **Dataset** | DUDE |
| **Complexity** | C2 |
| **Original question** | What are the values included in the second paragraph of the **real estate assessment**? |
| **Corrupted question** | What are the **Real Estate Assessments** included in the second paragraph of the **CC**? |
| **Gold answer** | *$140,167,165* |
| **Zero-shot answer** | *"$140,167,165 for the land, $221,206,700 for improvements"* |
| **Fine-tuned answer** | **Unable to determine** |
| **Document page** | `/mnt/beegfs/amartinelli/DUDE_images/c0787b13e291418819152f4e6d36fff8_0.jpg` |

**Why it illustrates the thesis:** This C2 example has two corrupted entities. The document has a real estate assessment section but no section called "CC". The zero-shot model extracts the real dollar figures as if the question were valid — a textbook hallucination. The fine-tuned model correctly determines that the "CC" section does not exist and refuses.

---

## Sample 5 — Entity Corruption (Person Name → Pronoun) | Gemma4 | SlideVQA | C1

| Field | Value |
|-------|-------|
| **Model** | Gemma4 |
| **Corruption type** | Entity — named person replaced with unresolvable pronoun |
| **Dataset** | SlideVQA |
| **Complexity** | C1 |
| **Original question** | What type of blood flow will Associate Professor **Marta R. Gerasymchuk** describe typical disorders of? |
| **Corrupted question** | What type of blood flow will Associate Professor **him** describe typical disorders of? |
| **Gold answer** | *Microcirculation* |
| **Zero-shot answer** | *"Microcirculation and peripheral blood flow"* (repeated across most windows) |
| **Fine-tuned answer** | Window 1 (title slide): *"Microcirculation and peripheral blood flow"* — Windows 2–20: **Unable to determine.** |
| **Document page** | `/mnt/beegfs/amartinelli/SlideVQA_images/typicaldisordersofperipheralbloodflow-131024050639-phpapp01_95_1.jpg` |

**Why it illustrates the thesis:** The zero-shot model resolves the generic pronoun "him" from visual context and answers confidently across almost all windows, failing to detect the referent ambiguity. The fine-tuned model is markedly more discriminating: only window 1 (the title slide, where the professor's name is visible) produces an answer; all 19 subsequent content slides elicit "Unable to determine" because the pronoun cannot be resolved from slide content alone. This graduated refusal behaviour is precisely the calibration the fine-tuning aims to install.

---

## Sample 6 — Element Corruption (Category Label) | Phi-4 | SlideVQA | C1

| Field | Value |
|-------|-------|
| **Model** | Phi-4 |
| **Corruption type** | Element — chart category label replaced with non-existent term |
| **Dataset** | SlideVQA |
| **Complexity** | C1 |
| **Original question** | What percent is shown for **SOMEWHAT EASIER**? |
| **Corrupted question** | What percent is shown for **children's book**? |
| **Gold answer** | *3%* |
| **Zero-shot answer** | *"0"* (i.e., 0%, fabricated for a non-existent chart category) |
| **Fine-tuned answer** | **Unable to determine** (all 20 windows) |
| **Document page** | `/mnt/beegfs/amartinelli/SlideVQA_images/stateofstartups2015-151130013757-lva1-app6891_95_15.jpg` |

**Why it illustrates the thesis:** The slide shows a percentage breakdown by category (including "SOMEWHAT EASIER" at 3%). "Children's book" appears nowhere in the document. The zero-shot model hallucinates a 0% figure — inventing a value for a category that does not exist in the chart. The fine-tuned model correctly refuses across all 20 sliding windows, demonstrating that LoRA training generalises the refusal behaviour to unseen SlideVQA document types.

---

## How these were selected

Cross-matched items from:
- ZS (Qwen): `qwen_all_val_300/{dataset}/zeroshot/augmented/Qwen_vqa_analysis_results_converted_augmented.json`
- ZS (Gemma4, Phi4): `{model}_all_val_300/{dataset}/zeroshot_ocr/predictions.json`
- FT (all models): `{model}_all_val_300/{dataset}/finetuned_ocr/predictions.json`

**Confirmed:** Qwen `augmented/` and `converted/` files are identical for all picks (same 300 items, same answer text, verified manually).

Selection criteria:
1. ZS first-window answer does **not** contain "Unable to determine"
2. FT answers contain "Unable to determine" in the majority of windows
3. Page image exists on `/mnt/beegfs/amartinelli/`
4. Coverage: entity / element / layout corruption types × DUDE / MPDocVQA / BDocs / SlideVQA datasets × Qwen2.5-VL / Gemma4 / Phi-4 models
