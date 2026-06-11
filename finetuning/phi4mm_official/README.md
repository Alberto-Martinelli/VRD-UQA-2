# Phi-4-multimodal fine-tuning — official-sample fallback

Use this path ONLY if `run_finetune_vlm.sh phi4mm smoke` (LLaMA-Factory, `phi4`
template) fails to train Phi-4-multimodal's vision path. Phi-4-mm ships internal
speech/vision LoRA adapters, which LLaMA-Factory may not handle.

## Procedure
1. Clone Microsoft's Phi-4-multimodal finetuning sample (from the model card /
   `microsoft/Phi-4-multimodal-instruct` repo) into this directory.
2. Convert `artifacts/finetuning/dataset/train.json` (+ `val.json`) — the same
   model-agnostic VRD-UQA SFT data registered as `vrd_uqa_train`/`vrd_uqa_val` —
   into the sample's expected format (image path + prompt + target).
3. Train a LoRA on the language path (keep the vision adapter frozen), batch_size 1,
   matching the LoRA rank/alpha/lr from `phi4mm_lora_sft.yaml` (16 / 32 / 2e-5) as
   closely as the sample allows.
4. Export the result so eval can consume it:
   - If it produces a PEFT adapter → copy to `artifacts/finetuning/phi4mm_lora_sft`
     and keep the eval `phi4_finetuned` entry as-is (adapter_path).
   - Else merge with `finetuning/merge_lora.py` → set `phi4_finetuned.model_name`
     to the merged dir and remove its `adapter_path`.

## Decision log
- [ ] LLaMA-Factory `phi4` template result: __pass / fail__ (fill after Task 3 Step 4)
- [ ] Fallback used: __yes / no__
