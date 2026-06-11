1) Build the dataset (already done, but if you change it):
    python finetuning/build_train_dataset.py

2) Submit the job:
    sbatch finetuning/run_finetune_qwen25vl.sh

3) Plot results after it finishes:
    python finetuning/plot_training.py /mnt/beegfs/amartinelli/finetune_out/qwen25vl_lora_smoke/trainer_log.jsonl

Output PNG lands next to the log file.

4) Find your trained LoRA weights at:
    /mnt/beegfs/amartinelli/finetune_out/qwen25vl_lora_smoke/

--- Multi-model LoRA fine-tuning ---
InternVL3.5 via LLaMA-Factory (append 'smoke' for a fast config check):
    sbatch --job-name=ft-internvl35 scripts/slurm/run_finetune_vlm.sh internvl35
Phi-4-multimodal via Microsoft's official path (LLaMA-Factory can't train its vision path):
    sbatch scripts/slurm/run_finetune_phi4mm_official.sh smoke   # then drop 'smoke' for the full run
Trained adapters/models land in artifacts/finetuning/ (consumed by the *_finetuned eval entries).
Qwen still uses its original script: sbatch scripts/slurm/run_finetune_qwen25vl.sh
