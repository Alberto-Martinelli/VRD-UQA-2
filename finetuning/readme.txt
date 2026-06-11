1) Build the dataset (already done, but if you change it):
    python finetuning/build_train_dataset.py

2) Submit the job:
    sbatch finetuning/run_finetune_qwen25vl.sh

3) Plot results after it finishes:
    python finetuning/plot_training.py /mnt/beegfs/amartinelli/finetune_out/qwen25vl_lora_smoke/trainer_log.jsonl

Output PNG lands next to the log file.

4) Find your trained LoRA weights at:
    /mnt/beegfs/amartinelli/finetune_out/qwen25vl_lora_smoke/

--- Multi-model LoRA fine-tuning (parallel, one model per job) ---
Run all three in parallel (no collision; each uses its own scratch dir + venv + HF cache):
    for M in internvl35 phi4mm; do
        sbatch --job-name=ft-$M scripts/slurm/run_finetune_vlm.sh $M
    done
Smoke first (fast config check): append 'smoke' ->  run_finetune_vlm.sh <M> smoke
Adapters land in: artifacts/finetuning/<M>_lora_sft/   (consumed by the *_finetuned eval entries)
Qwen still uses its original script: sbatch scripts/slurm/run_finetune_qwen25vl.sh
