1) Build the dataset (already done, but if you change it):
    python finetuning/build_train_dataset.py

2) Submit the job:
    sbatch finetuning/run_finetune_qwen25vl.sh

3) Plot results after it finishes:
    python finetuning/plot_training.py /mnt/beegfs/amartinelli/finetune_out/qwen25vl_lora_smoke/trainer_log.jsonl

Output PNG lands next to the log file.

4) Find your trained LoRA weights at:
    /mnt/beegfs/amartinelli/finetune_out/qwen25vl_lora_smoke/
