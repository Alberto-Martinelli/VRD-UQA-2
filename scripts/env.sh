#!/usr/bin/env bash
# Single source of truth for environment roots (bash side).
# Mirror of config/paths.py defaults — keep the two in sync.
# Override any value by exporting it BEFORE sourcing this file.
export SCRATCH_FLASH="${SCRATCH_FLASH:-/mnt/beegfs/amartinelli}"
export MPDOCVQA_SOURCE_QAS="${MPDOCVQA_SOURCE_QAS:-/home/amartinelli/MPDocVQA/MPDocVQA_complete/qas}"
export HF_HOME="$SCRATCH_FLASH/.cache/huggingface"
# Persistent repo root (rsync source / data root). On SLURM the running repo is
# a scratch copy with data/ excluded, so run_eval reads data/ from here instead.
export VRD_UQA_HOME="${VRD_UQA_HOME:-$HOME/VRD-UQA}"
