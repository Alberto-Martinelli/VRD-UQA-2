#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-23:59:00
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out

# Usage: sbatch --job-name=DUDE run_pipeline.sh DUDE val 300
#        sbatch --job-name=SlideVQA run_pipeline.sh SlideVQA val 300
#        sbatch --job-name=MPDocVQA run_pipeline.sh MPDocVQA val 300
#        sbatch --job-name=BDocs run_pipeline.sh BDocs val 300

set -euo pipefail

DATASET="${1:?Usage: sbatch run_pipeline.sh <DATASET> <SPLIT> <N>}"
SPLIT="${2:?}"
N="${3:?}"

# Map short dataset key to the pipeline's "name" field (BDocs has a space)
case "$DATASET" in
    BDocs)       DATASET_NAME="Bounding Docs" ;;
    *)           DATASET_NAME="$DATASET" ;;
esac

START_TIME=$SECONDS
echo "Job started at: $(date)"
echo "Dataset: $DATASET | Split: $SPLIT | N: $N"

module purge
module load miniconda3/3.13.25
module load nvhpc/25.1

export SCRATCH_FLASH="/mnt/beegfs/amartinelli"
export HF_HOME="$SCRATCH_FLASH/.cache/huggingface"

WORK_DIR="$SCRATCH_FLASH/${DATASET}_${SPLIT}_${N}"
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
rsync -aq --exclude='.git' --exclude='.venv' --exclude='corruption-scripts/results' \
    --exclude='data/DUDE' --exclude='data/BDocs' --exclude='data/SlideVQA' --exclude='data/MPDocVQA' \
    "$HOME/VRD-UQA/" "$WORK_DIR/"

# Re-include only the target dataset's data folder
rsync -a "$HOME/VRD-UQA/data/$DATASET/" "$WORK_DIR/data/$DATASET/"

cd "$WORK_DIR"

# Generate config from template
CONFIG="corruption-scripts/config.generated.json"
sed \
    -e "s|__DATASET__|$DATASET|g" \
    -e "s|__DATASET_NAME__|$DATASET_NAME|g" \
    -e "s|__SPLIT__|$SPLIT|g" \
    -e "s|__N__|$N|g" \
    corruption-scripts/config.template.json > "$CONFIG"

uv --version
export UV_LINK_MODE=copy
export UV_CACHE_DIR=/tmp/uv_cache_$SLURM_JOB_ID
uv venv --python python3
uv sync -qq

# # ------ CORRUPTION PIPELINE ------
# uv run corruption-scripts/corruption/main.py --config "$CONFIG"

# cp "corruption-scripts/results/${DATASET}_unanswerable_corrupted_questions_cleaned.json" "$HOME/VRD-UQA/"

# # ------ ANSWERABILITY VERIFIER ------
# uv run python corruption-scripts/verification/answerability_verifier.py --config "$CONFIG"

# cp "corruption-scripts/results/${DATASET}_unanswerable_corrupted_questions_verified.json" "$HOME/VRD-UQA/"

# # ------ JUST FALSE ------
# uv run python corruption-scripts/verification/just_false.py \
#     --input_file "corruption-scripts/results/${DATASET}_unanswerable_corrupted_questions_verified.json" \
#     --output_file "corruption-scripts/results/${DATASET}_unanswerable_corrupted_questions_just_false.json"

# cp "corruption-scripts/results/${DATASET}_unanswerable_corrupted_questions_just_false.json" "$HOME/VRD-UQA/"


# ------ EXACT UNANSWERABLE QUESTION GENERATION ORCHESTRATOR ------
# This dynamically corrupts, verifies, and check-points until exactly $N (300) verified unanswerable questions are reached.
uv run python corruption-scripts/corruption/generate_exact_dataset.py \
    --config "$CONFIG" \
    --target "$N" \
    --batch_size 25 \
    --output "corruption-scripts/results/${SPLIT}_${N}_${DATASET}_unanswerable_corrupted_questions_just_false.json"

# Copy the final perfect pool of exactly 300 verified questions back to your persistent workspace
cp "corruption-scripts/results/${SPLIT}_${N}_${DATASET}_unanswerable_corrupted_questions_just_false.json" "$HOME/VRD-UQA/"


mv "$HOME"/slurm* "$HOME/VRD-UQA/" 2>/dev/null || true

ELAPSED=$(( SECONDS - START_TIME ))
printf 'Job finished at: %s\nTotal execution time: %02d:%02d:%02d (%ds)\n' \
    "$(date)" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$ELAPSED"
