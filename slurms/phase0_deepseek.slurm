#!/bin/bash
#SBATCH --mail-user=pitp@student.unimelb.edu.au
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH -o /data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2/logs/phase0_deepseek-%j.out
#SBATCH -e /data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2/logs/phase0_deepseek-%j.err
#SBATCH -p gpu-h100
#SBATCH --gres=gpu:1

# ── Phase 0 full-dataset evaluation — DeepSeek-VL2 ────────────────────────
# Separate from the main array because DeepSeek lives in .venv-deepseek
# and uses a different Python interpreter.
# Estimated runtime: ~7h
# ──────────────────────────────────────────────────────────────────────────

PROJECT=/data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2
DATA=/data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2/data/ui_dataset

MANIFEST=$DATA/manifest.csv
GT_DIR=$DATA/ground_truth/pil
OUT_DIR=$PROJECT/outputs/phase0_metrics_full

mkdir -p $PROJECT/logs $OUT_DIR

export HF_HOME=/data/gpfs/projects/punim2826/Smith/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $PROJECT

echo "============================================================"
echo " Phase 0 full-dataset eval — DeepSeek-VL2"
echo " Node: $(hostname)"
echo " GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo " Time: $(date)"
echo "============================================================"

$PROJECT/.venv-deepseek/bin/python -m vlm_suppress.tests.phase0_eval \
    --model    deepseekvl2 \
    --manifest $MANIFEST \
    --gt-dir   $GT_DIR \
    --out-dir  $OUT_DIR

echo "Done: deepseek_vl2 at $(date)"
