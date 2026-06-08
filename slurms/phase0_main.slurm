#!/bin/bash
#SBATCH --mail-user=pitp@student.unimelb.edu.au
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH -o /data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2/logs/phase0_eval-%x-%j.out
#SBATCH -e /data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2/logs/phase0_eval-%x-%j.err
#SBATCH -p gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --array=0-3

# ── Phase 0 full-dataset evaluation — main surrogates ─────────────────────
# Array tasks:
#   0 = qwen2_5vl      (~5h  estimated)
#   1 = internvl3_5    (~6h  estimated)
#   2 = llama3_2vl       (~10h estimated)
#   3 = paligemma2     (~4h  estimated)
# ──────────────────────────────────────────────────────────────────────────

MODELS=(
    "qwen2_5vl"
    "internvl3_5"
    "llama3_2vl"
    "paligemma2"
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

PROJECT=/data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2
DATA=/data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2/data/ui_dataset

MANIFEST=$DATA/manifest.csv
GT_DIR=$DATA/ground_truth/pil
OUT_DIR=$PROJECT/outputs/phase0_metrics_full

mkdir -p $PROJECT/logs $OUT_DIR

# ── Environment ────────────────────────────────────────────────────────────
export HF_HOME=/data/gpfs/projects/punim2826/Smith/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $PROJECT

echo "============================================================"
echo " Phase 0 full-dataset eval"
echo " Model:    $MODEL"
echo " Task ID:  $SLURM_ARRAY_TASK_ID"
echo " Node:     $(hostname)"
echo " GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo " Time:     $(date)"
echo "============================================================"

uv run -m vlm_suppress.tests.phase0_eval \
    --model     $MODEL \
    --manifest  $MANIFEST \
    --gt-dir    $GT_DIR \
    --out-dir   $OUT_DIR

echo "Done: $MODEL at $(date)"
