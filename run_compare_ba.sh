#!/bin/bash

# --- Configurable Paths ---
IMAGE_B="/data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2/outputs/stage1/banking_0000_clean_w_decoy_v2.png" #this is the decoy

# Optional: You can also centralize these to make the script even cleaner
SCRIPT_PATH="/data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2/vlm_suppress/tests/Phase_1/compare_ba.py"
IMAGE_A="data/ui_dataset/images/pil/banking_0000.png"
GT_PATH="data/ui_dataset/ground_truth/pil/banking_0000.json"

# --- Model Execution ---

# Qwen
uv run python "$SCRIPT_PATH" \
    --model    qwen2_5vl \
    --image-a  "$IMAGE_A" \
    --image-b  "$IMAGE_B" \
    --gt       "$GT_PATH" \
    --label-a  "clean" \
    --label-b  "shuffled+decoy" \
    --device   cuda:0

# InternVL
uv run python "$SCRIPT_PATH" \
    --model    internvl3_5 \
    --image-a  "$IMAGE_A" \
    --image-b  "$IMAGE_B" \
    --gt       "$GT_PATH" \
    --label-a  "clean" \
    --label-b  "shuffled+decoy" \
    --device   cuda:0

# Llama
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python "$SCRIPT_PATH" \
    --model    llama3_2vl \
    --image-a  "$IMAGE_A" \
    --image-b  "$IMAGE_B" \
    --gt       "$GT_PATH" \
    --label-a  "clean" \
    --label-b  "shuffled+decoy" \
    --device   cuda:0

# DeepSeek
.venv-deepseek/bin/python "$SCRIPT_PATH" \
    --model    deepseekvl2 \
    --image-a  "$IMAGE_A" \
    --image-b  "$IMAGE_B" \
    --gt       "$GT_PATH" \
    --label-a  "clean" \
    --label-b  "shuffled+decoy" \
    --device   cuda:0