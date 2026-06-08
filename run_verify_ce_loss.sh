#!/bin/bash

# uv run python vlm_suppress/tests/verify_ce_loss_qwen.py \
#     --image  data/ui_dataset/images/pil/banking_0000.png \
#     --labels data/ui_dataset/labels_pil.jsonl \
#     --decoy  data/ui_dataset/images/pil/banking_0000.png \
#     --model_id Qwen/Qwen2.5-VL-7B-Instruct \
#     --device cuda:0

# .venv-deepseek/bin/python vlm_suppress/tests/verify_ce_loss_deepseek.py \
#     --image  data/ui_dataset/images/pil/banking_0000.png \
#     --labels data/ui_dataset/labels_pil.jsonl \
#     --decoy  data/ui_dataset/images/pil/banking_0001.png \
#     --model_id deepseek-ai/deepseek-vl2-small \
#     --device cuda:0


# uv run python vlm_suppress/tests/verify_ce_loss_internvl.py \
#     --image  data/ui_dataset/images/pil/banking_0000.png \
#     --labels data/ui_dataset/labels_pil.jsonl \
#     --decoy  data/ui_dataset/images/pil/banking_0001.png \
#     --model_id OpenGVLab/InternVL3_5-8B \
#     --device cuda:0


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python vlm_suppress/tests/verify_ce_loss_llama.py \
    --image  data/ui_dataset/images/pil/banking_0000.png \
    --labels data/ui_dataset/labels_pil.jsonl \
    --decoy  data/ui_dataset/images/pil/banking_0001.png \
    --model_id meta-llama/Llama-3.2-11B-Vision-Instruct \
    --device cuda:0


uv run python -m vlm_suppress.tests.phase0_test \
        --model llava1_6 \
        --data-path /data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2/data/ui_dataset/labels_pil.jsonl \
        --category banking \
        --out-dir outputs/phase0_test/llava1_6



    