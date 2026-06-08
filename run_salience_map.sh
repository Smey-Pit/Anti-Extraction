#!/bin/bash

uv run python vlm_suppress/tests/Phase_1/salience_map/salience_map_qwen.py \
    --image  data/ui_dataset/images/pil/banking_0000.png \
    --labels data/ui_dataset/labels_pil.jsonl \
    --gt     data/ui_dataset/ground_truth/pil/banking_0000.json \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --out    outputs/phase_1/salience_maps/qwen/ \
    --device cuda:0