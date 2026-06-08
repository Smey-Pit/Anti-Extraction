#!/bin/bash

# #TASK 1 : Creating ground truth from labels_pil
# uv run python dataset_generation/UI/build_ground_truth.py \
#     --labels /data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2/data/ui_dataset/labels_pil.jsonl \
#     --out-dir /data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2/data/ui_dataset/ground_truth/pil

# #TASK 2 : Running smoke test for phase0 eval, with Qwen2.5VL
# uv run -m vlm_suppress.tests.phase0_eval \
#     --model qwen2_5vl \
#     --manifest /data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2/data/ui_dataset/manifest.csv \
#     --gt-dir   /data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2/data/ui_dataset/ground_truth/pil \
#     --out-dir  /data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2/outputs/phase0_metrics \
#     --one-per-domain

# #TASK 3 : Running one-per-domain for phase0 eval for all surrogates
for m in paligemma2; do
    echo "=== $m ==="
    uv run -m vlm_suppress.tests.phase0_eval \
        --model $m \
        --manifest /data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2/data/ui_dataset/manifest_50.csv \
        --gt-dir   /data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2/data/ui_dataset/ground_truth/pil \
        --out-dir  /data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2/outputs/phase0_metrics_50
done

# .venv-deepseek/bin/python -m vlm_suppress.tests.phase0_eval \
#     --model deepseekvl2 \
#     --manifest /data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2/data/ui_dataset/manifest_50.csv \
#     --gt-dir   /data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2/data/ui_dataset/ground_truth/pil \
#     --out-dir  /data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2/outputs/phase0_metrics_50


# # Task 4: Create a sample manifest for testing phase 0
# uv run vlm_suppress/tests/sample_manifest.py \
#     --manifest /data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2/data/ui_dataset/manifest.csv \
#     --out      /data/gpfs/projects/punim2826/Smith/Anti-Extraction-v2/data/ui_dataset/manifest_50.csv \
#     --n 50 --seed 42