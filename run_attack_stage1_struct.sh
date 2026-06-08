
#!/bin/bash
IMAGE_A="data/ui_dataset/images/pil/banking_0000.png"
GT_PATH="data/ui_dataset/ground_truth/pil/banking_0000.json"

uv run python vlm_suppress/attack/stage1_struct.py \
    --image  $IMAGE_A \
    --labels data/ui_dataset/labels_pil.jsonl \
    --gt     $GT_PATH \
    --epsilon 255 \
    --steps  100 \
    --eval-every 10 \
    --out    outputs/stage1/