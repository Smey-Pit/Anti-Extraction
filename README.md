This is a README for anti-extraction text-in-image


#before regenerating new synthetic dataset
rm -rf data/synthetic/images data/synthetic/labels.jsonl data/synthetic/labels.json outputs/ data/sythetic/preview.png

#running probe
uv run python scripts/run_probe.py \
  --config configs/probe.yaml \
  --norm linf \
  --steps 50

#running attack
uv run python scripts/run_attack.py --config configs/attack.yaml