# Domain 2 — UI Web Screenshot Dataset

Paired synthetic web portal images with LLM-generated sensitive text content.
Designed for adversarial text suppression research (anti-scraping / anti-extraction).

---

## File overview

| File | Purpose |
|---|---|
| `backends.py` | LLM backend abstraction (Anthropic, Qwen-API, Qwen-local) |
| `generate_content.py` | **Step 1** — LLM calls only, saves `content_bank.json` |
| `pil_renderer.py` | Varied PIL image renderer (font, size, layout, colour) |
| `render_images.py` | **Step 2** — renders images from a saved content bank |

---

## Installation

```bash
# Core dependencies (always required)
pip install anthropic pillow openai --break-system-packages

# For Playwright HTML renders (optional, not needed on HPC)
pip install playwright --break-system-packages
playwright install chromium

# For local Qwen (GPU required)
pip install transformers torch accelerate --break-system-packages
pip install bitsandbytes --break-system-packages   # optional, for 4-bit quantisation
```

---

## Quickstart — 1000 images with Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Step 1: generate content (250 per category = 1000 total)
python generate_content.py --total 1000 --backend anthropic --out content_bank.json

# Step 2: render images
python render_images.py --content content_bank.json --out domain2_ui_dataset/
```

Output:
```
domain2_ui_dataset/
├── images/pil/
│   ├── banking_0001.png
│   ├── medical_0002.png
│   └── ...
├── labels_pil.jsonl     ← one JSON object per line, matches domain1 schema
└── labels_pil.json      ← pretty-printed full array
```

---

## Step 1 — generate_content.py

Calls the LLM backend and saves structured content to a JSON file.
**No images are produced here.** This is the only step that requires API access or GPU.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--total N` | — | Total items split evenly across all 4 categories |
| `--per-category CAT=N ...` | — | Custom split, e.g. `banking=300 medical=300 news=200 copyright=200` |
| `--out PATH` | `content_bank.json` | Output path |
| `--resume` | off | Append to existing bank instead of overwriting (safe after crash) |
| `--backend` | `anthropic` | `anthropic` / `qwen-api` / `qwen-local` |

Exactly one of `--total` or `--per-category` must be given. Default (neither) = 250 per category.

### Backend options

**Anthropic (default)**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python generate_content.py --total 1000 --backend anthropic
```

**Qwen via Ollama** (OpenAI-compatible local server)
```bash
ollama pull qwen2.5:7b
python generate_content.py --total 1000 --backend qwen-api \
    --api-base-url http://localhost:11434/v1 \
    --api-model qwen2.5:7b
```

**Qwen via vLLM** (OpenAI-compatible local server)
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct --port 8000

python generate_content.py --total 1000 --backend qwen-api \
    --api-base-url http://localhost:8000/v1 \
    --api-model Qwen/Qwen2.5-7B-Instruct
```

**Qwen local** (loads model directly via transformers, GPU required)
```bash
# Full precision (~15 GB VRAM)
python generate_content.py --total 1000 --backend qwen-local

# 4-bit quantised (~5 GB VRAM, fits a 3090)
python generate_content.py --total 1000 --backend qwen-local --load-in-4bit
```

**Together AI** (cloud hosted Qwen)
```bash
export TOGETHER_API_KEY=...
python generate_content.py --total 1000 --backend qwen-api \
    --api-base-url https://api.together.xyz/v1 \
    --api-model Qwen/Qwen2.5-7B-Instruct-Turbo \
    --api-key $TOGETHER_API_KEY
```

### Resume after crash

The script saves incrementally after each category batch completes, so if it
crashes mid-run you only lose the current batch (at most 5–10 items):

```bash
python generate_content.py --total 1000 --out content_bank.json --resume
```

---

## Step 2 — render_images.py

Renders images from the saved content bank. **No LLM calls. No GPU needed.**
Safe to run on HPC CPU nodes.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--content PATH` | `content_bank.json` | Content bank from Step 1 |
| `--out DIR` | `domain2_ui_dataset` | Output directory |
| `--seed N` | `42` | Layout seed — controls font, size, canvas, margins |
| `--categories` | all 4 | Subset to render: `banking medical news copyright` |
| `--mode LABEL` | `pil` | Label written into annotations (useful for paired renders) |
| `--dry-run` | off | Print plan without writing files |

### Basic render
```bash
python render_images.py --content content_bank.json --out domain2_ui_dataset/
```

### Paired renders from the same content

This is the key feature for paper-quality experiments. Run Step 2 multiple
times with the same content bank but different seeds or mode labels:

```bash
# Render A — seed 42
python render_images.py --content content_bank.json \
    --seed 42 --mode pil_s42 --out domain2_ui_dataset/

# Render B — seed 99 (same text, different fonts/layouts)
python render_images.py --content content_bank.json \
    --seed 99 --mode pil_s99 --out domain2_ui_dataset/
```

Both runs produce images of **identical text content** with **different visual
layouts**. This lets you isolate whether attack success comes from content
difficulty or visual rendering properties — a clean ablation the paper needs.

### Custom category split only
```bash
python render_images.py --content content_bank.json \
    --categories banking medical --out banking_medical_only/
```

### Dry run first (recommended for large jobs)
```bash
python render_images.py --content content_bank.json --dry-run
```

---

## What varies between renders

Each image is independently randomised from `(global_seed, content_id)`:

| Property | Options |
|---|---|
| Canvas width | 1024, 1152, 1280, 1440 px |
| Canvas height | 800, 900, 1024, 1100 px |
| Font family | sans-serif, serif, monospace |
| Body font size | 12, 13, 14, 15, 16 pt |
| Line spacing | 1.30×, 1.40×, 1.50×, 1.60×, 1.75× |
| Left margin | 24, 32, 40, 48, 56 px |
| Section gap | 8, 10, 12, 16, 20 px |
| Header colour | 4 variants per category |
| Background | white / off-white / light tint (3 options) |

This is what was causing the identical images in the original pipeline —
it had hardcoded `y` positions and a single DejaVu font at fixed sizes.

---

## Annotation schema (labels_pil.jsonl)

Drop-in compatible with domain1 `labels.jsonl`. Each line is a JSON object:

```json
{
  "image_id":            "banking_0001",
  "image_path":          "images/pil/banking_0001.png",
  "full_text":           "Northgate Bank\nJane Torres\n...",
  "domain":              "ui_web",
  "category":            "banking",
  "raw_content":         { "bank_name": "Northgate Bank", ... },
  "render_mode":         "pil",
  "render_seed":         42,
  "layout": {
    "W": 1280, "H": 900,
    "font_family": "serif",
    "body_size": 14,
    "line_spacing": 1.5,
    "margin_left": 40
  },
  "split":               "train",
  "has_ambiguous_chars": false,
  "layout_type":         "web_portal",
  "text_category":       "banking"
}
```

`full_text` is the ground-truth string for CER/WER computation.
`layout` lets you stratify results by font family or size in your analysis.

---

## HPC usage

Step 1 (LLM generation) requires internet or GPU — run on a login node or
GPU node. Step 2 (rendering) is CPU-only and can be parallelised:

```bash
#!/bin/bash
#SBATCH --job-name=render_ui
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00

python render_images.py \
    --content content_bank.json \
    --out domain2_ui_dataset/ \
    --seed 42
```

To parallelise by category across SLURM array jobs:

```bash
#!/bin/bash
#SBATCH --array=0-3

CATS=(banking medical news copyright)
CAT=${CATS[$SLURM_ARRAY_TASK_ID]}

python render_images.py \
    --content content_bank.json \
    --categories $CAT \
    --out domain2_ui_dataset/ \
    --seed 42
```

---

## Categories

| Category | Portal style | Sensitive content |
|---|---|---|
| `banking` | Online banking statement | Account numbers, transactions, balances |
| `medical` | Patient health portal | Diagnoses, medications, lab results |
| `news` | News / opinion site | Political sentiment, named sources |
| `copyright` | Document / ebook reader | Book prose, screenplay, feature journalism |