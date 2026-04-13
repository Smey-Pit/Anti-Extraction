"""
backends.py
===========
Pluggable LLM backend abstraction for the UI dataset generator.

All backends implement the same interface:

    backend = build_backend(cfg: BackendConfig) -> LLMBackend
    results = backend.generate(category: str, n: int) -> list[dict]

Supported backends
------------------
anthropic   — Claude via the Anthropic Python SDK (original behaviour).

qwen-local  — Qwen2.5-7B-Instruct (or any HF model) loaded locally via
              the `transformers` pipeline. Requires a CUDA or MPS GPU;
              falls back to CPU with a warning.

qwen-api    — Any model served behind an OpenAI-compatible /v1/chat/completions
              endpoint (Ollama, vLLM, Together AI, LM Studio, …). Pass the
              base URL and model name via BackendConfig.

Adding a new backend
--------------------
1. Subclass LLMBackend and implement generate().
2. Add it to build_backend().
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# SHARED PROMPT MATERIAL  (identical for all backends)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are a synthetic data generator for adversarial ML research.
    Generate realistic but entirely fictional sensitive text content.
    All names, account numbers, diagnoses, and details must be invented — never real people.
    Respond ONLY with a JSON array. No markdown, no preamble, no explanation.
""").strip()

CATEGORY_PROMPTS: dict[str, str] = {
    "banking": textwrap.dedent("""
        Generate {n} distinct fictional bank statement page contents.
        Each item is a JSON object with these fields:
          - bank_name: string (invented bank name)
          - account_holder: string (fictional full name)
          - account_number: string (masked, e.g. ****4821)
          - account_type: string (e.g. "Everyday Savings", "Business Cheque")
          - statement_period: string (e.g. "1 Oct – 31 Oct 2024")
          - opening_balance: string (e.g. "$4,821.33")
          - closing_balance: string
          - transactions: array of 4–7 objects, each with:
              date, description, amount (negative for debits), running_balance
          - summary_note: string (1–2 sentences, e.g. flagged overdraft, large transfer note)
        Make amounts, merchants, and patterns realistic and varied.
        Return a JSON array of {n} objects.
    """),
    "medical": textwrap.dedent("""
        Generate {n} distinct fictional patient portal page contents.
        Each item is a JSON object with these fields:
          - hospital_name: string (invented)
          - patient_name: string (fictional)
          - dob: string
          - patient_id: string
          - visit_date: string
          - attending_physician: string (fictional Dr name)
          - chief_complaint: string
          - diagnosis: string (realistic ICD-style, e.g. "Type 2 Diabetes Mellitus (E11.9)")
          - medications: array of strings (name, dose, frequency)
          - lab_results: array of objects with: test, value, reference_range, flag (Normal/High/Low)
          - clinical_notes: string (2–4 sentences of realistic clinical narrative)
          - follow_up: string
        Make diagnoses, medications, and lab values medically coherent.
        Return a JSON array of {n} objects.
    """),
    "news": textwrap.dedent("""
        Generate {n} distinct fictional news article page contents with political or opinion content.
        Each item is a JSON object with these fields:
          - outlet_name: string (invented newspaper/outlet)
          - headline: string (politically charged but fictional)
          - byline: string (fictional journalist name)
          - dateline: string (city, date)
          - category_tag: string (e.g. "Opinion", "Politics", "Analysis")
          - lead_paragraph: string (2–3 sentences, strong political framing)
          - body_paragraphs: array of 3–4 strings (continuation of article)
          - pull_quote: string (striking quote attributed to fictional person)
          - tags: array of strings (topic tags)
        Cover varied political angles — don't skew one direction.
        Make content feel like real journalism, not a parody.
        Return a JSON array of {n} objects.
    """),
    "copyright": textwrap.dedent("""
        Generate {n} distinct fictional copyright-style content page contents.
        Mix types: book excerpts, screenplay pages, newspaper features.
        Each item is a JSON object with these fields:
          - content_type: one of "book_excerpt", "screenplay", "newspaper_feature"
          - title: string (fictional work title)
          - author: string (fictional author name)
          - publisher: string (invented publisher)
          - copyright_line: string (e.g. "© 2023 Elara Voss. All rights reserved.")
          - page_number: integer
          - content: string (150–250 words of realistic fictional prose/script/feature)
          - chapter_or_scene: string (e.g. "Chapter 4: The Last Signal" or "INT. KITCHEN - NIGHT")
        For screenplays use proper slug lines, action lines, and dialogue.
        For books write literary prose. For newspaper features write feature journalism.
        Return a JSON array of {n} objects.
    """),
}


def _clean_json(raw: str) -> list[dict]:
    """Strip markdown fences and parse JSON. Returns list or raises ValueError."""
    raw = raw.strip()
    # Strip ```json ... ``` or ``` ... ```
    if raw.startswith("```"):
        parts = raw.split("```")
        # parts[1] is the content between first pair of fences
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data)}")
    return data


# ---------------------------------------------------------------------------
# BASE CLASS
# ---------------------------------------------------------------------------

class LLMBackend(ABC):
    """Abstract base — all backends implement generate()."""

    @abstractmethod
    def generate(self, category: str, n: int) -> list[dict]:
        """
        Generate n content dicts for the given category.
        Returns as many items as successfully parsed; may return fewer than n.
        """
        ...

    def _build_prompt(self, category: str, n: int) -> str:
        return CATEGORY_PROMPTS[category].strip().format(n=n)

    def _parse_with_retry(self, raw: str, category: str, n: int) -> list[dict]:
        """Try to parse; on failure return empty list with a warning."""
        try:
            return _clean_json(raw)
        except Exception as e:
            print(f"  [backend] JSON parse error ({category}, n={n}): {e}", file=sys.stderr)
            print(f"  [backend] Raw response head: {raw[:200]!r}", file=sys.stderr)
            return []


# ---------------------------------------------------------------------------
# ANTHROPIC BACKEND
# ---------------------------------------------------------------------------

class AnthropicBackend(LLMBackend):
    """Claude via the official Anthropic Python SDK."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 4000,
                 max_retries: int = 3):
        import anthropic as _anthropic
        self.client = _anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        print(f"  [backend] Anthropic — model={self.model}")

    def generate(self, category: str, n: int) -> list[dict]:
        prompt = self._build_prompt(category, n)
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text
                result = self._parse_with_retry(raw, category, n)
                if result:
                    return result
            except Exception as e:
                print(f"  [anthropic] Attempt {attempt+1} failed: {e}", file=sys.stderr)
                time.sleep(2 ** attempt)
        return []


# ---------------------------------------------------------------------------
# OPENAI-COMPATIBLE API BACKEND  (Ollama / vLLM / Together / LM Studio / …)
# ---------------------------------------------------------------------------

class OpenAICompatBackend(LLMBackend):
    """
    Any server that speaks /v1/chat/completions.

    Examples
    --------
    Ollama (local):
        base_url = "http://localhost:11434/v1"
        model    = "qwen2.5:7b"
        api_key  = "ollama"          # Ollama ignores the key

    vLLM (local):
        base_url = "http://localhost:8000/v1"
        model    = "Qwen/Qwen2.5-7B-Instruct"
        api_key  = "EMPTY"

    Together AI (cloud):
        base_url = "https://api.together.xyz/v1"
        model    = "Qwen/Qwen2.5-7B-Instruct-Turbo"
        api_key  = os.environ["TOGETHER_API_KEY"]
    """

    DEFAULT_MODEL   = "Qwen/Qwen2.5-7B-Instruct"
    DEFAULT_BASE_URL = "http://localhost:11434/v1"

    def __init__(
        self,
        base_url:    str = DEFAULT_BASE_URL,
        model:       str = DEFAULT_MODEL,
        api_key:     str = "ollama",
        max_tokens:  int = 4000,
        temperature: float = 0.7,
        max_retries: int = 3,
        timeout:     float = 120.0,
    ):
        try:
            from openai import OpenAI
            self.client = OpenAI(base_url=base_url, api_key=api_key,
                                 timeout=timeout)
        except ImportError:
            raise ImportError(
                "openai package required for qwen-api backend.\n"
                "Install with: pip install openai --break-system-packages"
            )
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        print(f"  [backend] OpenAI-compat — base_url={base_url} model={self.model}")

    def generate(self, category: str, n: int) -> list[dict]:
        prompt = self._build_prompt(category, n)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                raw = resp.choices[0].message.content or ""
                result = self._parse_with_retry(raw, category, n)
                if result:
                    return result
            except Exception as e:
                print(f"  [qwen-api] Attempt {attempt+1} failed: {e}", file=sys.stderr)
                time.sleep(2 ** attempt)
        return []


# ---------------------------------------------------------------------------
# LOCAL TRANSFORMERS BACKEND
# ---------------------------------------------------------------------------

class LocalTransformersBackend(LLMBackend):
    """
    Qwen2.5-7B-Instruct (or any chat model) loaded locally via HuggingFace
    transformers. The model is loaded once and reused across all generate() calls.

    GPU memory guide
    ----------------
    Qwen2.5-7B-Instruct in bfloat16  ≈ 15 GB VRAM  → fits a single A100/H100/4090
    Qwen2.5-7B-Instruct in int4      ≈  5 GB VRAM  → fits a 3090/A5000
    Pass load_in_4bit=True for quantised loading (requires bitsandbytes).
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    def __init__(
        self,
        model_name:   str   = DEFAULT_MODEL,
        max_new_tokens: int = 4096,
        temperature:  float = 0.7,
        load_in_4bit: bool  = False,
        device_map:   str   = "auto",
        max_retries:  int   = 2,
    ):
        self.model_name      = model_name
        self.max_new_tokens  = max_new_tokens
        self.temperature     = temperature
        self.max_retries     = max_retries
        self._pipe           = None          # lazy-loaded on first generate()
        self._load_kwargs    = dict(
            load_in_4bit=load_in_4bit,
            device_map=device_map,
        )
        print(f"  [backend] Local transformers — model={model_name} "
              f"4bit={load_in_4bit} device_map={device_map}")
        print(f"  [backend] Model will be downloaded/loaded on first generate() call.")

    def _load(self):
        """Lazy-load the pipeline on first use."""
        if self._pipe is not None:
            return
        try:
            import torch
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "transformers and torch are required for qwen-local backend.\n"
                "Install with: pip install transformers torch accelerate --break-system-packages\n"
                "For 4-bit: pip install bitsandbytes --break-system-packages"
            )

        print(f"  [local] Loading {self.model_name} …", flush=True)
        t0 = time.time()

        quant_cfg = None
        if self._load_kwargs.get("load_in_4bit"):
            from transformers import BitsAndBytesConfig
            quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self._load_kwargs.get("device_map", "auto"),
            quantization_config=quant_cfg,
            trust_remote_code=True,
        )
        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        print(f"  [local] Model loaded in {time.time()-t0:.1f}s", flush=True)

    def generate(self, category: str, n: int) -> list[dict]:
        self._load()
        prompt = self._build_prompt(category, n)

        # Build chat messages using the tokenizer's chat template
        tokenizer = self._pipe.tokenizer
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        for attempt in range(self.max_retries):
            try:
                outputs = self._pipe(
                    text,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    return_full_text=False,
                )
                raw = outputs[0]["generated_text"]
                result = self._parse_with_retry(raw, category, n)
                if result:
                    return result
            except Exception as e:
                print(f"  [qwen-local] Attempt {attempt+1} failed: {e}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# FACTORY
# ---------------------------------------------------------------------------

@dataclass
class BackendConfig:
    """All backend settings in one place — populated from CLI args."""
    backend:        str   = "anthropic"          # anthropic | qwen-local | qwen-api

    # Anthropic
    anthropic_model: str  = AnthropicBackend.DEFAULT_MODEL

    # OpenAI-compat (qwen-api)
    api_base_url:   str   = OpenAICompatBackend.DEFAULT_BASE_URL
    api_model:      str   = OpenAICompatBackend.DEFAULT_MODEL
    api_key:        str   = "ollama"

    # Local transformers (qwen-local)
    local_model:    str   = LocalTransformersBackend.DEFAULT_MODEL
    load_in_4bit:   bool  = False
    device_map:     str   = "auto"

    # Shared
    max_tokens:     int   = 4000
    temperature:    float = 0.7
    max_retries:    int   = 3


def build_backend(cfg: BackendConfig) -> LLMBackend:
    """Instantiate and return the requested backend."""
    b = cfg.backend.lower()

    if b == "anthropic":
        return AnthropicBackend(
            model=cfg.anthropic_model,
            max_tokens=cfg.max_tokens,
            max_retries=cfg.max_retries,
        )

    elif b == "qwen-api":
        # Also accept OPENAI_API_KEY / TOGETHER_API_KEY etc. from env
        api_key = cfg.api_key
        for env_var in ("OPENAI_API_KEY", "TOGETHER_API_KEY", "VLLM_API_KEY"):
            if os.environ.get(env_var):
                api_key = os.environ[env_var]
                break
        return OpenAICompatBackend(
            base_url=cfg.api_base_url,
            model=cfg.api_model,
            api_key=api_key,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            max_retries=cfg.max_retries,
        )

    elif b == "qwen-local":
        return LocalTransformersBackend(
            model_name=cfg.local_model,
            max_new_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            load_in_4bit=cfg.load_in_4bit,
            device_map=cfg.device_map,
            max_retries=cfg.max_retries,
        )

    else:
        raise ValueError(
            f"Unknown backend '{cfg.backend}'. "
            "Choose from: anthropic, qwen-local, qwen-api"
        )