"""
Central configuration schema for the entire project.

All experiment configs are dataclasses loaded from YAML via dacite.
Adding a new phase means adding a new dataclass here — nothing else changes.

Design principle: every field that affects a result must be in the config
and serialised to config.json at run time. No magic numbers in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ── Enums ──────────────────────────────────────────────────────────────────────

class Domain(str, Enum):
    SYNTHETIC = "synthetic"
    DOCUMENT  = "document"
    UI        = "ui"
    SCENE     = "scene"


class ProxyStage(str, Enum):
    STAGE1 = "stage1"   # L_edge + L_stroke
    STAGE2 = "stage2"   # L_edge + L_stroke + L_shape
    STAGE3 = "stage3"   # L_edge + L_stroke + L_shape + L_conf


class ObjectiveConfig(str, Enum):
    CE_ONLY      = "ce_only"
    ALIGN_ONLY   = "align_only"
    CE_AND_ALIGN = "ce_and_align"


class EnsembleWeighting(str, Enum):
    UNIFORM   = "uniform"
    DIVERSITY = "diversity"   # CKA-based, week 4+


# ── Sub-configs ────────────────────────────────────────────────────────────────

@dataclass
class SurrogateConfig:
    name:           str
    model_id:       str
    dtype:          str = "bfloat16"
    device_map:     str = "auto"
    max_new_tokens: int = 256
    alpha:          float = 1.0
    device:         str = ''


@dataclass
class AttackConfig:
    epsilon:        float = 4.0 / 255.0
    pgd_steps:      int   = 200
    pgd_step_size:  float = 1.0 / 255.0

    mu_init:    float = 0.1
    mu_growth:  float = 1.05
    kappa:      float = 0.05

    lambda_ce:    float = 1.0
    lambda_align: float = 0.5

    beta_edge:   float = 1.0
    beta_stroke: float = 1.0
    beta_shape:  float = 1.0
    beta_conf:   float = 0.0

    objective:          ObjectiveConfig   = ObjectiveConfig.CE_AND_ALIGN
    proxy_stage:        ProxyStage        = ProxyStage.STAGE1
    ensemble_weighting: EnsembleWeighting = EnsembleWeighting.UNIFORM

    # ── Region-aware perturbation budgets ──────────────────────────────────
    # Text region and background are allowed different l-inf budgets.
    # epsilon (above) is IGNORED when region_aware=True — the two budgets
    # below replace it entirely.
    #
    # epsilon_text > epsilon_bg is the expected operating regime:
    # text pixels receive the larger budget to concentrate suppression
    # energy where the model extracts content; background pixels receive
    # a small budget to avoid unnecessary readability disruption.
    # Setting both equal to epsilon reproduces the uniform baseline.
    region_aware:   bool  = True
    epsilon_text:   float = 0.06274510   # text pixels — large budget
    epsilon_bg:     float = 0.00392157   # background pixels — small budget
    # Mask dilation radius in pixels — expands the text mask slightly to
    # cover anti-aliasing fringe pixels around character edges.
    mask_dilation:  int   = 3

    # ── Salience-conditioned budget map ────────────────────────────────────
    # When True, eps_map is replaced by a gradient-salience-weighted map
    # computed once before the PGD loop. Requires word_boxes to be non-empty.
    # [ABLATION: uniform-text budget] is the active path when this is False.
    salience_budget: bool  = False
    epsilon_min:     float = 0.01568627   # budget floor for text pixels (salience mode), 4/255

    # Indices into the full surrogate list (opt + held-out) to use when
    # computing the salience map. None means use only the opt pool (surrogates).
    salience_surrogate_indices: Optional[list[int]] = None

    # If True, surrogates are loaded/unloaded each PGD step.
    # Slower but avoids OOM when surrogates cannot all fit in VRAM.
    # Salience surrogates always use lazy loading when
    # salience_surrogate_indices is set.
    lazy_loading: bool = False

    # If True, the salience surrogate pool loads/unloads each model
    # individually during the one-shot salience pass, regardless of
    # lazy_loading.  Prevents all salience surrogates from sitting in VRAM
    # simultaneously before the PGD loop starts.
    salience_lazy: bool = True


@dataclass
class DataConfig:
    data_dir:   Path   = Path("data/synthetic")
    data_dir_additional: Optional[str] = None
    domain:     Domain = Domain.SYNTHETIC
    n_images:   int    = 10
    # (H, W) — set to null in YAML to keep original image resolution
    image_size: Optional[tuple[int, int]] = (512, 512)
    # Optional filters — None disables each filter
    split_filter:    Optional[str] = None
    category_filter: Optional[str] = None
    contrast_filter: Optional[str] = None


@dataclass
class LogConfig:
    output_dir:       Path = Path("outputs")
    save_images:      bool = True
    save_trajectories: bool = True
    save_visual_diffs: bool = True


@dataclass
class FrontierConfig:
    openai_model:     str = "gpt-4o-2024-11-20"
    gemini_model:     str = "gemini-1.5-pro"
    anthropic_model:  str = "claude-3-5-sonnet-20241022"
    extraction_prompt: str = (
        "Transcribe exactly all visible text in this image. "
        "Preserve line breaks. Do not infer or complete missing text. "
        "Output only the transcribed text, nothing else."
    )
    max_retries: int   = 3
    temperature: float = 0.0


@dataclass
class ExperimentConfig:
    run_id: str = "run_001"
    phase:  str = "week1_sanity"
    seed:   int = 42

    data:    DataConfig   = field(default_factory=DataConfig)
    attack:  AttackConfig = field(default_factory=AttackConfig)
    log:     LogConfig    = field(default_factory=LogConfig)

    surrogates: list[SurrogateConfig] = field(default_factory=lambda: [
        SurrogateConfig(
            name="internvl2",
            model_id="OpenGVLab/InternVL2-8B",
        ),
    ])

    held_out_indices: list[int] = field(default_factory=list)
    
    cer_clean_threshold: float = 0.05

    frontier: Optional[FrontierConfig] = None

    epsilon_sweep: list[float] = field(default_factory=lambda: [
        2.0 / 255.0, 4.0 / 255.0, 8.0 / 255.0
    ])

    kappa_sweep: list[float] = field(default_factory=lambda: [0.03, 0.07])