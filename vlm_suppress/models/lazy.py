from __future__ import annotations

from typing import Optional, Union
import torch

from vlm_suppress.models.base import SurrogateModel

# Union accepted anywhere a surrogate is needed.
AnySurrogate = Union[SurrogateModel, "LazySurrogate"]


class LazySurrogate:
    """
    Deferred-load wrapper for a surrogate model.

    Holds config + class reference but allocates no GPU memory until used as a
    context manager.  .name and .device delegate to cfg so they are safe to
    call outside a context (e.g. for logging or index selection in pgd.py).
    """

    def __init__(self, surrogate_cfg, surrogate_cls) -> None:
        self.cfg  = surrogate_cfg
        self.cls  = surrogate_cls
        self._model: Optional[SurrogateModel] = None

    @property
    def name(self) -> str:
        return self.cfg.name

    @property
    def device(self) -> torch.device:
        return torch.device(self.cfg.device)

    def __enter__(self) -> SurrogateModel:
        if self._model is None:
            print(f"  [lazy] loading {self.cfg.name} → {self.cfg.device}")
            self._model = self.cls(self.cfg)
        return self._model

    def __exit__(self, *args) -> None:
        if self.cls is None:
            return  # from_eager wrapping — caller owns the model
        print(f"  [lazy] unloading {self.cfg.name}")
        del self._model
        self._model = None
        torch.cuda.empty_cache()

    @classmethod
    def from_eager(cls, model: SurrogateModel) -> "LazySurrogate":
        """
        Wrap an already-loaded SurrogateModel for use in a lazy loop without
        unloading/reloading it.  __enter__ returns the model directly;
        __exit__ is a no-op.  cfg is duck-typed: .name and .device are read
        from the model so the salience print statement works correctly.
        """
        instance = cls.__new__(cls)
        instance.cfg    = model   # model.name / model.device satisfy the duck type
        instance.cls    = None    # sentinel: __exit__ checks cls is None → no-op
        instance._model = model
        return instance


class OffloadSurrogate(LazySurrogate):
    """
    Wraps an eager surrogate for one-model-at-a-time VRAM use during salience
    or importance passes.

    The wrapped model's weights live on CPU between passes.  __enter__ moves
    them to the target GPU; __exit__ moves them back to CPU and clears the
    CUDA cache.  Inherits from LazySurrogate so salience.py's isinstance check
    routes it through the context-manager path automatically.

    Usage
    -----
        wrappers = [OffloadSurrogate(m) for m in eager_models]
        for m in eager_models:
            m.model.to("cpu")
        torch.cuda.empty_cache()
        build_salience_budget_map(..., surrogates=wrappers, ...)
        # weights restored on each __exit__; call .restore() when done
    """

    def __init__(self, model: SurrogateModel) -> None:
        self._model      = model
        self._gpu_device = model.device
        self.cfg         = model   # duck-types .name and .device for LazySurrogate base
        self.cls         = None    # prevents base-class __exit__ from trying to unload

    def __enter__(self) -> SurrogateModel:
        self._model.model.to(self._gpu_device)
        return self._model

    def __exit__(self, *args) -> None:
        self._model.model.to("cpu")
        torch.cuda.empty_cache()

    def restore(self) -> None:
        """Move weights back to GPU after the salience/importance pass is done."""
        self._model.model.to(self._gpu_device)
