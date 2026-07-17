"""LoRA modules and injection utilities (PEFT)."""
from __future__ import annotations

from typing import Iterable, Optional
from collections.abc import Iterable as IterableABC

import torch


def _as_list(val: Optional[Iterable[str]]) -> list[str]:
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, IterableABC) and not isinstance(val, (str, bytes, dict)):
        return list(val)
    return [str(val)]


def inject_lora(model: torch.nn.Module, cfg: dict) -> torch.nn.Module:
    """Inject LoRA layers into the model using PEFT."""
    if not cfg.get("enabled", True):
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("peft is required for LoRA injection") from exc

    r = int(cfg.get("r", 8))
    alpha = int(cfg.get("alpha", 16))
    dropout = float(cfg.get("dropout", 0.0))
    target_modules = _as_list(cfg.get("target_modules", []))
    bias = cfg.get("bias", "none")

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=cfg.get("task_type", "FEATURE_EXTRACTION"),
    )
    model = get_peft_model(model, lora_cfg)

    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True

    return model
