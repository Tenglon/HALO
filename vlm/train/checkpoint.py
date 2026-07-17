"""Checkpoint helpers."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch


def checkpoint_path(output_dir: str, step: int) -> str:
    return os.path.join(output_dir, f"checkpoint_step_{step}.pt")


def _is_aux_trainable_key(key: str) -> bool:
    lkey = key.lower()
    return ("logit_scale" in lkey) or ("curv_param" in lkey)


def _is_lora_only_state_dict(state_dict: Dict[str, Any]) -> bool:
    if not state_dict:
        return False
    return all(("lora_" in str(key)) or _is_aux_trainable_key(str(key)) for key in state_dict.keys())


def save_checkpoint(
    output_dir: str,
    step: int,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    model_state = {
        k: v
        for k, v in model.state_dict().items()
        if ("lora_" in k) or _is_aux_trainable_key(k)
    }
    if not model_state:
        raise ValueError("No LoRA/aux-trainable parameters were found in model.state_dict().")
    payload: Dict[str, Any] = {
        "step": int(step),
        "model": model_state,
        "model_format": "lora_only",
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload["extra"] = dict(extra)
    path = checkpoint_path(output_dir, step)
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    raw = torch.load(path, map_location=map_location)
    if isinstance(raw, dict) and "model" in raw:
        payload = dict(raw)
        state_dict = payload["model"]
    elif isinstance(raw, dict):
        payload = {"model": raw}
        state_dict = raw
    else:
        raise ValueError(f"Unexpected checkpoint format at {path}: {type(raw)}")

    if not isinstance(state_dict, dict):
        raise ValueError(f"Invalid model state_dict in checkpoint: {path}")

    model_format = payload.get("model_format", None)
    if model_format is None:
        model_format = "lora_only" if _is_lora_only_state_dict(state_dict) else "full"
    model_format = str(model_format)
    if model_format not in {"full", "lora_only"}:
        raise ValueError(f"Unsupported checkpoint model_format={model_format} at {path}")

    strict = model_format != "lora_only"
    incompat = model.load_state_dict(state_dict, strict=strict)
    if model_format == "lora_only":
        unexpected_keys = list(getattr(incompat, "unexpected_keys", []))
        if unexpected_keys:
            preview = ", ".join(unexpected_keys[:8])
            raise RuntimeError(
                f"Unexpected keys while loading LoRA-only checkpoint {path}: {preview}"
            )

    if optimizer is not None and "optimizer" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except ValueError as exc:
            message = str(exc)
            # Allow resume across compatible model weights even when optimizer grouping changed.
            # Some torch versions use "parameter group" (singular), others "parameter groups".
            if "parameter group" in message.lower():
                payload["optimizer_load_error"] = message
            else:
                raise
    payload["model_format"] = model_format
    return payload
