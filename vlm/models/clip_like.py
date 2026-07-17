"""CLIP-like dual-encoder model backed by HuggingFace."""
from __future__ import annotations

import io
import logging
import os
import warnings
from contextlib import redirect_stderr, redirect_stdout
from typing import Dict, Any

import torch
import torch.nn.functional as F

from vlm.losses.contrastive_lorentz import contrastive_lorentz_loss, matching_score
from vlm.models.lora import inject_lora


def _disable_hf_progress() -> None:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
    except Exception:
        pass


def _load_hf_clip_quiet(model_name: str) -> torch.nn.Module:
    from transformers import CLIPModel

    _disable_hf_progress()
    prev_logging_disable = logging.root.manager.disable
    buf = io.StringIO()
    try:
        logging.disable(logging.CRITICAL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with redirect_stdout(buf), redirect_stderr(buf):
                model = CLIPModel.from_pretrained(model_name)
    finally:
        logging.disable(prev_logging_disable)
    print(f"[load] clip_model: {model_name}")
    return model


class ClipLikeModel(torch.nn.Module):
    """CLIP-like dual-encoder model."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg["model"]
        model_name = model_cfg.get("hf_model_name", "openai/clip-vit-base-patch32")

        self.clip = _load_hf_clip_quiet(model_name)
        if not hasattr(self.clip, "logit_scale"):
            self.logit_scale = torch.nn.Parameter(torch.tensor(2.6592))

        lorentz_cfg = model_cfg.get("lorentz", {})
        self.curv_param = torch.nn.Parameter(
            torch.tensor(float(lorentz_cfg.get("curv", 1.0)), dtype=torch.float32)
        )

        if model_cfg.get("gradient_checkpointing", False) and hasattr(self.clip, "gradient_checkpointing_enable"):
            self.clip.gradient_checkpointing_enable()

        if cfg.get("lora", {}).get("enabled", True):
            self.clip = inject_lora(self.clip, cfg["lora"])

        _freeze_trainable_params(self, cfg)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        if hasattr(self.clip, "get_image_features"):
            out = self.clip.get_image_features(pixel_values=images)
            return self._coerce_features(out, is_image=True)
        raise AttributeError("HF model does not expose get_image_features")

    def encode_text(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        if hasattr(self.clip, "get_text_features"):
            out = self.clip.get_text_features(**tokens)
            return self._coerce_features(out, is_image=False)
        raise AttributeError("HF model does not expose get_text_features")

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        global_step: int | None = None,
        total_steps: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        image_emb = self.encode_image(batch["image"])
        text_emb = self.encode_text(batch["tokens"])
        image_norm = torch.linalg.vector_norm(image_emb, dim=-1)
        text_norm = torch.linalg.vector_norm(text_emb, dim=-1)

        logit_scale = self._get_logit_scale()
        curv = self._get_curv()
        model_cfg = self.cfg["model"]
        geometry = model_cfg.get("geometry", "lorentz")
        score_type = model_cfg.get("matching_score", "lorentz_inner_product")
        hybrid_cfg = model_cfg.get("hybrid", {})
        entailment_cfg = model_cfg.get("entailment", {})
        entail_enabled = bool(entailment_cfg.get("enabled", False))
        entail_margin = float(entailment_cfg.get("margin", 0.0))
        # Monitor entailment violation before weighting, for debugging stability.
        entail_violation = F.relu((image_norm + entail_margin) - text_norm).mean()
        loss = contrastive_lorentz_loss(
            image_emb=image_emb,
            text_emb=text_emb,
            logit_scale=logit_scale,
            curv=curv,
            rank=int(self.cfg.get("rank", 0)),
            world_size=int(self.cfg.get("world_size", 1)),
            geometry=geometry,
            score_type=score_type,
            hybrid_cfg=hybrid_cfg,
            entailment_cfg=entailment_cfg,
            current_step=global_step,
            total_steps=total_steps,
        )
        return {
            "loss": loss,
            "logging": {
                "loss": loss.detach(),
                "logit_scale": logit_scale.detach(),
                "score_mean": matching_score(
                    image_emb,
                    text_emb,
                    geometry=geometry,
                    score_type=score_type,
                    curv=curv,
                    hybrid_cfg=hybrid_cfg,
                    current_step=global_step,
                    total_steps=total_steps,
                )
                .diag()
                .mean()
                .detach(),
                "curv": curv.detach(),
                "image_norm_mean": image_norm.mean().detach(),
                "image_norm_std": image_norm.std(unbiased=False).detach(),
                "text_norm_mean": text_norm.mean().detach(),
                "text_norm_std": text_norm.std(unbiased=False).detach(),
                "entailment_violation_mean": entail_violation.detach(),
                "entailment_enabled": torch.tensor(
                    1.0 if entail_enabled else 0.0,
                    device=image_emb.device,
                ),
            },
        }

    def _get_logit_scale(self) -> torch.Tensor:
        if hasattr(self.clip, "logit_scale"):
            return self.clip.logit_scale
        return self.logit_scale

    def _get_curv(self) -> torch.Tensor:
        return F.softplus(self.curv_param) + 1e-6

    def _coerce_features(self, out: Any, is_image: bool) -> torch.Tensor:
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, dict):
            key = "image_embeds" if is_image else "text_embeds"
            if key in out:
                return out[key]
            return self._extract_pooled(out)
        if hasattr(out, "image_embeds") and is_image:
            return out.image_embeds
        if hasattr(out, "text_embeds") and not is_image:
            return out.text_embeds
        return self._extract_pooled(out)

    def _extract_pooled(self, out: Any) -> torch.Tensor:
        if hasattr(out, "pooler_output"):
            return out.pooler_output
        if isinstance(out, dict) and "pooler_output" in out:
            return out["pooler_output"]
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0]
        if isinstance(out, dict) and "last_hidden_state" in out:
            return out["last_hidden_state"][:, 0]
        raise ValueError("Unable to extract pooled features from model output")

def _freeze_trainable_params(model: torch.nn.Module, cfg: Dict[str, Any]) -> None:
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    lorentz_cfg = model_cfg.get("lorentz", {})
    train_logit_scale = bool(train_cfg.get("train_logit_scale", True))
    learn_curv = bool(lorentz_cfg.get("learn_curv", False))

    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        lname = name.lower()
        if "lora_" in lname:
            p.requires_grad = True
            continue
        if train_logit_scale and "logit_scale" in lname:
            p.requires_grad = True
            continue
        if learn_curv and "curv_param" in lname:
            p.requires_grad = True
