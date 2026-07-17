"""Lorentz inner-product contrastive loss (CLIP-style)."""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from vlm.geom.euclidean import euclidean_inner_product, euclidean_distance
from vlm.geom.lorentz import lift_to_lorentz, lorentz_distance, matching_score as lorentz_matching_score


def _concat_all_gather(x: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return x
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return x
    gathered = [torch.zeros_like(x) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, x)
    gathered[rank] = x
    return torch.cat(gathered, dim=0)


def _resolve_hybrid_cfg(
    hybrid_cfg: Dict[str, Any] | None,
    current_step: int | None = None,
    total_steps: int | None = None,
) -> Tuple[str, str, float, float]:
    cfg = hybrid_cfg or {}
    euclidean_score = str(cfg.get("euclidean_score", "euclidean_inner_product"))
    lorentz_score = str(cfg.get("lorentz_score", "negative_lorentz_distance"))
    base_euclidean_weight = float(cfg.get("euclidean_weight", 0.5))
    base_lorentz_weight = float(cfg.get("lorentz_weight", 0.5))

    euclidean_weight = base_euclidean_weight
    lorentz_weight = base_lorentz_weight
    dyn_cfg = cfg.get("dynamic", {}) if isinstance(cfg.get("dynamic", {}), dict) else {}
    if bool(dyn_cfg.get("enabled", False)) and current_step is not None:
        # Dynamic schedule for hybrid blending:
        # interpolate (euclidean, lorentz) from start_* to end_* over [start_step, end_step].
        mode = str(dyn_cfg.get("mode", "linear")).lower()

        total_steps_resolved = None
        if total_steps is not None:
            total_steps_resolved = max(int(total_steps), 1)

        start_step = int(dyn_cfg.get("start_step", 0))
        end_step_default = total_steps_resolved if total_steps_resolved is not None else max(start_step + 1, 1)
        end_step = int(dyn_cfg.get("end_step", end_step_default))

        if total_steps_resolved is not None:
            if "start_ratio" in dyn_cfg:
                start_step = int(float(dyn_cfg["start_ratio"]) * total_steps_resolved)
            if "end_ratio" in dyn_cfg:
                end_step = int(float(dyn_cfg["end_ratio"]) * total_steps_resolved)

        if end_step <= start_step:
            end_step = start_step + 1

        start_eu = float(dyn_cfg.get("start_euclidean_weight", base_euclidean_weight))
        start_lo = float(dyn_cfg.get("start_lorentz_weight", base_lorentz_weight))
        end_eu = float(dyn_cfg.get("end_euclidean_weight", base_euclidean_weight))
        end_lo = float(dyn_cfg.get("end_lorentz_weight", base_lorentz_weight))

        step = int(current_step)
        if step <= start_step:
            euclidean_weight, lorentz_weight = start_eu, start_lo
        elif step >= end_step:
            euclidean_weight, lorentz_weight = end_eu, end_lo
        else:
            t = float(step - start_step) / float(end_step - start_step)
            t = max(0.0, min(t, 1.0))
            if mode == "cosine":
                t = 0.5 * (1.0 - math.cos(math.pi * t))
            euclidean_weight = (1.0 - t) * start_eu + t * end_eu
            lorentz_weight = (1.0 - t) * start_lo + t * end_lo

    euclidean_weight = max(euclidean_weight, 0.0)
    lorentz_weight = max(lorentz_weight, 0.0)
    weight_sum = euclidean_weight + lorentz_weight
    if weight_sum <= 0.0:
        return euclidean_score, lorentz_score, 0.5, 0.5
    return (
        euclidean_score,
        lorentz_score,
        euclidean_weight / weight_sum,
        lorentz_weight / weight_sum,
    )


def matching_score(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    geometry: str,
    score_type: str,
    curv: float,
    hybrid_cfg: Dict[str, Any] | None = None,
    lorentz_input_is_lifted: bool = False,
    current_step: int | None = None,
    total_steps: int | None = None,
) -> torch.Tensor:
    if geometry == "hybrid":
        if score_type not in {"hybrid_euclidean_lorentz", "hybrid"}:
            raise ValueError(f"Unknown hybrid matching_score: {score_type}")
        euclidean_score, lorentz_score, euclidean_weight, lorentz_weight = _resolve_hybrid_cfg(
            hybrid_cfg,
            current_step=current_step,
            total_steps=total_steps,
        )
        euclidean_sims = matching_score(
            image_emb,
            text_emb,
            geometry="euclidean",
            score_type=euclidean_score,
            curv=curv,
            hybrid_cfg=None,
            current_step=current_step,
            total_steps=total_steps,
        )
        image_lifted = lift_to_lorentz(image_emb, curv)
        text_lifted = lift_to_lorentz(text_emb, curv)
        lorentz_sims = matching_score(
            image_lifted,
            text_lifted,
            geometry="lorentz",
            score_type=lorentz_score,
            curv=curv,
            hybrid_cfg=None,
            lorentz_input_is_lifted=True,
            current_step=current_step,
            total_steps=total_steps,
        )
        return euclidean_weight * euclidean_sims + lorentz_weight * lorentz_sims
    if geometry == "lorentz":
        if score_type == "lorentz_inner_product":
            return lorentz_matching_score(image_emb, text_emb, curv, input_is_lifted=lorentz_input_is_lifted)
        if score_type == "negative_lorentz_distance":
            return -lorentz_distance(image_emb, text_emb, curv, input_is_lifted=lorentz_input_is_lifted)
        raise ValueError(f"Unknown lorentz matching_score: {score_type}")
    if geometry == "euclidean":
        if score_type == "euclidean_inner_product":
            # CLIP-style Euclidean scoring is cosine similarity.
            image_emb = F.normalize(image_emb, dim=-1)
            text_emb = F.normalize(text_emb, dim=-1)
            return euclidean_inner_product(image_emb, text_emb)
        if score_type == "euclidean_raw_inner_product":
            # Unnormalized dot product for ablation against cosine-style branch.
            return euclidean_inner_product(image_emb, text_emb)
        if score_type == "negative_euclidean_distance":
            return -euclidean_distance(image_emb, text_emb)
        raise ValueError(f"Unknown euclidean matching_score: {score_type}")
    raise ValueError(f"Unknown geometry: {geometry}")


def contrastive_lorentz_loss(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    logit_scale: torch.Tensor,
    curv: float,
    rank: int,
    world_size: int,
    geometry: str = "lorentz",
    score_type: str = "lorentz_inner_product",
    hybrid_cfg: Dict[str, Any] | None = None,
    entailment_cfg: Dict[str, Any] | None = None,
    current_step: int | None = None,
    total_steps: int | None = None,
) -> torch.Tensor:
    """Compute CLIP-style contrastive loss."""
    all_text = _concat_all_gather(text_emb, rank, world_size)
    all_image = _concat_all_gather(image_emb, rank, world_size)

    scale = logit_scale.clamp(max=math.log(100)).exp()
    batch_size = image_emb.shape[0]
    targets = torch.arange(batch_size, device=image_emb.device) + (rank * batch_size)

    def _entailment_loss_text_entails_image(
        image_feats: torch.Tensor,
        text_feats: torch.Tensor,
    ) -> torch.Tensor:
        """A small directional regularizer: text should be at least as 'specific' as image.

        For Lorentz-space interpretation, use space component norm ordering:
        enforce ||text_space|| >= ||image_space|| + margin (hinge).
        """
        cfg = entailment_cfg or {}
        if not bool(cfg.get("enabled", False)):
            return image_feats.new_zeros(())
        if str(cfg.get("direction", "text_entails_image")) != "text_entails_image":
            return image_feats.new_zeros(())
        weight = float(cfg.get("weight", 0.0))
        if weight <= 0.0:
            return image_feats.new_zeros(())
        margin = float(cfg.get("margin", 0.0))
        image_norm = torch.linalg.vector_norm(image_feats, dim=-1)
        text_norm = torch.linalg.vector_norm(text_feats, dim=-1)
        penalty = F.relu((image_norm + margin) - text_norm).mean()
        return image_feats.new_tensor(weight) * penalty

    if geometry == "hybrid":
        if score_type not in {"hybrid_euclidean_lorentz", "hybrid"}:
            raise ValueError(f"Unknown hybrid matching_score: {score_type}")
        euclidean_score, lorentz_score, euclidean_weight, lorentz_weight = _resolve_hybrid_cfg(
            hybrid_cfg,
            current_step=current_step,
            total_steps=total_steps,
        )
        image_lifted = lift_to_lorentz(image_emb, curv)
        text_lifted = lift_to_lorentz(text_emb, curv)
        all_text_lifted = _concat_all_gather(text_lifted, rank, world_size)
        all_image_lifted = _concat_all_gather(image_lifted, rank, world_size)

        eu_logits_i2t = scale * matching_score(
            image_emb,
            all_text,
            geometry="euclidean",
            score_type=euclidean_score,
            curv=curv,
            hybrid_cfg=None,
            current_step=current_step,
            total_steps=total_steps,
        )
        eu_logits_t2i = scale * matching_score(
            text_emb,
            all_image,
            geometry="euclidean",
            score_type=euclidean_score,
            curv=curv,
            hybrid_cfg=None,
            current_step=current_step,
            total_steps=total_steps,
        )
        lo_logits_i2t = scale * matching_score(
            image_lifted,
            all_text_lifted,
            geometry="lorentz",
            score_type=lorentz_score,
            curv=curv,
            hybrid_cfg=None,
            lorentz_input_is_lifted=True,
            current_step=current_step,
            total_steps=total_steps,
        )
        lo_logits_t2i = scale * matching_score(
            text_lifted,
            all_image_lifted,
            geometry="lorentz",
            score_type=lorentz_score,
            curv=curv,
            hybrid_cfg=None,
            lorentz_input_is_lifted=True,
            current_step=current_step,
            total_steps=total_steps,
        )

        eu_loss = 0.5 * (F.cross_entropy(eu_logits_i2t, targets) + F.cross_entropy(eu_logits_t2i, targets))
        lo_loss = 0.5 * (F.cross_entropy(lo_logits_i2t, targets) + F.cross_entropy(lo_logits_t2i, targets))
        base_loss = euclidean_weight * eu_loss + lorentz_weight * lo_loss
        ent_loss = _entailment_loss_text_entails_image(image_emb, text_emb)
        return base_loss + ent_loss

    logits_i2t = scale * matching_score(
        image_emb,
        all_text,
        geometry=geometry,
        score_type=score_type,
        curv=curv,
        hybrid_cfg=hybrid_cfg,
        current_step=current_step,
        total_steps=total_steps,
    )
    logits_t2i = scale * matching_score(
        text_emb,
        all_image,
        geometry=geometry,
        score_type=score_type,
        curv=curv,
        hybrid_cfg=hybrid_cfg,
        current_step=current_step,
        total_steps=total_steps,
    )

    base_loss = 0.5 * (F.cross_entropy(logits_i2t, targets) + F.cross_entropy(logits_t2i, targets))
    ent_loss = _entailment_loss_text_entails_image(image_emb, text_emb)
    return base_loss + ent_loss
