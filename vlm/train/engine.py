"""Training engine."""
from __future__ import annotations

import math
from typing import Any, Dict

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf

from vlm.data.cc3m import build_cc3m_dataset
from vlm.data.dataloader import build_dataloader
from vlm.data.tokenizer import build_tokenizer
from vlm.data.transforms import build_transforms
from vlm.models.clip_like import ClipLikeModel
from vlm.train.checkpoint import load_checkpoint, save_checkpoint


def _count_params(model: torch.nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def _build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Build AdamW with optional no-weight-decay groups for norms/bias/scalars."""
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"]["wd"])
    beta1 = float(cfg["train"].get("beta1", 0.9))
    beta2 = float(cfg["train"].get("beta2", 0.999))

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        is_no_decay = (
            param.ndim <= 1
            or lname.endswith(".bias")
            or "bias" in lname
            or "gain" in lname
            or "logit_scale" in lname
            or "curv" in lname
        )
        if is_no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": wd})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer")

    return torch.optim.AdamW(param_groups, lr=lr, betas=(beta1, beta2))


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    total_steps: int,
    start_step: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    train_cfg = cfg.get("train", {})
    scheduler_name = str(train_cfg.get("scheduler", "cosine")).lower()
    if scheduler_name in {"none", "off", "constant"}:
        return None

    warmup = max(int(train_cfg.get("warmup", 0)), 0)
    total_steps = max(int(total_steps), 1)
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.0))
    min_lr_ratio = max(0.0, min(min_lr_ratio, 1.0))

    if warmup >= total_steps:
        warmup = max(total_steps - 1, 0)

    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    def lr_lambda(current_step: int) -> float:
        if warmup > 0 and current_step < warmup:
            return max(float(current_step + 1) / float(warmup), 1e-8)

        if total_steps <= warmup:
            return 1.0

        progress = float(current_step - warmup) / float(total_steps - warmup)
        progress = max(0.0, min(progress, 1.0))

        if scheduler_name == "linear":
            decay = 1.0 - progress
        else:
            # Default cosine decay for smoother late-stage updates.
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
        last_epoch=start_step - 1,
    )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().cpu())
    return float(value)


def train(cfg: Dict[str, Any]) -> Any:
    """Train loop entry."""
    grad_accum = int(cfg["train"].get("grad_accum", 1))
    precision = cfg["train"].get("precision", "no")
    fallback_msg = None
    if precision == "bf16":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            precision = "no"
            fallback_msg = "bf16 not supported on this device; falling back to fp32."
    accelerator = Accelerator(
        mixed_precision=precision,
        gradient_accumulation_steps=grad_accum,
    )
    if fallback_msg and accelerator.is_main_process:
        accelerator.print(fallback_msg)

    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["rank"] = accelerator.process_index
    cfg["world_size"] = accelerator.num_processes
    cfg.setdefault("data", {})
    if cfg["data"].get("batch_size") is None:
        cfg["data"]["batch_size"] = int(cfg["train"].get("batch_size", 256))

    seed = int(cfg["train"].get("seed", 42))
    torch.manual_seed(seed)

    tokenizer = build_tokenizer(cfg["data"]["tokenizer"])
    transforms = build_transforms(cfg["data"])
    dataset = build_cc3m_dataset(cfg["data"], transforms=transforms)
    dataloader = build_dataloader(dataset, cfg["data"], tokenizer=tokenizer)

    model = ClipLikeModel(cfg)

    optimizer = _build_optimizer(model, cfg)

    start_step = 0
    resume_from = cfg["train"].get("resume_from", None)
    if resume_from:
        payload = load_checkpoint(str(resume_from), model=model, optimizer=optimizer, map_location="cpu")
        start_step = int(payload.get("step", 0))
        if accelerator.is_main_process:
            accelerator.print(f"Resumed from checkpoint: {resume_from} (step={start_step})")
            optimizer_warn = payload.get("optimizer_load_error")
            if optimizer_warn:
                accelerator.print(
                    "Warning: optimizer state was not restored during resume; "
                    f"continuing with fresh optimizer state ({optimizer_warn})."
                )

    max_steps = int(cfg["train"]["max_steps"])
    schedule_total_steps_raw = cfg["train"].get("schedule_total_steps", None)
    if schedule_total_steps_raw is None:
        schedule_total_steps = max_steps
    else:
        schedule_total_steps = int(schedule_total_steps_raw)
    lr_scheduler = _build_scheduler(
        optimizer=optimizer,
        cfg=cfg,
        total_steps=schedule_total_steps,
        start_step=start_step,
    )

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    if accelerator.is_main_process:
        count_model = accelerator.unwrap_model(model)
        total_params = _count_params(count_model)
        trainable_params = _count_params(count_model, trainable_only=True)
        pct = 0.0 if total_params == 0 else 100.0 * trainable_params / total_params
        accelerator.print(f"Total params: {total_params}")
        accelerator.print(f"Trainable params: {trainable_params} ({pct:.2f}%)")
        if lr_scheduler is not None:
            accelerator.print(
                f"LR scheduler: {cfg['train'].get('scheduler', 'cosine')} "
                f"(warmup={cfg['train'].get('warmup', 0)}, total_steps={schedule_total_steps})"
            )
        else:
            accelerator.print("LR scheduler: constant")

    save_every = int(cfg["logging"].get("save_every", 0))
    output_dir = str(cfg["logging"]["output_dir"])
    log_every = int(cfg["logging"].get("log_every", 50))
    norm_log_every = 100
    step = start_step
    model.train()

    epoch = 0
    while step < max_steps:
        sampler = getattr(dataloader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        made_progress = False
        for batch in dataloader:
            with accelerator.accumulate(model):
                out = model(
                    batch,
                    global_step=step + 1,
                    total_steps=schedule_total_steps,
                )
                loss = out["loss"]
                if not torch.isfinite(loss).all():
                    raise RuntimeError(
                        f"Non-finite loss detected at step={step + 1}: {float(loss.detach().cpu())}"
                    )
                accelerator.backward(loss)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue

            made_progress = True
            step += 1

            if accelerator.is_main_process and step % log_every == 0:
                log_dict = out.get("logging", {})
                logit_scale_val = _as_float(log_dict.get("logit_scale"))
                curv_val = _as_float(log_dict.get("curv"))
                log_payload: Dict[str, Any] = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
                if logit_scale_val is not None:
                    log_payload["logit_scale"] = logit_scale_val
                if curv_val is not None:
                    log_payload["curv"] = curv_val
                accelerator.print(
                    log_payload
                )

            if accelerator.is_main_process and step % norm_log_every == 0:
                log_dict = out.get("logging", {})
                image_norm_mean = _as_float(log_dict.get("image_norm_mean"))
                image_norm_std = _as_float(log_dict.get("image_norm_std"))
                text_norm_mean = _as_float(log_dict.get("text_norm_mean"))
                text_norm_std = _as_float(log_dict.get("text_norm_std"))
                if None not in (image_norm_mean, image_norm_std, text_norm_mean, text_norm_std):
                    accelerator.print(
                        {
                            "step": step,
                            "image_norm": f"{image_norm_mean:.4f} +- {image_norm_std:.4f}",
                            "text_norm": f"{text_norm_mean:.4f} +- {text_norm_std:.4f}",
                        }
                    )

            if save_every > 0 and step % save_every == 0 and accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                save_checkpoint(output_dir, step, unwrapped, optimizer=optimizer)
            if step >= max_steps:
                break

        if not made_progress:
            raise RuntimeError(
                "No optimizer steps were completed in an epoch. "
                "Check batch size, grad_accum, and dataset size."
            )
        epoch += 1

    accelerator.wait_for_everyone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return None
