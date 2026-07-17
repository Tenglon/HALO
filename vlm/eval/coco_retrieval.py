"""COCO retrieval evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Dict, Iterable, List, Tuple

import torch


@dataclass
class RetrievalStats:
    r1: float
    r5: float
    r10: float


def _normalize(x: torch.Tensor) -> torch.Tensor:
    if hasattr(x, "pooler_output"):
        x = x.pooler_output
    elif hasattr(x, "last_hidden_state"):
        x = x.last_hidden_state[:, 0]
    return torch.nn.functional.normalize(x, dim=-1)


def _identity(x: torch.Tensor) -> torch.Tensor:
    if hasattr(x, "pooler_output"):
        return x.pooler_output
    if hasattr(x, "last_hidden_state"):
        return x.last_hidden_state[:, 0]
    return x


def _batch(iterable: List, size: int) -> Iterable[List]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def _compute_recall_at_k(
    sims: torch.Tensor,
    targets: List[int],
    k: int,
) -> float:
    if sims.numel() == 0:
        return 0.0
    k = min(k, sims.shape[-1])
    if k <= 0:
        return 0.0
    topk = sims.topk(k, dim=-1).indices
    hits = 0
    for row, target in zip(topk, targets):
        if target in row:
            hits += 1
    return 100.0 * hits / len(targets)


def _compute_recall_at_k_multi(
    sims: torch.Tensor,
    targets: List[List[int]],
    k: int,
) -> float:
    if sims.numel() == 0:
        return 0.0
    k = min(k, sims.shape[-1])
    if k <= 0:
        return 0.0
    topk = sims.topk(k, dim=-1).indices
    hits = 0
    for row, target_list in zip(topk, targets):
        if any(t in row for t in target_list):
            hits += 1
    return 100.0 * hits / len(targets)


def _coerce_image_id(raw_image_id: Any, fallback: int) -> Any:
    if raw_image_id is None:
        return fallback
    if hasattr(raw_image_id, "item"):
        try:
            raw_image_id = raw_image_id.item()
        except Exception:
            pass
    try:
        hash(raw_image_id)
        return raw_image_id
    except Exception:
        return str(raw_image_id)


def _extract_captions(raw_caption: Any) -> List[str]:
    if raw_caption is None:
        return []
    if isinstance(raw_caption, str):
        return [raw_caption]
    if isinstance(raw_caption, dict):
        for key in ("caption", "captions", "text", "text_input", "raw", "sentence", "sent"):
            value = raw_caption.get(key)
            if value is not None:
                return _extract_captions(value)
        return []
    if isinstance(raw_caption, list):
        captions: List[str] = []
        for item in raw_caption:
            captions.extend(_extract_captions(item))
        return captions
    return [str(raw_caption)]


def _disable_hf_progress() -> None:
    import os

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    try:
        from datasets.utils.logging import disable_progress_bar as disable_ds_progress_bar

        disable_ds_progress_bar()
    except Exception:
        pass
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
    except Exception:
        pass


def _from_pretrained_quiet(
    cls: Any,
    model_name: str,
    success_label: str,
    **kwargs: Any,
) -> Any:
    import io
    import logging
    import sys
    import warnings
    from contextlib import redirect_stderr, redirect_stdout

    _disable_hf_progress()
    buffer = io.StringIO()
    prev_logging_disable = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with redirect_stdout(buffer), redirect_stderr(buffer):
                obj = cls.from_pretrained(model_name, **kwargs)
    except Exception:
        captured = buffer.getvalue().strip()
        if captured:
            print(captured, file=sys.stderr)
        raise
    finally:
        logging.disable(prev_logging_disable)
    print(f"[load] {success_label}: {model_name}")
    return obj


def evaluate_coco_retrieval(cfg: Dict) -> Dict[str, RetrievalStats]:
    from datasets import load_dataset

    _disable_hf_progress()

    eval_cfg = cfg["eval"]
    device_str = eval_cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    model_source = eval_cfg.get("model_source", "hf")
    dataset_name = eval_cfg.get("dataset_name", "jxie/coco_captions")
    dataset_config = eval_cfg.get("dataset_config", None)
    dataset_kwargs = dict(eval_cfg.get("dataset_kwargs", {}))
    trust_remote_code = bool(eval_cfg.get("trust_remote_code", False))
    split = eval_cfg.get("split", "validation")
    batch_size = int(eval_cfg.get("batch_size", 64))
    max_samples = int(eval_cfg.get("max_samples", 0))
    image_column = eval_cfg.get("image_column", None)
    caption_column = eval_cfg.get("caption_column", None)
    image_id_column = eval_cfg.get("image_id_column", None)

    try:
        if dataset_config is None:
            ds = load_dataset(dataset_name, split=split, trust_remote_code=trust_remote_code, **dataset_kwargs)
        else:
            ds = load_dataset(
                dataset_name,
                dataset_config,
                split=split,
                trust_remote_code=trust_remote_code,
                **dataset_kwargs,
            )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load retrieval dataset. "
            "Set eval.dataset_name / eval.dataset_config / eval.dataset_kwargs in your config. "
            "Your datasets version may not support loading-script datasets."
        ) from exc
    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    image_encoder: Callable[[List[Any]], torch.Tensor]
    text_encoder: Callable[[List[str]], torch.Tensor]
    sim_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    if model_source == "hycoclip":
        image_encoder, text_encoder, sim_fn = _build_hycoclip_encoders(eval_cfg, device)
        normalize_fn = _identity
    elif model_source == "vlm_checkpoint":
        image_encoder, text_encoder, sim_fn = _build_vlm_checkpoint_encoders(eval_cfg, device)
        normalize_fn = _identity
    else:
        image_encoder, text_encoder, sim_fn = _build_hf_clip_encoders(eval_cfg, device)
        normalize_fn = _normalize

    image_records: Dict[Any, Any] = {}
    image_ids: List[Any] = []
    captions: List[str] = []
    caption_image_ids: List[Any] = []

    for row_idx, row in enumerate(ds):
        if image_id_column:
            raw_image_id = row.get(image_id_column)
        else:
            raw_image_id = row.get("image_id", row.get("id", row.get("img_id", row.get("filename", None))))
        image_id = _coerce_image_id(raw_image_id, row_idx)

        if image_column:
            image = row.get(image_column)
        else:
            image = row.get("image", None)
        if image is None:
            continue

        if image_id not in image_records:
            image_records[image_id] = image
            image_ids.append(image_id)

        if caption_column:
            cap = row.get(caption_column)
        else:
            cap = row.get(
                "caption",
                row.get("captions", row.get("sentences", row.get("text", row.get("text_input", None)))),
            )

        for c in _extract_captions(cap):
            captions.append(c)
            caption_image_ids.append(image_id)

    if not image_ids:
        raise RuntimeError("No images found in eval dataset. Check eval.image_column and eval.split.")
    if not captions:
        raise RuntimeError("No captions found in eval dataset. Check eval.caption_column and eval.split.")

    image_tensors: List[Any] = []
    image_id_list: List[Any] = []
    for image_id, image in zip(image_ids, [image_records[i] for i in image_ids]):
        image_tensors.append(image)
        image_id_list.append(image_id)

    with torch.no_grad():
        image_embeds: List[torch.Tensor] = []
        for batch in _batch(image_tensors, batch_size):
            feats = image_encoder(batch)
            image_embeds.append(normalize_fn(feats).cpu())
        image_embeds_t = torch.cat(image_embeds, dim=0)

        text_embeds: List[torch.Tensor] = []
        for batch in _batch(captions, batch_size):
            feats = text_encoder(batch)
            text_embeds.append(normalize_fn(feats).cpu())
        text_embeds_t = torch.cat(text_embeds, dim=0)

    sim_t2i = sim_fn(text_embeds_t, image_embeds_t)
    image_id_to_index = {iid: idx for idx, iid in enumerate(image_id_list)}
    target_image_idx = [image_id_to_index[iid] for iid in caption_image_ids]

    t2i = RetrievalStats(
        r1=_compute_recall_at_k(sim_t2i, target_image_idx, 1),
        r5=_compute_recall_at_k(sim_t2i, target_image_idx, 5),
        r10=_compute_recall_at_k(sim_t2i, target_image_idx, 10),
    )

    image_to_caption_indices: Dict[Any, List[int]] = {}
    for idx, image_id in enumerate(caption_image_ids):
        image_to_caption_indices.setdefault(image_id, []).append(idx)
    targets_i2t = [image_to_caption_indices[iid] for iid in image_id_list]

    sim_i2t = sim_fn(image_embeds_t, text_embeds_t)
    i2t = RetrievalStats(
        r1=_compute_recall_at_k_multi(sim_i2t, targets_i2t, 1),
        r5=_compute_recall_at_k_multi(sim_i2t, targets_i2t, 5),
        r10=_compute_recall_at_k_multi(sim_i2t, targets_i2t, 10),
    )

    return {"text_to_image": t2i, "image_to_text": i2t}


def _build_hf_clip_encoders(
    eval_cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[
    Callable[[List[Any]], torch.Tensor],
    Callable[[List[str]], torch.Tensor],
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
]:
    from transformers import CLIPModel, CLIPProcessor

    clip_name = eval_cfg.get("clip_model", "openai/clip-vit-base-patch32")
    clip = _from_pretrained_quiet(CLIPModel, clip_name, "clip_model").to(device)
    processor = _from_pretrained_quiet(CLIPProcessor, clip_name, "clip_processor")
    clip.eval()

    def image_encoder(images: List[Any]) -> torch.Tensor:
        inputs = processor(images=images, text=[""] * len(images), return_tensors="pt").to(device)
        return clip.get_image_features(pixel_values=inputs["pixel_values"])

    def text_encoder(texts: List[str]) -> torch.Tensor:
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        return clip.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
        )

    def sim_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a @ b.T

    return image_encoder, text_encoder, sim_fn


def _build_vlm_checkpoint_encoders(
    eval_cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[
    Callable[[List[Any]], torch.Tensor],
    Callable[[List[str]], torch.Tensor],
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
]:
    from omegaconf import OmegaConf
    from transformers import CLIPProcessor

    from vlm.losses.contrastive_lorentz import matching_score as train_matching_score
    from vlm.models.clip_like import ClipLikeModel
    from vlm.train.checkpoint import load_checkpoint

    train_config_path = eval_cfg.get("vlm_train_config", None)
    checkpoint_path = eval_cfg.get("vlm_checkpoint", None)
    if not train_config_path or not checkpoint_path:
        raise ValueError("vlm_train_config and vlm_checkpoint are required for model_source=vlm_checkpoint")

    train_cfg = OmegaConf.to_container(OmegaConf.load(train_config_path), resolve=True)
    if not isinstance(train_cfg, dict):
        raise ValueError(f"Invalid train config: {train_config_path}")
    train_cfg["rank"] = 0
    train_cfg["world_size"] = 1

    model = ClipLikeModel(train_cfg).to(device)
    ckpt_payload = load_checkpoint(checkpoint_path, model=model, map_location=device)
    model.eval()
    ckpt_mode = ckpt_payload.get("model_format", "full")
    if ckpt_mode == "lora_only":
        print(f"[load] vlm_checkpoint: {checkpoint_path} (lora_only)")
    else:
        print(f"[load] vlm_checkpoint: {checkpoint_path}")

    model_cfg = train_cfg.get("model", {})
    geometry = model_cfg.get("geometry", "lorentz")
    score_type = model_cfg.get("matching_score", "lorentz_inner_product")
    hybrid_cfg = model_cfg.get("hybrid", {})
    default_curv = float(model_cfg.get("lorentz", {}).get("curv", 1.0))
    train_cfg_train = train_cfg.get("train", {})
    schedule_total_steps = int(
        train_cfg_train.get("schedule_total_steps")
        or train_cfg_train.get("max_steps")
        or 0
    )
    checkpoint_step = None
    m = re.search(r"checkpoint_step_(\d+)\.pt$", str(checkpoint_path))
    if m:
        checkpoint_step = int(m.group(1))

    clip_name = train_cfg.get("model", {}).get("hf_model_name", "openai/clip-vit-base-patch32")
    processor = _from_pretrained_quiet(CLIPProcessor, clip_name, "clip_processor")

    def image_encoder(images: List[Any]) -> torch.Tensor:
        inputs = processor(images=images, return_tensors="pt").to(device)
        return model.encode_image(inputs["pixel_values"])

    def text_encoder(texts: List[str]) -> torch.Tensor:
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        tokens = {"input_ids": inputs["input_ids"]}
        if inputs.get("attention_mask", None) is not None:
            tokens["attention_mask"] = inputs["attention_mask"]
        return model.encode_text(tokens)

    def sim_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        curv_value: float | torch.Tensor = default_curv
        if hasattr(model, "_get_curv"):
            with torch.no_grad():
                curv_tensor = model._get_curv()  # type: ignore[attr-defined]
            if isinstance(curv_tensor, torch.Tensor):
                curv_value = float(curv_tensor.detach().float().mean().cpu())
        return train_matching_score(
            a,
            b,
            geometry=geometry,
            score_type=score_type,
            curv=curv_value,
            hybrid_cfg=hybrid_cfg,
            current_step=checkpoint_step,
            total_steps=schedule_total_steps if schedule_total_steps > 0 else None,
        )

    return image_encoder, text_encoder, sim_fn


def _build_hycoclip_encoders(
    eval_cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[
    Callable[[List[Any]], torch.Tensor],
    Callable[[List[str]], torch.Tensor],
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
]:
    import sys
    from pathlib import Path

    import torchvision.transforms as T

    repo_root = Path(__file__).resolve().parents[2]
    hycoclip_root = repo_root / "hycoclip"
    if hycoclip_root.exists() and str(hycoclip_root) not in sys.path:
        sys.path.insert(0, str(hycoclip_root))

    if "loguru" not in sys.modules:
        class _DummyLogger:
            def __getattr__(self, name):
                def _noop(*args, **kwargs):
                    return None
                return _noop
        sys.modules["loguru"] = type(sys)("loguru")
        sys.modules["loguru"].logger = _DummyLogger()

    L = None
    try:
        from other_models import lorentz as L  # type: ignore
    except Exception:
        try:
            from hycoclip import lorentz as L  # type: ignore
        except Exception:
            L = None

    if L is None:
        class _Lorentz:
            @staticmethod
            def pairwise_inner(x: torch.Tensor, y: torch.Tensor, curv: float | torch.Tensor = 1.0) -> torch.Tensor:
                x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
                y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
                return x @ y.T - x_time @ y_time.T

        L = _Lorentz
    from hycoclip.config import LazyConfig, LazyFactory
    from hycoclip.models import HyCoCLIP, MERU
    from hycoclip.tokenizer import Tokenizer
    from hycoclip.utils.checkpointing import CheckpointManager

    train_config = eval_cfg.get("hycoclip_train_config", None)
    checkpoint_path = eval_cfg.get("hycoclip_checkpoint", None)
    if not train_config or not checkpoint_path:
        raise ValueError("hycoclip_train_config and hycoclip_checkpoint are required for model_source=hycoclip")

    cfg_train = LazyConfig.load(train_config)
    model = LazyFactory.build_model(cfg_train, device=device).eval()

    allow_pickle = bool(eval_cfg.get("hycoclip_allow_pickle", True))
    if allow_pickle:
        import omegaconf

        torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model" in state:
                model.load_state_dict(state["model"])
            else:
                model.load_state_dict(state)
        except Exception:
            CheckpointManager(model=model).load(checkpoint_path)
    else:
        CheckpointManager(model=model).load(checkpoint_path)

    image_size = int(eval_cfg.get("image_size", 224))
    image_transform = T.Compose(
        [T.Resize((image_size, image_size), T.InterpolationMode.BICUBIC), T.ToTensor()]
    )
    tokenizer = Tokenizer()

    def _to_rgb(img: Any) -> Any:
        if hasattr(img, "convert"):
            return img.convert("RGB")
        return img

    def image_encoder(images: List[Any]) -> torch.Tensor:
        batch = torch.stack([image_transform(_to_rgb(img)) for img in images], dim=0).to(device)
        return model.encode_image(batch, project=True)

    def text_encoder(texts: List[str]) -> torch.Tensor:
        tokens = tokenizer(texts)
        return model.encode_text(tokens, project=True)

    def sim_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if isinstance(model, (HyCoCLIP, MERU)):
            curv = model.curv.exp().to(a.device)
            return L.pairwise_inner(a, b, curv)
        return a @ b.T

    return image_encoder, text_encoder, sim_fn
