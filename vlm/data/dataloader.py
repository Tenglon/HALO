"""DataLoader helpers."""
from __future__ import annotations

from typing import Any, Iterable, Dict, List

import torch
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset


def _get_rank_world_size() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _collate_fn(batch: List[Dict[str, Any]], tokenizer: Any, max_length: int) -> Dict[str, Any]:
    batch = [b for b in batch if b.get("image") is not None and b.get("text") is not None]
    if not batch:
        raise ValueError("Empty batch after filtering invalid samples")
    images = torch.stack([b["image"] for b in batch], dim=0)
    texts = [b["text"] for b in batch]
    ids = [b.get("id") for b in batch]
    tokens = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokens = {k: v for k, v in tokens.items()}
    return {"image": images, "text": texts, "tokens": tokens, "id": ids}


def build_dataloader(dataset: Iterable[Any], cfg: dict, tokenizer: Any) -> DataLoader:
    """Return a torch DataLoader."""
    batch_size = int(cfg.get("batch_size", 256))
    num_workers = int(cfg.get("num_workers", 8))
    pin_memory = bool(cfg.get("pin_memory", True))
    drop_last = bool(cfg.get("drop_last", True))
    persistent_workers = bool(cfg.get("persistent_workers", True))
    max_text_len = int(cfg.get("max_text_len", 77))
    collate_fn = lambda b: _collate_fn(b, tokenizer, max_text_len)
    use_persistent_workers = persistent_workers and (num_workers > 0)

    if isinstance(dataset, IterableDataset):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=use_persistent_workers,
            collate_fn=collate_fn,
        )

    rank, world_size = _get_rank_world_size()
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=cfg.get("shuffle", True))
        shuffle = False
    else:
        sampler = None
        shuffle = cfg.get("shuffle", True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=use_persistent_workers,
        collate_fn=collate_fn,
    )
