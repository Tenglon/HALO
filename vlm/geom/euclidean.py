"""Euclidean geometry utilities."""
from __future__ import annotations

import torch


def euclidean_inner_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Euclidean inner product."""
    if x.dim() == 2 and y.dim() == 2:
        return x @ y.t()
    return (x * y).sum(dim=-1)


def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Euclidean distance."""
    if x.dim() == 2 and y.dim() == 2:
        # Use cdist to avoid materializing [N, M, D] which can OOM in eval.
        return torch.cdist(x, y, p=2)
    return torch.norm(x - y, dim=-1)
