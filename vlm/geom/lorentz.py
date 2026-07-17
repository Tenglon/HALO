"""Lorentz geometry utilities."""
from __future__ import annotations

import torch


def lift_x0(x: torch.Tensor, curv: float) -> torch.Tensor:
    """Compute time component from spatial component."""
    curv_t = torch.as_tensor(curv, device=x.device, dtype=x.dtype)
    spatial_sq = (x * x).sum(dim=-1)
    inside = (1.0 / curv_t) + spatial_sq
    inside = torch.clamp(inside, min=1e-6)
    return torch.sqrt(inside)


def lift_to_lorentz(x: torch.Tensor, curv: float) -> torch.Tensor:
    """Lift spatial features to explicit Lorentz coordinates [x0, x]."""
    x0 = lift_x0(x, curv)
    return torch.cat([x0.unsqueeze(-1), x], dim=-1)


def _lorentz_inner_product_lifted(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2 and y.dim() == 2:
        spatial = x[:, 1:] @ y[:, 1:].t()
        return spatial - (x[:, 0:1] * y[None, :, 0])
    spatial = (x[..., 1:] * y[..., 1:]).sum(dim=-1)
    return spatial - (x[..., 0] * y[..., 0])


def lorentz_inner_product(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: float,
    input_is_lifted: bool = False,
) -> torch.Tensor:
    """Compute Lorentz inner product."""
    if input_is_lifted:
        return _lorentz_inner_product_lifted(x, y)
    if x.dim() == 2 and y.dim() == 2:
        x0 = lift_x0(x, curv)
        y0 = lift_x0(y, curv)
        spatial = x @ y.t()
        return spatial - (x0[:, None] * y0[None, :])
    x0 = lift_x0(x, curv)
    y0 = lift_x0(y, curv)
    spatial = (x * y).sum(dim=-1)
    return spatial - (x0 * y0)


def lorentz_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: float,
    eps: float = 1e-6,
    input_is_lifted: bool = False,
) -> torch.Tensor:
    """Compute Lorentz distance."""
    curv_t = torch.as_tensor(curv, device=x.device, dtype=x.dtype)
    if x.dim() == 2 and y.dim() == 2:
        lip = lorentz_inner_product(x, y, curv, input_is_lifted=input_is_lifted)
        c = (-curv_t * lip).clamp(min=1.0 + eps)
        return torch.acosh(c) / torch.sqrt(curv_t)
    lip = lorentz_inner_product(x, y, curv, input_is_lifted=input_is_lifted)
    c = (-curv_t * lip).clamp(min=1.0 + eps)
    return torch.acosh(c) / torch.sqrt(curv_t)


def matching_score(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: float,
    input_is_lifted: bool = False,
) -> torch.Tensor:
    """Compute matching score s(x,y) = -curv * <x,y>_L."""
    curv_t = torch.as_tensor(curv, device=x.device, dtype=x.dtype)
    return -curv_t * lorentz_inner_product(x, y, curv, input_is_lifted=input_is_lifted)
