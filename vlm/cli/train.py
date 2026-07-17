"""Training entrypoint (Hydra-compatible, Accelerate-friendly)."""
from __future__ import annotations

import logging
import warnings

import hydra
from omegaconf import DictConfig, OmegaConf

from vlm.train.engine import train


def _configure_quiet_logging() -> None:
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub.repocard").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=".*Repo card metadata block was not found.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*gemm_and_bias error: CUBLAS_STATUS_NOT_INITIALIZED.*",
        category=UserWarning,
        module=r"torch\.nn\.modules\.linear",
    )
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    _configure_quiet_logging()

    OmegaConf.set_struct(cfg, False)
    train(cfg)


if __name__ == "__main__":
    main()
