"""Evaluation entrypoint."""
from __future__ import annotations

import logging
import warnings

import hydra
from omegaconf import DictConfig, OmegaConf

from vlm.eval.coco_retrieval import evaluate_coco_retrieval


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


@hydra.main(version_base=None, config_path="../configs", config_name="eval_coco_retrieval")
def main(cfg: DictConfig) -> None:
    _configure_quiet_logging()

    OmegaConf.set_struct(cfg, False)
    task = cfg.get("eval", {}).get("task", "coco_retrieval")
    if task == "coco_retrieval":
        results = evaluate_coco_retrieval(cfg)
        print(
            {
                "text_to_image": results["text_to_image"].__dict__,
                "image_to_text": results["image_to_text"].__dict__,
            }
        )
        return
    raise ValueError(f"Unknown eval task: {task}")


if __name__ == "__main__":
    main()
