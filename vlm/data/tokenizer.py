"""Tokenizer utilities."""
from __future__ import annotations

from typing import Any, List


def build_tokenizer(cfg: dict) -> Any:
    """Return tokenizer instance."""
    provider = cfg.get("provider", "transformers")
    name = cfg.get("name", "openai/clip-vit-base-patch32")
    if provider == "transformers":
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name)
    raise ValueError(f"Unsupported tokenizer provider: {provider}")


def tokenize(tokenizer: Any, texts: List[str], max_length: int) -> Any:
    """Tokenize text list."""
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
