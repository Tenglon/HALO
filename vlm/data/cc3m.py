"""CC3M dataset adapters (HuggingFace datasets or JSONL/Parquet)."""
from __future__ import annotations

from typing import Iterable, Iterator, Dict, Any, List, Optional

import os
from io import BytesIO

import torch
from torch.utils.data import IterableDataset, Dataset

from PIL import Image


class CC3MFilesDataset(Dataset):
    """Map-style dataset backed by JSONL/Parquet with image paths."""

    def __init__(self, records: List[Dict[str, Any]], transforms: Optional[Any] = None) -> None:
        self._records = records
        self._transforms = transforms

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self._records[idx]
        image_path = rec["image_path"]
        text = rec["caption"]
        image = Image.open(image_path).convert("RGB")
        if self._transforms is not None:
            image = self._transforms(image)
        return {"image": image, "text": text, "id": rec.get("id")}


class CC3MDummyDataset(Dataset):
    """Synthetic dataset for smoke tests."""

    def __init__(self, num_samples: int, image_size: int) -> None:
        self._num_samples = num_samples
        self._image_size = image_size

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image = torch.rand(3, self._image_size, self._image_size)
        text = f"dummy caption {idx}"
        return {"image": image, "text": text, "id": idx}


class CC3MHFDataset(Dataset):
    """Map-style HuggingFace dataset wrapper."""

    def __init__(
        self,
        dataset: Any,
        image_column: str,
        text_column: str,
        id_column: Optional[str],
        transforms: Optional[Any] = None,
    ) -> None:
        self._dataset = dataset
        self._image_column = image_column
        self._text_column = text_column
        self._id_column = id_column
        self._transforms = transforms

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._dataset[idx]
        return _hf_sample_to_dict(
            sample=sample,
            image_column=self._image_column,
            text_column=self._text_column,
            id_column=self._id_column,
            transforms=self._transforms,
        )


class CC3MHFIterableDataset(IterableDataset):
    """Iterable HuggingFace dataset wrapper (streaming)."""

    def __init__(
        self,
        dataset: Any,
        image_column: str,
        text_column: str,
        id_column: Optional[str],
        transforms: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._image_column = image_column
        self._text_column = text_column
        self._id_column = id_column
        self._transforms = transforms

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for sample in self._dataset:
            text = sample.get(self._text_column)
            if text is None or len(str(text).strip()) == 0:
                continue
            yield _hf_sample_to_dict(
                sample=sample,
                image_column=self._image_column,
                text_column=self._text_column,
                id_column=self._id_column,
                transforms=self._transforms,
            )


def _load_records(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        import json

        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if not obj.get("caption"):
                    continue
                records.append(
                    {
                        "image_path": obj["image_path"],
                        "caption": obj["caption"],
                        "id": obj.get("id"),
                    }
                )
        return records
    if path.endswith(".parquet"):
        import pandas as pd

        df = pd.read_parquet(path)
        df = df[df["caption"].notna() & (df["caption"].str.len() > 0)]
        records = df[["image_path", "caption"]].to_dict(orient="records")
        return records
    raise ValueError(f"Unsupported data file: {path}")


def _maybe_cast_image_column(dataset: Any, image_column: str) -> Any:
    try:
        from datasets import Image as HFImage
    except Exception:
        return dataset
    features = getattr(dataset, "features", None)
    if features is None:
        return dataset
    if image_column not in features:
        return dataset
    feature = features[image_column]
    if feature.__class__.__name__ == "Image":
        return dataset
    try:
        return dataset.cast_column(image_column, HFImage())
    except Exception:
        return dataset


def _hf_sample_to_dict(
    sample: Dict[str, Any],
    image_column: str,
    text_column: str,
    id_column: Optional[str],
    transforms: Optional[Any],
) -> Dict[str, Any]:
    sample_id = sample.get(id_column) if id_column else sample.get("id")
    image = sample.get(image_column)
    if image is None:
        return {"image": None, "text": None, "id": sample_id}
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            image = Image.open(BytesIO(image["bytes"]))
        elif image.get("path") is not None:
            image = Image.open(image["path"])
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    elif isinstance(image, str):
        image = Image.open(image).convert("RGB")
    if transforms is not None:
        image = transforms(image)
    text = sample.get(text_column)
    if text is None or len(str(text).strip()) == 0:
        return {"image": None, "text": None, "id": sample_id}
    return {"image": image, "text": text, "id": sample_id}


def build_cc3m_dataset(cfg: Dict[str, Any], transforms: Optional[Any] = None) -> Iterable[Dict[str, Any]]:
    """Factory for CC3M dataset."""
    data_format = cfg.get("format", "hf")
    if data_format == "dummy":
        num_samples = int(cfg.get("dummy_num_samples", 64))
        image_size = int(cfg.get("image_size", 224))
        return CC3MDummyDataset(num_samples=num_samples, image_size=image_size)
    if data_format in {"hf", "datasets"}:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("datasets is required for HuggingFace datasets") from exc
        name = cfg.get("dataset_name", "conceptual_captions")
        config = cfg.get("dataset_config", "cc3m")
        split = cfg.get("split", "train")
        streaming = bool(cfg.get("streaming", False))
        data_dir = cfg.get("data_dir")
        data_files = cfg.get("data_files")
        load_kwargs = {
            "split": split,
            "streaming": streaming,
            "data_dir": data_dir,
            "data_files": data_files,
        }
        if config:
            dataset = load_dataset(name, config, **load_kwargs)
        else:
            dataset = load_dataset(name, **load_kwargs)
        image_column = cfg.get("image_column", "image")
        text_column = cfg.get("text_column", "caption")
        id_column = cfg.get("id_column")
        if not streaming:
            dataset = dataset.filter(
                lambda x: x.get(image_column) is not None
                and x.get(text_column) is not None
                and len(str(x.get(text_column)).strip()) > 0
            )
        dataset = _maybe_cast_image_column(dataset, image_column)
        if streaming:
            if cfg.get("shuffle", True):
                buffer_size = int(cfg.get("buffer_size", 1000))
                dataset = dataset.shuffle(buffer_size=buffer_size, seed=int(cfg.get("seed", 42)))
            return CC3MHFIterableDataset(
                dataset=dataset,
                image_column=image_column,
                text_column=text_column,
                id_column=id_column,
                transforms=transforms,
            )
        return CC3MHFDataset(
            dataset=dataset,
            image_column=image_column,
            text_column=text_column,
            id_column=id_column,
            transforms=transforms,
        )
    if data_format in {"jsonl", "parquet"}:
        path = cfg.get("path")
        if not path:
            raise ValueError("No path provided for JSONL/Parquet dataset")
        if os.path.isdir(path):
            files = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.endswith(f".{data_format}")
            ]
        else:
            files = [path]
        records: List[Dict[str, Any]] = []
        for fpath in files:
            records.extend(_load_records(fpath))
        return CC3MFilesDataset(records=records, transforms=transforms)
    raise ValueError(f"Unknown data format: {data_format}")
