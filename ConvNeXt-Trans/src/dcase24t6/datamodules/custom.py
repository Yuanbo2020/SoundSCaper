#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import pickle
import json

from torch import Tensor
import torch
import numpy as np
from typing_extensions import NotRequired, TypedDict

from aac_datasets.datasets.base import AACDataset

pylog = logging.getLogger(__name__)

class CustomItem(TypedDict):
    """Custom dataset item structure"""
    index: int
    subset: str
    dataset: str
    fname: str
    frame_embs: Tensor
    frame_embs_shape: Tensor
    captions: List[str]
    duration: NotRequired[float]
    # Add other custom fields as needed

class CustomDataset(AACDataset[CustomItem]):
    def __init__(
        self,
        log_mel_path: Union[str, Path],
        captions_path: Union[str, Path],
        subset_path: Union[str, Path],
        subset: str = "train",
        transform: Optional[Callable[[CustomItem], Any]] = None,
        verbose: int = 0,
        *,
        custom_columns: Optional[Dict[str, Any]] = None,
    ):
        """
        :param log_mel_path: Path to pickle file containing {filename: log_mel_ndarray}
        :param captions_path: Path to JSON file containing {filename: caption_info}
        :param subset_path: Path to JSON file containing {split: [filenames]}
        :param subset: Which subset to load (train/val/test)
        """
        # Load raw data
        with open(log_mel_path, 'rb') as f:
            log_mel_data = pickle.load(f)  # {filename: ndarray}

        with open(captions_path, 'r') as f:
            captions_data = json.load(f)  # {filename: caption_info}

        with open(subset_path, 'r') as f:
            subset_info = json.load(f)  # {"train": [fnames], "val": [fnames], "test": [fnames]}

        # Get filenames for target subset
        filenames = subset_info.get(subset, [])
        if not filenames:
            raise ValueError(f"Subset '{subset}' not found in subset_info keys: {subset_info.keys()}")

        # Verify data consistency
        missing_files = [
            fname for fname in filenames 
            if fname not in log_mel_data or fname not in captions_data
        ]
        if missing_files:
            raise RuntimeError(
                f"Missing data for {len(missing_files)} files in subset '{subset}'. "
                f"First 5 missing files: {missing_files[:5]}"
            )

        # Build ordered data lists
        ordered_data = []
        for idx, fname in enumerate(filenames):
            # Get log-mel features (convert numpy array to tensor)
            log_mel = log_mel_data[fname]  # numpy array
            if not isinstance(log_mel, np.ndarray):
                raise TypeError(
                    f"Expected numpy array for log-mel features, got {type(log_mel)} "
                    f"for file {fname}"
                )

            # Get captions (ensure list format)
            caption_info = captions_data[fname]
            if isinstance(caption_info, str):
                captions = [caption_info]
            elif isinstance(caption_info, list):
                captions = caption_info
            else:
                raise TypeError(
                    f"Unexpected caption type {type(caption_info)} for file {fname}"
                )

            # Build item base
            item = {
                "index": idx,
                "subset": subset,
                "dataset": "CustomDataset",
                "fname": fname,
                "frame_embs": log_mel,
                "frame_embs_shape": torch.tensor(log_mel.shape),  # Will be converted to tensor later
                "captions": captions
            }
            item["duration"] = 30.0

            ordered_data.append(item)

        # Convert to raw_data format
        raw_data = {
            "index": [d["index"] for d in ordered_data],
            "subset": [d["subset"] for d in ordered_data],
            "dataset": [d["dataset"] for d in ordered_data],
            "fname": [d["fname"] for d in ordered_data],
            "frame_embs": [d["frame_embs"] for d in ordered_data],
            "frame_embs_shape": [d["frame_embs_shape"] for d in ordered_data],
            "captions": [d["captions"] for d in ordered_data]
        }
        # 在初始化时添加默认 duration
        if "duration" not in raw_data:
            raw_data["duration"] = [30.0] * len(raw_data["index"])

        # Add custom columns
        if custom_columns is not None:
            for col_name, default_val in custom_columns.items():
                raw_data[col_name] = [default_val] * len(ordered_data)

        # Define column names
        column_names = list(CustomItem.__required_keys__) + list(CustomItem.__optional_keys__)

        super().__init__(
            raw_data=raw_data,
            transform=transform,
            column_names=column_names,
            verbose=verbose,
        )

        # Store metadata
        self._log_mel_path = Path(log_mel_path)
        self._captions_path = Path(captions_path)
        self._subset_path = Path(subset_path)
        self._subset = subset

    @property
    def filenames(self) -> List[str]:
        """Get ordered list of filenames in the dataset"""
        return self.raw_data["filename"]

    @property
    def feature_shape(self) -> tuple:
        """Shape of the log-mel features (time, n_mels)"""
        if len(self) == 0:
            return (0, 0)
        sample = self[0]["log_mel"]
        return sample.shape if isinstance(sample, np.ndarray) else (0, 0)

    def __repr__(self) -> str:
        return (
            f"CustomDataset(subset={self._subset}, "
            f"samples={len(self)}, "
            f"feature_shape={self.feature_shape}, "
            f"caption_counts={[len(c) for c in self.raw_data['captions'][:3]]}...)"
        )

# Example usage:
if __name__ == "__main__":
    dataset = CustomDataset(
        log_mel_path="data/log_mel_features.pkl",
        captions_path="data/captions.json",
        subset_path="data/splits.json",
        subset="train",
        custom_columns={"sound_id": "default_id"}
    )
    
    print(dataset)
    print("First item keys:", dataset[0].keys())
    print("Feature shape:", dataset.feature_shape)