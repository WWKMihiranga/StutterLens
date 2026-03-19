"""
PyTorch Dataset classes for the stuttering detection pipeline.

* ``StutterDatasetClipLevel``  — Stage 1 (MIL): returns (audio, clip_label).
* ``StutterDatasetFrameLevel`` — Stage 2 (self-training): returns
  (audio, clip_label, frame_label, has_pseudo).
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.data.preprocessor import AudioPreprocessor
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — clip-level labels only
# ─────────────────────────────────────────────────────────────────────────────
class StutterDatasetClipLevel(Dataset):
    """Memory-efficient dataset that loads audio on demand."""

    def __init__(self, csv_path: str, preprocessor: AudioPreprocessor,
                 label2idx: dict, max_samples: Optional[int] = None,
                 augment: bool = False):
        self.df = pd.read_csv(csv_path, usecols=["file_path", "class"])
        if max_samples and len(self.df) > max_samples:
            self.df = self.df.sample(max_samples, random_state=42).reset_index(drop=True)
        self.preprocessor = preprocessor
        self.label2idx = label2idx
        self.num_classes = len(label2idx)
        self.augment = augment
        self._compute_class_weights()

    def _compute_class_weights(self):
        counts = self.df["class"].value_counts()
        total = len(self.df)
        self.class_weights = {}
        for cls, idx in self.label2idx.items():
            cnt = counts.get(cls, 1)
            w = total / (self.num_classes * cnt)
            self.class_weights[idx] = min(w, 10.0)  # higher cap for extreme imbalance
        # Negative samples get weight 1.0
        self.class_weights[-1] = 1.0

    def get_sample_weights(self) -> torch.Tensor:
        """Return per-sample weights for WeightedRandomSampler."""
    
        classes = self.df["class"].values

        weights = [
            1.0 if cls == "__negative__"
            else self.class_weights.get(self.label2idx.get(cls, 0), 1.0)
            for cls in classes
        ]

        # IMPORTANT:
        # sampler weights MUST stay on CPU
        return torch.tensor(weights, dtype=torch.float32, device="cpu")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio = self.preprocessor.load_and_preprocess(
            row["file_path"], augment=self.augment
        )
        if audio is None:
            audio = np.zeros(self.preprocessor.target_length, dtype=np.float32)
        audio_tensor = torch.from_numpy(audio).float()

        # Negative samples get all-zero labels
        clip_label = torch.zeros(self.num_classes, dtype=torch.float32)
        cls = row["class"]
        if cls != "__negative__" and cls in self.label2idx:
            clip_label[self.label2idx[cls]] = 1.0

        if cls == "__negative__":
            weight = torch.tensor(1.0, dtype=torch.float32)
        else:
            weight = torch.tensor(
                self.class_weights.get(self.label2idx.get(cls, 0), 1.0),
                dtype=torch.float32,
            )
        return {
            "audio": audio_tensor,
            "clip_label": clip_label,
            "class_name": cls,
            "class_weight": weight,
            "file_path": row["file_path"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — frame-level pseudo-labels
# ─────────────────────────────────────────────────────────────────────────────
class StutterDatasetFrameLevel(Dataset):
    """Wraps a clip-level dataset and augments each sample with pre-computed
    frame-level pseudo-labels loaded from a ``.npz`` archive."""

    def __init__(self, clip_dataset: StutterDatasetClipLevel,
                 pseudo_labels_path: Optional[str] = None,
                 expected_seq_len: int = 149):
        self.clip_dataset = clip_dataset
        self.num_classes = clip_dataset.num_classes
        self.expected_seq_len = expected_seq_len
        self.pseudo_labels = {}

        if pseudo_labels_path and os.path.exists(pseudo_labels_path):
            data = np.load(pseudo_labels_path, allow_pickle=True)
            self.pseudo_labels = data["pseudo_labels"].item()
            print(f"Loaded pseudo-labels for {len(self.pseudo_labels)} samples")

    def __len__(self):
        return len(self.clip_dataset)

    def __getitem__(self, idx):
        sample = self.clip_dataset[idx]

        fp = sample["file_path"]
        if fp in self.pseudo_labels:
            fl = self.pseudo_labels[fp]
            frame_label = torch.from_numpy(fl).float()
            has_pseudo = True
        else:
            frame_label = torch.zeros(self.expected_seq_len, self.num_classes,
                                      dtype=torch.float32)
            has_pseudo = False

        sample["frame_label"] = frame_label
        sample["has_pseudo_label"] = has_pseudo
        return sample
