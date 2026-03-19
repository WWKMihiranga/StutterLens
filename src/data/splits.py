"""
Dataset discovery, speaker-ID extraction, and speaker-disjoint splitting
for the SEP-28k stuttering dataset.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List


def discover_dataset(dataset_path: str, class_mapping: Dict[str, str],
                     non_stutter_classes: List[str],
                     include_negatives: bool = False,
                     negative_ratio: float = 0.15,
                     negative_class_name: str = "NoStutteredWords") -> pd.DataFrame:
    """Scan *dataset_path* for class sub-folders and build an index DataFrame.

    Only folders whose names appear in *class_mapping* are kept.
    Uses case-insensitive matching so "Interjection" matches correctly.

    If *include_negatives* is True, also samples a fraction of files from
    the *negative_class_name* folder and labels them as ``__negative__``
    (all-zero label vector). This teaches the model what "no stutter" sounds
    like and dramatically improves precision.

    Returns
    -------
    pd.DataFrame with columns [file_path, filename, class, class_label].
    """
    records = []
    negative_records = []
    all_items = sorted(
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    )

    # Build case-insensitive lookup for class mapping
    ci_mapping = {k.lower(): v for k, v in class_mapping.items()}
    ci_non_stutter = {c.lower() for c in non_stutter_classes}

    for folder_name in all_items:
        folder_path = os.path.join(dataset_path, folder_name)
        wav_files = sorted(glob.glob(os.path.join(folder_path, "*.wav")))

        # Check if this is the negative class we want to sample from
        if include_negatives and folder_name.lower() == negative_class_name.lower():
            for fp in wav_files:
                negative_records.append({
                    "file_path": fp,
                    "filename": os.path.basename(fp),
                    "class": "__negative__",
                    "class_label": "__negative__",
                })
            continue

        if folder_name.lower() in ci_non_stutter:
            continue
        if folder_name.lower() not in ci_mapping and folder_name not in class_mapping:
            continue

        canonical = class_mapping.get(folder_name,
                                       ci_mapping.get(folder_name.lower()))
        if canonical is None:
            continue

        for fp in wav_files:
            records.append({
                "file_path": fp,
                "filename": os.path.basename(fp),
                "class": canonical,
                "class_label": canonical,
            })

    df = pd.DataFrame(records)

    # Add negative samples if requested
    if include_negatives and negative_records:
        n_stutter = len(df)
        n_neg = int(n_stutter * negative_ratio)
        neg_df = pd.DataFrame(negative_records)
        if len(neg_df) > n_neg:
            neg_df = neg_df.sample(n_neg, random_state=42).reset_index(drop=True)
        df = pd.concat([df, neg_df], ignore_index=True)
        print(f"  Added {len(neg_df)} negative (NoStutteredWords) samples "
              f"({negative_ratio*100:.0f}% of {n_stutter} stutter samples)")

    print(f"Dataset discovery: {len(df):,} files across "
          f"{df['class'].nunique()} classes")
    for cls in sorted(df["class"].unique()):
        print(f"  {cls:20s}: {(df['class'] == cls).sum():5d} files")

    # Validate all expected classes are present
    expected = set(class_mapping.values())
    found = set(df["class"].unique()) - {"__negative__"}
    missing = expected - found
    if missing:
        print(f"  ⚠ WARNING: Missing classes in data: {missing}")
    return df


def extract_speaker_id(filename: str) -> str:
    """Derive a speaker / episode ID from an SEP-28k filename.

    Convention: ``<Podcast>_<Episode>_<Clip>.wav``
    We use ``<podcast>_<episode>`` as the speaker-group key.
    """
    name = filename.replace(".wav", "")
    parts = name.split("_")
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}".lower()
    if len(parts) == 2:
        return f"{parts[0]}_unknown".lower()
    return "unknown_speaker"


def create_speaker_disjoint_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign each row to train / val / test so that no speaker appears in
    more than one split.

    Returns the DataFrame with a new ``split`` column.
    """
    rng = np.random.RandomState(seed)

    df = df.copy()
    df["speaker_id"] = df["filename"].apply(extract_speaker_id)

    speakers = df["speaker_id"].unique().tolist()
    rng.shuffle(speakers)

    n_train = int(len(speakers) * train_ratio)
    n_val = int(len(speakers) * val_ratio)

    train_speakers = set(speakers[:n_train])
    val_speakers = set(speakers[n_train : n_train + n_val])
    # remaining go to test
    test_speakers = set(speakers[n_train + n_val :])

    def _assign(sid):
        if sid in train_speakers:
            return "train"
        if sid in val_speakers:
            return "val"
        return "test"

    df["split"] = df["speaker_id"].apply(_assign)

    for split in ["train", "val", "test"]:
        n = (df["split"] == split).sum()
        s = df.loc[df["split"] == split, "speaker_id"].nunique()
        print(f"  {split:5s}: {n:5d} clips  ({s} speakers)")

    return df


def save_splits(df: pd.DataFrame, processed_dir: str,
                classes: List[str]) -> Dict:
    """Persist train / val / test CSVs and label mappings."""
    os.makedirs(processed_dir, exist_ok=True)

    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {str(i): c for i, c in enumerate(classes)}
    # Map stutter classes to indices; negatives get -1
    df["label_idx"] = df["class"].map(label2idx).fillna(-1).astype(int)

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        out = os.path.join(processed_dir, f"{split}_split.csv")
        split_df.to_csv(out, index=False)
        # Report per-class counts
        stutter_counts = split_df[split_df['class'] != '__negative__']['class'].value_counts()
        neg_count = (split_df['class'] == '__negative__').sum()
        print(f"  {split:5s}: {len(split_df):5d} clips "
              f"({len(split_df) - neg_count} stutter + {neg_count} negative)")

    mappings = {"label2idx": label2idx, "idx2label": idx2label, "classes": classes}
    map_path = os.path.join(processed_dir, "label_mappings.json")
    with open(map_path, "w") as f:
        json.dump(mappings, f, indent=2)

    print(f"Splits and label mappings saved to {processed_dir}")
    return mappings
