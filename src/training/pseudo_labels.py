"""
Pseudo-label generation from a trained Stage 1 model.

Key fixes:
  - Uses ADAPTIVE per-class thresholds (percentile-based) instead of a
    single fixed threshold.  This is critical because some classes (block)
    produce lower max-probabilities than others (interjection).
  - Uses the clip-level label as a PRIOR: only generates frame labels for
    the class that the clip is labelled with (prevents cross-class leakage).
  - Reports per-class statistics so you can verify coverage.
"""

import os
import numpy as np
import torch
from tqdm.auto import tqdm

from src.data.preprocessor import AudioPreprocessor


@torch.no_grad()
def generate_pseudo_labels(model, dataframe, preprocessor, config,
                           confidence_threshold=None,
                           min_event_length=None,
                           max_samples=None):
    """Generate frame-level pseudo-labels.

    Uses a two-pass approach with cached inference:
      Pass 1 — run model on all samples, cache probs, collect max-prob stats.
      Pass 2 — threshold cached probs using per-class adaptive thresholds.
    """
    min_event_length = min_event_length or config.MIN_EVENT_LENGTH
    model.eval()

    df = dataframe.copy()
    if max_samples and len(df) > max_samples:
        df = df.head(max_samples)

    label2idx = {c: i for i, c in enumerate(config.STUTTER_TYPES)}
    num_classes = config.NUM_CLASSES

    # ── Pass 1: run inference ONCE and cache all probabilities ──
    print("Pseudo-labels Pass 1: running inference and collecting statistics...")
    per_class_maxprobs = {c: [] for c in range(num_classes)}
    cached_probs = {}  # file_path → (T, C) numpy array

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pass 1", leave=False):
        fp = row["file_path"]
        audio = preprocessor.load_and_preprocess(fp)
        if audio is None:
            continue

        audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(config.DEVICE)
        logits = model(audio_t)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (T, C)

        cached_probs[fp] = probs

        for c in range(num_classes):
            max_p = probs[:, c].max()
            per_class_maxprobs[c].append(max_p)

    # Compute per-class thresholds (use configurable percentile of max-probs)
    percentile = getattr(config, 'PSEUDO_LABEL_PERCENTILE', 65)
    min_thresh = getattr(config, 'PSEUDO_LABEL_MIN_THRESH', 0.35)
    thresholds = {}
    for c in range(num_classes):
        if per_class_maxprobs[c]:
            arr = np.array(per_class_maxprobs[c])
            thresholds[c] = max(float(np.percentile(arr, percentile)), min_thresh)
        else:
            thresholds[c] = 0.5

    print("Per-class adaptive thresholds:")
    for c in range(num_classes):
        name = config.STUTTER_TYPES[c]
        print(f"  {name:20s}: {thresholds[c]:.3f}")

    # ── Pass 2: generate pseudo-labels from CACHED probs (no re-inference) ──
    print("Pass 2: generating pseudo-labels from cached probabilities...")
    pseudo_labels = {}
    stats = {"total": 0, "with_events": 0}
    per_class_events = {c: 0 for c in range(num_classes)}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pass 2", leave=False):
        fp = row["file_path"]
        cls_name = row.get("class", None)

        if fp not in cached_probs:
            continue

        probs = cached_probs[fp]
        frame_labels = np.zeros_like(probs, dtype=np.float32)

        # Use clip label as prior: only label the class the clip belongs to
        if cls_name and cls_name in label2idx:
            target_c = label2idx[cls_name]
            binary = (probs[:, target_c] >= thresholds[target_c]).astype(np.float32)
            binary = _remove_short_events(binary, min_event_length)
            frame_labels[:, target_c] = binary

            # Also allow secondary detections if very confident
            for c in range(num_classes):
                if c == target_c:
                    continue
                high_thresh = min(thresholds[c] + 0.15, 0.8)
                sec = (probs[:, c] >= high_thresh).astype(np.float32)
                sec = _remove_short_events(sec, min_event_length + 2)
                frame_labels[:, c] = sec
        else:
            # No clip label available: threshold all classes
            for c in range(num_classes):
                binary = (probs[:, c] >= thresholds[c]).astype(np.float32)
                binary = _remove_short_events(binary, min_event_length)
                frame_labels[:, c] = binary

        pseudo_labels[fp] = frame_labels
        stats["total"] += 1
        if frame_labels.sum() > 0:
            stats["with_events"] += 1
            for c in range(num_classes):
                if frame_labels[:, c].sum() > 0:
                    per_class_events[c] += 1

    # Free cached probs
    del cached_probs

    pct = stats["with_events"] / max(stats["total"], 1) * 100
    print(f"\nPseudo-labels: {stats['total']} samples, "
          f"{stats['with_events']} with events ({pct:.1f}%)")
    print("Per-class coverage:")
    for c in range(num_classes):
        name = config.STUTTER_TYPES[c]
        print(f"  {name:20s}: {per_class_events[c]} samples with events")

    return pseudo_labels


def _remove_short_events(binary, min_len):
    diff = np.diff(np.concatenate([[0], binary, [0]]))
    onsets = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0]
    out = np.zeros_like(binary)
    for on, off in zip(onsets, offsets):
        if off - on >= min_len:
            out[on:off] = 1.0
    return out


def save_pseudo_labels(pseudo_labels, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, pseudo_labels=pseudo_labels)
    print(f"Pseudo-labels saved → {path}")


def load_pseudo_labels(path):
    data = np.load(path, allow_pickle=True)
    return data["pseudo_labels"].item()
