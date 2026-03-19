"""
Evaluation metrics for stuttering event detection.

Covers all metrics specified in the PPRS:
  * **Event-level F1** with IoU threshold (localization quality).
  * **Onset / Offset RMSE** in milliseconds (boundary precision).
  * **Clip-level classification** metrics (accuracy, precision, recall, F1).
  * **Multi-label metrics** — Hamming loss, subset accuracy (exact match),
    sample-averaged F1 for proper multi-label evaluation.
  * **Probability calibration** — post-hoc temperature scaling (Platt scaling)
    to fix uncalibrated sigmoid outputs.
  * **Pearson's r** for gate–feature correlation (explanation fidelity).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import (
    precision_recall_fscore_support, classification_report,
    hamming_loss as sk_hamming_loss,
)


# ─────────────────────────────────────────────────────────────────────────────
# Event detection from frame predictions
# ─────────────────────────────────────────────────────────────────────────────
def detect_events(frame_preds: np.ndarray,
                  min_event_length: int = 3,
                  median_filter_size: int = 0,
                  merge_gap: int = 0) -> List[Dict]:
    """Extract contiguous events from a binary frame-prediction matrix.

    Parameters
    ----------
    frame_preds : (T, C) binary array
    min_event_length : minimum frames for a valid event
    median_filter_size : if > 0, apply median filter to smooth predictions
        before event extraction (reduces boundary jitter).
    merge_gap : if > 0, merge events of the same class separated by fewer
        than this many frames (reduces fragmentation).

    Returns
    -------
    list of dicts with keys: class_idx, onset, offset, duration
    """
    from scipy.ndimage import median_filter as _medfilt

    T, C = frame_preds.shape
    events = []
    for c in range(C):
        col = frame_preds[:, c].copy()

        # Optional median filter for smoother boundaries
        if median_filter_size > 1 and T > median_filter_size:
            col = _medfilt(col, size=median_filter_size)
            col = (col > 0.5).astype(float)

        diff = np.diff(np.concatenate([[0], col, [0]]))
        onsets = np.where(diff == 1)[0]
        offsets = np.where(diff == -1)[0]

        # Merge close events
        if merge_gap > 0 and len(onsets) > 1:
            merged_on, merged_off = [onsets[0]], [offsets[0]]
            for k in range(1, len(onsets)):
                if onsets[k] - merged_off[-1] <= merge_gap:
                    merged_off[-1] = offsets[k]  # extend previous event
                else:
                    merged_on.append(onsets[k])
                    merged_off.append(offsets[k])
            onsets, offsets = np.array(merged_on), np.array(merged_off)

        for on, off in zip(onsets, offsets):
            dur = off - on
            if dur >= min_event_length:
                events.append({
                    "class_idx": c,
                    "onset": int(on),
                    "offset": int(off),
                    "duration": int(dur),
                })
    return events


# ─────────────────────────────────────────────────────────────────────────────
# IoU and event-level F1
# ─────────────────────────────────────────────────────────────────────────────
def _iou(ev_a: Dict, ev_b: Dict) -> float:
    """Temporal IoU between two events (same class assumed)."""
    inter_start = max(ev_a["onset"], ev_b["onset"])
    inter_end = min(ev_a["offset"], ev_b["offset"])
    inter = max(0, inter_end - inter_start)
    union = ((ev_a["offset"] - ev_a["onset"])
             + (ev_b["offset"] - ev_b["onset"]) - inter)
    return inter / max(union, 1e-8)


def event_level_f1(pred_events: List[Dict], gt_events: List[Dict],
                   iou_threshold: float = 0.3) -> Dict:
    """Compute event-level precision, recall, F1 at a given IoU threshold."""
    matched_gt = set()
    tp = 0

    for pe in pred_events:
        best_iou, best_idx = 0.0, -1
        for gi, ge in enumerate(gt_events):
            if ge["class_idx"] != pe["class_idx"]:
                continue
            if gi in matched_gt:
                continue
            score = _iou(pe, ge)
            if score > best_iou:
                best_iou = score
                best_idx = gi
        if best_iou >= iou_threshold and best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)

    precision = tp / max(len(pred_events), 1)
    recall = tp / max(len(gt_events), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp,
            "fp": len(pred_events) - tp, "fn": len(gt_events) - tp}


# ─────────────────────────────────────────────────────────────────────────────
# Onset / offset RMSE
# ─────────────────────────────────────────────────────────────────────────────
def boundary_rmse(pred_events: List[Dict], gt_events: List[Dict],
                  ms_per_frame: float = 20.0,
                  iou_threshold: float = 0.1) -> Dict:
    """RMSE of onset and offset errors (in ms) for matched events."""
    onset_errs, offset_errs = [], []
    matched_gt = set()

    for pe in pred_events:
        best_iou, best_idx = 0.0, -1
        for gi, ge in enumerate(gt_events):
            if ge["class_idx"] != pe["class_idx"] or gi in matched_gt:
                continue
            score = _iou(pe, ge)
            if score > best_iou:
                best_iou = score
                best_idx = gi
        if best_iou >= iou_threshold and best_idx >= 0:
            matched_gt.add(best_idx)
            ge = gt_events[best_idx]
            onset_errs.append((pe["onset"] - ge["onset"]) * ms_per_frame)
            offset_errs.append((pe["offset"] - ge["offset"]) * ms_per_frame)

    def _rmse(arr):
        return float(np.sqrt(np.mean(np.square(arr)))) if arr else float("nan")

    return {
        "onset_rmse_ms": _rmse(onset_errs),
        "offset_rmse_ms": _rmse(offset_errs),
        "num_matched": len(onset_errs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Clip-level classification report
# ─────────────────────────────────────────────────────────────────────────────
def clip_level_metrics(all_preds: np.ndarray, all_targets: np.ndarray,
                       class_names: List[str], threshold: float = 0.5) -> Dict:
    """Multi-label classification report with proper metrics.

    In addition to macro-averaged precision/recall/F1, this now reports:
      - **Hamming loss**: fraction of wrong labels (lower is better).
      - **Subset accuracy**: fraction of samples where ALL labels match exactly.
      - **Sample F1**: F1 averaged per-sample (proper multi-label metric).
      - Per-class metrics including per-class accuracy.

    The ``accuracy`` field reports Hamming accuracy (1 - hamming_loss),
    which is more informative for multi-label tasks than subset accuracy.
    """
    preds_bin = (all_preds > threshold).astype(int)
    targets_bin = all_targets.astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        targets_bin, preds_bin, average="macro", zero_division=0
    )
    report = classification_report(
        targets_bin, preds_bin, target_names=class_names, zero_division=0
    )

    # Multi-label specific metrics
    h_loss = float(sk_hamming_loss(targets_bin, preds_bin))
    hamming_acc = 1.0 - h_loss
    subset_acc = float((preds_bin == targets_bin).all(axis=1).mean())

    # Sample-averaged F1 (per-sample, then averaged — proper multi-label metric)
    sample_f1s = []
    for i in range(len(targets_bin)):
        tp_i = ((preds_bin[i] == 1) & (targets_bin[i] == 1)).sum()
        fp_i = ((preds_bin[i] == 1) & (targets_bin[i] == 0)).sum()
        fn_i = ((preds_bin[i] == 0) & (targets_bin[i] == 1)).sum()
        p_i = tp_i / max(tp_i + fp_i, 1)
        r_i = tp_i / max(tp_i + fn_i, 1)
        f1_i = 2 * p_i * r_i / max(p_i + r_i, 1e-8)
        sample_f1s.append(f1_i)
    sample_f1 = float(np.mean(sample_f1s))

    # Per-class metrics
    per_class = {}
    p_pc, r_pc, f1_pc, sup_pc = precision_recall_fscore_support(
        targets_bin, preds_bin, average=None, zero_division=0
    )
    for i, name in enumerate(class_names):
        # Per-class accuracy: fraction of samples correct for this class
        cls_acc = float((preds_bin[:, i] == targets_bin[:, i]).mean())
        per_class[name] = {
            "precision": float(p_pc[i]),
            "recall": float(r_pc[i]),
            "f1": float(f1_pc[i]),
            "support": int(sup_pc[i]),
            "accuracy": cls_acc,
        }

    return {
        "accuracy": hamming_acc,       # Hamming accuracy (multi-label appropriate)
        "subset_accuracy": subset_acc,  # Exact match ratio
        "hamming_loss": h_loss,
        "sample_f1": sample_f1,         # Per-sample averaged F1
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "report": report,
        "per_class": per_class,
    }


def optimize_per_class_thresholds(all_preds: np.ndarray, all_targets: np.ndarray,
                                   class_names: List[str],
                                   search_range: Tuple[float, float] = (0.1, 0.9),
                                   steps: int = 50) -> Dict:
    """Find per-class decision thresholds that maximise per-class F1.

    Instead of using a fixed 0.5 threshold for all classes, this searches
    for the best threshold per class on the validation set.  This is
    especially important for minority classes that may have lower confidence.

    Parameters
    ----------
    all_preds   : (N, C) float probabilities
    all_targets : (N, C) binary targets
    class_names : list of class name strings
    search_range: (min_thresh, max_thresh) to search
    steps       : number of threshold candidates

    Returns
    -------
    dict with 'thresholds' (per-class), 'best_metrics' (per-class F1),
    and 'overall' (macro metrics using optimised thresholds).
    """
    thresholds = np.linspace(search_range[0], search_range[1], steps)
    num_classes = all_preds.shape[1]
    best_thresholds = np.full(num_classes, 0.5)
    best_f1s = np.zeros(num_classes)

    for c in range(num_classes):
        for t in thresholds:
            preds_c = (all_preds[:, c] > t).astype(int)
            targets_c = all_targets[:, c].astype(int)
            if targets_c.sum() == 0:
                continue
            p, r, f1, _ = precision_recall_fscore_support(
                targets_c, preds_c, average="binary", zero_division=0
            )
            if f1 > best_f1s[c]:
                best_f1s[c] = f1
                best_thresholds[c] = t

    # Compute overall metrics with optimised thresholds
    preds_opt = np.zeros_like(all_preds, dtype=int)
    for c in range(num_classes):
        preds_opt[:, c] = (all_preds[:, c] > best_thresholds[c]).astype(int)
    targets_bin = all_targets.astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        targets_bin, preds_opt, average="macro", zero_division=0
    )
    report = classification_report(
        targets_bin, preds_opt, target_names=class_names, zero_division=0
    )
    acc = (preds_opt == targets_bin).mean()

    result = {
        "thresholds": {class_names[c]: float(best_thresholds[c]) for c in range(num_classes)},
        "per_class_f1": {class_names[c]: float(best_f1s[c]) for c in range(num_classes)},
        "overall": {
            "accuracy": float(acc),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "report": report,
        },
    }
    return result


def apply_per_class_thresholds(all_preds: np.ndarray, all_targets: np.ndarray,
                                class_names: List[str],
                                thresholds: Dict[str, float]) -> Dict:
    """Apply pre-computed per-class thresholds (e.g. from validation set) to new data.

    Unlike ``optimize_per_class_thresholds``, this does NOT search — it just
    applies the given thresholds, making it safe to use on a held-out test set.
    """
    num_classes = all_preds.shape[1]
    preds_bin = np.zeros_like(all_preds, dtype=int)
    for c, name in enumerate(class_names):
        t = thresholds.get(name, 0.5)
        preds_bin[:, c] = (all_preds[:, c] > t).astype(int)

    targets_bin = all_targets.astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        targets_bin, preds_bin, average="macro", zero_division=0
    )
    report = classification_report(
        targets_bin, preds_bin, target_names=class_names, zero_division=0
    )
    acc = (preds_bin == targets_bin).mean()

    # Per-class
    per_class_f1 = {}
    p_pc, r_pc, f1_pc, _ = precision_recall_fscore_support(
        targets_bin, preds_bin, average=None, zero_division=0
    )
    for i, name in enumerate(class_names):
        per_class_f1[name] = float(f1_pc[i])

    return {
        "overall": {
            "accuracy": float(acc),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "report": report,
        },
        "per_class_f1": per_class_f1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Probability calibration — post-hoc temperature scaling
# ─────────────────────────────────────────────────────────────────────────────
def calibrate_probabilities(val_preds: np.ndarray, val_targets: np.ndarray,
                            class_names: List[str]) -> Dict:
    """Fit per-class temperature scaling on validation predictions.

    For each class, finds a temperature T that maps raw probabilities p
    to calibrated probabilities via: p_cal = sigmoid(logit(p) / T).
    This fixes the common problem where sigmoid outputs are poorly
    calibrated (e.g., true positives cluster at 0.15 instead of > 0.5).

    Returns a dict with per-class temperatures and a diagnostic report.
    """
    from scipy.special import logit as sp_logit
    from scipy.optimize import minimize_scalar

    temperatures = {}
    diagnostics = {}

    for c, name in enumerate(class_names):
        probs_c = np.clip(val_preds[:, c], 1e-7, 1 - 1e-7)
        targets_c = val_targets[:, c]

        if targets_c.sum() == 0:
            temperatures[name] = 1.0
            diagnostics[name] = {"mean_pos_prob": 0.0, "mean_neg_prob": 0.0, "temp": 1.0}
            continue

        logits_c = sp_logit(probs_c)
        mean_pos = float(probs_c[targets_c == 1].mean())
        mean_neg = float(probs_c[targets_c == 0].mean()) if (targets_c == 0).sum() > 0 else 0.0

        # Find T that minimises BCE on validation set
        def neg_f1_at_temp(t):
            cal_probs = 1.0 / (1.0 + np.exp(-logits_c / max(t, 0.01)))
            # Use threshold search to find best F1
            best_f1 = 0.0
            for thresh in np.linspace(0.2, 0.8, 30):
                preds_t = (cal_probs > thresh).astype(int)
                _, _, f1, _ = precision_recall_fscore_support(
                    targets_c, preds_t, average="binary", zero_division=0
                )
                best_f1 = max(best_f1, f1)
            return -best_f1  # minimise negative F1

        result = minimize_scalar(neg_f1_at_temp, bounds=(0.1, 5.0), method='bounded')
        best_temp = float(result.x)
        temperatures[name] = best_temp
        diagnostics[name] = {
            "mean_pos_prob": mean_pos,
            "mean_neg_prob": mean_neg,
            "temp": best_temp,
            "separation": mean_pos - mean_neg,
        }

    return {"temperatures": temperatures, "diagnostics": diagnostics}


def apply_calibration(preds: np.ndarray, temperatures: Dict[str, float],
                      class_names: List[str]) -> np.ndarray:
    """Apply fitted temperature scaling to probability predictions."""
    from scipy.special import logit as sp_logit

    calibrated = np.zeros_like(preds)
    for c, name in enumerate(class_names):
        probs_c = np.clip(preds[:, c], 1e-7, 1 - 1e-7)
        logits_c = sp_logit(probs_c)
        temp = temperatures.get(name, 1.0)
        calibrated[:, c] = 1.0 / (1.0 + np.exp(-logits_c / max(temp, 0.01)))
    return calibrated


# ─────────────────────────────────────────────────────────────────────────────
# Pearson's r for gate–feature correlation (RQ2)
# ─────────────────────────────────────────────────────────────────────────────
def gate_feature_correlation(gate_weights: np.ndarray,
                             rule_scores: np.ndarray) -> Dict:
    """Pearson's r between each rule's gate weight and its activation score.

    Parameters
    ----------
    gate_weights : (N, R+1)  — columns [neural, burst_gate, voicing_gate, rhythm_gate]
    rule_scores  : (N, R)    — columns [burst_score, voicing_score, rhythm_score]

    Returns
    -------
    dict mapping rule name → Pearson r value.
    """
    rule_names = ["burst", "voicing", "rhythm"]
    num_rules = rule_scores.shape[1] if rule_scores.ndim > 1 else 1
    results = {}
    for i in range(min(num_rules, len(rule_names))):
        name = rule_names[i]
        gate_col = gate_weights[:, i + 1]  # skip neural column
        score_col = rule_scores[:, i]
        if gate_col.std() < 1e-8 or score_col.std() < 1e-8:
            results[name] = 0.0
        else:
            r = np.corrcoef(gate_col, score_col)[0, 1]
            results[name] = float(r)
    return results
