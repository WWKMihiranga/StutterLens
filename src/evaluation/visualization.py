"""
Visualization utilities for the stuttering detection project.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_training_curves(history: Dict, title: str = "Training Curves",
                         save_path: Optional[str] = None):
    """Plot loss and accuracy curves from a training history dict."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(history["train_loss"], label="Train Loss", marker="o", markersize=3)
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val Loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title} — Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    if "val_accuracy" in history:
        ax.plot(history["val_accuracy"], label="Val Macro F1",
                marker="o", markersize=3, color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"{title} — Validation Macro F1")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_class_distribution(class_counts: Dict[str, int],
                            title: str = "SEP-28k: Class Distribution",
                            save_path: Optional[str] = None):
    """Bar chart of samples per class."""
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(classes, counts, color="steelblue", edgecolor="black")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5,
                f"{int(b.get_height()):,}", ha="center", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Files")
    ax.set_xlabel("Stutter Type")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_rule_contributions(overall: Dict, save_path: Optional[str] = None):
    """Bar + pie chart of neural vs. rule contributions."""
    components = ["Neural", "Burst", "Voicing", "Rhythm"]
    values = [overall.get("neural", 0),
              overall.get("burst", 0),
              overall.get("voicing", 0),
              overall.get("rhythm", 0)]
    colors = ["#4C72B0", "#C44E52", "#55A868", "#CCB974"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    bars = ax.bar(components, values, color=colors, edgecolor="black")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{v:.3f}", ha="center")
    ax.set_ylabel("Average Gate Weight")
    ax.set_title("Component Contributions")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    pos = [max(0, v) for v in values]
    if sum(pos) > 0:
        ax.pie(pos, labels=components, autopct="%1.1f%%",
               startangle=90, colors=colors)
    ax.set_title("Relative Importance")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_frame_predictions(frame_probs: np.ndarray, rule_scores: np.ndarray,
                           gate_weights: np.ndarray,
                           class_names: List[str],
                           save_path: Optional[str] = None):
    """Three-panel visualisation of frame-level model outputs.

    Panels:
      1. Frame probabilities per class with decision threshold.
      2. Rule scores (burst, voicing, rhythm).
      3. Gate weights over time.
    """
    T = frame_probs.shape[0]
    frames = np.arange(T)
    rule_names = ["Burst", "Voicing", "Rhythm"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 1 — Probabilities
    ax = axes[0]
    for c, name in enumerate(class_names):
        if c < frame_probs.shape[1]:
            ax.plot(frames, frame_probs[:, c], label=name, linewidth=1.5)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Threshold")
    ax.set_ylabel("Probability")
    ax.set_title("Frame-Level Class Probabilities")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # # 2 — Predictions vs. targets
    # ax = axes[1]
    # for c, name in enumerate(class_names):
    #     if c < frame_preds.shape[1]:
    #         ax.fill_between(frames, c + frame_preds[:, c] * 0.8, c,
    #                         alpha=0.4, label=f"Pred {name}")
    #         ax.fill_between(frames, c + frame_targets[:, c] * 0.8, c,
    #                         alpha=0.2, hatch="//", label=f"GT {name}")
    # ax.set_ylabel("Class")
    # ax.set_title("Predictions (solid) vs. Ground Truth (hatched)")
    # ax.grid(alpha=0.3)

    # 3 — Rule scores
    ax = axes[1]
    colors_r = ["#C44E52", "#55A868", "#CCB974"]
    for r, (name, col) in enumerate(zip(rule_names, colors_r)):
        if r < rule_scores.shape[1]:
            ax.plot(frames, rule_scores[:, r], label=name, color=col,
                    linewidth=1.5)
    ax.set_ylabel("Score")
    ax.set_title("Soft-Rule Activations")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # 4 — Gate weights
    ax = axes[2]
    gate_labels = ["Neural"] + rule_names
    colors_g = ["#4C72B0", "#C44E52", "#55A868", "#CCB974"]
    for g, (name, col) in enumerate(zip(gate_labels, colors_g)):
        if g < gate_weights.shape[1]:
            ax.plot(frames, gate_weights[:, g], label=name, color=col,
                    linewidth=1.5)
    ax.set_ylabel("Weight")
    ax.set_xlabel("Frame")
    ax.set_title("Adaptive Gate Weights")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_ablation_comparison(full_f1: float, no_rules_f1: float,
                             save_path: Optional[str] = None):
    """Side-by-side bar chart for ablation test results."""
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Full Model\n(with rules)", "Neural Only\n(rules disabled)"]
    values = [full_f1, no_rules_f1]
    colors = ["#55A868", "#C44E52"]

    bars = ax.bar(labels, values, color=colors, edgecolor="black", width=0.5)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{v:.4f}", ha="center", fontweight="bold")
    ax.set_ylabel("Event-Level F1")
    ax.set_title("Ablation Test: Rules ON vs. OFF (RQ3)")
    ax.set_ylim(0, max(values) * 1.3 + 0.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_calibration_analysis(val_preds: np.ndarray, val_preds_cal: np.ndarray,
                              val_targets: np.ndarray, class_names: List[str],
                              temperatures: Dict[str, float],
                              thresholds: Dict[str, float],
                              save_path: Optional[str] = None):
    """Multi-panel calibration diagnostic plot.

    Panel 1: Reliability diagram (before vs. after calibration).
    Panel 2: Per-class probability distributions (positive vs. negative).
    Panel 3: Per-class temperature and threshold summary.
    """
    num_classes = len(class_names)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Panel 1: Reliability diagram (averaged across classes) ──
    ax = axes[0]
    n_bins = 10
    for label, preds, color, ls in [
        ("Before Calibration", val_preds, "steelblue", "--"),
        ("After Calibration", val_preds_cal, "coral", "-"),
    ]:
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_accs, bin_confs, bin_counts = [], [], []
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            mask = (preds >= lo) & (preds < hi)
            if mask.sum() > 0:
                bin_accs.append(val_targets[mask].mean())
                bin_confs.append(preds[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_accs.append(np.nan)
                bin_confs.append((lo + hi) / 2)
                bin_counts.append(0)
        ax.plot(bin_confs, bin_accs, marker='o', label=label, color=color,
                linestyle=ls, markersize=5)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Reliability Diagram')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # ── Panel 2: Per-class probability distributions ──
    ax = axes[1]
    for c, name in enumerate(class_names):
        pos_mask = val_targets[:, c] == 1
        neg_mask = val_targets[:, c] == 0
        if pos_mask.sum() > 0:
            pos_probs = val_preds_cal[pos_mask, c]
            ax.boxplot([pos_probs], positions=[c], widths=0.35,
                       patch_artist=True,
                       boxprops=dict(facecolor='coral', alpha=0.6),
                       medianprops=dict(color='darkred'),
                       showfliers=False)
        if neg_mask.sum() > 0:
            neg_probs = val_preds_cal[neg_mask, c]
            ax.boxplot([neg_probs], positions=[c + 0.4], widths=0.35,
                       patch_artist=True,
                       boxprops=dict(facecolor='steelblue', alpha=0.6),
                       medianprops=dict(color='darkblue'),
                       showfliers=False)
    # Legend proxies
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(facecolor='coral', alpha=0.6, label='Positive'),
        mpatches.Patch(facecolor='steelblue', alpha=0.6, label='Negative'),
    ], fontsize=8)
    ax.set_xticks(np.arange(num_classes) + 0.2)
    ax.set_xticklabels(class_names, rotation=15, fontsize=8)
    ax.set_ylabel('Calibrated Probability')
    ax.set_title('Calibrated Prob. Distribution (Pos vs Neg)')
    ax.grid(axis='y', alpha=0.3)

    # ── Panel 3: Temperature & threshold table ──
    ax = axes[2]
    ax.axis('off')
    table_data = [['Class', 'Temperature', 'Threshold']]
    for cls in class_names:
        t = temperatures.get(cls, 1.0)
        th = thresholds.get(cls, 0.5)
        table_data.append([cls, f'{t:.2f}', f'{th:.2f}'])

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    # Header row styling
    for j in range(3):
        table[0, j].set_facecolor('#4C72B0')
        table[0, j].set_text_props(color='white', fontweight='bold')
    ax.set_title('Per-Class Calibration Parameters', pad=20)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_multilabel_summary(clip_metrics: Dict, class_names: List[str],
                            save_path: Optional[str] = None):
    """Dashboard showing multi-label specific metrics.

    Panels:
      1. Per-class accuracy, precision, recall, F1 grouped bar chart.
      2. Summary table of multi-label metrics (hamming loss, subset acc, sample F1).
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # ── Panel 1: Per-class grouped bars ──
    ax = axes[0]
    x = np.arange(len(class_names))
    width = 0.2
    metrics_to_plot = ['precision', 'recall', 'f1', 'accuracy']
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    for j, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        vals = [clip_metrics['per_class'][c][metric] for c in class_names]
        ax.bar(x + j * width, vals, width, label=metric.capitalize(), color=color)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(class_names, rotation=15, fontsize=8)
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # ── Panel 2: Multi-label summary table ──
    ax = axes[1]
    ax.axis('off')
    summary_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['Hamming Loss', f"{clip_metrics['hamming_loss']:.4f}",
         'Fraction of wrong labels (lower=better)'],
        ['Hamming Accuracy', f"{clip_metrics['accuracy']:.4f}",
         '1 - Hamming Loss'],
        ['Subset Accuracy', f"{clip_metrics['subset_accuracy']:.4f}",
         'Exact-match ratio (all labels correct)'],
        ['Sample F1', f"{clip_metrics['sample_f1']:.4f}",
         'Per-sample F1 averaged (multi-label)'],
        ['Macro F1', f"{clip_metrics['f1']:.4f}",
         'Per-class F1 averaged'],
        ['Macro Precision', f"{clip_metrics['precision']:.4f}",
         'Per-class precision averaged'],
        ['Macro Recall', f"{clip_metrics['recall']:.4f}",
         'Per-class recall averaged'],
    ]
    table = ax.table(cellText=summary_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    for j in range(3):
        table[0, j].set_facecolor('#4C72B0')
        table[0, j].set_text_props(color='white', fontweight='bold')
    ax.set_title('Multi-Label Evaluation Summary', pad=20)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
