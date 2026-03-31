import os
import json
import copy
import numpy as np
import torch
from typing import Dict, List

from src.evaluation.metrics import detect_events, event_level_f1, boundary_rmse
from src.utils.helpers import make_json_serializable


# Ablation test (RQ3)
@torch.no_grad()
def ablation_test(model, dataset, config, num_samples: int = 200) -> Dict:

    model.eval()
    results = {"full_model": [], "no_rules": []}
    per_class_full = {}
    per_class_nr = {}

    medfilt = getattr(config, 'EVENT_MEDIAN_FILTER_SIZE', 5)
    merge_gap = getattr(config, 'EVENT_MERGE_GAP', 2)

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        audio = sample["audio"].unsqueeze(0).to(config.DEVICE)
        frame_target = sample["frame_label"].numpy()

        gt_events = detect_events(frame_target, min_event_length=3)
        if not gt_events:
            continue

        # Full model
        model.use_rules = True
        logits_full = model(audio)
        preds_full = (torch.sigmoid(logits_full).squeeze(0).cpu().numpy() > 0.5
                      ).astype(float)
        pred_events_full = detect_events(preds_full, min_event_length=3,
                                         median_filter_size=medfilt,
                                         merge_gap=merge_gap)
        f1_full = event_level_f1(pred_events_full, gt_events)

        # No rules
        model.use_rules = False
        logits_nr = model(audio)
        preds_nr = (torch.sigmoid(logits_nr).squeeze(0).cpu().numpy() > 0.5
                    ).astype(float)
        pred_events_nr = detect_events(preds_nr, min_event_length=3,
                                       median_filter_size=medfilt,
                                       merge_gap=merge_gap)
        f1_nr = event_level_f1(pred_events_nr, gt_events)

        results["full_model"].append(f1_full["f1"])
        results["no_rules"].append(f1_nr["f1"])

        # Track per-class
        cls_name = sample.get("class_name", "unknown")
        per_class_full.setdefault(cls_name, []).append(f1_full["f1"])
        per_class_nr.setdefault(cls_name, []).append(f1_nr["f1"])

    model.use_rules = True  # restore

    avg_full = float(np.mean(results["full_model"])) if results["full_model"] else 0.0
    avg_nr = float(np.mean(results["no_rules"])) if results["no_rules"] else 0.0
    delta = avg_full - avg_nr

    print(f"Ablation test ({len(results['full_model'])} samples):")
    print(f"  Full model F1:   {avg_full:.4f}")
    print(f"  No-rules F1:     {avg_nr:.4f}")
    print(f"  Δ F1 (rules):    {delta:+.4f}")

    # Per-class breakdown
    per_class_delta = {}
    for cls in sorted(set(list(per_class_full.keys()) + list(per_class_nr.keys()))):
        if cls == "__negative__":
            continue
        f_mean = float(np.mean(per_class_full.get(cls, [0])))
        n_mean = float(np.mean(per_class_nr.get(cls, [0])))
        d = f_mean - n_mean
        per_class_delta[cls] = d
        n_samp = len(per_class_full.get(cls, []))
        print(f"    {cls:20s}: full={f_mean:.4f}  no_rules={n_mean:.4f}  "
              f"Δ={d:+.4f}  (n={n_samp})")

    return {"full_model_f1": avg_full, "no_rules_f1": avg_nr, "delta_f1": delta,
            "num_samples": len(results["full_model"]),
            "per_class_delta": per_class_delta}


# Rule contribution analysis (RQ2)
@torch.no_grad()
def analyze_rule_contributions(model, dataset, config,
                               num_samples: int = 30) -> Dict:
    """Compute average gate weights grouped by class.

    Uses stratified sampling to ensure all stutter classes are represented,
    not just whichever class appears first in the dataset index order.
    """
    model.eval()
    rule_names = ["burst", "voicing", "rhythm"]
    per_class = {}

    # Stratified index selection
    # Group dataset indices by class, then sample equally from each group.
    class_indices = {}
    for i in range(len(dataset)):
        cls = dataset[i]["class_name"]
        if cls == "__negative__":
            continue
        class_indices.setdefault(cls, []).append(i)

    # Allocate samples per class (round-robin if uneven)
    n_classes = max(len(class_indices), 1)
    per_cls_budget = max(num_samples // n_classes, 1)
    selected_indices = []
    for cls, indices in class_indices.items():
        rng = np.random.RandomState(config.SEED)
        chosen = rng.choice(indices, size=min(per_cls_budget, len(indices)),
                            replace=False).tolist()
        selected_indices.extend(chosen)

    # Run inference on selected samples
    for i in selected_indices:
        sample = dataset[i]
        audio = sample["audio"].unsqueeze(0).to(config.DEVICE)
        cls = sample["class_name"]

        _, details = model(audio, return_details=True)
        gw = details["gate_weights"].squeeze(0).cpu().numpy()  # (T, R+1)
        avg = gw.mean(axis=0)  # (R+1,)

        if cls not in per_class:
            per_class[cls] = {"neural": [], **{r: [] for r in rule_names}}
        per_class[cls]["neural"].append(float(avg[0]))
        for j, rn in enumerate(rule_names):
            if j + 1 < len(avg):
                per_class[cls][rn].append(float(avg[j + 1]))

    # Aggregate
    summary = {}
    for cls, data in per_class.items():
        summary[cls] = {k: float(np.mean(v)) if v else 0.0 for k, v in data.items()}

    # Overall
    all_neural = [v for d in per_class.values() for v in d["neural"]]
    overall = {"neural": float(np.mean(all_neural))}
    for rn in rule_names:
        vals = [v for d in per_class.values() for v in d[rn]]
        overall[rn] = float(np.mean(vals)) if vals else 0.0
    most_important = max(rule_names, key=lambda r: overall[r])

    print(f"Rule contributions (stratified, {len(selected_indices)} samples):")
    for cls, s in summary.items():
        parts = "  ".join(f"{k}={v:.3f}" for k, v in s.items())
        print(f"  {cls:20s}: {parts}")
    print(f"  Overall most important rule: {most_important}")

    return {"per_class": summary, "overall": overall,
            "most_important_rule": most_important}


# Clinical metrics
def compute_clinical_metrics(events: List[Dict], duration_seconds: float,
                             ms_per_frame: float = 20.0) -> Dict:
    """Compute events/minute and average event duration."""
    if not events:
        return {"events_per_minute": 0.0, "avg_event_duration_ms": 0.0,
                "total_events": 0, "distribution": {}}
    durations_ms = [e["duration"] * ms_per_frame for e in events]
    dist = {}
    for e in events:
        c = e["class_idx"]
        dist[c] = dist.get(c, 0) + 1

    return {
        "events_per_minute": len(events) / max(duration_seconds / 60, 1e-8),
        "avg_event_duration_ms": float(np.mean(durations_ms)),
        "total_events": len(events),
        "distribution": dist,
    }


# Single-audio demo
@torch.no_grad()
def demo_single_audio(model, audio_path: str, preprocessor, config,
                      idx2label: Dict, calibration_info: Dict = None) -> Dict:
    
    model.eval()
    audio = preprocessor.load_and_preprocess(audio_path)
    if audio is None:
        return {"error": f"Could not load {audio_path}"}

    audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(config.DEVICE)
    logits, details = model(audio_t, return_details=True)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    # Apply calibration if available
    thresholds = np.full(probs.shape[1], 0.5)
    calibrated = False
    if calibration_info is not None:
        from src.evaluation.metrics import apply_calibration
        # apply_calibration expects (N, C), so add batch dim
        probs = apply_calibration(
            probs, calibration_info['temperatures'], config.STUTTER_TYPES
        )
        for c, name in enumerate(config.STUTTER_TYPES):
            thresholds[c] = calibration_info['thresholds'].get(name, 0.5)
        calibrated = True

    # Apply per-class thresholds
    preds = np.zeros_like(probs)
    for c in range(probs.shape[1]):
        preds[:, c] = (probs[:, c] > thresholds[c]).astype(float)

    gate_w = details["gate_weights"].squeeze(0).cpu().numpy()

    events = detect_events(preds, min_event_length=3)
    dur_s = len(audio) / config.SAMPLE_RATE
    clinical = compute_clinical_metrics(events, dur_s)

    # Frame-to-ms conversion factor (Wav2Vec2 base: ~20 ms per frame)
    ms_per_frame = (dur_s * 1000) / probs.shape[0]

    event_list = []
    for ev in events:
        cn = idx2label.get(str(ev["class_idx"]), f"class_{ev['class_idx']}")
        event_list.append({
            "type": cn,
            "onset_ms": round(ev["onset"] * ms_per_frame, 1),
            "offset_ms": round(ev["offset"] * ms_per_frame, 1),
            "duration_ms": round(ev["duration"] * ms_per_frame, 1),
        })

    avg_gate = gate_w.mean(axis=0)
    rule_analysis = {"neural_contribution": float(avg_gate[0])}
    rule_names_demo = ["burst", "voicing", "rhythm"]
    for i, rn in enumerate(rule_names_demo):
        if i + 1 < len(avg_gate):
            rule_analysis[rn] = float(avg_gate[i + 1])
    report = {
        "audio_file": os.path.basename(audio_path),
        "duration_seconds": round(dur_s, 2),
        "total_events": len(events),
        "calibrated": calibrated,
        "thresholds": {config.STUTTER_TYPES[c]: float(thresholds[c])
                       for c in range(len(config.STUTTER_TYPES))},
        "events": event_list,
        "clinical_metrics": make_json_serializable(clinical),
        "rule_analysis": rule_analysis,
    }
    return report
