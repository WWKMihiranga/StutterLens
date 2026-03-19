# StutterLens — Interpretable Weakly-Supervised Stuttering Detection

> **BSc (Hons) Computer Science — Final Year Project**  
> W.W. Kavindu Mihiranga · University of Westminster / IIT Sri Lanka  
> Supervisor: Mr. Rathesan Sivagnanalingam

## Overview

StutterLens is a neuro-symbolic system that detects stuttering events from
speech audio and provides **millisecond-level temporal boundaries**
(onset/offset times) with **human-readable explanations** for each detection.
The model is trained using only coarse **clip-level labels** from the SEP-28k
dataset — no frame-by-frame annotations are needed.

### Key capabilities

- Detects 3 stuttering types: **interjection**, **prolongation**, **word repetition**
- Recovers event boundaries with ~50 ms accuracy from weak supervision alone
- Provides per-event explanations via differentiable acoustic rules
- Supports audio of **any length** (automatic chunking)
- Includes post-training probability calibration for reliable thresholds

## Results

Results on the held-out test set (1 331 clips, speaker-disjoint):

| Metric | Value |
|--------|-------|
| Clip-level macro F1 (calibrated + optimised thresholds) | **0.623** |
| Clip-level macro F1 (raw, fixed 0.5 threshold) | 0.612 |
| Hamming accuracy | 0.809 |
| Event-level F1 (IoU ≥ 0.3) | **0.549** |
| Onset RMSE | 50.5 ms |
| Offset RMSE | 40.1 ms |
| Ablation Δ F1 (rules ON − OFF) | +0.026 |

### Per-class breakdown

| Class | Support | F1 (A) | F1 (B) | Threshold |
|-------|--------:|-------:|-------:|----------:|
| Interjection | 443 | 0.671 | **0.710** | 0.44 |
| Prolongation | 309 | 0.605 | **0.621** | 0.48 |
| Word repetition | 306 | 0.560 | **0.537** | 0.51 |

*(A) = raw probs, fixed 0.5 · (B) = calibrated + per-class optimised thresholds*

## Architecture

```
Raw audio (any length, 16 kHz)
    │
    │   ┌─── chunked into 3 s segments ───┐
    ▼                                      │
┌───────────────────────────┐              │
│  Wav2Vec2 Encoder         │              │
│  facebook/wav2vec2-base   │  (top 2 layers fine-tuned)
└────────────┬──────────────┘              │
             │  (B, 149, 768)              │
        ┌────┴────┐                        │
        ▼         ▼                        │
┌──────────┐ ┌───────────────────────────┐ │
│ BiLSTM   │ │ Differentiable            │ │
│ Temporal │ │ Soft-Rule Module          │ │
│ Head     │ │  1. Energy burst          │ │
│ (2-layer,│ │  2. Voicing continuity    │ │
│  256-dim)│ │     (duration-gated)      │ │
│          │ │  3. Rhythmic pattern      │ │
│          │ │     (multi-scale lag 2–5) │ │
└────┬─────┘ └────────────┬──────────────┘ │
     │                    │                │
     │   ┌────────────────┘                │
     ▼   ▼                                 │
┌───────────────────────┐                  │
│ Adaptive Gating       │  → per-frame     │
│ Network               │    explanations  │
│ (sigmoid gates,       │                  │
│  anti-collapse floor) │                  │
└───────────┬───────────┘                  │
            ▼                              │
   Final logits (B, T, 3)                  │
            │                              │
            ├── merge events across chunks ─┘
            ▼
   Temporal boundaries + explanations + clinical metrics
```

### Key design decisions

- **Encoder fine-tuning**: last 2 of 12 Wav2Vec2 layers unfrozen with separate
  LR (5e-6) to adapt to disfluent speech without catastrophic forgetting.
- **3 stutter classes**: block and sound repetition were dropped after empirical
  evaluation showed poor annotation quality (block separation: 0.066,
  sound repetition F1: 0.346).
- **Negative sampling**: 25% fluent (NoStutteredWords) clips included as
  explicit negatives to improve precision.

## Differentiable Soft-Rule Module

Three interpretable acoustic rules, each aligned with one stutter class:

| Rule | Target class | Acoustic principle | Key feature |
|------|--------------|--------------------|-------------|
| Energy burst | Interjection | Sudden energy spike relative to local context | Per-clip normalisation |
| Voicing continuity | Prolongation | Low frame-to-frame spectral change | Duration gate: fires only when low change sustained ≥ 7 frames (~140 ms) |
| Rhythmic pattern | Word repetition | Periodic self-similarity at a lag | Multi-scale lags (2–5) with learnable weights and max-pooling |

## Adaptive Gating Network

Combines neural and rule outputs with **independent sigmoid gates** (not
softmax) to avoid zero-sum competition:

- **Frame-probability input**: detached `sigmoid(neural_logits)` fed to gate
  network for context-dependent rule weighting
- **Minimum rule floor**: rule gates clamped ≥ 0.03 during training
- **Entropy annealing**: λ decays from 0.5 → 0.15 over training
- **Residual bypass**: learnable fraction of rule logits added directly to output

## Training Pipeline

| Stage | Description | Supervision | Key details |
|-------|-------------|-------------|-------------|
| **0** | Pre-train soft rules on synthetic data | Synthetic frame labels | 5 000 samples, 20 epochs |
| **1** | Multi-Instance Learning (MIL) | Clip-level weak labels | 25 epochs, focal BCE + class weights, mixed pooling, warmup + cosine LR |
| **cal** | Post-training calibration | Validation set | Per-class temperature scaling + threshold optimisation |
| **2** | Self-training with Mean Teacher | Pseudo frame labels | 20 epochs, EMA 0.999, confidence-weighted boundary loss |

## Project Structure

```
StutterLens/
├── notebooks/
│   └── main_pipeline.ipynb          # Orchestrator (12 phases)
├── src/
│   ├── config.py                    # Centralised configuration
│   ├── data/
│   │   ├── preprocessor.py          # Audio loading, normalisation, 6-type augmentation
│   │   ├── dataset.py               # PyTorch datasets (clip-level + frame-level)
│   │   └── splits.py                # Speaker-disjoint splitting
│   ├── models/
│   │   ├── soft_rules.py            # Differentiable soft-rule module
│   │   ├── gating.py                # Adaptive sigmoid gating network
│   │   ├── temporal_head.py         # BiLSTM temporal detection head
│   │   └── neurosymbolic.py         # Complete NeuroSymbolicStutterDetector
│   ├── training/
│   │   ├── losses.py                # Focal MIL, boundary, consistency, gate entropy
│   │   ├── stage0_rule_pretrain.py  # Synthetic data + rule pre-training
│   │   ├── stage1_mil.py            # MIL training + calibration
│   │   ├── stage2_self_training.py  # Mean Teacher self-training
│   │   └── pseudo_labels.py         # Adaptive-threshold pseudo-label generation
│   ├── evaluation/
│   │   ├── metrics.py               # Event F1, RMSE, calibration, Pearson r
│   │   ├── interpretability.py      # Rule analysis, ablation, clinical metrics
│   │   └── visualization.py         # Training curves, calibration, frame predictions
│   └── utils/
│       └── helpers.py               # Seed, logger, JSON serialisation
├── data/
│   ├── raw/                         # SEP-28k class folders
│   └── processed/                   # Split CSVs, label mappings, pseudo-labels
├── models/checkpoints/              # Saved weights + calibration
├── outputs/
│   ├── visualizations/              # Generated figures
│   └── demos/                       # Single-audio demo JSON reports
├── logs/                            # Timestamped training logs
├── export_for_deployment.py         # Export model for Hugging Face deployment
├── requirements.txt
└── README.md
```

## Dataset

**SEP-28k** (Stuttering Events in Podcasts): ~28 000 three-second clips.
After filtering to 3 target classes + negative sampling:

| Class | Train | Val | Test |
|-------|------:|----:|-----:|
| Interjection | ~2 300 | ~425 | 443 |
| Prolongation | ~1 634 | ~396 | 309 |
| Word repetition | ~1 402 | ~301 | 306 |
| \_\_negative\_\_ (fluent) | ~800 | ~166 | ~162 |
| **Total** | **6 676** | **1 391** | **1 331** |

Splits are **speaker-disjoint** (70 / 15 / 15).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Place SEP-28k class folders under data/raw/
# Then run every cell in the notebook:
jupyter notebook notebooks/main_pipeline.ipynb
```

The notebook runs 12 sequential phases — from dataset discovery through
training, calibration, evaluation, and a single-audio demo.

## Deployment

A Streamlit web application is available for interactive use:

```bash
# 1. Export the trained model for deployment
python export_for_deployment.py

# 2. The exported file is at models/checkpoints/cpu_final_model.pth
#    Upload it to your Hugging Face Space (see deployment guide below)
```

The web app supports audio of **any length** by automatically chunking into
3-second segments, analysing each independently, and merging results across
chunk boundaries. Clinical metrics are computed on the full recording.

**Live demo**: [Hugging Face Spaces](https://huggingface.co/spaces/w2kavindumihiranga/stuttering-detection)

## Evaluation Metrics

### Clip-level (classification)

| Metric | Description |
|--------|-------------|
| Macro F1 | Per-class F1 averaged (primary) |
| Hamming accuracy | 1 − fraction of wrong labels |
| Subset accuracy | Exact-match ratio |
| Sample F1 | Per-sample F1 averaged |

### Event-level (temporal localisation)

| Metric | Description |
|--------|-------------|
| Event-level F1 (IoU ≥ 0.3) | Localisation quality |
| Onset / Offset RMSE (ms) | Boundary precision |

### Interpretability

| Metric | Description |
|--------|-------------|
| Pearson r | Gate weight–rule score correlation |
| Ablation Δ F1 | F1 drop when rules are disabled |

### Clinical

| Metric | Description |
|--------|-------------|
| Events per minute | Stuttering frequency |
| Average event duration (ms) | Mean event length |

## Hardware & Software

| Requirement | Purpose |
|-------------|---------|
| Apple M4 MacBook (or CUDA GPU) | Training (MPS / CUDA acceleration) |
| Python 3.9+ | Primary language |
| PyTorch ≥ 2.0, Transformers ≥ 4.30 | Model and encoder |
| Librosa, SciPy, scikit-learn | Audio processing and evaluation |

Training was performed on Apple Silicon (M4) using the MPS backend.
Total training time: ~15 hours (Stage 1: ~7h, Stage 2: ~7h).

## License

This project is submitted in partial fulfilment of the requirements for the
BSc (Hons) Computer Science degree at the University of Westminster.
