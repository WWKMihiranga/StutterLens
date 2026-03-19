# Interpretable Weakly-Supervised Stuttering Detection

> **BSc (Hons) Computer Science — Final Year Project**
> W.W. Kavindu Mihiranga · University of Westminster / IIT Sri Lanka
> Supervisor: Mr. Rathesan Sivagnanalingam

## Overview

An interpretable, weakly-supervised system for detecting stuttering events that
provides **millisecond-level temporal boundaries** (onset/offset times) and
generates **human-readable explanations** for each detection.  The model is
trained using only coarse **clip-level labels** from the SEP-28k dataset.

## Architecture

```
Raw Audio (3 s, 16 kHz)
    │
    ▼
┌───────────────────────────┐
│  Wav2Vec2 Encoder         │  (top 2 layers fine-tuned)
│  facebook/wav2vec2-base   │
└────────────┬──────────────┘
             │  (B, T, 768)
        ┌────┴────┐
        ▼         ▼
┌──────────┐ ┌───────────────────────────┐
│ BiLSTM   │ │ Differentiable            │
│ Temporal │ │ Soft-Rule Module (v2)     │
│ Head     │ │  1. Energy Burst          │
│ (2-layer,│ │  2. Voicing Continuity    │
│  256-dim)│ │     (duration-gated)      │
│          │ │  3. Rhythmic Pattern      │
│          │ │     (multi-scale lag 2–5) │
└────┬─────┘ └────────────┬──────────────┘
     │                    │
     │   ┌────────────────┘
     ▼   ▼
┌───────────────────────┐
│ Adaptive Gating       │  → per-frame explanations
│ Network               │
│ (sigmoid gates,       │
│  frame-prob input,    │
│  anti-collapse floor) │
└───────────┬───────────┘
            ▼
   Final Logits (B, T, 3)
```

### Key Design Decisions

- **Encoder fine-tuning**: the last 2 of 12 Wav2Vec2 transformer layers are
  unfrozen with a separate, lower learning rate (5e-6) to adapt representations
  to disfluent speech without catastrophic forgetting.
- **3 stutter classes**: interjection, prolongation, and word repetition.
  Block and sound repetition were dropped after empirical evaluation showed
  poor positive/negative separation (block: 0.066) and low F1
  (sound repetition: 0.346) on the 5-class setting.
- **Negative sampling**: 25 % of fluent (NoStutteredWords) clips are included as
  explicit negative examples to improve precision.

## Differentiable Soft-Rule Module (v2)

Three interpretable acoustic rules, each aligned with one stutter class:

| Rule | Target Class | Acoustic Principle | v2 Improvement |
|------|--------------|--------------------|----------------|
| **Energy Burst** | Interjection | Sudden energy spike relative to local context | — |
| **Voicing Continuity** | Prolongation | Low frame-to-frame spectral change | Duration gate: only fires when low change is sustained ≥ 7 frames (~140 ms); per-clip normalisation of change rate |
| **Rhythmic Pattern** | Word Repetition | Periodic self-similarity at a lag | Multi-scale lags (2, 3, 4, 5) with learnable per-lag weights and max-pooling across lags |

Each rule has learnable parameters (thresholds, weights, temperatures) and a
lightweight linear projection, keeping it end-to-end trainable yet
interpretable.

## Adaptive Gating Network

The gating network combines neural and rule outputs with **independent sigmoid
gates** (not softmax) to avoid zero-sum competition.  Key features:

- **Frame-probability input**: `sigmoid(neural_logits)` is fed to the gate
  network (detached) so it can distinguish stutter frames from fluent frames
  and assign context-dependent rule weights.
- **Minimum rule floor**: rule gates are clamped ≥ 0.03 during training to
  prevent early collapse.
- **Entropy annealing**: gate entropy regularisation starts strong (λ = 0.5)
  and linearly decays to 0.15, allowing rules to specialise in later epochs.
- **Residual bypass**: a learnable fraction of rule logits is added directly
  to the output, scaled by the mean rule gate weight.

## Training Pipeline

| Stage | Description | Supervision | Key Details |
|-------|-------------|-------------|-------------|
| **0** | Pre-train soft rules on synthetic stutters | Synthetic frame labels | 5 000 samples, 20 epochs; prolongations ≥ 8 frames, repetitions with variable lag 2–5 |
| **1** | Multi-Instance Learning (MIL) | Clip-level weak labels | 25 epochs, focal BCE + class weights (cap 15), mixed pooling (0.7 max + 0.3 mean), margin separation loss, warmup + cosine LR |
| **2** | Self-training with Mean Teacher | Pseudo frame labels | 15 epochs, EMA 0.999; confidence-weighted boundary loss; pseudo-label threshold at 65th percentile |

### Post-Training Calibration

After Stage 1, per-class **temperature scaling** (Platt scaling) is fitted on
the validation set, followed by per-class **threshold optimisation**.  These
calibration parameters carry through to Stage 2 validation and final test
evaluation.

## Project Structure

```
Code_Base/
├── notebooks/
│   └── main_pipeline.ipynb       # Orchestrator notebook (12 phases)
├── src/
│   ├── config.py                 # Centralised configuration
│   ├── data/
│   │   ├── preprocessor.py       # Audio loading, normalisation & augmentation
│   │   ├── dataset.py            # PyTorch Dataset classes (clip & frame level)
│   │   └── splits.py             # Speaker-disjoint splitting
│   ├── models/
│   │   ├── soft_rules.py         # Differentiable Soft-Rule Module (v2)
│   │   ├── gating.py             # Adaptive Gating Network (frame-prob input)
│   │   ├── temporal_head.py      # BiLSTM Temporal Detection Head
│   │   └── neurosymbolic.py      # Complete NeuroSymbolicStutterDetector
│   ├── training/
│   │   ├── losses.py             # MIL (focal + margin sep.), boundary (confidence-weighted), consistency, gate entropy
│   │   ├── stage0_rule_pretrain.py   # Synthetic data generation & rule pre-training
│   │   ├── stage1_mil.py             # MIL training with warmup + entropy annealing
│   │   ├── stage2_self_training.py   # Mean Teacher self-training
│   │   └── pseudo_labels.py          # Adaptive-threshold pseudo-label generation
│   ├── evaluation/
│   │   ├── metrics.py            # Event F1, RMSE, clip metrics, calibration, Pearson r
│   │   ├── interpretability.py   # Stratified rule analysis, ablation, clinical metrics
│   │   └── visualization.py      # Training curves, calibration plots, frame predictions
│   └── utils/
│       └── helpers.py            # Seed, logger, JSON helpers
├── data/raw/                     # SEP-28k class folders (Interjection/, Prolongation/, WordRep/, NoStutteredWords/)
├── data/processed/               # Split CSVs, label mappings, pseudo-labels
├── models/checkpoints/           # Saved model weights & calibration params
├── outputs/
│   ├── visualizations/           # All generated figures
│   └── demos/                    # Single-audio demo JSON reports
├── logs/                         # Timestamped training logs
├── requirements.txt
└── README.md
```

## Dataset

**SEP-28k** (Stuttering Events in Podcasts): ~28 000 three-second clips with
clip-level labels.  After filtering to the 3 target classes + negative
sampling, the working dataset comprises ~8 600 clips:

| Class | Description | Train | Val | Test |
|-------|-------------|------:|----:|-----:|
| Interjection | Filler words inserted to overcome blocks | ~2 300 | ~425 | ~443 |
| Prolongation | Stretched / sustained sounds | ~1 634 | ~396 | ~309 |
| Word Repetition | Repeated whole words | ~1 402 | ~301 | ~306 |
| \_\_negative\_\_ | Fluent speech (NoStutteredWords, 25 %) | ~800 | ~166 | ~162 |

Splits are **speaker-disjoint** (70 / 15 / 15) to prevent data leakage.

## Quick Start

```bash
pip install -r requirements.txt
```

Place the SEP-28k class folders under `data/raw/`, then run every cell in
`notebooks/main_pipeline.ipynb`.

The notebook is organised into 12 sequential phases:

1. Environment setup & configuration
2. Dataset discovery & exploration
3. Data preprocessing & speaker-disjoint splits
4. Model architecture construction
5. Stage 0 — rule pre-training
6. Stage 1 — MIL training
7. Pseudo-label generation
8. Stage 2 — self-training with Mean Teacher
9. Probability calibration, evaluation & metrics
10. Interpretability analysis (RQ2)
11. Ablation tests (RQ3)
12. Demo & final summary

## Evaluation Metrics

### Clip-Level (Classification)

| Metric | Description |
|--------|-------------|
| **Macro F1** | Per-class F1 averaged (primary metric) |
| **Hamming Accuracy** | 1 − fraction of wrong individual labels |
| **Subset Accuracy** | Exact-match ratio (all labels correct) |
| **Sample F1** | Per-sample F1 averaged (multi-label) |

### Event-Level (Temporal Localisation)

| Metric | Description |
|--------|-------------|
| **Event-level F1** (IoU ≥ 0.3) | Temporal localisation quality |
| **Onset / Offset RMSE** (ms) | Boundary precision |

Event detection uses **median filtering** (window 5) and **event merging**
(gap ≤ 2 frames) for smoother, more clinically relevant boundaries.

### Interpretability (Explanation Fidelity)

| Metric | Description |
|--------|-------------|
| **Pearson's r** | Gate weight – rule score correlation per rule |
| **Ablation Δ F1** | F1 drop when the rule module is disabled |
| **Per-class gate breakdown** | Average neural vs. rule gate weights by stutter type |

### Clinical

| Metric | Description |
|--------|-------------|
| **Events per minute** | Stuttering frequency |
| **Average event duration** (ms) | Mean length of detected events |

## Hardware & Software

| Requirement | Purpose |
|-------------|---------|
| Apple M4 MacBook (or CUDA GPU) | Training — MPS or CUDA acceleration |
| Python 3.9+ | Primary language |
| PyTorch ≥ 2.0, Transformers ≥ 4.30 | Model & encoder |
| Librosa, SciPy, scikit-learn | Audio processing & evaluation |

## License

This project is submitted in partial fulfilment of the requirements for the
BSc (Hons) Computer Science degree at the University of Westminster.
