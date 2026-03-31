import copy
import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from src.training.losses import MILLoss, GateEntropyLoss

# NEW: import calibration & threshold utilities
from src.evaluation.metrics import (
    calibrate_probabilities, optimize_per_class_thresholds
)


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001, mode="max"):
        self.patience, self.min_delta = patience, min_delta
        self.mode = mode
        self.counter = 0
        self.best = float("-inf") if mode == "max" else float("inf")

    def __call__(self, metric):
        if self.mode == "max":
            improved = metric > self.best + self.min_delta
        else:
            improved = metric < self.best - self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def _compute_class_weights(dataset, num_classes: int, cap: float = 10.0) -> tuple:
    
    if hasattr(dataset, 'df') and 'class' in dataset.df.columns:
        counts = np.zeros(num_classes)
        label2idx = dataset.label2idx
        df_stutter = dataset.df[dataset.df['class'] != '__negative__']
        for cls, idx in label2idx.items():
            counts[idx] = (df_stutter['class'] == cls).sum()
    else:
        counts = np.zeros(num_classes)
        for i in range(len(dataset)):
            sample = dataset[i]
            label = sample["clip_label"].numpy()
            counts += label

    counts = np.maximum(counts, 1)
    total_stutter = counts.sum()

    # Inter-class weights (inverse frequency)
    weights = total_stutter / (num_classes * counts)
    weights = np.minimum(weights, cap)

    print(f"Class weights (cap={cap}): {dict(zip(range(num_classes), weights.round(2)))}")
    return (
        torch.tensor(weights, dtype=torch.float32),
        torch.ones(num_classes, dtype=torch.float32),  # placeholder, not used
    )


@torch.no_grad()
def _validate_with_per_class(model, loader, criterion, device, num_classes, class_names,
                              pooling='mixed', mixed_alpha=0.7):
    model.eval()
    total_loss, n = 0.0, 0
    per_class_tp = np.zeros(num_classes)
    per_class_fp = np.zeros(num_classes)
    per_class_total = np.zeros(num_classes)

    for batch in tqdm(loader, desc="Val", leave=False):
        audio = batch["audio"].to(device)
        targets = batch["clip_label"].to(device)

        output = model(audio, return_details=True)
        if isinstance(output, tuple):
            logits, details = output
        else:
            logits = output
            details = None

        loss = criterion(logits, targets)
        total_loss += loss.item()
        n += 1

        # Use same pooling as training loss
        if pooling == "mixed":
            max_logits, _ = torch.max(logits, dim=1)
            mean_logits = torch.mean(logits, dim=1)
            clip_logits = mixed_alpha * max_logits + (1 - mixed_alpha) * mean_logits
        elif pooling == "mean":
            clip_logits = torch.mean(logits, dim=1)
        else:
            clip_logits, _ = torch.max(logits, dim=1)

        preds = (torch.sigmoid(clip_logits) > 0.5).float().cpu().numpy()
        tgts = targets.cpu().numpy()

        for c in range(num_classes):
            pos_mask = tgts[:, c] == 1
            neg_mask = tgts[:, c] == 0
            per_class_total[c] += pos_mask.sum()
            per_class_tp[c] += (preds[pos_mask, c] == 1).sum() if pos_mask.sum() > 0 else 0
            per_class_fp[c] += (preds[neg_mask, c] == 1).sum() if neg_mask.sum() > 0 else 0

    recall = np.divide(per_class_tp, np.maximum(per_class_total, 1))
    precision = np.divide(per_class_tp, np.maximum(per_class_tp + per_class_fp, 1))
    f1 = np.divide(2 * precision * recall, np.maximum(precision + recall, 1e-8))
    avg_loss = total_loss / max(n, 1)
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    recall_str = "  ".join(f"{class_names[i]}={recall[i]:.2f}" for i in range(num_classes))
    return avg_loss, macro_recall, macro_f1, recall_str


def train_stage1(model, train_loader, val_loader, config,
                 num_epochs=None, lambda_gate=0.5):
    
    num_epochs = num_epochs or config.NUM_EPOCHS_STAGE1

    # Compute class weights from training data
    weight_cap = getattr(config, 'CLASS_WEIGHT_CAP', 10.0)
    class_weights, pos_weight = _compute_class_weights(
        train_loader.dataset, config.NUM_CLASSES, cap=weight_cap
    )

    # Use mixed pooling for better class-wise accuracy
    pooling = getattr(config, 'MIL_POOLING', 'mixed')
    mixed_alpha = getattr(config, 'MIL_MIXED_ALPHA', 0.7)
    label_smoothing = getattr(config, 'LABEL_SMOOTHING', 0.0)
    focal_gamma = getattr(config, 'FOCAL_GAMMA', 2.0)
    criterion = MILLoss(
        class_weights=class_weights, focal_gamma=focal_gamma,
        pooling=pooling, mixed_alpha=mixed_alpha,
        label_smoothing=label_smoothing,
        pos_weight=pos_weight,
    )
    gate_loss = GateEntropyLoss(min_rule_weight=0.05)

    # Separate parameter groups: encoder (if unfrozen), rules+gating, other
    encoder_params, rule_params, other_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "encoder" in name:
            encoder_params.append(p)
        elif "soft_rules" in name or "gating" in name:
            rule_params.append(p)
        else:
            other_params.append(p)

    encoder_lr = getattr(config, 'ENCODER_LR', config.LEARNING_RATE * 0.1)
    param_groups = [
        {"params": other_params, "lr": config.LEARNING_RATE},
        {"params": rule_params, "lr": config.LEARNING_RATE * 5},  # 5x LR for rules+gating
    ]
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": encoder_lr})
        print(f"  Encoder params: {sum(p.numel() for p in encoder_params):,} "
              f"(lr={encoder_lr})")

    optimizer = AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)

    # Warmup + cosine annealing scheduler
    warmup_steps = getattr(config, 'WARMUP_STEPS', 200)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    early_stop = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, mode="max")

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_f1 = 0.0
    best_state = None

    class_names = config.STUTTER_TYPES

    # Gate entropy annealing: start strong, decay to allow specialisation
    gate_lambda_start = getattr(config, 'GATE_ENTROPY_LAMBDA_START', lambda_gate)
    gate_lambda_end = getattr(config, 'GATE_ENTROPY_LAMBDA_END', lambda_gate * 0.3)

    print(f"\n{'='*60}")
    print(f"STAGE 1 — MIL TRAINING  ({num_epochs} epochs)")
    print(f"  Focal gamma: {criterion.focal_gamma}")
    print(f"  Gate entropy lambda: {gate_lambda_start} → {gate_lambda_end} (annealed)")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"{'='*60}")

    step_count = 0  # global step counter for warmup scheduler

    for epoch in range(num_epochs):
        # Anneal gate entropy lambda linearly over epochs
        progress = epoch / max(num_epochs - 1, 1)
        current_gate_lambda = gate_lambda_start + (gate_lambda_end - gate_lambda_start) * progress

        # Train
        model.train()
        total_loss, n = 0.0, 0
        for batch in tqdm(train_loader, desc="Train", leave=False):
            audio = batch["audio"].to(config.DEVICE)
            targets = batch["clip_label"].to(config.DEVICE)

            logits, details = model(audio, return_details=True)
            mil_loss = criterion(logits, targets)

            # Gate entropy regularisation (annealed)
            gw = details["gate_weights"]
            g_loss = gate_loss(gw)

            loss = mil_loss + current_gate_lambda * g_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Step the warmup scheduler per-batch during warmup phase
            step_count += 1
            if step_count <= warmup_steps:
                scheduler.step()

            total_loss += loss.item()
            n += 1

        t_loss = total_loss / max(n, 1)

        # Validate
        v_loss, macro_recall, macro_f1, recall_str = _validate_with_per_class(
            model, val_loader, criterion, config.DEVICE,
            config.NUM_CLASSES, class_names,
            pooling=pooling, mixed_alpha=mixed_alpha,
        )
        # Step cosine scheduler per-epoch (after warmup)
        if step_count > warmup_steps:
            scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_accuracy"].append(macro_f1)

        improved = ""
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            best_state = copy.deepcopy(model.state_dict())
            improved = "  ★"

        # Report gate weights
        with torch.no_grad():
            sample = next(iter(val_loader))
            _, det = model(sample["audio"][:1].to(config.DEVICE),
                           return_details=True)
            gw_avg = det["gate_weights"].squeeze(0).mean(0)
            gate_labels = ["N", "Br", "Vc", "Rh"][:len(gw_avg)]
            gate_str = "gates=[" + " ".join(
                f"{gl}:{gw_avg[i]:.2f}" for i, gl in enumerate(gate_labels)
            ) + "]"

        print(f"Ep {epoch+1:2d}/{num_epochs}  loss={t_loss:.4f}  "
              f"val_loss={v_loss:.4f}  F1={macro_f1:.3f}  recall={macro_recall:.3f}  "
              f"{gate_str}  [{recall_str}]{improved}")

        if early_stop(macro_f1):
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model (val_macro_F1={best_val_f1:.4f})")

    return model, history


# Post-training calibration & threshold optimisation
@torch.no_grad()
def collect_val_predictions(model, val_loader, config):
    model.eval()
    all_preds, all_targets = [], []
    pooling = getattr(config, 'MIL_POOLING', 'mixed')
    mixed_alpha = getattr(config, 'MIL_MIXED_ALPHA', 0.7)

    for batch in val_loader:
        audio = batch["audio"].to(config.DEVICE)
        targets = batch["clip_label"]
        logits = model(audio)  # (B, T, C)

        if pooling == "mixed":
            max_logits, _ = torch.max(logits, dim=1)
            mean_logits = torch.mean(logits, dim=1)
            clip_logits = mixed_alpha * max_logits + (1 - mixed_alpha) * mean_logits
        elif pooling == "mean":
            clip_logits = torch.mean(logits, dim=1)
        else:
            clip_logits, _ = torch.max(logits, dim=1)

        probs = torch.sigmoid(clip_logits).cpu().numpy()
        all_preds.append(probs)
        all_targets.append(targets.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def calibrate_and_optimize(model, val_loader, config):
    from src.evaluation.metrics import apply_calibration

    class_names = config.STUTTER_TYPES

    # Step 1: Collect predictions
    val_preds, val_targets = collect_val_predictions(model, val_loader, config)

    # Step 2: Calibrate probabilities
    cal_result = calibrate_probabilities(val_preds, val_targets, class_names)
    temperatures = cal_result['temperatures']
    val_preds_cal = apply_calibration(val_preds, temperatures, class_names)

    # Step 3: Optimize thresholds on CALIBRATED probabilities
    opt_result = optimize_per_class_thresholds(
        val_preds_cal, val_targets, class_names
    )
    thresholds = opt_result['thresholds']

    # Report
    print(f"\n{'='*60}")
    print(f"  POST-TRAINING CALIBRATION & THRESHOLD OPTIMIZATION")
    print(f"{'='*60}")
    for cls in class_names:
        t = temperatures[cls]
        th = thresholds[cls]
        diag = cal_result['diagnostics'][cls]
        print(f"  {cls:20s}  temp={t:.2f}  thresh={th:.2f}  "
              f"pos_prob={diag['mean_pos_prob']:.3f}  "
              f"neg_prob={diag['mean_neg_prob']:.3f}  "
              f"sep={diag.get('separation', 0):.3f}")

    # Also report improvement
    from src.evaluation.metrics import clip_level_metrics
    raw_metrics = clip_level_metrics(val_preds, val_targets, class_names)
    cal_metrics = clip_level_metrics(val_preds_cal, val_targets, class_names, threshold=0.5)
    print(f"\n  Val F1 (raw, fixed 0.5):         {raw_metrics['f1']:.4f}")
    print(f"  Val F1 (calibrated, fixed 0.5):   {cal_metrics['f1']:.4f}")
    print(f"  Val F1 (calibrated, optimised):   {opt_result['overall']['f1']:.4f}")

    return {
        'temperatures': temperatures,
        'thresholds': thresholds,
        'val_preds': val_preds,
        'val_preds_cal': val_preds_cal,
        'val_targets': val_targets,
        'diagnostics': cal_result['diagnostics'],
    }
