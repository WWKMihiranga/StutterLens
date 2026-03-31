import copy
import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from src.training.losses import Stage2CombinedLoss, GateEntropyLoss


class MeanTeacher:
    def __init__(self, student_model, ema_decay=0.99):
        self.student = student_model
        self.ema_decay = ema_decay
        self.teacher = copy.deepcopy(student_model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self):
        for tp, sp in zip(self.teacher.parameters(), self.student.parameters()):
            tp.data.mul_(self.ema_decay).add_(sp.data, alpha=1 - self.ema_decay)

    @torch.no_grad()
    def teacher_forward(self, audio, return_details=False):
        self.teacher.eval()
        return self.teacher(audio, return_details=return_details)


def train_stage2(mean_teacher, train_loader, val_loader, config,
                 num_epochs=None, lambda_gate=0.4, calibration_info=None):
    num_epochs = num_epochs or config.NUM_EPOCHS_STAGE2
    student = mean_teacher.student

    # Compute class weights from the training data (same as Stage 1)
    from src.training.stage1_mil import _compute_class_weights
    weight_cap = getattr(config, 'CLASS_WEIGHT_CAP', 10.0)
    class_weights, pos_weight = _compute_class_weights(
        train_loader.dataset.clip_dataset if hasattr(train_loader.dataset, 'clip_dataset')
        else train_loader.dataset,
        config.NUM_CLASSES, cap=weight_cap,
    )

    pooling = getattr(config, 'MIL_POOLING', 'mixed')
    mixed_alpha = getattr(config, 'MIL_MIXED_ALPHA', 0.7)
    focal_gamma = getattr(config, 'FOCAL_GAMMA', 2.0)

    criterion = Stage2CombinedLoss(
        alpha=config.BOUNDARY_LOSS_ALPHA,
        beta=config.BOUNDARY_LOSS_BETA,
        lam_sup=config.LAMBDA_SUPERVISED,
        lam_cons=config.LAMBDA_CONSISTENCY,
        lam_weak=0.5,
        temperature=config.CONSISTENCY_TEMPERATURE,
        class_weights=class_weights,
        pos_weight=pos_weight,
        focal_gamma=focal_gamma,
        pooling=pooling,
        mixed_alpha=mixed_alpha,
    )
    gate_loss = GateEntropyLoss(min_rule_weight=0.05)

    # Separate parameter groups like Stage 1 to preserve encoder fine-tuning
    encoder_params, rule_params, other_params = [], [], []
    for name, p in student.named_parameters():
        if not p.requires_grad:
            continue
        if "encoder" in name:
            encoder_params.append(p)
        elif "soft_rules" in name or "gating" in name:
            rule_params.append(p)
        else:
            other_params.append(p)

    base_lr = config.STAGE2_LR
    encoder_lr = base_lr * 0.25  # very conservative for already fine-tuned encoder
    param_groups = [
        {"params": other_params, "lr": base_lr},
        {"params": rule_params, "lr": base_lr * 3},  # rules/gating need higher LR
    ]
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": encoder_lr})
        print(f"  Stage 2 encoder LR: {encoder_lr}")

    optimizer = AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_f1 = 0.0
    best_state = None

    print(f"\n{'='*60}")
    print(f"STAGE 2 — SELF-TRAINING WITH MEAN TEACHER  ({num_epochs} epochs)")
    print(f"  Lambda consistency: {config.LAMBDA_CONSISTENCY}")
    print(f"  Lambda weak: {criterion.lam_weak}")
    print(f"  EMA decay: {mean_teacher.ema_decay}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        student.train()
        epoch_loss, n = 0.0, 0

        for batch in tqdm(train_loader, desc=f"S2 Ep {epoch+1}", leave=False):
            audio = batch["audio"].to(config.DEVICE)
            clip_labels = batch["clip_label"].to(config.DEVICE)
            frame_labels = batch["frame_label"].to(config.DEVICE)
            has_pseudo = batch["has_pseudo_label"]  # (B,) boolean

            s_logits, s_details = student(audio, return_details=True)
            t_logits = mean_teacher.teacher_forward(audio)

            min_t = min(s_logits.shape[1], frame_labels.shape[1], t_logits.shape[1])

            # Convert has_pseudo to a boolean mask on the correct device
            has_pseudo_mask = has_pseudo.bool().to(config.DEVICE)

            loss, _ = criterion(
                s_logits[:, :min_t], t_logits[:, :min_t],
                frame_labels[:, :min_t], clip_labels,
                has_pseudo_mask=has_pseudo_mask,
            )

            # Gate entropy reg
            g_loss = gate_loss(s_details["gate_weights"])
            loss = loss + lambda_gate * g_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            mean_teacher.update_teacher()

            epoch_loss += loss.item()
            n += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n, 1)

        val_f1, val_loss = _validate_stage2(student, val_loader, config, calibration_info)
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_f1)

        improved = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(student.state_dict())
            improved = "  ★"

        print(f"Ep {epoch+1:2d}/{num_epochs}  loss={avg_loss:.4f}  "
              f"val_F1={val_f1:.4f}{improved}")

    if best_state is not None:
        student.load_state_dict(best_state)
        mean_teacher.update_teacher()
        print(f"Loaded best student (val_F1={best_f1:.4f})")

    return mean_teacher, history


@torch.no_grad()
def _validate_stage2(model, loader, config, calibration_info=None):
    from src.training.losses import MILLoss
    model.eval()
    per_class_tp = np.zeros(config.NUM_CLASSES)
    per_class_fp = np.zeros(config.NUM_CLASSES)
    per_class_total = np.zeros(config.NUM_CLASSES)

    # Use same MIL settings as training for consistent loss scale
    pooling = getattr(config, 'MIL_POOLING', 'mixed')
    mixed_alpha = getattr(config, 'MIL_MIXED_ALPHA', 0.7)
    focal_gamma = getattr(config, 'FOCAL_GAMMA', 2.0)
    val_criterion = MILLoss(
        focal_gamma=focal_gamma, pooling=pooling, mixed_alpha=mixed_alpha,
    )
    total_loss, n_batches = 0.0, 0

    # Resolve thresholds
    thresholds = np.full(config.NUM_CLASSES, 0.5)
    temperatures = None
    if calibration_info is not None:
        class_names = config.STUTTER_TYPES
        temperatures = calibration_info.get('temperatures')
        thresh_dict = calibration_info.get('thresholds', {})
        for c, name in enumerate(class_names):
            thresholds[c] = thresh_dict.get(name, 0.5)

    for batch in loader:
        audio = batch["audio"].to(config.DEVICE)
        targets = batch["clip_label"]
        logits = model(audio)

        # Compute validation loss with same settings as training
        val_loss_batch = val_criterion(logits, targets.to(config.DEVICE))
        total_loss += val_loss_batch.item()
        n_batches += 1

        # Use same pooling as training
        if pooling == "mixed":
            max_logits, _ = torch.max(logits, dim=1)
            mean_logits = torch.mean(logits, dim=1)
            clip_logits = mixed_alpha * max_logits + (1 - mixed_alpha) * mean_logits
        elif pooling == "mean":
            clip_logits = torch.mean(logits, dim=1)
        else:
            clip_logits, _ = torch.max(logits, dim=1)

        probs = torch.sigmoid(clip_logits).cpu().numpy()

        # Apply calibration if available
        if temperatures is not None:
            from src.evaluation.metrics import apply_calibration
            probs = apply_calibration(probs, temperatures, config.STUTTER_TYPES)

        # Per-class thresholding
        preds = np.zeros_like(probs)
        for c in range(config.NUM_CLASSES):
            preds[:, c] = (probs[:, c] > thresholds[c]).astype(float)

        tgts = targets.numpy()

        for c in range(config.NUM_CLASSES):
            pos_mask = tgts[:, c] == 1
            neg_mask = tgts[:, c] == 0
            per_class_total[c] += pos_mask.sum()
            if pos_mask.sum() > 0:
                per_class_tp[c] += (preds[pos_mask, c] == 1).sum()
            if neg_mask.sum() > 0:
                per_class_fp[c] += (preds[neg_mask, c] == 1).sum()

    recall = np.divide(per_class_tp, np.maximum(per_class_total, 1))
    precision = np.divide(per_class_tp, np.maximum(per_class_tp + per_class_fp, 1))
    f1 = np.divide(2 * precision * recall, np.maximum(precision + recall, 1e-8))
    avg_val_loss = total_loss / max(n_batches, 1)
    return float(f1.mean()), avg_val_loss
