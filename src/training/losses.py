import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# Stage 0 — Rule pre-training
class RulePretrainingLoss(nn.Module):
    def forward(self, rule_scores, rule_targets):
        return F.binary_cross_entropy(rule_scores, rule_targets, reduction="mean")


# Stage 1 — Multi-Instance Learning (CLASS-WEIGHTED FOCAL BCE)
class MILLoss(nn.Module):

    def __init__(self, class_weights: Optional[torch.Tensor] = None,
                 focal_gamma: float = 2.0,
                 pooling: str = "mixed", mixed_alpha: float = 0.7,
                 label_smoothing: float = 0.0,
                 pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.pooling = pooling
        self.mixed_alpha = mixed_alpha
        self.label_smoothing = label_smoothing
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(1),
        )
        # pos_weight accepted for backward compatibility but NOT used
        # in the loss computation — this is the key change that fixes
        # the recall saturation problem.

    def forward(self, logits, targets):
        """logits: (B, T, C),  targets: (B, C)"""
        if self.pooling == "max":
            clip_logits, _ = torch.max(logits, dim=1)  # (B, C)
        elif self.pooling == "mean":
            clip_logits = torch.mean(logits, dim=1)     # (B, C)
        elif self.pooling == "mixed":
            max_logits, _ = torch.max(logits, dim=1)
            mean_logits = torch.mean(logits, dim=1)
            clip_logits = self.mixed_alpha * max_logits + (1 - self.mixed_alpha) * mean_logits
        elif self.pooling == "attention":
            attn_weights = torch.softmax(logits.mean(dim=-1, keepdim=True), dim=1)
            clip_logits = (logits * attn_weights).sum(dim=1)
        else:
            clip_logits, _ = torch.max(logits, dim=1)

        # Apply label smoothing
        smooth_targets = targets
        if self.label_smoothing > 0:
            smooth_targets = targets * (1 - self.label_smoothing) + self.label_smoothing / 2

        # Standard BCE without pos_weight — balanced precision/recall
        bce = F.binary_cross_entropy_with_logits(
            clip_logits, smooth_targets, reduction="none"
        )  # (B, C)

        # Symmetric focal modulation — same gamma for positives and negatives.
        # Down-weights easy examples equally on both sides.
        probs = torch.sigmoid(clip_logits)
        is_positive = (targets > 0.5).float()
        pt = is_positive * probs + (1.0 - is_positive) * (1.0 - probs)
        focal = (1.0 - pt) ** self.focal_gamma

        # Class weights (inter-class balancing)
        cw = self.class_weights.to(bce.device)
        if cw.dim() == 1 and cw.shape[0] > 1:
            cw = cw.unsqueeze(0)  # (1, C)

        focal_loss = (focal * cw * bce).mean()

        # Margin separation penalty: encourage positive probabilities to be
        # higher than negative probabilities by at least `target_margin`.
        # This directly addresses the poor pos/neg separation issue.
        target_margin = 0.3
        sep_loss = torch.tensor(0.0, device=logits.device)
        num_classes = probs.shape[1]
        for c in range(num_classes):
            pos_mask = targets[:, c] > 0.5
            neg_mask = targets[:, c] < 0.5
            if pos_mask.any() and neg_mask.any():
                mean_pos = probs[pos_mask, c].mean()
                mean_neg = probs[neg_mask, c].mean()
                gap = mean_pos - mean_neg
                sep_loss = sep_loss + F.relu(target_margin - gap)
        sep_loss = sep_loss / max(num_classes, 1) * 0.3  # moderate weight

        return focal_loss + sep_loss


# Gate entropy regulariser — PREVENTS GATING COLLAPSE (v2)
class GateEntropyLoss(nn.Module):

    def __init__(self, min_rule_weight: float = 0.03):
        super().__init__()
        self.min_rule_weight = min_rule_weight

    def forward(self, gate_weights):
        """gate_weights: (B, T, R+1)"""
        eps = 1e-8
        entropy = -(gate_weights * torch.log(gate_weights + eps)).sum(dim=-1)
        max_ent = torch.log(torch.tensor(float(gate_weights.shape[-1]),
                                         device=gate_weights.device))
        ent_loss = (max_ent - entropy).mean()

        # Direct floor on rule gates (gentle — just prevent complete death)
        rule_gates = gate_weights[:, :, 1:]  # skip neural column
        shortfall = F.relu(self.min_rule_weight - rule_gates)
        floor_penalty = shortfall.mean() * 10.0

        # Rule total penalty: sum of all rule gates should be >= 0.10
        rule_total = rule_gates.sum(dim=-1)  # (B, T)
        total_shortfall = F.relu(0.10 - rule_total).mean() * 5.0

        # Diversity penalty: encourage different rules to fire differently
        rule_var = rule_gates.var(dim=-1).mean()
        diversity_penalty = F.relu(0.001 - rule_var) * 3.0

        return ent_loss + floor_penalty + total_shortfall + diversity_penalty


# Stage 2 — Boundary regression + consistency + weak MIL
class BoundaryRegressionLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        self.alpha, self.beta = alpha, beta

    def forward(self, frame_logits, frame_targets, has_pseudo_mask=None):
        if has_pseudo_mask is not None and has_pseudo_mask.any():
            # Only compute supervised loss on samples with real pseudo-labels
            masked_logits = frame_logits[has_pseudo_mask]
            masked_targets = frame_targets[has_pseudo_mask]
            if masked_logits.numel() == 0:
                bce = torch.tensor(0.0, device=frame_logits.device)
            else:
                # Confidence weighting: pseudo-labels near 0 or 1 are more
                # reliable than those near 0.5.  Weight each frame's loss by
                # how confident the pseudo-label is (distance from 0.5).
                confidence = (2.0 * torch.abs(masked_targets - 0.5)).clamp(min=0.1)
                per_frame_bce = F.binary_cross_entropy_with_logits(
                    masked_logits, masked_targets, reduction="none")
                bce = (per_frame_bce * confidence).mean()
        elif has_pseudo_mask is not None and not has_pseudo_mask.any():
            bce = torch.tensor(0.0, device=frame_logits.device)
        else:
            bce = F.binary_cross_entropy_with_logits(
                frame_logits, frame_targets, reduction="mean")

        if frame_logits.shape[1] > 1 and frame_logits.shape[2] >= 3:
            smooth_classes = [1, 2]  # prolongation, word_repetition
            smooth_logits = frame_logits[:, :, smooth_classes]
            smooth = torch.abs(smooth_logits[:, 1:] - smooth_logits[:, :-1]).mean()
        elif frame_logits.shape[1] > 1:
            smooth = torch.abs(frame_logits[:, 1:] - frame_logits[:, :-1]).mean()
        else:
            smooth = torch.tensor(0.0, device=frame_logits.device)
        return self.alpha * bce + self.beta * smooth, {
            "frame_bce": bce.item(), "smoothness": smooth.item()
        }


class ConsistencyLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        s = torch.sigmoid(student_logits / self.temperature)
        t = torch.sigmoid(teacher_logits / self.temperature)
        return F.mse_loss(s, t)


class Stage2CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, lam_sup=1.0, lam_cons=0.3,
                 lam_weak=0.5, temperature=1.0,
                 class_weights=None, pos_weight=None, focal_gamma=2.0,
                 pooling="mixed", mixed_alpha=0.7):
        super().__init__()
        self.boundary = BoundaryRegressionLoss(alpha, beta)
        self.consistency = ConsistencyLoss(temperature)
        self.mil = MILLoss(
            class_weights=class_weights, pos_weight=pos_weight,
            focal_gamma=focal_gamma, pooling=pooling, mixed_alpha=mixed_alpha,
        )
        self.lam_sup = lam_sup
        self.lam_cons = lam_cons
        self.lam_weak = lam_weak

    def forward(self, student_logits, teacher_logits, frame_targets, clip_targets,
                has_pseudo_mask=None):
        sup, sup_d = self.boundary(student_logits, frame_targets, has_pseudo_mask)
        cons = self.consistency(student_logits, teacher_logits)
        weak = self.mil(student_logits, clip_targets)
        total = self.lam_sup * sup + self.lam_cons * cons + self.lam_weak * weak
        return total, {"supervised": sup.item(), "consistency": cons.item(),
                       "weak": weak.item(), **sup_d}
