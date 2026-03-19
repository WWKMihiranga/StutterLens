"""
Adaptive Gating Network — v3: anti-collapse sigmoid gating.

Key design changes vs. softmax gating:
  1. Uses **sigmoid gates** (not softmax) — each rule gate is independent.
     This prevents the zero-sum competition where the neural path "steals"
     all probability mass from rules.
  2. **Minimum rule floor**: each rule gate is clamped to >= min_gate during
     training, ensuring rules always receive gradient signal.
  3. **Stronger residual bypass**: rule logits are added directly to the
     output with a learnable mixing weight, so even if gates are low the
     rules still contribute to the loss landscape.
  4. **Rule score normalisation**: LayerNorm on projected rule features
     prevents vanishing activations from drowning out rule signals.
  5. Gate initialisation biases rules to start at ~0.15 each.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGatingNetwork(nn.Module):

    def __init__(self, feature_dim: int = 512, num_rules: int = 3,
                 num_classes: int = 3, gate_hidden: int = 128,
                 residual_rule_weight: float = 0.15,
                 min_gate: float = 0.03):
        super().__init__()
        self.num_rules = num_rules
        self.num_classes = num_classes
        self.min_gate = min_gate

        # Learnable residual weight (starts at residual_rule_weight, trainable)
        self.residual_rule_weight = nn.Parameter(
            torch.tensor(residual_rule_weight))

        # ── Gate network ─────────────────────────────────────────────────
        # Project neural features down to balance with rule scores
        self.neural_proj = nn.Sequential(
            nn.Linear(feature_dim, num_rules * 4),
            nn.LayerNorm(num_rules * 4),
            nn.ReLU(),
        )

        # gate_in includes: projected neural feats + rule scores + frame probs
        # The frame-probability signal lets the gate network distinguish
        # stutter frames (high prob) from fluent frames (low prob) and assign
        # different rule weights accordingly — addressing the static-gate issue.
        gate_in = num_rules * 4 + num_rules + num_classes  # +C for frame probs
        # Separate gate heads: one for neural, one per rule
        self.neural_gate_head = nn.Sequential(
            nn.Linear(gate_in, gate_hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(gate_hidden, 1),
        )
        self.rule_gate_head = nn.Sequential(
            nn.Linear(gate_in, gate_hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(gate_hidden, num_rules),
        )

        # Bias initialisation: neural starts moderate, rules start meaningful
        with torch.no_grad():
            # Neural gate: sigmoid(0.5) ≈ 0.62
            self.neural_gate_head[-1].bias.fill_(0.5)
            # Rule gates: sigmoid(-0.2) ≈ 0.45 each — start high to prevent
            # early collapse, will be regularised down if not useful
            self.rule_gate_head[-1].bias.fill_(-0.2)

        # ── Per-rule classifiers (deeper for expressiveness) ─────────────
        self.rule_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 32),
                nn.LayerNorm(32),
                nn.ReLU(),
                nn.Linear(32, num_classes),
            )
            for _ in range(num_rules)
        ])

    def forward(self, neural_features, rule_scores, neural_logits):
        """
        neural_features : (B, T, D)
        rule_scores     : (B, T, R)
        neural_logits   : (B, T, C)
        Returns: combined_logits (B, T, C), gate_weights (B, T, R+1)
        """
        # Project neural features down to balance with rule scores
        neural_proj = self.neural_proj(neural_features)  # (B, T, R*4)
        # Frame-probability signal: sigmoid of neural logits provides a
        # frame-level "stutter confidence" that makes gate behaviour
        # context-dependent rather than static.
        frame_probs = torch.sigmoid(neural_logits).detach()  # (B, T, C) — detached
        gate_input = torch.cat([neural_proj, rule_scores, frame_probs], dim=-1)

        # Independent sigmoid gates (not softmax — avoids zero-sum collapse)
        neural_gate = torch.sigmoid(self.neural_gate_head(gate_input))  # (B,T,1)
        rule_gates = torch.sigmoid(self.rule_gate_head(gate_input))     # (B,T,R)

        # Enforce minimum rule gate during training to keep rules alive
        if self.training and self.min_gate > 0:
            rule_gates = torch.clamp(rule_gates, min=self.min_gate)

        # Normalise gates to sum to 1 (for interpretability)
        all_gates = torch.cat([neural_gate, rule_gates], dim=-1)  # (B,T,R+1)
        gate_sum = all_gates.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        gate_weights = all_gates / gate_sum  # (B, T, R+1)

        # Gated combination
        combined = gate_weights[:, :, 0:1] * neural_logits

        rule_logits_list = []
        for i, clf in enumerate(self.rule_classifiers):
            r_logit = clf(rule_scores[:, :, i:i+1])  # (B, T, C)
            combined = combined + gate_weights[:, :, i+1:i+2] * r_logit
            rule_logits_list.append(r_logit)

        # Residual bypass: add a trainable fraction of rule logits directly.
        # Scale by the mean rule gate weight so the residual contribution is
        # proportional to how much the gating network trusts rules for this frame.
        # When gates are low (fluent speech), the residual shrinks to near zero.
        if rule_logits_list:
            rule_avg = torch.stack(rule_logits_list, dim=0).mean(dim=0)
            alpha = torch.sigmoid(self.residual_rule_weight)  # bound to (0,1)
            rule_gate_mean = gate_weights[:, :, 1:].mean(dim=-1, keepdim=True)  # (B,T,1)
            combined = combined + alpha * rule_gate_mean * rule_avg

        return combined, gate_weights
