"""
Differentiable Soft-Rule Module — 3-rule version (v2).

Implements three interpretable acoustic rules as differentiable functions,
one per class:
  1. **Energy Burst**       — detects interjections (sudden energy spikes).
  2. **Voicing Continuity** — detects prolongations (sustained low-change over
     multiple consecutive frames, with per-clip normalisation to reduce
     false positives on normal sustained vowels).
  3. **Rhythmic Pattern**   — detects word repetitions (multi-scale periodic
     structure using lags 2-5 to handle variable word lengths).

Rule order matches STUTTER_TYPES: [interjection, prolongation, word_repetition].

Changes from v1
----------------
- Voicing rule: added cumulative low-change duration gate so short sustained
  sounds (normal vowels) don't trigger the rule — only sustained low-change
  spanning several frames activates the rule.  Per-clip normalisation of the
  change rate makes the threshold relative to the clip's own dynamics.
- Rhythm rule: uses multi-scale lags (2, 3, 4, 5) with learnable per-lag
  weights and max-pools across lags to capture variable-length repetitions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableSoftRules(nn.Module):
    """Three differentiable acoustic rules with learnable parameters and temperature."""

    def __init__(self, feature_dim: int = 768, num_rules: int = 3,
                 projection_dim: int = 64):
        super().__init__()
        self.num_rules = num_rules
        self.feature_dim = feature_dim

        # ── Rule 1: Energy burst (interjections) ─────────────────────────
        self.burst_threshold = nn.Parameter(torch.tensor(0.5))
        self.burst_weight = nn.Parameter(torch.tensor(2.0))
        self.burst_temp = nn.Parameter(torch.tensor(1.0))
        self.burst_proj = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        # ── Rule 2: Voicing continuity (prolongations) — duration-aware ──
        self.voicing_threshold = nn.Parameter(torch.tensor(0.3))
        self.voicing_weight = nn.Parameter(torch.tensor(2.0))
        self.voicing_temp = nn.Parameter(torch.tensor(1.0))
        self.voicing_proj = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        # Learnable sustain gate: how many consecutive low-change frames
        # are needed before the rule fires (initialised ~ 5 frames = 100ms)
        self.voicing_sustain_threshold = nn.Parameter(torch.tensor(4.0))
        self.voicing_sustain_temp = nn.Parameter(torch.tensor(1.0))

        # ── Rule 3: Rhythmic pattern (word repetitions) — multi-scale ────
        self.rhythm_threshold = nn.Parameter(torch.tensor(0.4))
        self.rhythm_weight = nn.Parameter(torch.tensor(2.0))
        self.rhythm_temp = nn.Parameter(torch.tensor(1.0))
        self.rhythm_proj = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        # Learnable per-lag weights for lags 2, 3, 4, 5
        self.rhythm_lag_weights = nn.Parameter(torch.tensor([0.5, 1.0, 0.8, 0.5]))
        self.rhythm_lags = [2, 3, 4, 5]

    # ── Individual rules ─────────────────────────────────────────────────
    def energy_burst_rule(self, features: torch.Tensor) -> torch.Tensor:
        """High output on sudden energy spikes relative to local context.

        Per-clip normalisation of burst_ratio ensures the learnable threshold
        is relative to the clip's own energy dynamics, preventing the rule from
        being biased by overall clip loudness.
        """
        B, T, _ = features.shape
        proj = F.relu(self.burst_proj(features))
        energy = torch.mean(proj ** 2, dim=-1)  # (B, T)

        if T > 4:
            energy_pad = F.pad(energy.unsqueeze(1), (2, 2), mode='reflect')
            local_mean = F.avg_pool1d(energy_pad, kernel_size=5, stride=1).squeeze(1)
        else:
            local_mean = energy.mean(dim=1, keepdim=True).expand_as(energy)

        burst_ratio = energy / (local_mean + 1e-8) - 1.0

        # Per-clip normalisation: zero-mean burst ratio so the threshold
        # is relative to the clip's own energy dynamics
        if T > 1:
            clip_mean = burst_ratio.mean(dim=1, keepdim=True)
            burst_ratio = burst_ratio - clip_mean

        temp = self.burst_temp.clamp(min=0.1)
        return torch.sigmoid(self.burst_weight * (burst_ratio - self.burst_threshold) / temp)

    def voicing_continuity_rule(self, features: torch.Tensor) -> torch.Tensor:
        """High output when adjacent frames are similar for a *sustained* duration.

        Per-clip normalisation ensures the threshold is relative to the clip's
        own dynamics.  A causal duration gate requires low change to persist
        for several frames before firing.
        """
        B, T, _ = features.shape
        proj = F.relu(self.voicing_proj(features))

        if T > 1:
            diff = proj[:, 1:, :] - proj[:, :-1, :]
            change_rate = torch.mean(torch.abs(diff), dim=-1)  # (B, T-1)
            change_rate = torch.cat([change_rate, change_rate[:, -1:]], dim=1)
        else:
            change_rate = torch.zeros(B, T, device=features.device)

        # Per-clip normalisation: zero-mean change rate
        if T > 1:
            clip_mean = change_rate.mean(dim=1, keepdim=True)
            change_rate_norm = change_rate - clip_mean
        else:
            change_rate_norm = change_rate

        # Base continuity score
        temp = self.voicing_temp.clamp(min=0.1)
        base_score = torch.sigmoid(
            self.voicing_weight * (self.voicing_threshold - change_rate_norm) / temp
        )

        # Duration gate: causal average of base_score
        if T > 3:
            kernel_size = min(7, T)
            padded = F.pad(base_score.unsqueeze(1), (kernel_size - 1, 0),
                           mode='constant', value=0.0)
            cumul = F.avg_pool1d(padded, kernel_size=kernel_size, stride=1).squeeze(1)
        else:
            cumul = base_score

        sustain_temp = self.voicing_sustain_temp.clamp(min=0.1)
        sustain_gate = torch.sigmoid(
            (cumul - torch.sigmoid(self.voicing_sustain_threshold)) / sustain_temp
        )

        return base_score * sustain_gate

    def rhythmic_pattern_rule(self, features: torch.Tensor) -> torch.Tensor:
        """High output when periodic patterns are detected (word repetitions).

        Uses multiple lags (2-5) with learnable per-lag weights and max-pools
        across lags so the rule fires when *any* periodic structure is detected.
        """
        B, T, _ = features.shape
        proj = F.relu(self.rhythm_proj(features))
        normed = F.normalize(proj, p=2, dim=-1, eps=1e-8)

        # Adjacent similarity (baseline)
        if T > 1:
            adj_sim = torch.sum(normed[:, 1:] * normed[:, :-1], dim=-1)
        else:
            return torch.zeros(B, T, device=features.device)

        # Multi-scale lag similarities
        lag_weights = F.softmax(self.rhythm_lag_weights, dim=0)
        lag_scores = []

        for i, lag in enumerate(self.rhythm_lags):
            if T > lag:
                lag_sim = torch.sum(normed[:, lag:] * normed[:, :-lag], dim=-1)
                min_len = min(lag_sim.shape[1], adj_sim.shape[1])
                score = lag_sim[:, :min_len] - adj_sim[:, :min_len] * 0.5
                pad_len = T - score.shape[1]
                score = F.pad(score, (0, pad_len), value=0.0)
                lag_scores.append(score * lag_weights[i])

        if lag_scores:
            stacked = torch.stack(lag_scores, dim=-1)
            rhythm_score, _ = stacked.max(dim=-1)
        else:
            rhythm_score = torch.zeros(B, T, device=features.device)

        temp = self.rhythm_temp.clamp(min=0.1)
        return torch.sigmoid(self.rhythm_weight * (rhythm_score - self.rhythm_threshold) / temp)

    # ── Forward ──────────────────────────────────────────────────────────
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply all rules.

        Returns rule_scores : (B, T, 3)
        Order: [burst (interjection), voicing (prolongation), rhythm (word_rep)]
        """
        burst = self.energy_burst_rule(features)
        voicing = self.voicing_continuity_rule(features)
        rhythm = self.rhythmic_pattern_rule(features)
        return torch.stack([burst, voicing, rhythm], dim=-1)
