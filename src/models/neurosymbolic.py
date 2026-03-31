import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

from src.models.temporal_head import TemporalDetectionHead
from src.models.soft_rules import DifferentiableSoftRules
from src.models.gating import AdaptiveGatingNetwork
from typing import Optional


class NeuroSymbolicStutterDetector(nn.Module):

    def __init__(self, config, wav2vec2_model: Optional[Wav2Vec2Model] = None,
                 freeze_encoder: bool = True, use_rules: bool = True):
        super().__init__()
        self.config = config
        self.use_rules = use_rules

        # 1. Encoder
        if wav2vec2_model is not None:
            self.encoder = wav2vec2_model
        else:
            self.encoder = Wav2Vec2Model.from_pretrained(config.WAV2VEC2_MODEL)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

            # Optionally unfreeze the last N transformer layers for fine-tuning
            unfreeze_layers = getattr(config, 'UNFREEZE_ENCODER_LAYERS', 0)
            if unfreeze_layers > 0 and hasattr(self.encoder, 'encoder') and hasattr(self.encoder.encoder, 'layers'):
                total_layers = len(self.encoder.encoder.layers)
                for layer in self.encoder.encoder.layers[total_layers - unfreeze_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True
                print(f"  Unfroze last {unfreeze_layers} of {total_layers} encoder layers")

        # 2. Temporal head
        self.temporal_head = TemporalDetectionHead(
            input_dim=config.HIDDEN_DIM,
            hidden_dim=config.TEMPORAL_HIDDEN_DIM,
            num_layers=config.TEMPORAL_NUM_LAYERS,
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT,
        )

        # 3. Soft rules
        self.soft_rules = DifferentiableSoftRules(
            feature_dim=config.HIDDEN_DIM,
            num_rules=config.NUM_RULES,
            projection_dim=config.RULE_PROJECTION_DIM,
        )

        # 4. Gating network
        self.gating_network = AdaptiveGatingNetwork(
            feature_dim=config.TEMPORAL_HIDDEN_DIM * 2,  # BiLSTM output
            num_rules=config.NUM_RULES,
            num_classes=config.NUM_CLASSES,
            gate_hidden=config.GATE_HIDDEN_DIM,
        )

        # Cache whether any encoder params require grad (avoids scanning ~94M
        # params on every forward pass)
        self._encoder_has_grad = any(
            p.requires_grad for p in self.encoder.parameters()
        )

    # Forward
    def forward(self, audio_input: torch.Tensor, return_details: bool = False):

        # 1 — Wav2Vec2 features
        if not self._encoder_has_grad:
            with torch.no_grad():
                features = self.encoder(audio_input).last_hidden_state
        else:
            features = self.encoder(audio_input).last_hidden_state

        # 2 — Neural logits
        neural_logits, temporal_feats = self.temporal_head(features)

        # 3 — Rule scores
        rule_scores = self.soft_rules(features)

        # 4 — Gated combination
        if self.use_rules:
            final_logits, gate_weights = self.gating_network(
                temporal_feats, rule_scores, neural_logits
            )
        else:
            # Ablation mode — rules disabled
            final_logits = neural_logits
            gate_weights = torch.zeros(
                *neural_logits.shape[:2], self.config.NUM_RULES + 1,
                device=neural_logits.device,
            )

        if return_details:
            details = {
                "wav2vec_features": features.detach(),
                "neural_logits": neural_logits.detach(),
                "rule_scores": rule_scores.detach(),
                "gate_weights": gate_weights.detach(),
                "temporal_features": temporal_feats.detach(),
            }
            return final_logits, details
        return final_logits

    # Utilities
    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
