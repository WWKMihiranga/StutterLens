import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm.auto import tqdm


class SyntheticStutterDataset(Dataset):

    def __init__(self, num_samples=3000, seq_len=50, feature_dim=768):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.feature_dim = feature_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx * 7 + 13)

        # Generate temporally smooth base features (random walk)
        features = np.zeros((self.seq_len, self.feature_dim), dtype=np.float32)
        features[0] = rng.randn(self.feature_dim).astype(np.float32) * 0.5
        for t in range(1, self.seq_len):
            features[t] = 0.85 * features[t - 1] + 0.15 * rng.randn(
                self.feature_dim
            ).astype(np.float32)

        targets = np.zeros((self.seq_len, 3), dtype=np.float32)

        # Choose event type: cycle through 4 types (0=fluent, 1-3=stutter)
        event_type = idx % 4  # deterministic cycling for balance
        onset = rng.randint(8, self.seq_len - 18)

        if event_type == 1:
            # Interjection: sudden energy burst (spike in magnitude)
            duration = rng.randint(4, 10)
            offset = min(onset + duration, self.seq_len - 1)
            for t in range(onset, offset):
                features[t] = features[t] * 3.0 + rng.randn(
                    self.feature_dim
                ).astype(np.float32) * 0.5
            targets[onset:offset, 0] = 1.0  # burst rule

        elif event_type == 2:
            # Prolongation: very low frame-to-frame change, LONGER duration
            # (min 8 frames ≈ 160ms) so the duration-gated voicing rule learns
            # to require sustained continuity.
            duration = rng.randint(8, 18)
            offset = min(onset + duration, self.seq_len - 1)
            base = features[onset - 1].copy()
            for t in range(onset, offset):
                features[t] = base + rng.randn(self.feature_dim).astype(
                    np.float32
                ) * 0.01
            targets[onset:offset, 1] = 1.0  # voicing rule

        elif event_type == 3:
            # Word repetition: periodic pattern with VARIABLE lag (2-5)
            # to match multi-scale detection in the improved rhythm rule.
            period = rng.randint(2, 6)  # was fixed at 3
            duration = rng.randint(max(6, period * 2), 15)
            offset = min(onset + duration, self.seq_len - 1)
            pattern = features[onset - 1].copy()
            for t in range(onset, offset):
                cycle_pos = (t - onset) % period
                if cycle_pos == 0:
                    features[t] = pattern + rng.randn(
                        self.feature_dim
                    ).astype(np.float32) * 0.03
                else:
                    features[t] = features[t - 1] * 0.5 + rng.randn(
                        self.feature_dim
                    ).astype(np.float32) * 0.2
            targets[onset:offset, 2] = 1.0  # rhythm rule

        # event_type == 0 → fluent, all-zero targets

        return {
            "features": torch.from_numpy(features),
            "rule_targets": torch.from_numpy(targets),
        }


def pretrain_rules(soft_rules_module, config,
                   num_epochs=None, num_samples=None):
    from src.training.losses import RulePretrainingLoss

    num_epochs = num_epochs or config.STAGE0_NUM_EPOCHS
    num_samples = num_samples or config.STAGE0_SYNTHETIC_SAMPLES

    dataset = SyntheticStutterDataset(
        num_samples=num_samples,
        seq_len=50,
        feature_dim=config.HIDDEN_DIM,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = RulePretrainingLoss()
    optimizer = Adam(soft_rules_module.parameters(), lr=config.STAGE0_LR)

    history = {"loss": []}
    print(f"Stage 0: Pre-training rules on {num_samples} synthetic samples "
          f"for {num_epochs} epochs")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(loader, desc=f"Stage 0 Ep {epoch+1}", leave=False):
            feats = batch["features"].to(config.DEVICE)
            targets = batch["rule_targets"].to(config.DEVICE)

            rule_scores = soft_rules_module(feats)
            loss = criterion(rule_scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        history["loss"].append(avg)
        print(f"  Epoch {epoch+1}/{num_epochs}  Loss: {avg:.4f}")

    # Verify rules are actually discriminating
    soft_rules_module.eval()
    with torch.no_grad():
        test_batch = next(iter(loader))
        scores = soft_rules_module(test_batch["features"].to(config.DEVICE))
        tgts = test_batch["rule_targets"].to(config.DEVICE)
        for i, name in enumerate(["Burst", "Voicing", "Rhythm"]):
            pos_mask = tgts[:, :, i] > 0.5
            neg_mask = tgts[:, :, i] < 0.5
            if pos_mask.any() and neg_mask.any():
                pos_mean = scores[:, :, i][pos_mask].mean().item()
                neg_mean = scores[:, :, i][neg_mask].mean().item()
                print(f"  {name}: positive={pos_mean:.3f}  negative={neg_mean:.3f}  "
                      f"gap={pos_mean - neg_mean:.3f}")

    soft_rules_module.train()
    print("Stage 0 complete.")
    return history
