import os
import numpy as np
import librosa
from typing import Optional


class AudioPreprocessor:

    def __init__(self, target_sr: int = 16_000, clip_duration: float = 3.0,
                 seed: int = 42):
        self.target_sr = target_sr
        self.clip_duration = clip_duration
        self.target_length = int(target_sr * clip_duration)
        self._aug_seed = seed
        self._aug_counter = 0

    def load_and_preprocess(self, file_path: str, augment: bool = False) -> Optional[np.ndarray]:
        try:
            audio, _ = librosa.load(
                file_path, sr=self.target_sr, mono=True, duration=self.clip_duration
            )

            # Peak normalisation
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.9

            # Pad or centre-crop to target_length
            if len(audio) > self.target_length:
                start = (len(audio) - self.target_length) // 2
                audio = audio[start : start + self.target_length]
            elif len(audio) < self.target_length:
                audio = np.pad(audio, (0, self.target_length - len(audio)))

            # Training-time augmentation
            if augment:
                audio = self._augment(audio)

            return audio.astype(np.float32)

        except Exception as e:
            print(f"  [AudioPreprocessor] Error loading {os.path.basename(file_path)}: {e}")
            return None

    def _augment(self, audio: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self._aug_seed + self._aug_counter)
        self._aug_counter += 1

        # 1. Additive noise
        if rng.random() < 0.5:
            snr_db = rng.uniform(15, 30)
            sig_power = np.mean(audio ** 2) + 1e-10
            noise_power = sig_power / (10 ** (snr_db / 10))
            audio = audio + rng.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)

        # 2. Time masking (1-3 contiguous zero-masks)
        if rng.random() < 0.4:
            num_masks = rng.integers(1, 4)
            for _ in range(num_masks):
                mask_len = rng.integers(int(0.02 * len(audio)), int(0.1 * len(audio)) + 1)
                start = rng.integers(0, max(len(audio) - mask_len, 1))
                audio[start : start + mask_len] = 0.0

        # 3. Gain perturbation
        if rng.random() < 0.5:
            gain_db = rng.uniform(-4, 4)
            audio = audio * (10 ** (gain_db / 20))

        # 4. Time shift
        if rng.random() < 0.3:
            shift = rng.integers(-int(0.1 * len(audio)), int(0.1 * len(audio)) + 1)
            audio = np.roll(audio, shift)
            if shift > 0:
                audio[:shift] = 0.0
            elif shift < 0:
                audio[shift:] = 0.0

        # 5. Speed perturbation (stretch/compress time axis)
        if rng.random() < 0.3:
            speed_factor = rng.uniform(0.9, 1.1)
            try:
                stretched = librosa.effects.time_stretch(audio, rate=speed_factor)
                if len(stretched) >= self.target_length:
                    audio = stretched[:self.target_length]
                else:
                    audio = np.pad(stretched, (0, self.target_length - len(stretched)))
            except Exception:
                pass  # skip if librosa fails

        # 6. Pitch shift (small, ±1 semitone)
        if rng.random() < 0.2:
            n_steps = rng.uniform(-1, 1)
            try:
                audio = librosa.effects.pitch_shift(
                    audio, sr=self.target_sr, n_steps=n_steps
                )
            except Exception:
                pass  # skip if librosa fails

        # Ensure correct length
        if len(audio) > self.target_length:
            audio = audio[:self.target_length]
        elif len(audio) < self.target_length:
            audio = np.pad(audio, (0, self.target_length - len(audio)))

        # Re-normalise
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.9

        return audio
