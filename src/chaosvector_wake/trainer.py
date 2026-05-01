"""Training pipeline for custom wake word models.

Workflow:
  1. Load positive samples (recordings of the wake word).
  2. Load negative samples (background noise, TV, music, other speech).
  3. Extract mel spectrograms.
  4. Augment (speed, pitch, noise injection, room impulse response).
  5. Train a small CNN classifier.
  6. Export to ONNX (<2MB, <5ms inference on Pi CPU).

Requires torch (optional dependency) -- import is deferred.
"""

from __future__ import annotations

import logging
import random
import struct
import wave
from pathlib import Path
from typing import Any

import numpy as np

from chaosvector_wake.features import (
    SAMPLE_RATE,
    N_MELS,
    mel_spectrogram,
)

log = logging.getLogger(__name__)

# Target spectrogram length for fixed-size model input (1.5s of audio)
TARGET_FRAMES = 148  # ceil(1.5s * 16000 / hop_length=160) + padding


# ---------------------------------------------------------------------------
# WAV loading
# ---------------------------------------------------------------------------

def load_wav(path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a WAV file as float32 mono at the target sample rate.

    Uses only stdlib `wave` -- no external audio libs required.
    Supports 16-bit PCM only. Resampling is not implemented (files must
    already be at 16kHz or close enough).
    """
    with wave.open(str(path), "rb") as wf:
        assert wf.getsampwidth() == 2, f"Only 16-bit WAV supported, got {wf.getsampwidth() * 8}-bit"
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    samples = np.array(struct.unpack(f"<{n_frames * n_channels}h", raw), dtype=np.float32)

    # Mix to mono if stereo
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

class Augmenter:
    """Collection of audio augmentation transforms."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def speed_perturb(self, audio: np.ndarray, factor: float | None = None) -> np.ndarray:
        """Change speed by resampling. factor > 1 = faster/higher pitch."""
        if factor is None:
            factor = self.rng.uniform(0.9, 1.1)
        indices = np.round(np.arange(0, len(audio), factor)).astype(int)
        indices = indices[indices < len(audio)]
        return audio[indices]

    def pitch_shift(self, audio: np.ndarray, semitones: float | None = None) -> np.ndarray:
        """Approximate pitch shift via speed change + length correction.

        A proper pitch shifter would use phase vocoder, but for wake word
        training this approximation is sufficient.
        """
        if semitones is None:
            semitones = self.rng.uniform(-1.5, 1.5)
        factor = 2.0 ** (semitones / 12.0)
        shifted = self.speed_perturb(audio, factor)
        # Trim or pad to original length
        if len(shifted) > len(audio):
            shifted = shifted[: len(audio)]
        elif len(shifted) < len(audio):
            shifted = np.pad(shifted, (0, len(audio) - len(shifted)))
        return shifted

    def add_noise(self, audio: np.ndarray, noise: np.ndarray, snr_db: float | None = None) -> np.ndarray:
        """Mix noise into audio at a given SNR."""
        if snr_db is None:
            snr_db = self.rng.uniform(5.0, 20.0)

        # Loop noise to match audio length
        if len(noise) < len(audio):
            reps = int(np.ceil(len(audio) / len(noise)))
            noise = np.tile(noise, reps)
        noise = noise[: len(audio)]

        sig_power = np.mean(audio ** 2) + 1e-10
        noise_power = np.mean(noise ** 2) + 1e-10
        target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        scale = np.sqrt(target_noise_power / noise_power)

        return audio + scale * noise

    def apply_rir(self, audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
        """Convolve audio with a room impulse response for reverb."""
        # Normalize RIR
        rir = rir / (np.max(np.abs(rir)) + 1e-10)
        convolved = np.convolve(audio, rir, mode="full")[: len(audio)]
        return convolved.astype(np.float32)

    def random_gain(self, audio: np.ndarray) -> np.ndarray:
        """Random volume change in dB."""
        gain_db = self.rng.uniform(-6.0, 6.0)
        return audio * (10.0 ** (gain_db / 20.0))

    def augment(self, audio: np.ndarray, noise_pool: list[np.ndarray] | None = None) -> np.ndarray:
        """Apply a random chain of augmentations."""
        if self.rng.random() < 0.5:
            audio = self.speed_perturb(audio)
        if self.rng.random() < 0.4:
            audio = self.pitch_shift(audio)
        if noise_pool and self.rng.random() < 0.6:
            noise = self.rng.choice(noise_pool)
            audio = self.add_noise(audio, noise)
        if self.rng.random() < 0.5:
            audio = self.random_gain(audio)
        return audio


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def _pad_or_trim_features(features: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    """Pad or trim spectrogram to fixed number of frames."""
    if features.shape[0] > target_frames:
        features = features[:target_frames]
    elif features.shape[0] < target_frames:
        pad = np.zeros((target_frames - features.shape[0], features.shape[1]), dtype=np.float32)
        features = np.concatenate([features, pad], axis=0)
    return features


def prepare_dataset(
    positive_dir: Path,
    negative_dir: Path,
    augmenter: Augmenter | None = None,
    augment_factor: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and prepare training data.

    Returns:
        (X, y) where X has shape (N, TARGET_FRAMES, N_MELS) and y is (N,) with 0/1 labels.
    """
    augmenter = augmenter or Augmenter()

    positive_files = sorted(positive_dir.glob("*.wav"))
    negative_files = sorted(negative_dir.glob("*.wav"))

    if not positive_files:
        raise FileNotFoundError(f"No WAV files in {positive_dir}")
    if not negative_files:
        raise FileNotFoundError(f"No WAV files in {negative_dir}")

    log.info("Found %d positive, %d negative samples", len(positive_files), len(negative_files))

    # Load all raw audio
    pos_audio = [load_wav(f) for f in positive_files]
    neg_audio = [load_wav(f) for f in negative_files]

    # Use negatives as a noise pool for augmentation
    noise_pool = neg_audio.copy()

    features_list: list[np.ndarray] = []
    labels: list[int] = []

    # Positive samples + augmented variants
    for audio in pos_audio:
        mel = mel_spectrogram(audio)
        features_list.append(_pad_or_trim_features(mel))
        labels.append(1)

        for _ in range(augment_factor):
            aug = augmenter.augment(audio, noise_pool=noise_pool)
            mel = mel_spectrogram(aug)
            features_list.append(_pad_or_trim_features(mel))
            labels.append(1)

    # Negative samples + augmented variants (fewer augmentations)
    for audio in neg_audio:
        mel = mel_spectrogram(audio)
        features_list.append(_pad_or_trim_features(mel))
        labels.append(0)

        for _ in range(augment_factor // 2 + 1):
            aug = augmenter.augment(audio)
            mel = mel_spectrogram(aug)
            features_list.append(_pad_or_trim_features(mel))
            labels.append(0)

    X = np.stack(features_list, axis=0)
    y = np.array(labels, dtype=np.float32)
    log.info("Dataset: %d samples (positive=%d, negative=%d)", len(y), int(y.sum()), int(len(y) - y.sum()))
    return X, y


# ---------------------------------------------------------------------------
# Model definition (torch)
# ---------------------------------------------------------------------------

def _build_model() -> Any:
    """Build a small CNN classifier for wake word detection.

    Architecture: 3 conv blocks + global avg pool + FC sigmoid.
    Target: <2MB parameters, <5ms inference on Pi CPU.
    """
    import torch
    import torch.nn as nn

    class WakeWordCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1: (1, 148, 40) -> (16, 74, 20)
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # Block 2: (16, 74, 20) -> (32, 37, 10)
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # Block 3: (32, 37, 10) -> (32, 18, 5)
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = WakeWordCNN()
    param_count = sum(p.numel() for p in model.parameters())
    size_mb = param_count * 4 / (1024 * 1024)
    log.info("Model: %d params, ~%.2f MB", param_count, size_mb)
    return model


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class WakeWordTrainer:
    """End-to-end training pipeline."""

    def __init__(self, positive_dir: Path, negative_dir: Path) -> None:
        self.positive_dir = positive_dir
        self.negative_dir = negative_dir
        self._model: Any = None
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def train(self, epochs: int = 50, batch_size: int = 32, lr: float = 1e-3) -> None:
        """Train the wake word model."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        log.info("Preparing dataset...")
        X, y = prepare_dataset(self.positive_dir, self.negative_dir)
        self._X, self._y = X, y

        # Shuffle
        idx = np.random.permutation(len(y))
        X, y = X[idx], y[idx]

        # Train/val split (90/10)
        split = int(0.9 * len(y))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Convert to tensors: (N, 1, T, M) for conv2d
        X_train_t = torch.from_numpy(X_train[:, np.newaxis, :, :])
        y_train_t = torch.from_numpy(y_train[:, np.newaxis])
        X_val_t = torch.from_numpy(X_val[:, np.newaxis, :, :])
        y_val_t = torch.from_numpy(y_val[:, np.newaxis])

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model = _build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for xb, yb in train_dl:
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_ds)

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
                val_acc = ((val_pred > 0.5).float() == y_val_t).float().mean().item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == 0:
                log.info(
                    "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f",
                    epoch + 1, epochs, train_loss, val_loss, val_acc,
                )

        if best_state is not None:
            model.load_state_dict(best_state)
        self._model = model
        log.info("Training complete. Best val_loss=%.4f", best_val_loss)

    def export_onnx(self, output_path: Path) -> None:
        """Export trained model to ONNX format."""
        import torch

        if self._model is None:
            raise RuntimeError("Must call train() before export_onnx()")

        self._model.eval()
        dummy_input = torch.randn(1, 1, TARGET_FRAMES, N_MELS)

        torch.onnx.export(
            self._model,
            dummy_input,
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch"},
                "output": {0: "batch"},
            },
            opset_version=13,
        )

        size_kb = output_path.stat().st_size / 1024
        log.info("Exported ONNX model to %s (%.1f KB)", output_path, size_kb)
        if size_kb > 2048:
            log.warning("Model is %.1f KB -- target is <2MB", size_kb)
