"""Audio feature extraction -- numpy-only, no librosa dependency.

Produces mel spectrograms suitable for the small CNN wake word classifier.
Parameters tuned for 16kHz speech: 40 mel bins, 25ms window, 10ms hop.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
N_FFT = 400          # 25ms window at 16kHz
HOP_LENGTH = 160     # 10ms hop
N_MELS = 40
FMIN = 60.0
FMAX = 7800.0


# ---------------------------------------------------------------------------
# Mel filterbank (computed once, cached)
# ---------------------------------------------------------------------------
_MEL_FILTERBANK: np.ndarray | None = None


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank(
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """Build a mel filterbank matrix (n_mels, n_fft//2 + 1)."""
    n_freqs = n_fft // 2 + 1

    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])

    # Convert Hz to FFT bin indices
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        for j in range(left, center):
            if center != left:
                fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                fb[i, j] = (right - j) / (right - center)

    return fb


def get_mel_filterbank() -> np.ndarray:
    """Return the cached mel filterbank, building it on first call."""
    global _MEL_FILTERBANK
    if _MEL_FILTERBANK is None:
        _MEL_FILTERBANK = _build_mel_filterbank()
    return _MEL_FILTERBANK


# ---------------------------------------------------------------------------
# Windowing and STFT
# ---------------------------------------------------------------------------

def _hann_window(length: int) -> np.ndarray:
    """Periodic Hann window."""
    n = np.arange(length)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / length)).astype(np.float32)


_WINDOW: np.ndarray | None = None


def _get_window() -> np.ndarray:
    global _WINDOW
    if _WINDOW is None:
        _WINDOW = _hann_window(N_FFT)
    return _WINDOW


def stft_magnitude(signal: np.ndarray) -> np.ndarray:
    """Compute STFT magnitude spectrogram.

    Args:
        signal: 1-D float32 audio signal.

    Returns:
        Magnitude spectrogram of shape (n_frames, n_fft//2 + 1).
    """
    win = _get_window()
    n_freqs = N_FFT // 2 + 1

    # Pad signal so we get complete frames
    pad_len = N_FFT // 2
    padded = np.pad(signal, (pad_len, pad_len), mode="reflect")

    n_frames = 1 + (len(padded) - N_FFT) // HOP_LENGTH
    frames = np.zeros((n_frames, N_FFT), dtype=np.float32)
    for i in range(n_frames):
        start = i * HOP_LENGTH
        frames[i] = padded[start : start + N_FFT] * win

    # Real FFT
    spectrum = np.fft.rfft(frames, n=N_FFT, axis=1)
    return np.abs(spectrum).astype(np.float32)


# ---------------------------------------------------------------------------
# Public feature extraction
# ---------------------------------------------------------------------------

def mel_spectrogram(signal: np.ndarray) -> np.ndarray:
    """Compute log-mel spectrogram from raw audio.

    Args:
        signal: 1-D float32 audio (16kHz).

    Returns:
        Log-mel spectrogram of shape (n_frames, N_MELS).
    """
    mag = stft_magnitude(signal)
    fb = get_mel_filterbank()  # (n_mels, n_freqs)
    mel = mag @ fb.T           # (n_frames, n_mels)

    # Log compression with floor to avoid log(0)
    mel = np.log(np.maximum(mel, 1e-10))
    return mel


def rms_energy(signal: np.ndarray) -> float:
    """Root mean square energy of an audio chunk."""
    return float(np.sqrt(np.mean(signal ** 2)))


def extract_features(signal: np.ndarray) -> tuple[np.ndarray, float]:
    """Extract mel spectrogram and energy from an audio chunk.

    Args:
        signal: 1-D float32 audio (16kHz).

    Returns:
        (mel_spec, energy) -- mel_spec shape is (n_frames, N_MELS).
    """
    mel = mel_spectrogram(signal)
    energy = rms_energy(signal)
    return mel, energy
