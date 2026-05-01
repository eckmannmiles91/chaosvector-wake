"""Wake word detector -- processes streaming audio and fires detections.

Audio contract: 16kHz mono S16LE, fed in 20ms chunks (320 samples).
Internally we accumulate a sliding window long enough for one spectrogram
pass (~1.5s) and run inference every step.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np

from chaosvector_wake.features import (
    N_MELS,
    SAMPLE_RATE,
    extract_features,
    rms_energy,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------

class Detection(NamedTuple):
    detected: bool
    confidence: float
    energy: float


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

# How much audio the model sees per inference pass.
WINDOW_SECONDS = 1.5
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)  # 24000

# 20ms chunk at 16kHz
CHUNK_SAMPLES = int(SAMPLE_RATE * 0.020)  # 320


class WakeWordDetector:
    """Streaming wake word detector with ONNX inference.

    Parameters
    ----------
    model_path:
        Path to the exported ONNX model file.
    threshold:
        Confidence threshold for a positive detection (0-1).
    trigger_level:
        Number of consecutive above-threshold frames before firing.
    energy_gate:
        Minimum RMS energy to bother running inference (skip silence).
    """

    def __init__(
        self,
        model_path: Path,
        threshold: float = 0.85,
        trigger_level: int = 3,
        energy_gate: float = 50.0,
    ) -> None:
        self.threshold = threshold
        self.trigger_level = trigger_level
        self.energy_gate = energy_gate

        # Sliding audio buffer
        self._buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)

        # Consecutive positive count
        self._consecutive = 0

        # Refractory period -- suppress re-triggers for N chunks after firing
        self._refractory_chunks = int(1.0 / 0.020)  # 1 second
        self._refractory_counter = 0

        # Load ONNX model
        self._session = self._load_model(model_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(model_path: Path):
        """Load an ONNX Runtime inference session.

        Falls back to a dummy scorer if onnxruntime is unavailable or the
        model file doesn't exist yet (useful during development).
        """
        try:
            import onnxruntime as ort

            if not model_path.exists():
                log.warning("Model %s not found -- using dummy scorer", model_path)
                return None

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            session = ort.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            log.info("Loaded ONNX model from %s", model_path)
            return session
        except ImportError:
            log.warning("onnxruntime not installed -- using dummy scorer")
            return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(self, features: np.ndarray) -> float:
        """Run the ONNX model on extracted features and return confidence.

        Args:
            features: Mel spectrogram of shape (n_frames, N_MELS).

        Returns:
            Confidence score in [0, 1].
        """
        if self._session is None:
            # Dummy: always return 0.0 when no model loaded
            return 0.0

        # Model expects (batch, 1, n_frames, n_mels) -- single-channel image
        inp = features[np.newaxis, np.newaxis, :, :].astype(np.float32)

        input_name = self._session.get_inputs()[0].name
        output_name = self._session.get_outputs()[0].name

        result = self._session.run([output_name], {input_name: inp})
        # Assume sigmoid output: single float in [0, 1]
        score = float(result[0].flat[0])
        return score

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------

    def process_chunk(self, chunk: np.ndarray) -> Detection:
        """Process a single 20ms audio chunk (320 float32 samples).

        Call this repeatedly with sequential chunks. Returns a Detection
        indicating whether the wake word was detected on this step.
        """
        if chunk.shape[0] != CHUNK_SAMPLES:
            log.warning(
                "Expected %d samples, got %d -- padding/truncating",
                CHUNK_SAMPLES, chunk.shape[0],
            )
            if chunk.shape[0] < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - chunk.shape[0]))
            else:
                chunk = chunk[:CHUNK_SAMPLES]

        # Shift buffer left and append new chunk
        self._buffer = np.roll(self._buffer, -CHUNK_SAMPLES)
        self._buffer[-CHUNK_SAMPLES:] = chunk

        energy = rms_energy(chunk)

        # Refractory period after a trigger
        if self._refractory_counter > 0:
            self._refractory_counter -= 1
            self._consecutive = 0
            return Detection(detected=False, confidence=0.0, energy=energy)

        # Energy gate -- skip inference on silence
        if energy < self.energy_gate:
            self._consecutive = 0
            return Detection(detected=False, confidence=0.0, energy=energy)

        # Extract features from the full sliding window
        features, _ = extract_features(self._buffer)
        confidence = self._run_inference(features)

        # Threshold + trigger level logic
        if confidence >= self.threshold:
            self._consecutive += 1
        else:
            self._consecutive = 0

        detected = self._consecutive >= self.trigger_level
        if detected:
            self._consecutive = 0
            self._refractory_counter = self._refractory_chunks
            log.info("DETECTED  confidence=%.3f  energy=%.1f", confidence, energy)

        return Detection(detected=detected, confidence=confidence, energy=energy)

    def reset(self) -> None:
        """Reset internal state (buffer, counters)."""
        self._buffer[:] = 0.0
        self._consecutive = 0
        self._refractory_counter = 0
