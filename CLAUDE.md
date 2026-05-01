# chaosvector-wake

## Architecture

- **detector.py** -- Streaming wake word detector. Accumulates a 1.5s sliding window of 16kHz audio, extracts mel spectrograms via features.py, runs ONNX inference. Confidence threshold + consecutive-detection trigger level reduces false positives. Energy gate skips inference on silence.

- **features.py** -- Numpy-only mel spectrogram extraction (no librosa). 40 mel bins, 25ms window, 10ms hop. Also provides RMS energy. The mel filterbank is computed once and cached.

- **trainer.py** -- Loads WAV recordings, augments (speed, pitch, noise, gain), extracts features, trains a small CNN (3 conv blocks + global avg pool), exports to ONNX. Target: <2MB model, <5ms inference on Pi.

- **wyoming_server.py** -- Wyoming protocol server so Home Assistant sees this as a wake word provider. Drop-in replacement for wyoming-openwakeword.

- **__main__.py** -- CLI entry point with subcommands: detect, train, serve.

## Audio contract

- 16kHz mono S16LE
- 20ms chunks (320 samples, 640 bytes)
- Detector accumulates into a 1.5s sliding window (24000 samples)

## Model architecture

Small CNN: Conv2d(1->16) -> Conv2d(16->32) -> Conv2d(32->32), each with BN + ReLU + MaxPool2d(2). Then AdaptiveAvgPool2d(1) -> Linear(32,1) -> Sigmoid. Input shape: (batch, 1, 148, 40).

## Deployment targets

- Raspberry Pi 4/5 (primary)
- Any Linux box with onnxruntime
- Wyoming protocol integration with Home Assistant

## Dependencies

- Runtime: numpy, wyoming, onnxruntime
- Training: torch (optional)
- No librosa, no torchaudio at runtime

## Development

```bash
pip install -e ".[dev,train]"
pytest
ruff check src/
```
