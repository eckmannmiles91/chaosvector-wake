# Changes from Upstream

ChaosVector Wake replaces [wyoming-openwakeword](https://github.com/rhasspy/wyoming-openwakeword) (MIT) with a trainable wake word engine.

## Why we forked
- openwakeword uses pre-trained models that can't be customized for specific voices or environments
- The "hey_jarvis" model triggers on TV audio (RMS 204-230) and misses real speech at 12ft (RMS 290-337)
- The service crashes after ~5 days of continuous uptime
- No way to train on family voice recordings for better accuracy

## What we changed

### Custom Training Pipeline
- **Upstream:** Pre-trained models only. Take it or leave it.
- **Ours:** Full training pipeline — record family members saying "Hey Jarvis" at different distances and rooms, augment with speed/pitch/noise/room-impulse-response variations, train a small CNN classifier, export to ONNX. Target: <2MB model, <5ms inference on Pi CPU.

### Built-in Energy Gate
- **Upstream:** Energy threshold is handled externally by the satellite code
- **Ours:** Energy gate integrated into the detector. RMS check happens before feature extraction — saves compute on silent/quiet frames. The threshold tuning (275 for 12ft range, rejecting TV at 204-230) is part of the detector config, not scattered across the satellite.

### Numpy-Only Feature Extraction
- **Upstream:** Depends on tensorflow-lite or onnxruntime for mel spectrogram
- **Ours:** Pure numpy mel spectrogram (40 mel bins, 25ms window, 10ms hop). No heavy ML framework dependency for feature extraction. Faster startup, smaller memory footprint.

### Streaming Detection with Trigger Level
- **Upstream:** Single-frame detection
- **Ours:** Configurable trigger level — require N consecutive positive detections before firing. Reduces false positives from transient audio events.

## Pain Points This Solves
- TV false wakes (train on negative samples from your actual TV audio)
- 12ft range rejection (train on positive samples at actual distances)
- 5-day uptime crashes (simpler architecture, no tensorflow)
- Per-family accuracy (each household trains their own model)
