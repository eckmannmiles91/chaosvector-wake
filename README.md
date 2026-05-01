# chaosvector-wake

Custom wake word detection engine that lets you train on family voice recordings for much better accuracy than pre-trained models (openwakeword). Runs on Raspberry Pi with ONNX inference.

## Why

openwakeword uses pre-trained models that can't be customized to your voices, rooms, or distances. This project trains on your actual recordings, yielding fewer false positives and better recall across the house.

## Quick start

```bash
# Install (inference only)
pip install .

# Install with training support
pip install ".[train]"

# Train a model
chaosvector-wake train \
    --positive-dir ./data/hey_jarvis/ \
    --negative-dir ./data/negative/ \
    --output wake_model.onnx

# Run detection from stdin (16kHz S16LE)
arecord -r 16000 -f S16_LE -c 1 | chaosvector-wake detect --model wake_model.onnx

# Run as Wyoming server for Home Assistant
chaosvector-wake serve --model wake_model.onnx --uri tcp://0.0.0.0:10400
```

## Recording tips

- Record the wake word from 1m, 3m, and across the room
- Include every family member
- Record in different rooms (kitchen, living room, bedroom)
- 50+ positive samples recommended
- Negative samples: TV audio, music, normal conversation without the wake word
