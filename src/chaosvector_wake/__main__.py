"""Entry point for chaosvector-wake."""

import argparse
import logging
import sys
from pathlib import Path

from chaosvector_wake import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Custom wake word detection engine"
    )
    parser.add_argument(
        "--version", action="version", version=f"chaosvector-wake {__version__}"
    )

    sub = parser.add_subparsers(dest="command")

    # --- detect ---
    det = sub.add_parser("detect", help="Run wake word detection on mic or stdin")
    det.add_argument("--model", type=Path, required=True, help="Path to ONNX model")
    det.add_argument(
        "--threshold", type=float, default=0.85,
        help="Confidence threshold (default: 0.85)",
    )
    det.add_argument(
        "--trigger-level", type=int, default=3,
        help="Consecutive detections before trigger (default: 3)",
    )
    det.add_argument(
        "--energy-gate", type=float, default=50.0,
        help="RMS energy gate to skip silence (default: 50.0)",
    )

    # --- train ---
    tr = sub.add_parser("train", help="Train a custom wake word model")
    tr.add_argument("--positive-dir", type=Path, required=True, help="Dir of positive WAV samples")
    tr.add_argument("--negative-dir", type=Path, required=True, help="Dir of negative WAV samples")
    tr.add_argument("--output", type=Path, default=Path("wake_model.onnx"), help="Output ONNX path")
    tr.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    tr.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")

    # --- serve ---
    srv = sub.add_parser("serve", help="Run Wyoming protocol server")
    srv.add_argument("--model", type=Path, required=True, help="Path to ONNX model")
    srv.add_argument("--uri", default="tcp://0.0.0.0:10400", help="Wyoming URI (default: tcp://0.0.0.0:10400)")
    srv.add_argument("--wake-word-name", default="hey_jarvis", help="Wake word identifier")
    srv.add_argument("--threshold", type=float, default=0.85)
    srv.add_argument("--trigger-level", type=int, default=3)
    srv.add_argument("--energy-gate", type=float, default=50.0)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "detect":
        from chaosvector_wake.detector import WakeWordDetector

        detector = WakeWordDetector(
            model_path=args.model,
            threshold=args.threshold,
            trigger_level=args.trigger_level,
            energy_gate=args.energy_gate,
        )
        _run_stdin_detection(detector)

    elif args.command == "train":
        from chaosvector_wake.trainer import WakeWordTrainer

        trainer = WakeWordTrainer(
            positive_dir=args.positive_dir,
            negative_dir=args.negative_dir,
        )
        trainer.train(epochs=args.epochs, batch_size=args.batch_size)
        trainer.export_onnx(args.output)

    elif args.command == "serve":
        import asyncio
        from chaosvector_wake.wyoming_server import run_server

        asyncio.run(run_server(args))

    else:
        parser.print_help()
        sys.exit(1)


def _run_stdin_detection(detector) -> None:
    """Read raw 16kHz S16LE audio from stdin and detect wake words."""
    import numpy as np

    log = logging.getLogger("detect")
    log.info("Reading 16kHz mono S16LE from stdin, chunk=640 bytes (20ms)")

    chunk_bytes = 640  # 320 samples * 2 bytes = 20ms at 16kHz
    try:
        while True:
            raw = sys.stdin.buffer.read(chunk_bytes)
            if len(raw) < chunk_bytes:
                break
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            detected, confidence, energy = detector.process_chunk(audio)
            if detected:
                log.info("WAKE WORD  confidence=%.3f  energy=%.1f", confidence, energy)
    except KeyboardInterrupt:
        pass
    log.info("Stopped.")


if __name__ == "__main__":
    main()
