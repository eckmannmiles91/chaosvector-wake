"""Wyoming protocol server for wake word detection.

Exposes the wake word detector over the Wyoming protocol so Home Assistant
can use it as a drop-in replacement for wyoming-openwakeword.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from chaosvector_wake import __version__
from chaosvector_wake.detector import CHUNK_SAMPLES, WakeWordDetector

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wyoming event handler
# ---------------------------------------------------------------------------

class WakeWordEventHandler:
    """Handles a single Wyoming client connection."""

    def __init__(
        self,
        detector: WakeWordDetector,
        wake_word_name: str,
        wyoming_info: object,
    ) -> None:
        self.detector = detector
        self.wake_word_name = wake_word_name
        self._wyoming_info = wyoming_info

    async def handle_event(self, event) -> asyncio.Event | None:
        """Process a Wyoming protocol event.

        Handles:
          - AudioChunk: feed to detector, emit Detection on trigger
          - Describe: respond with server info
          - AudioStart: reset detector state
          - AudioStop: reset detector state
        """
        from wyoming.audio import AudioChunk, AudioStart, AudioStop
        from wyoming.event import Event
        from wyoming.wake import Detect, Detection as WyomingDetection, NotDetected

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            import numpy as np

            audio = np.frombuffer(chunk.audio, dtype=np.int16).astype(np.float32)

            # Process in CHUNK_SAMPLES-sized pieces
            offset = 0
            while offset + CHUNK_SAMPLES <= len(audio):
                segment = audio[offset : offset + CHUNK_SAMPLES]
                result = self.detector.process_chunk(segment)
                if result.detected:
                    log.info(
                        "Wake word '%s' detected (confidence=%.3f)",
                        self.wake_word_name, result.confidence,
                    )
                    return WyomingDetection(
                        name=self.wake_word_name,
                        timestamp=None,
                    ).event()
                offset += CHUNK_SAMPLES

        elif AudioStart.is_type(event.type):
            self.detector.reset()

        elif AudioStop.is_type(event.type):
            self.detector.reset()

        elif event.type == "describe":
            return self._wyoming_info

        return None


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------

def _build_wyoming_info(wake_word_name: str):
    """Build the Wyoming info/describe response."""
    from wyoming.info import Attribution, Describe, Info, WakeModel, WakeProgram

    return Describe(
        info=Info(
            wake=[
                WakeProgram(
                    name="chaosvector-wake",
                    description="Custom wake word detection with family voice training",
                    installed=True,
                    version=__version__,
                    attribution=Attribution(
                        name="chaosvector",
                        url="https://github.com/eckma/chaosvector-wake",
                    ),
                    models=[
                        WakeModel(
                            name=wake_word_name,
                            description=f"Custom wake word: {wake_word_name}",
                            installed=True,
                            languages=["en"],
                            attribution=Attribution(
                                name="chaosvector",
                                url="https://github.com/eckma/chaosvector-wake",
                            ),
                        )
                    ],
                )
            ]
        )
    ).event()


async def run_server(args: argparse.Namespace) -> None:
    """Run the Wyoming protocol server.

    This follows the same pattern as other Wyoming wake word servers.
    It listens for TCP connections, receives audio chunks, and sends
    Detection events when the wake word is detected.
    """
    from wyoming.server import AsyncServer

    log.info(
        "Starting Wyoming server on %s (model=%s, wake_word=%s)",
        args.uri, args.model, args.wake_word_name,
    )

    detector = WakeWordDetector(
        model_path=args.model,
        threshold=args.threshold,
        trigger_level=args.trigger_level,
        energy_gate=args.energy_gate,
    )

    wyoming_info = _build_wyoming_info(args.wake_word_name)

    handler = WakeWordEventHandler(
        detector=detector,
        wake_word_name=args.wake_word_name,
        wyoming_info=wyoming_info,
    )

    server = AsyncServer.from_uri(args.uri)
    log.info("Wyoming server listening on %s", args.uri)

    try:
        await server.run(
            partial_handler=lambda: handler,
        )
    except asyncio.CancelledError:
        pass
    finally:
        log.info("Server stopped.")
