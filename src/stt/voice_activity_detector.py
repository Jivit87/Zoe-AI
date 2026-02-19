"""
Voice Activity Detection using Silero-VAD (Enhanced)
=====================================================
Detects when the user is speaking vs silent.

Enhancements over Week 1:
  - Hysteresis thresholding: high threshold (0.85) to START detecting speech,
    low threshold (0.30) to STOP. Prevents rapid flickering.
  - Pre-roll ring buffer (captures audio just before speech starts)
  - Silence duration tracking for proactive responses
"""

import torch
import numpy as np
from collections import deque
from silero_vad import load_silero_vad


class VoiceActivityDetector:
    def __init__(
        self,
        sample_rate: int = 16000,
        start_threshold: float = 0.85,
        stop_threshold: float = 0.30,
    ):
        # Load Silero VAD model
        print("🔇 Loading Silero VAD model...")
        self.model = load_silero_vad()
        print("✓ VAD model loaded!")

        self.sample_rate = sample_rate

        # Hysteresis thresholds — prevents flickering
        self.start_threshold = start_threshold  # High: confident speech start
        self.stop_threshold = stop_threshold     # Low: keeps detecting once started

        # Ring buffer for pre-roll (~100ms of audio before speech starts)
        self.ring_buffer = deque(maxlen=3)  # 3 chunks ≈ 96ms at 512 samples
        self.is_speaking = False

        # Track silence duration for proactive responses
        self.silence_chunks = 0
        self.silence_threshold = 250  # ~8 seconds at 32ms per chunk

        # Track the last raw probability (useful for barge-in detector)
        self.last_probability = 0.0

    def is_speech(self, audio_chunk: np.ndarray) -> tuple[bool, float]:
        """
        Check if an audio chunk contains speech using hysteresis thresholding.

        Uses a HIGH threshold (0.85) to start detecting speech and a LOW
        threshold (0.30) to stop detecting. This prevents rapid
        speech/not-speech flickering which causes false barge-ins.

        Args:
            audio_chunk: numpy array of 512 samples (32ms at 16kHz)

        Returns:
            (is_speech: bool, probability: float)
        """
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_chunk).float()

        # Get speech probability from Silero-VAD
        speech_prob = self.model(audio_tensor, self.sample_rate).item()
        self.last_probability = speech_prob

        # Hysteresis logic
        if not self.is_speaking:
            # Not currently speaking — require HIGH confidence to start
            detected = speech_prob > self.start_threshold
        else:
            # Currently speaking — keep detecting with LOW threshold
            detected = speech_prob > self.stop_threshold

        # Track silence
        if not detected:
            self.silence_chunks += 1
        else:
            self.silence_chunks = 0

        return detected, speech_prob

    def get_silence_duration_seconds(self) -> float:
        """Get how long the user has been silent (approx)."""
        chunk_duration = 512 / self.sample_rate  # ~32ms per chunk
        return self.silence_chunks * chunk_duration

    def is_prolonged_silence(self) -> bool:
        """Check if user has been silent long enough for a proactive response."""
        return self.silence_chunks >= self.silence_threshold

    def reset(self):
        """Reset state for a new session."""
        self.model.reset_states()
        self.is_speaking = False
        self.silence_chunks = 0
        self.last_probability = 0.0
        self.ring_buffer.clear()
