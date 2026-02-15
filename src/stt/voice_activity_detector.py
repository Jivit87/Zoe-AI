"""
Voice Activity Detection using Silero-VAD
==========================================
Detects when the user is speaking vs silent.
Uses a ring buffer for pre-roll audio (captures audio just before speech starts).
"""

import torch
import numpy as np
from collections import deque
from silero_vad import load_silero_vad


class VoiceActivityDetector:
    def __init__(self, sample_rate=16000, threshold=0.5):
        # Load Silero VAD model from the installed pip package
        print("🔇 Loading Silero VAD model...")
        self.model = load_silero_vad()
        print("✓ VAD model loaded!")

        self.sample_rate = sample_rate
        self.threshold = threshold

        # Ring buffer for pre-roll (~100ms of audio before speech starts)
        self.ring_buffer = deque(maxlen=3)  # 3 chunks ≈ 96ms at 512 samples
        self.is_speaking = False

        # Track silence duration for proactive responses
        self.silence_chunks = 0
        self.silence_threshold = 250  # ~8 seconds at 32ms per chunk

    def is_speech(self, audio_chunk):
        """
        Check if an audio chunk contains speech.

        Args:
            audio_chunk: numpy array of 512 samples (32ms at 16kHz)

        Returns:
            (is_speech: bool, probability: float)
        """
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_chunk).float()

        # Get speech probability from Silero-VAD
        speech_prob = self.model(audio_tensor, self.sample_rate).item()

        # Track silence
        if speech_prob < self.threshold:
            self.silence_chunks += 1
        else:
            self.silence_chunks = 0

        return speech_prob > self.threshold, speech_prob

    def get_silence_duration_seconds(self):
        """Get how long the user has been silent (approx)."""
        chunk_duration = 512 / self.sample_rate  # ~32ms per chunk
        return self.silence_chunks * chunk_duration

    def is_prolonged_silence(self):
        """Check if user has been silent long enough for a proactive response."""
        return self.silence_chunks >= self.silence_threshold

    def reset(self):
        """Reset state for a new session."""
        self.model.reset_states()
        self.is_speaking = False
        self.silence_chunks = 0
        self.ring_buffer.clear()
