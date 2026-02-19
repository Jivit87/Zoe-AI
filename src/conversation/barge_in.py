"""
Barge-In Detection
===================
Monitors the microphone during TTS playback to detect genuine user
interruptions while filtering out echoes, noise, and backchannels.

Uses four conditions that must ALL hold simultaneously:
  1. Grace period elapsed (ignore first 200ms of echo)
  2. Audio energy above threshold (above background noise)
  3. VAD probability above threshold (speech, not noise)
  4. Speech sustained for 300ms+ (not a transient)
"""

import time
import numpy as np


class BargeInDetector:
    """Detects real user interruptions during TTS playback."""

    def __init__(
        self,
        grace_period_ms: float = 200,
        energy_threshold: float = 0.03,
        vad_threshold: float = 0.7,
        min_speech_duration_ms: float = 300,
    ):
        self.grace_period_ms = grace_period_ms
        self.energy_threshold = energy_threshold
        self.vad_threshold = vad_threshold
        self.min_speech_duration_ms = min_speech_duration_ms

        # Internal state
        self._tts_start_time: float = 0
        self._speech_start_time: float = 0
        self._is_monitoring: bool = False

    def on_tts_start(self):
        """Call when TTS playback begins."""
        self._tts_start_time = time.time()
        self._speech_start_time = 0
        self._is_monitoring = True

    def on_tts_stop(self):
        """Call when TTS playback ends."""
        self._is_monitoring = False
        self._speech_start_time = 0

    def check(self, audio_chunk: np.ndarray, vad_probability: float) -> bool:
        """
        Check if the current audio represents a real barge-in.

        Args:
            audio_chunk: Raw float32 audio samples
            vad_probability: Speech probability from Silero-VAD (0.0–1.0)

        Returns:
            True if this is a genuine user interruption
        """
        if not self._is_monitoring:
            return False

        # Condition 1: Grace period — ignore echo right after TTS starts
        elapsed_ms = (time.time() - self._tts_start_time) * 1000
        if elapsed_ms < self.grace_period_ms:
            return False

        # Condition 2: Sufficient audio energy (above background noise)
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        if energy < self.energy_threshold:
            self._speech_start_time = 0  # Reset sustain timer
            return False

        # Condition 3: VAD confidence above threshold
        if vad_probability < self.vad_threshold:
            self._speech_start_time = 0
            return False

        # Condition 4: Time-gating — speech must be sustained
        now = time.time()
        if self._speech_start_time == 0:
            self._speech_start_time = now
            return False  # Just started, wait for sustain

        sustained_ms = (now - self._speech_start_time) * 1000
        if sustained_ms < self.min_speech_duration_ms:
            return False  # Not sustained long enough

        # All four conditions met — genuine barge-in
        return True

    def reset(self):
        """Reset all state."""
        self._tts_start_time = 0
        self._speech_start_time = 0
        self._is_monitoring = False
