"""
Real-time Speech Recognition with Faster-Whisper (Enhanced)
=============================================================
Listens to the microphone, uses VAD to detect speech boundaries,
then transcribes complete utterances using Faster-Whisper.

Enhancements over Week 1:
  - Mic stays active during TTS (for barge-in detection)
  - BargeInDetector integration for real interruptions
  - BackchannelClassifier to filter "mm-hmm" from real turns
  - Audio duration tracking for emotion detection
"""

import numpy as np
import sounddevice as sd
import threading
import time
from faster_whisper import WhisperModel
from .voice_activity_detector import VoiceActivityDetector


class SpeechRecognizer:
    def __init__(self, model_size="base", device="cpu"):
        """
        Args:
            model_size: Whisper model size ("small" recommended)
            device: "cpu" or "cuda" (use "cpu" on macOS)
        """
        # Load Faster-Whisper
        print(f"🎤 Loading Faster-Whisper ({model_size}) on {device}...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type="int8",  # int8 works well on CPU/macOS
        )
        print("✓ STT model loaded!")

        # Initialize VAD (now with hysteresis)
        self.vad = VoiceActivityDetector()

        # Audio settings
        self.sample_rate = 16000
        self.chunk_size = 512  # 32ms chunks at 16kHz

        # State
        self.is_listening = False
        self.is_paused = False
        self.audio_buffer = bytearray()

        # Minimum audio length to process (avoid tiny fragments)
        self.min_audio_seconds = 0.5

        # End-of-speech detection: require N consecutive non-speech chunks
        self.silence_after_speech_chunks = 0
        self.silence_required = 15  # ~480ms of silence to confirm end-of-speech

        # Track audio duration for emotion detection
        self._speech_start_time: float = 0

        # Callbacks
        self.on_speech_detected = None   # (transcription: str, audio_duration: float)
        self.on_barge_in = None          # () — called when barge-in detected during TTS

        # External components (set by SaraAI)
        self.barge_in_detector = None
        self.backchannel_classifier = None
        self.state_manager = None

        # Buffer for audio captured DURING barge-in detection (the 300ms sustain)
        # so no words are lost when the barge-in finally confirms
        self._barge_in_audio_buffer = bytearray()

    def audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk from the microphone."""
        if self.is_paused:
            return  # Fully paused (e.g. during greeting)

        if status:
            pass  # Suppress noisy warnings

        # Convert to float32 mono
        audio_chunk = indata[:, 0].astype(np.float32)

        # Check if chunk contains speech
        is_speech, prob = self.vad.is_speech(audio_chunk)

        # --- Barge-in detection during SPEAKING state ---
        if (
            self.state_manager
            and self.state_manager.is_speaking
            and self.barge_in_detector
        ):
            # Buffer speech-like audio during barge-in detection
            # so we don't lose the first words of the interruption
            if prob > 0.3:
                self._barge_in_audio_buffer.extend(audio_chunk.tobytes())
            else:
                # Non-speech → sustain timer resets, clear stale buffer
                self._barge_in_audio_buffer = bytearray()

            if self.barge_in_detector.check(audio_chunk, prob):
                # Genuine barge-in confirmed — prepend buffered audio
                # to main buffer so the full utterance gets transcribed
                self.audio_buffer = self._barge_in_audio_buffer + self.audio_buffer
                self._barge_in_audio_buffer = bytearray()
                self.vad.is_speaking = True
                self._speech_start_time = time.time() - 0.3  # account for buffered audio
                if self.on_barge_in:
                    self.on_barge_in()
            return  # Barge-in detector is in control during SPEAKING

        # --- Normal listening mode ---
        if is_speech:
            self.silence_after_speech_chunks = 0

            if not self.vad.is_speaking:
                # Speech just started — add pre-roll from ring buffer
                self.vad.is_speaking = True
                self._speech_start_time = time.time()
                for buffered_chunk in self.vad.ring_buffer:
                    self.audio_buffer.extend(buffered_chunk.tobytes())

            # Add current chunk to buffer
            self.audio_buffer.extend(audio_chunk.tobytes())

        else:
            # Not speech — add to ring buffer for pre-roll
            self.vad.ring_buffer.append(audio_chunk.copy())

            if self.vad.is_speaking:
                # We were speaking, now counting silence
                self.silence_after_speech_chunks += 1
                # Keep buffering a bit of silence for natural cutoff
                self.audio_buffer.extend(audio_chunk.tobytes())

                if self.silence_after_speech_chunks >= self.silence_required:
                    # Confirmed end of speech — process it
                    audio_duration = time.time() - self._speech_start_time
                    self.vad.is_speaking = False
                    self.silence_after_speech_chunks = 0
                    # Process in a separate thread to avoid blocking audio stream
                    audio_copy = bytes(self.audio_buffer)
                    self.audio_buffer = bytearray()
                    threading.Thread(
                        target=self._process_audio,
                        args=(audio_copy, audio_duration),
                        daemon=True,
                    ).start()

    def _process_audio(self, audio_bytes: bytes, audio_duration: float = 0):
        """Transcribe collected audio buffer."""
        # Convert bytes back to numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

        # Skip if too short
        duration = len(audio_array) / self.sample_rate
        if duration < self.min_audio_seconds:
            return

        # Transcribe with Faster-Whisper
        try:
            segments, info = self.model.transcribe(
                audio_array,
                language="en",
                beam_size=5,
                vad_filter=True,
            )

            # Collect full transcription
            transcription = " ".join(
                segment.text for segment in segments
            ).strip()

            # Skip empty
            if not transcription:
                return

            # Check if this is just a backchannel
            if self.backchannel_classifier and self.backchannel_classifier.is_backchannel(transcription):
                # It's just "yeah" or "mm-hmm" — don't treat as a new turn
                return

            # Fire callback with transcription + duration
            if self.on_speech_detected:
                self.on_speech_detected(transcription, audio_duration)

        except Exception as e:
            print(f"❌ Transcription error: {e}")

    def start_listening(self):
        """Start listening to the microphone (blocking)."""
        self.is_listening = True
        self.is_paused = False

        print("🎧 Listening... (Press Ctrl+C to stop)")

        with sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype="float32",
            callback=self.audio_callback,
        ):
            while self.is_listening:
                sd.sleep(100)

    def stop_listening(self):
        """Stop listening."""
        self.is_listening = False

    def pause(self):
        """Pause listening (temporary deafen — used during greeting only)."""
        self.is_paused = True
        self.audio_buffer = bytearray()
        self._barge_in_audio_buffer = bytearray()
        self.vad.ring_buffer.clear()

    def resume(self):
        """Resume listening."""
        self.is_paused = False
        self.audio_buffer = bytearray()
        self._barge_in_audio_buffer = bytearray()
        self.vad.ring_buffer.clear()

