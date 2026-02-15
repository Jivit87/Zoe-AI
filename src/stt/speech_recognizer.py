"""
Real-time Speech Recognition with Faster-Whisper
==================================================
Listens to the microphone, uses VAD to detect speech boundaries,
then transcribes complete utterances using Faster-Whisper.

Key Features:
- VAD for clean segmentation
- Background transcription (non-blocking)
- Pause/Resume support to prevent self-listening (echoes)
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
            model_size: Whisper model size ("base" recommended for Week 1)
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

        # Initialize VAD
        self.vad = VoiceActivityDetector()

        # Audio settings
        self.sample_rate = 16000
        self.chunk_size = 512  # 32ms chunks at 16kHz

        # State
        self.is_listening = False
        self.is_paused = False  # New flag for temporary deafening
        self.audio_buffer = bytearray()

        # Minimum audio length to process (avoid tiny fragments)
        self.min_audio_seconds = 0.5

        # End-of-speech detection: require N consecutive non-speech chunks
        self.silence_after_speech_chunks = 0
        self.silence_required = 15  # ~480ms of silence to confirm end-of-speech

        # Callback: called with (transcription_text: str)
        self.on_speech_detected = None

    def audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk from the microphone."""
        if self.is_paused:
            return  # Ignore audio while paused (e.g. while speaking)

        if status:
            print(f"⚠️  Audio input warning: {status}")

        # Convert to float32 mono
        audio_chunk = indata[:, 0].astype(np.float32)

        # Check if chunk contains speech
        is_speech, prob = self.vad.is_speech(audio_chunk)

        if is_speech:
            self.silence_after_speech_chunks = 0

            if not self.vad.is_speaking:
                # Speech just started — add pre-roll from ring buffer
                self.vad.is_speaking = True
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
                    self.vad.is_speaking = False
                    self.silence_after_speech_chunks = 0
                    # Process in a separate thread to avoid blocking audio stream
                    audio_copy = bytes(self.audio_buffer)
                    self.audio_buffer = bytearray()
                    threading.Thread(
                        target=self._process_audio,
                        args=(audio_copy,),
                        daemon=True,
                    ).start()

    def _process_audio(self, audio_bytes):
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

            # Fire callback if we got meaningful text
            if transcription and self.on_speech_detected:
                self.on_speech_detected(transcription)

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
        """Pause listening (temporary deafen)."""
        self.is_paused = True
        # Clear buffers so we don't process echoes
        self.audio_buffer = bytearray()
        self.vad.ring_buffer.clear()
        # print("⏸️  STT Paused (Deafened)")

    def resume(self):
        """Resume listening."""
        self.is_paused = False
        # Clear buffers again to be safe
        self.audio_buffer = bytearray()
        self.vad.ring_buffer.clear()
        # print("▶️  STT Resumed (Listening)")
