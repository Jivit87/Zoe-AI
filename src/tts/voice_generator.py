"""
Voice Generator using Kokoro-82M (ONNX) — Enhanced
=====================================================
Ultra-fast, high-quality TTS optimized for local inference.

Enhancements:
  - Interruptible speak_stream(): checks cancel_event between chunks
  - Thinking sounds: pre-recorded filler phrases to eliminate dead silence
  - Returns spoken/remaining text on interruption for context tracking
"""

import sounddevice as sd
import soundfile as sf
from typing import Iterator, Optional
import numpy as np
import os
import random
import tempfile
import threading


# Pre-defined thinking sounds — played while LLM generates
THINKING_SOUNDS = [
    "Hmm...",
    "Well...",
    "Let me think...",
    "Mmm.",
    "Yeah...",
]


class VoiceGenerator:
    """Text-to-Speech using Kokoro-82M (ONNX) with interruption support."""

    DEFAULT_VOICE = "af_bella"
    DEFAULT_LANG = "en-us"

    def __init__(self, voice=None, use_edge_fallback=True):
        self.voice = voice or self.DEFAULT_VOICE
        self.temp_dir = tempfile.mkdtemp(prefix="sara_tts_")
        self.use_edge_fallback = use_edge_fallback
        self.kokoro = None
        self._edge_tts = None

        # Cancel event — set by StateManager during barge-in
        self.cancel_event: Optional[threading.Event] = None

        # Absolute path to model files
        self.model_path = os.path.abspath("src/models/kokoro/kokoro-v0_19.onnx")
        self.voices_path = os.path.abspath("src/models/kokoro/voices.bin")

        self._load_model()

    def _load_model(self):
        """Load Kokoro-82M ONNX model."""
        try:
            from kokoro_onnx import Kokoro

            if not os.path.exists(self.model_path) or not os.path.exists(self.voices_path):
                raise FileNotFoundError("Kokoro model files not found in src/models/kokoro/")

            print(f"🔊 Loading Kokoro-82M from {self.model_path}...")
            self.kokoro = Kokoro(self.model_path, self.voices_path)
            print(f"✓ Kokoro-82M ready! (voice: {self.voice})")

        except Exception as e:
            print(f"⚠️  Kokoro-82M failed to load: {e}")
            if self.use_edge_fallback:
                print("   Falling back to Edge-TTS...")
                self._setup_edge_fallback()
            else:
                raise

    def _setup_edge_fallback(self):
        """Set up Edge-TTS as fallback."""
        self._edge_tts = True
        print("🔊 Edge-TTS fallback ready (voice: en-US-AvaNeural)")

    def generate_audio(self, text):
        """Generate audio from text. Returns path to WAV file."""
        if self.kokoro is not None:
            return self._generate_kokoro(text)
        elif self._edge_tts:
            return self._generate_edge(text)
        else:
            raise RuntimeError("No TTS engine available")

    def _generate_kokoro(self, text):
        """Generate audio using Kokoro-ONNX."""
        output_path = os.path.join(self.temp_dir, "sara_speech.wav")
        samples, sample_rate = self.kokoro.create(
            text, voice=self.voice, speed=1, lang=self.DEFAULT_LANG
        )
        sf.write(output_path, samples, sample_rate)
        return output_path

    def _generate_edge(self, text):
        """Fallback: generate audio using Edge-TTS."""
        import edge_tts
        import asyncio

        output_path = os.path.join(self.temp_dir, "sara_speech.mp3")

        async def _gen():
            communicate = edge_tts.Communicate(
                text, "en-US-AvaNeural", rate="-5%", pitch="-2Hz"
            )
            await communicate.save(output_path)

        asyncio.run(_gen())
        return output_path

    # ─── Core playback methods ────────────────────────────────────────

    def speak(self, text):
        """Generate speech and play it through speakers (blocking, non-interruptible)."""
        try:
            audio_file = self.generate_audio(text)
            data, samplerate = sf.read(audio_file)
            sd.play(data, samplerate)
            sd.wait()
            try:
                os.remove(audio_file)
            except OSError:
                pass
        except Exception as e:
            print(f"❌ TTS error: {e}")

    def speak_stream(self, chunks: Iterator[str]) -> dict:
        """
        Play text chunks sequentially, with barge-in interruption support.

        Between each chunk, checks self.cancel_event. If set, stops
        immediately and returns what was spoken vs what remains.

        Args:
            chunks: Iterator of text strings from streaming LLM

        Returns:
            dict with:
                "full_text":  All text (spoken + unspoken)
                "spoken":     Text that was actually played
                "remaining":  Text that was NOT played (interrupted)
                "interrupted": True if playback was interrupted
        """
        spoken_parts = []
        remaining_parts = []
        interrupted = False
        chunk_index = 0

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            # Check cancel BEFORE synthesizing this chunk
            if self.cancel_event and self.cancel_event.is_set():
                remaining_parts.append(chunk)
                interrupted = True
                # Drain remaining chunks without playing
                for remaining in chunks:
                    remaining = remaining.strip()
                    if remaining:
                        remaining_parts.append(remaining)
                break

            chunk_index += 1

            # Synthesize and play this chunk
            try:
                audio_file = self.generate_audio(chunk)
                data, samplerate = sf.read(audio_file)
                sd.play(data, samplerate)

                # Wait for playback, but check cancel periodically
                while sd.get_stream().active:
                    if self.cancel_event and self.cancel_event.is_set():
                        sd.stop()  # Stop audio immediately
                        spoken_parts.append(chunk)  # Partially spoken
                        interrupted = True
                        # Drain remaining chunks
                        for remaining in chunks:
                            remaining = remaining.strip()
                            if remaining:
                                remaining_parts.append(remaining)
                        break
                    sd.sleep(50)  # Check every 50ms

                if interrupted:
                    break

                spoken_parts.append(chunk)

                try:
                    os.remove(audio_file)
                except OSError:
                    pass

            except Exception as e:
                print(f"❌ TTS stream error on chunk {chunk_index}: {e}")

        spoken = " ".join(spoken_parts)
        remaining = " ".join(remaining_parts)

        return {
            "full_text": (spoken + " " + remaining).strip(),
            "spoken": spoken,
            "remaining": remaining,
            "interrupted": interrupted,
        }

    # ─── Thinking sounds ──────────────────────────────────────────────

    def play_thinking_sound(self):
        """
        Play a random filler sound ("Hmm...", "Well...") to eliminate
        dead silence while the LLM is generating its response.

        Short and quick — typically ~200ms of audio.
        """
        sound = random.choice(THINKING_SOUNDS)
        try:
            audio_file = self.generate_audio(sound)
            data, samplerate = sf.read(audio_file)
            sd.play(data, samplerate)
            sd.wait()
            try:
                os.remove(audio_file)
            except OSError:
                pass
        except Exception as e:
            # Non-critical — just skip if it fails
            pass

    # ─── Utility ──────────────────────────────────────────────────────

    def speak_and_save(self, text, output_path=None):
        """Generate speech, play it, and save the WAV file."""
        try:
            audio_file = self.generate_audio(text)
            data, samplerate = sf.read(audio_file)

            if output_path is None:
                output_path = os.path.join(self.temp_dir, "sara_speech_saved.wav")
            sf.write(output_path, data, samplerate)

            sd.play(data, samplerate)
            sd.wait()

            try:
                if audio_file != output_path:
                    os.remove(audio_file)
            except OSError:
                pass

            return output_path

        except Exception as e:
            print(f"❌ TTS error: {e}")
            return None
