"""
Voice Generator using Kokoro-82M (ONNX)
=======================================
Ultra-fast, high-quality TTS optimized for local inference.
Uses the 82M parameter model via ONNX Runtime.

Falls back to Edge-TTS if Kokoro fails to load.
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import tempfile
import sys


class VoiceGenerator:
    """Text-to-Speech using Kokoro-82M (ONNX)."""

    # Available voices in voices.bin:
    #   af, af_bella, af_nicole, af_sarah, af_sky
    #   am, am_adam, am_michael
    #   bf_emma, bf_isabella
    #   bm_george, bm_lewis
    DEFAULT_VOICE = "af_bella"  # Natural, conversational female voice
    DEFAULT_LANG = "en-us"

    def __init__(self, voice=None, use_edge_fallback=True):
        self.voice = voice or self.DEFAULT_VOICE
        self.temp_dir = tempfile.mkdtemp(prefix="sara_tts_")
        self.use_edge_fallback = use_edge_fallback
        self.kokoro = None
        self._edge_tts = None

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
        """
        Generate audio from text.

        Returns:
            Path to generated WAV file
        """
        if self.kokoro is not None:
            return self._generate_kokoro(text)
        elif self._edge_tts:
            return self._generate_edge(text)
        else:
            raise RuntimeError("No TTS engine available")

    def _generate_kokoro(self, text):
        """Generate audio using Kokoro-ONNX."""
        output_path = os.path.join(self.temp_dir, "sara_speech.wav")

        # Generate audio (returns samples, sample_rate)
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

    def speak(self, text):
        """Generate speech and play it through speakers."""
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
