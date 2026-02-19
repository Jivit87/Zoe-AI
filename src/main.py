"""
Sara AI — Full Voice Conversation Loop (Week 1 MVP)
=====================================================
The complete pipeline:
  🎤 Hear → 📝 Transcribe → 🧠 Think → 💾 Remember → 🔊 Speak

Usage:
    python -m src.main

Press Ctrl+C to stop.
"""

import threading
import time

from src.stt.speech_recognizer import SpeechRecognizer
from src.llm.sara_brain import SaraBrain
from src.tts.voice_generator import VoiceGenerator


class SaraAI:
    """Week 1 MVP: Full voice-to-voice conversation loop."""

    def __init__(self):
        print()
        print("✨ Initializing Sara AI...\n")

        # Core components
        self.brain = SaraBrain()
        self.stt = SpeechRecognizer(model_size="small", device="cpu")
        self.tts = VoiceGenerator()

        # Wire up speech callback
        self.stt.on_speech_detected = self.handle_user_speech

        # State
        self.is_active = False
        self.is_speaking = False

        # Silence monitoring for proactive responses
        self.last_interaction_time = time.time()
        self.proactive_cooldown = 30  # seconds between proactive checks
        self._silence_thread = None

        print("\n✓ Sara AI ready!\n")

    def handle_user_speech(self, transcription):
        """Called when the user finishes speaking — uses streaming TTS."""
        # Don't process if Sara is currently speaking
        if self.is_speaking:
            return

        self.last_interaction_time = time.time()

        print(f"\n🎤 You: {transcription}")

        # --- Streaming path ---
        # Pause STT first so we don't hear ourselves
        self.is_speaking = True
        self.stt.pause()

        try:
            # Stream LLM tokens → speak each sentence chunk as it arrives
            chunks = self.brain.generate_response_streaming(
                user_input=transcription,
                emotional_state="neutral",
            )
            full_response = self.tts.speak_stream(chunks)
            print(f"💬 Sara: {full_response}")
        except Exception as e:
            print(f"❌ Streaming error: {e}")
        finally:
            self.is_speaking = False
            self.stt.resume()
            self.last_interaction_time = time.time()

    def _speak(self, text):
        """Speak Sara's response (blocks audio input while speaking)."""
        self.is_speaking = True
        self.stt.pause()  # Stop listening to avoid hearing ourselves
        try:
            self.tts.speak(text)
        except Exception as e:
            print(f"❌ Error speaking: {e}")
        finally:
            self.is_speaking = False
            self.stt.resume()  # Start listening again
            self.last_interaction_time = time.time()

    def _monitor_silence(self):
        """
        Background thread: if the user is silent for too long,
        Sara can proactively check in.
        """
        while self.is_active:
            time.sleep(5)  # check every 5 seconds

            if self.is_speaking:
                continue

            silence = time.time() - self.last_interaction_time

            # If silent for 30+ seconds, check in (once)
            if silence > self.proactive_cooldown:
                self.last_interaction_time = time.time()  # reset
                self.proactive_cooldown = 120  # longer cooldown after first check

                print("\n💭 [Sara notices the silence...]")

                response = self.brain.generate_response(
                    user_input="[The user has been silent for a while]",
                    emotional_state="neutral",
                )

                print(f"💬 Sara: {response}")
                self._speak(response)

    def start(self):
        """Start the full voice conversation loop."""
        self.is_active = True
        self.last_interaction_time = time.time()

        print("=" * 60)
        print("  ✨ SARA IS NOW ACTIVE ✨")
        print("=" * 60)
        print()
        print("  🎧 Listening for your voice...")
        print("  💡 Speak naturally. Sara responds when you pause.")
        print("  🛑 Press Ctrl+C to stop.")
        print()

        # Start silence monitor in background
        self._silence_thread = threading.Thread(
            target=self._monitor_silence, daemon=True
        )
        self._silence_thread.start()

        # Start listening (blocking)
        try:
            # Proactive Greeting (from ideal.md: "Don't wait passively")
            print("\n💬 Sara (Starting): Hello Sir, I'm online. How are you feeling right now?")
            self._speak("Hello Sir, I'm online. How are you feeling right now?")
            
            self.stt.start_listening()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Gracefully stop Sara."""
        print("\n\n🛑 Stopping Sara...")

        self.is_active = False
        self.stt.stop_listening()

        # Save conversation to markdown
        self.brain.memory.save_session_to_markdown()

        print("\n💙 Sara: Until next time, Sir. Take care.")
        print("\n✓ Session saved to conversations/\n")


if __name__ == "__main__":
    sara = SaraAI()
    sara.start()
