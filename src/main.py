"""
Sara AI — Full Voice Conversation Loop (Enhanced)
====================================================
The complete pipeline:
  🎤 Hear → 📝 Transcribe → 🧠 Think → 💾 Remember → 🔊 Speak

Enhancements:
  - Conversation state machine (IDLE/LISTENING/PROCESSING/SPEAKING/INTERRUPTED)
  - Barge-in: user can interrupt Sara mid-speech
  - Thinking sounds: filler audio while LLM generates
  - Progressive silence: multi-tier proactive check-ins
  - Emotion detection: text + speech timing → dynamic LLM adaptation
  - Hysteresis VAD: reduced false positives
  - Backchannel filtering: "yeah"/"mm-hmm" don't trigger new turns

Usage:
    python -m src.main

Press Ctrl+C to stop.
"""

import random
import threading
import time

from src.stt.speech_recognizer import SpeechRecognizer
from src.llm.sara_brain import SaraBrain
from src.tts.voice_generator import VoiceGenerator
from src.conversation.state_machine import StateManager, ConversationState
from src.conversation.barge_in import BargeInDetector
from src.conversation.backchannel import BackchannelClassifier
from src.emotion.emotion_detector import EmotionDetector


# Progressive silence tiers — (seconds_threshold, response_style)
SILENCE_TIERS = [
    (15, "light"),     # 15s  → light check-in
    (30, "warm"),      # 30s  → warmer check-in
    (60, "patient"),   # 60s  → patient presence
    (120, "pause"),    # 120s → assume session pause
]

# Context-aware silence responses
SILENCE_RESPONSES = {
    "light": {
        "distressed": "It's okay to take a moment.",
        "excited": "What are you thinking?",
        "default": "Take your time.",
    },
    "warm": {
        "distressed": "I'm right here with you.",
        "excited": "Still buzzing about it?",
        "default": "Hey, still here with you.",
    },
    "patient": {
        "distressed": "Whenever you're ready. No rush at all.",
        "excited": "I'm here when you want to pick back up.",
        "default": "Whenever you're ready, I'm listening.",
    },
}


class SaraAI:
    """Enhanced voice-to-voice conversation with natural turn-taking."""

    def __init__(self):
        print()
        print("✨ Initializing Sara AI...\n")

        # ─── Core components ─────────────────────────────────────────
        self.brain = SaraBrain()
        self.stt = SpeechRecognizer(model_size="small", device="cpu")
        self.tts = VoiceGenerator()

        # ─── Conversation management ─────────────────────────────────
        self.state = StateManager()
        self.barge_in = BargeInDetector()
        self.backchannel = BackchannelClassifier()
        self.emotion = EmotionDetector()

        # Wire TTS cancel event to state manager
        self.tts.cancel_event = self.state.cancel_event

        # Wire STT to conversation components
        self.stt.state_manager = self.state
        self.stt.barge_in_detector = self.barge_in
        self.stt.backchannel_classifier = self.backchannel
        self.stt.on_speech_detected = self.handle_user_speech
        self.stt.on_barge_in = self.handle_barge_in

        # ─── State ───────────────────────────────────────────────────
        self.is_active = False
        self._silence_thread = None
        self._last_silence_tier = -1  # Track which tier was last triggered
        self._last_emotional_state = "neutral"

        print("\n✓ Sara AI ready!\n")

    # ─── User speech handling ────────────────────────────────────────

    def handle_user_speech(self, transcription: str, audio_duration: float = 0):
        """Called when the user finishes speaking — streaming + thinking sounds."""
        # Don't process if Sara is currently speaking
        if self.state.is_speaking:
            return

        self.state.touch()
        self._last_silence_tier = -1  # Reset silence tiers

        print(f"\n🎤 You: {transcription}")

        # Detect emotion from text + speech timing
        word_count = len(transcription.split())
        emotion_result = self.emotion.analyze(transcription, audio_duration, word_count)
        emotional_state = emotion_result["state"]
        self._last_emotional_state = emotional_state

        if emotional_state != "neutral":
            print(f"   💡 Detected: {emotional_state}")

        # Transition to PROCESSING
        self.state.transition(ConversationState.PROCESSING)

        try:
            # Play a thinking sound to eliminate dead silence
            self.tts.play_thinking_sound()

            # Index user message into RAG memory
            self.brain.rag.remember(
                speaker="user",
                text=transcription,
                emotional_state=emotional_state,
            )

            # Transition to SPEAKING
            self.state.transition(ConversationState.SPEAKING)
            self.barge_in.on_tts_start()

            # Stream LLM tokens → speak each sentence chunk as it arrives
            chunks = self.brain.generate_response_streaming(
                user_input=transcription,
                emotional_state=emotional_state,
            )
            result = self.tts.speak_stream(chunks)

            self.barge_in.on_tts_stop()

            # Collect full response text for RAG indexing
            full_response = result['full_text'] if not result['interrupted'] else result['spoken']

            if result["interrupted"]:
                print(f"💬 Sara: {result['spoken']} [interrupted]")
                self.state.store_interruption_context(
                    result["spoken"], result["remaining"]
                )
            else:
                print(f"💬 Sara: {result['full_text']}")
                self.state.clear_interruption_context()

            # Index Sara's response into RAG memory
            if full_response.strip():
                self.brain.rag.remember(
                    speaker="sara",
                    text=full_response.strip(),
                    emotional_state=emotional_state,
                )

        except Exception as e:
            print(f"❌ Streaming error: {e}")
        finally:
            self.state.transition(ConversationState.IDLE)
            self.state.touch()

    # ─── Barge-in handling ───────────────────────────────────────────

    def handle_barge_in(self):
        """Called by SpeechRecognizer when a genuine barge-in is detected."""
        print("\n⚡ [Barge-in detected — stopping Sara]")
        self.state.transition(ConversationState.INTERRUPTED)
        self.barge_in.on_tts_stop()

    # ─── Simple speak (for greeting / proactive) ─────────────────────

    def _speak(self, text):
        """Speak Sara's response — simple blocking mode (greeting/proactive)."""
        self.state.transition(ConversationState.SPEAKING)
        self.stt.pause()  # Full pause for greeting only
        try:
            self.tts.speak(text)
        except Exception as e:
            print(f"❌ Error speaking: {e}")
        finally:
            self.state.transition(ConversationState.IDLE)
            self.stt.resume()
            self.state.touch()

    # ─── Progressive silence monitoring ──────────────────────────────

    def _monitor_silence(self):
        """
        Background thread: multi-tier progressive silence handling.

        Tiers:
          15s  → Light check-in ("Take your time.")
          30s  → Warm check-in ("Hey, still here with you.")
          60s  → Patient presence ("Whenever you're ready.")
          120s → Assume session pause, stop checking.
        """
        while self.is_active:
            time.sleep(3)  # Check every 3 seconds

            if self.state.is_speaking:
                continue

            silence = self.state.get_silence_duration()

            # Find the highest tier we should trigger
            for tier_index, (threshold, style) in enumerate(SILENCE_TIERS):
                if silence >= threshold and tier_index > self._last_silence_tier:
                    self._last_silence_tier = tier_index

                    if style == "pause":
                        # Session pause — stop actively checking
                        print("\n💤 [Session pause — Sara is resting]")
                        break

                    # Get context-aware response
                    responses = SILENCE_RESPONSES.get(style, SILENCE_RESPONSES["light"])
                    emotional_key = self._last_emotional_state if self._last_emotional_state in responses else "default"
                    response = responses.get(emotional_key, responses["default"])

                    print(f"\n💭 [Sara notices the silence... ({style})]")
                    print(f"💬 Sara: {response}")
                    self._speak(response)
                    break  # Only trigger one tier per check

    # ─── Main loop ───────────────────────────────────────────────────

    def start(self):
        """Start the full voice conversation loop."""
        self.is_active = True
        self.state.touch()

        print("=" * 60)
        print("  ✨ SARA IS NOW ACTIVE ✨")
        print("=" * 60)
        print()
        print("  🎧 Listening for your voice...")
        print("  💡 Speak naturally. Sara responds when you pause.")
        print("  ⚡ Interrupt Sara anytime — she'll stop and listen.")
        print("  🛑 Press Ctrl+C to stop.")
        print()

        # Start silence monitor in background
        self._silence_thread = threading.Thread(
            target=self._monitor_silence, daemon=True
        )
        self._silence_thread.start()

        # Start listening (blocking)
        try:
            # Proactive Greeting
            greeting = "Hello Sir, I'm online. How are you feeling right now?"
            print(f"\n💬 Sara (Starting): {greeting}")
            self._speak(greeting)

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

        # Flush RAG session — creates long-term session summary
        self.brain.rag.flush_session()

        print("\n💙 Sara: Until next time, Sir. Take care.")
        print("\n✓ Session saved to conversations/\n")


if __name__ == "__main__":
    sara = SaraAI()
    sara.start()
