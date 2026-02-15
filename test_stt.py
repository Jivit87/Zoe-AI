"""
Day 3-4 Test: Real-time Speech-to-Text
========================================
Listens to your microphone, detects when you speak,
and transcribes your speech in real-time.

Usage:
    python test_stt.py

Press Ctrl+C to stop.
"""

from src.stt.speech_recognizer import SpeechRecognizer


def on_speech(text):
    print(f"\n🎤 You said: {text}\n")


if __name__ == "__main__":
    print("=" * 50)
    print("  Sara STT Test — Speak and see your words!")
    print("=" * 50)
    print()

    recognizer = SpeechRecognizer(model_size="base", device="cpu")
    recognizer.on_speech_detected = on_speech

    print()
    print("💡 Speak naturally. Sara will transcribe when you pause.")
    print("   Press Ctrl+C to stop.\n")

    try:
        recognizer.start_listening()
    except KeyboardInterrupt:
        recognizer.stop_listening()
        print("\n👋 Stopped listening. STT test complete!")
