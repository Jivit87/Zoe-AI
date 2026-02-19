"""
Test: Streaming TTS Pipeline
==============================
Tests that Sara's streaming path works correctly — without a microphone.

Sends a text prompt to the brain's streaming generator, prints each chunk
as it arrives with timing info, and plays it via TTS.

Usage:
    python test_streaming.py

You should see chunks printed and played one at a time, with the first
audio starting significantly before the full response is ready.
"""

import time
from src.llm.sara_brain import SaraBrain
from src.tts.voice_generator import VoiceGenerator

TEST_PROMPTS = [
    "Tell me something interesting about the universe.",
    "I'm feeling a bit stressed today.",
    "What do you think about rainy days?",
]


def test_streaming(prompt: str):
    brain = SaraBrain()
    tts = VoiceGenerator()

    print(f"\n{'=' * 60}")
    print(f"  Prompt: \"{prompt}\"")
    print(f"{'=' * 60}\n")

    t_start = time.time()
    print(f"[{0.00:.2f}s] Sent to Groq (streaming)...\n")

    first_chunk_time = None
    chunk_index = 0
    full_text_parts = []

    for chunk in brain.generate_response_streaming(prompt):
        chunk = chunk.strip()
        if not chunk:
            continue

        chunk_index += 1
        elapsed = time.time() - t_start
        full_text_parts.append(chunk)

        if first_chunk_time is None:
            first_chunk_time = elapsed
            print(f"[{elapsed:.2f}s] ▶ First chunk ready (TTS starts NOW):")

        print(f"         Chunk {chunk_index}: \"{chunk}\"")
        print(f"         Playing...")

        tts.speak(chunk)

        print(f"         ✓ Done playing chunk {chunk_index}")

    total_time = time.time() - t_start
    full_response = " ".join(full_text_parts)

    print(f"\n{'─' * 60}")
    print(f"💬 Sara: {full_response}")
    print(f"{'─' * 60}")
    print(f"  ⏱  Time to FIRST audio : {first_chunk_time:.2f}s")
    print(f"  ⏱  Total response time : {total_time:.2f}s")
    print(f"  📝 Chunks spoken       : {chunk_index}")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    print("\n🔊 Sara Streaming TTS Test")
    print("   (no microphone needed — text mode)\n")

    # Run one test prompt
    test_streaming(TEST_PROMPTS[0])

    print("\nDone! Try other prompts by editing TEST_PROMPTS in this file.\n")
