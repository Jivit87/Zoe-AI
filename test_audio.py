"""
Day 1-2 Test: Verify Audio I/O (Microphone + Speakers)
========================================================
Records 3 seconds from your microphone, saves to test_recording.wav,
and plays it back to verify both input and output work.

Usage:
    python test_audio.py
"""

import sounddevice as sd
import soundfile as sf
import numpy as np

DURATION = 3       # seconds
SAMPLE_RATE = 16000  # 16kHz (what Whisper expects)
OUTPUT_FILE = "test_recording.wav"

# ── Record ──────────────────────────────────────────
print(f"🎤 Recording {DURATION} seconds... Speak now!")
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
)
sd.wait()
print("✓ Recording complete!")

# ── Save ────────────────────────────────────────────
sf.write(OUTPUT_FILE, audio, SAMPLE_RATE)
print(f"✓ Saved to {OUTPUT_FILE}")

# ── Playback ────────────────────────────────────────
print("🔊 Playing back...")
data, fs = sf.read(OUTPUT_FILE)
sd.play(data, fs)
sd.wait()
print("✓ Playback complete!")

print(f"\n────────────────────────────────────")
print(f"✓ Audio I/O works!")
print(f"  Sample rate : {SAMPLE_RATE} Hz")
print(f"  Duration    : {DURATION}s")
print(f"  File        : {OUTPUT_FILE}")
print(f"────────────────────────────────────")
