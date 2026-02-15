"""
Day 1-2 Test: Verify Groq API Connection
==========================================
Tests that the Groq API key is valid and llama-3.3-70b-versatile responds
with Sara's personality.

Usage:
    python test_groq.py
"""

from groq import Groq
from dotenv import load_dotenv
import os
import time

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key or api_key == "your_groq_api_key_here":
    print("❌ Error: GROQ_API_KEY not set!")
    print("   1. Get a free key at https://console.groq.com")
    print("   2. Paste it into .env file")
    exit(1)

client = Groq(api_key=api_key)

print("🔗 Testing Groq API connection...\n")

start_time = time.time()

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": (
                "You are Sara, an emotionally intelligent AI companion. "
                "You address the user as 'Sir' affectionately. "
                "Keep your response to 2-3 sentences."
            ),
        },
        {"role": "user", "content": "Hello Sara, introduce yourself."},
    ],
    temperature=0.85,
    max_tokens=200,
)

elapsed_ms = (time.time() - start_time) * 1000

sara_reply = response.choices[0].message.content
model_used = response.model

print(f"Sara: {sara_reply}")
print(f"\n────────────────────────────────────")
print(f"✓ Groq API connection works!")
print(f"  Model : {model_used}")
print(f"  Latency: {elapsed_ms:.0f}ms")
print(f"────────────────────────────────────")
