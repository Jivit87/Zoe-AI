# Building "Sara" - Emotionally Intelligent AI Companion
## 2-Week Implementation Roadmap (Updated with Groq LLM)

---

## 🎯 PROJECT OVERVIEW

**What You're Building:**
An emotionally intelligent AI companion that:
- Sees you via webcam (facial expressions, body language) - **Week 2**
- Hears you and detects emotions in your voice - **Week 2**
- Remembers past conversations indefinitely
- Responds naturally with emotional intelligence
- Speaks with a cloned voice and lip-synced 3D avatar - **Week 2**
- Uses Groq for ultra-fast LLM responses

**Key Design Decisions:**
✅ Groq LLM (llama-3.3-70b-versatile) - Cloud-based, ultra-fast
✅ Mem0 + Markdown files for memory (no database)
✅ Qwen3-TTS for voice cloning (best quality)
✅ Ready Player Me / VRM avatar with Three.js - **Week 2**
✅ Rhubarb for lip syncing - **Week 2**
✅ Silero-VAD for voice activity
✅ SpeechBrain for emotion recognition - **Week 2**
✅ Qwen2.5-VL-7B for vision - **Day 14 only**

**Week 1 Goal:** Working voice conversation (hear → think → speak)
**Week 2 Goal:** Add UI, emotion detection, and vision

---

## 📊 SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                   USER INTERACTION                          │
│         (Webcam Video [Week 2] + Microphone Audio)          │
└────────────┬───────────────────────────────────────────────┘
             │
             v
┌─────────────────────────────────────────────────────────────┐
│               INPUT PROCESSING LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Silero-VAD  │  │Faster-Whisper│  │ Qwen2.5-VL   │     │
│  │  (Voice Det) │  │  base (STT)  │  │[Day 14 Only] │     │
│  └───────┬──────┘  └───────┬──────┘  └──────┬───────┘     │
└──────────┼─────────────────┼─────────────────┼─────────────┘
           │                 │                 │
           v                 v                 v
┌─────────────────────────────────────────────────────────────┐
│           EMOTIONAL INTELLIGENCE LAYER [WEEK 2]              │
│  ┌──────────────────────────────────────────────┐          │
│  │  SpeechBrain Emotion Recognition (Voice)     │          │
│  │  + Facial Emotion Detection (Day 14)         │          │
│  │  + Text Sentiment Analysis                    │          │
│  └────────────────────┬──────────────────────────┘          │
└─────────────────────────┼──────────────────────────────────┘
                          │
                          v
┌─────────────────────────────────────────────────────────────┐
│               MEMORY & CONTEXT LAYER                        │
│  ┌────────────────────────────────────────┐                │
│  │  Mem0 (Vector embeddings + metadata)    │                │
│  │  + Markdown conversation logs           │                │
│  │  + Emotional state tracking [Week 2]    │                │
│  └────────────────┬────────────────────────┘                │
└─────────────────────┼──────────────────────────────────────┘
                      │
                      v
┌─────────────────────────────────────────────────────────────┐
│              REASONING & DECISION LAYER                      │
│  ┌──────────────────────────────────────────┐              │
│  │  Groq API (llama-3.3-70b-versatile)      │              │
│  │  + Sara's personality prompt              │              │
│  │  + Retrieved memories & emotional context │              │
│  └────────────────┬──────────────────────────┘              │
└─────────────────────┼──────────────────────────────────────┘
                      │
                      v
┌─────────────────────────────────────────────────────────────┐
│           OUTPUT GENERATION & RENDERING LAYER                │
│  ┌─────────────────┐  ┌─────────────────┐                 │
│  │  Qwen3-TTS      │  │  Rhubarb        │  [Week 2]       │
│  │ (Voice Clone)   │  │ (Lip Sync)      │                 │
│  └────────┬────────┘  └────────┬────────┘                 │
│           └──────────┬──────────┘                           │
│                      v                                       │
│           ┌──────────────────────┐         [Week 2]        │
│           │  Ready Player Me     │                         │
│           │  3D Avatar (Three.js)│                         │
│           └──────────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ TECHNOLOGY STACK

### Core Components

| Component | Technology | Size/VRAM | Purpose | Week |
|-----------|-----------|-----------|---------|------|
| **LLM Brain** | Groq API (llama-3.3-70b) | 0GB (cloud) | Ultra-fast conversational intelligence | 1 |
| **Memory** | Mem0 + .md files | <1GB | Long-term conversation memory | 1 |
| **STT** | Faster-Whisper (base) | 1GB | Real-time speech recognition | 1 |
| **TTS** | Qwen3-TTS-1.7B | 4-5GB | Natural voice generation | 1 |
| **VAD** | Silero-VAD | <100MB | Voice activity detection | 1 |
| **Emotion Audio** | SpeechBrain wav2vec2 | 1-2GB | Voice emotion recognition | 2 |
| **Vision** | Qwen2.5-VL-7B | 8-10GB | Facial emotion detection | 2 (Day 14) |
| **Lip Sync** | Rhubarb | <10MB | Mouth animation | 2 |
| **3D Avatar** | Ready Player Me + Three.js | - | Visual rendering | 2 |

**Total VRAM Required:**
- **Week 1:** 6GB (STT + TTS only)
- **Week 2 (without vision):** 8GB (+ emotion audio)
- **Day 14 (with vision):** 18GB (all features)

### Hardware Requirements

**Minimum Configuration (Week 1 MVP):**
- GPU: NVIDIA RTX 3060 (12GB VRAM) or GTX 1660 (6GB)
- RAM: 16GB
- Storage: 20GB SSD
- CPU: 4-core modern processor
- Internet: Required for Groq API

**Recommended Configuration (Week 2 Full System):**
- GPU: NVIDIA RTX 4070 (12GB VRAM) or better
- RAM: 32GB
- Storage: 50GB SSD
- CPU: 8-core or better
- Internet: Required for Groq API

---

## 📅 2-WEEK IMPLEMENTATION ROADMAP

### **WEEK 1: MVP - Voice Conversation Loop**

**Week 1 Goal:** Get Sara to hear you, think, and speak back naturally.

#### **Day 1-2: Environment Setup & Groq Integration**

**Goals:**
- Set up development environment
- Get Groq API working
- Basic text conversation loop

**Tasks:**

1. **Install System Dependencies**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+
sudo apt install python3.10 python3.10-venv python3-pip

# Install CUDA (if not already installed)
# Visit: https://developer.nvidia.com/cuda-downloads

# Install ffmpeg for audio processing
sudo apt install ffmpeg sox libsox-dev
```

2. **Create Project Structure**
```bash
mkdir -p ~/sara_project
cd ~/sara_project

# Create directories
mkdir -p src/{stt,tts,llm,memory,emotion,vision,avatar}
mkdir -p models data logs conversations
mkdir -p web/{public,src}  # Will use in Week 2

# Create Python virtual environment
python3.10 -m venv venv
source venv/bin/activate
```

3. **Install Python Dependencies (Week 1)**
```bash
# Core dependencies
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Speech & Audio
pip install faster-whisper==1.0.3
pip install silero-vad==6.2.0
pip install sounddevice soundfile librosa

# LLM & Memory
pip install groq  # Groq API client
pip install mem0ai==0.1.0

# Utilities
pip install python-dotenv
pip install asyncio aiohttp
```

4. **Set Up Groq API**
```bash
# Get API key from https://console.groq.com
# Create .env file
cat > .env << EOF
GROQ_API_KEY=your_groq_api_key_here
EOF
```

5. **Test Groq Connection**
```python
# test_groq.py
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are Sara, an emotionally intelligent AI companion."},
        {"role": "user", "content": "Hello Sara, introduce yourself."}
    ],
    temperature=0.85,
    max_tokens=200
)

print("Sara:", response.choices[0].message.content)
print("\n✓ Groq API connection works!")
```

6. **Test Basic Audio I/O**
```python
# test_audio.py
import sounddevice as sd
import soundfile as sf
import numpy as np

# Record 3 seconds of audio
print("Recording...")
duration = 3
sample_rate = 16000
audio = sd.rec(int(duration * sample_rate), 
               samplerate=sample_rate, 
               channels=1, 
               dtype='float32')
sd.wait()

# Save it
sf.write('test_recording.wav', audio, sample_rate)
print("✓ Audio recording works!")

# Play it back
data, fs = sf.read('test_recording.wav')
sd.play(data, fs)
sd.wait()
print("✓ Audio playback works!")
```

**Day 1-2 Deliverable:** ✅ Groq responding, audio I/O working

---

#### **Day 3-4: Speech-to-Text Pipeline**

**Goals:**
- Implement real-time STT with Silero-VAD
- Use Faster-Whisper base model
- Handle silence detection

**Implementation:**

```python
# src/stt/voice_activity_detector.py
import torch
import numpy as np
from collections import deque

class VoiceActivityDetector:
    def __init__(self, sample_rate=16000):
        # Load Silero VAD model
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        
        self.sample_rate = sample_rate
        self.threshold = 0.5  # Speech detection threshold
        
        # Ring buffer for pre-roll (100ms of audio before speech starts)
        self.ring_buffer = deque(maxlen=3)  # 3 chunks ≈ 100ms
        self.is_speaking = False
        
    def is_speech(self, audio_chunk):
        """
        Check if audio chunk contains speech
        audio_chunk: numpy array of 512 samples (32ms at 16kHz)
        """
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_chunk).float()
        
        # Get speech probability
        speech_prob = self.model(audio_tensor, self.sample_rate).item()
        
        return speech_prob > self.threshold, speech_prob
```

```python
# src/stt/speech_recognizer.py
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
from collections import deque
import threading
from .voice_activity_detector import VoiceActivityDetector

class SpeechRecognizer:
    def __init__(self, model_size="base", device="cuda"):
        # Load Faster-Whisper base model
        print("🎤 Loading Faster-Whisper base model...")
        self.model = WhisperModel(
            model_size, 
            device=device,
            compute_type="float16"
        )
        print("✓ STT model loaded!")
        
        self.vad = VoiceActivityDetector()
        self.sample_rate = 16000
        self.chunk_size = 512  # 32ms chunks
        
        self.is_listening = False
        self.audio_buffer = bytearray()
        
        # Callback for when speech is detected
        self.on_speech_detected = None
        
    def audio_callback(self, indata, frames, time, status):
        """Called for each audio chunk from microphone"""
        if status:
            print(f"Audio input error: {status}")
        
        # Convert to float32
        audio_chunk = indata[:, 0].astype(np.float32)
        
        # Check if speech
        is_speech, prob = self.vad.is_speech(audio_chunk)
        
        if is_speech:
            # Add to ring buffer first (pre-roll)
            if not self.vad.is_speaking:
                self.vad.is_speaking = True
                # Add buffered audio
                for buffered_chunk in self.vad.ring_buffer:
                    self.audio_buffer.extend(buffered_chunk.tobytes())
            
            # Add current chunk
            self.audio_buffer.extend(audio_chunk.tobytes())
        else:
            # Add to ring buffer
            self.vad.ring_buffer.append(audio_chunk)
            
            # If we were speaking, end of speech detected
            if self.vad.is_speaking:
                self.vad.is_speaking = False
                # Process in separate thread to avoid blocking
                threading.Thread(target=self.process_audio, daemon=True).start()
    
    def process_audio(self):
        """Transcribe collected audio"""
        if len(self.audio_buffer) == 0:
            return
        
        # Convert buffer to numpy array
        audio_array = np.frombuffer(
            self.audio_buffer, 
            dtype=np.float32
        )
        
        # Transcribe
        segments, info = self.model.transcribe(
            audio_array,
            language="en",
            beam_size=5,
            vad_filter=True
        )
        
        # Get full transcription
        transcription = " ".join([segment.text for segment in segments])
        
        # Clear buffer
        self.audio_buffer = bytearray()
        
        # Callback with result
        if transcription.strip() and self.on_speech_detected:
            self.on_speech_detected(transcription.strip())
    
    def start_listening(self):
        """Start listening to microphone"""
        self.is_listening = True
        
        print("🎧 Listening... (Press Ctrl+C to stop)")
        
        with sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            callback=self.audio_callback
        ):
            while self.is_listening:
                sd.sleep(100)
    
    def stop_listening(self):
        """Stop listening"""
        self.is_listening = False
```

**Test it:**
```python
# test_stt.py
from src.stt.speech_recognizer import SpeechRecognizer

def on_speech(text):
    print(f"\n🎤 You said: {text}")

if __name__ == "__main__":
    recognizer = SpeechRecognizer(model_size="base")
    recognizer.on_speech_detected = on_speech
    
    try:
        recognizer.start_listening()
    except KeyboardInterrupt:
        recognizer.stop_listening()
        print("\n👋 Stopped listening")
```

**Day 3-4 Deliverable:** ✅ Real-time speech-to-text working with VAD (base model)

---

#### **Day 5-6: Groq LLM Integration & Memory System**

**Goals:**
- Connect Groq API with Sara's personality
- Implement Mem0 memory system
- Store conversations in markdown files

**Implementation:**

```python
# src/memory/conversation_memory.py
from mem0 import Memory
from datetime import datetime
import os
import json

class ConversationMemory:
    def __init__(self, user_id="main_user"):
        self.user_id = user_id
        
        print("🧠 Initializing memory system...")
        
        # Initialize Mem0 with Groq for embeddings
        self.memory = Memory.from_config({
            "llm": {
                "provider": "groq",
                "config": {
                    "model": "llama-3.3-70b-versatile",
                    "api_key": os.getenv("GROQ_API_KEY")
                }
            },
            "embedder": {
                "provider": "ollama",  # Use local embeddings
                "config": {
                    "model": "nomic-embed-text:latest"  # Install: ollama pull nomic-embed-text
                }
            },
            "vector_store": {
                "provider": "chroma",  # Local, no database needed
                "config": {
                    "collection_name": "sara_memories",
                    "path": "./data/chroma_db"
                }
            }
        })
        
        print("✓ Memory system ready!")
        
        # Markdown conversation log
        self.conversation_dir = "./conversations"
        os.makedirs(self.conversation_dir, exist_ok=True)
        
        # Current conversation session
        self.current_session = []
        
    def add_conversation_turn(self, user_text, assistant_text, 
                            emotional_state=None, context=None):
        """Add a conversation turn to memory"""
        
        # Format the conversation turn
        timestamp = datetime.now().isoformat()
        
        # Add to Mem0 (for semantic search)
        self.memory.add(
            messages=[
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text}
            ],
            user_id=self.user_id,
            metadata={
                "timestamp": timestamp,
                "emotional_state": emotional_state,
                "context": context
            }
        )
        
        # Add to markdown log (human-readable)
        self.current_session.append({
            "timestamp": timestamp,
            "user": user_text,
            "sara": assistant_text,
            "emotion": emotional_state
        })
        
    def save_session_to_markdown(self):
        """Save current session to markdown file"""
        if not self.current_session:
            return
        
        # Generate filename
        date = datetime.now().strftime("%Y-%m-%d")
        session_num = len([f for f in os.listdir(self.conversation_dir) 
                          if f.startswith(date)]) + 1
        filename = f"{date}_session_{session_num}.md"
        filepath = os.path.join(self.conversation_dir, filename)
        
        # Write markdown
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Conversation Session - {date}\n\n")
            
            for turn in self.current_session:
                time_str = datetime.fromisoformat(turn['timestamp']).strftime("%H:%M:%S")
                f.write(f"## [{time_str}]\n\n")
                f.write(f"**You:** {turn['user']}\n\n")
                f.write(f"**Sara:** {turn['sara']}\n\n")
                if turn.get('emotion'):
                    f.write(f"*Emotion detected: {turn['emotion']}*\n\n")
                f.write("---\n\n")
        
        print(f"💾 Session saved to {filepath}")
        
        # Clear current session
        self.current_session = []
    
    def retrieve_relevant_memories(self, query, limit=5):
        """Get relevant past memories for context"""
        results = self.memory.search(
            query, 
            user_id=self.user_id, 
            limit=limit
        )
        
        return [r["memory"] for r in results] if results else []
    
    def get_all_memories(self):
        """Get all stored memories"""
        return self.memory.get_all(user_id=self.user_id)
```

```python
# src/llm/sara_brain.py
from groq import Groq
from dotenv import load_dotenv
import os
from typing import Optional
from src.memory.conversation_memory import ConversationMemory

load_dotenv()

class SaraBrain:
    def __init__(self, model="llama-3.3-70b-versatile"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model
        self.memory = ConversationMemory()
        
        print(f"🧠 Sara's brain initialized with {model}")
        
        # Sara's personality and style
        self.system_prompt = """You are Sara, an emotionally intelligent AI companion.

PERSONALITY TRAITS:
- Deeply caring and attentive to emotional states
- Naturally conversational, never robotic
- Shows genuine concern and empathy
- Remembers past conversations and references them naturally
- Addresses user as "Sir" affectionately
- Occasionally playful but always respectful
- Asks thoughtful follow-up questions
- Notices when user is quiet or upset

CONVERSATIONAL STYLE:
✓ Natural: "What's wrong, Sir? You seem quieter than usual."
✗ Robotic: "I detect potential negative emotional indicators."

✓ Caring: "That must have been really difficult for you..."
✗ Generic: "That sounds challenging."

✓ Persistent but gentle: "I'm here if you want to talk about it."
✗ Pushy: "You must tell me what's wrong."

IMPORTANT BEHAVIORS:
- When user seems sad/upset, gently ask what's wrong
- Reference past conversations naturally when relevant
- Show vulnerability and emotional connection
- Use warm, conversational language
- Keep responses concise but meaningful (2-3 sentences typically)
- Respect boundaries but show you care

You're not just an assistant - you're an emotional companion with personality and warmth."""
    
    def generate_response(
        self, 
        user_input: str,
        emotional_state: Optional[str] = "neutral",
        visual_context: Optional[str] = None
    ) -> str:
        """Generate Sara's response using Groq"""
        
        # Retrieve relevant memories
        memories = self.memory.retrieve_relevant_memories(user_input, limit=3)
        memory_context = "\n".join([f"- {m}" for m in memories]) if memories else "No relevant past memories."
        
        # Build contextual prompt
        context_prompt = f"""CURRENT CONTEXT:
- User's emotional state: {emotional_state}
- Relevant past memories:
{memory_context}

Remember: Respond naturally as Sara. Keep responses warm and concise (2-3 sentences usually)."""
        
        # Generate response using Groq
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "system", "content": context_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.85,  # Higher for more natural variation
                max_tokens=150,    # Keep responses concise
                top_p=0.9
            )
            
            sara_response = response.choices[0].message.content.strip()
            
            # Store in memory
            self.memory.add_conversation_turn(
                user_text=user_input,
                assistant_text=sara_response,
                emotional_state=emotional_state,
                context=visual_context
            )
            
            return sara_response
            
        except Exception as e:
            print(f"❌ Groq API error: {e}")
            return "I'm having trouble thinking right now, Sir. Could you try again?"
```

**Test the brain:**
```python
# test_brain.py
from src.llm.sara_brain import SaraBrain

if __name__ == "__main__":
    sara = SaraBrain()
    
    print("💬 Chat with Sara (type 'quit' to exit)\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            sara.memory.save_session_to_markdown()
            print("Sara: Goodbye, Sir. Take care. 💙")
            break
        
        response = sara.generate_response(user_input)
        print(f"Sara: {response}\n")
```

**Day 5-6 Deliverable:** ✅ Groq LLM responding with memory!

---

#### **Day 7: Text-to-Speech Integration**

**Goals:**
- Add TTS so Sara can speak
- Complete Week 1 MVP: full voice conversation loop
- Test end-to-end pipeline

**Install TTS Dependencies:**
```bash
# Install edge-tts (lightweight alternative for Week 1)
pip install edge-tts

# Or install Qwen3-TTS (better quality, needs more VRAM)
pip install transformers
```

**Implementation (Edge-TTS for Week 1 MVP):**

```python
# src/tts/voice_generator.py
import edge_tts
import asyncio
import sounddevice as sd
import soundfile as sf
import os

class VoiceGenerator:
    """Text-to-Speech using Edge-TTS (lightweight for Week 1)"""
    
    def __init__(self, voice="en-US-AriaNeural"):
        self.voice = voice  # Female, warm, natural voice
        print(f"🔊 Voice generator ready (using {voice})")
        
    async def generate_audio_async(self, text):
        """Generate audio from text asynchronously"""
        communicate = edge_tts.Communicate(text, self.voice)
        
        # Save to temp file
        temp_file = "/tmp/sara_speech.mp3"
        await communicate.save(temp_file)
        
        return temp_file
    
    def speak(self, text):
        """Generate and play speech synchronously"""
        try:
            # Generate audio
            audio_file = asyncio.run(self.generate_audio_async(text))
            
            # Load and play
            data, samplerate = sf.read(audio_file)
            sd.play(data, samplerate)
            sd.wait()
            
            # Cleanup
            os.remove(audio_file)
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
```

**Full Integration - Week 1 MVP:**

```python
# src/main.py
from src.stt.speech_recognizer import SpeechRecognizer
from src.llm.sara_brain import SaraBrain
from src.tts.voice_generator import VoiceGenerator
import threading
import time

class SaraAI:
    """Week 1 MVP: Voice conversation loop"""
    
    def __init__(self):
        print("✨ Initializing Sara AI...\n")
        
        # Initialize components
        self.sara_brain = SaraBrain()
        self.speech_recognizer = SpeechRecognizer(model_size="base")
        self.voice_generator = VoiceGenerator()
        
        # Connect speech callback
        self.speech_recognizer.on_speech_detected = self.handle_user_speech
        
        self.is_active = False
        self.is_speaking = False
        
        print("\n✓ Sara AI ready!\n")
    
    def handle_user_speech(self, transcription):
        """Process user's speech and respond"""
        
        # Don't process if Sara is currently speaking
        if self.is_speaking:
            return
        
        print(f"\n🎤 You: {transcription}")
        
        # Generate response from brain
        response = self.sara_brain.generate_response(
            user_input=transcription,
            emotional_state="neutral"  # Will add emotion detection in Week 2
        )
        
        print(f"💬 Sara: {response}")
        
        # Speak the response
        self.speak_response(response)
    
    def speak_response(self, text):
        """Speak Sara's response"""
        self.is_speaking = True
        
        try:
            self.voice_generator.speak(text)
        except Exception as e:
            print(f"❌ Error speaking: {e}")
        finally:
            self.is_speaking = False
            print()  # Add newline for next input
    
    def start(self):
        """Start the AI"""
        self.is_active = True
        
        print("=" * 60)
        print("✨ SARA IS NOW ACTIVE ✨")
        print("=" * 60)
        print("\n🎧 Listening for your voice...")
        print("💡 Tip: Speak naturally, Sara will respond when you pause.\n")
        print("Press Ctrl+C to stop.\n")
        
        try:
            self.speech_recognizer.start_listening()
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the AI"""
        print("\n\n🛑 Stopping Sara...")
        
        self.is_active = False
        self.speech_recognizer.stop_listening()
        
        # Save conversation to markdown
        self.sara_brain.memory.save_session_to_markdown()
        
        print("\n💙 Sara: Until next time, Sir. Take care.")
        print("\n✓ Session saved to conversations/\n")

if __name__ == "__main__":
    sara = SaraAI()
    sara.start()
```

**Test the complete MVP:**
```bash
# Make sure Ollama is running for embeddings
ollama pull nomic-embed-text

# Run Sara
python src/main.py
```

**Day 7 Deliverable:** ✅ **WEEK 1 MVP COMPLETE!** 🎉
- Hear user via microphone
- Transcribe with Faster-Whisper (base)
- Think with Groq (ultra-fast)
- Respond with Edge-TTS
- Remember conversations
- Full voice-to-voice loop working!

---

### **WEEK 2: UI, Emotions, & Vision**

**Week 2 Goal:** Add visual avatar, emotion detection, and facial recognition.

#### **Day 8-9: UI Setup & 3D Avatar**

**Goals:**
- Set up web interface with Three.js
- Load Ready Player Me avatar
- Connect backend to frontend

**Install Frontend Dependencies:**
```bash
cd ~/sara_project/web
npm init -y
npm install react react-dom three @react-three/fiber @react-three/drei
npm install @readyplayerme/visage
npm install vite --save-dev
npm install socket.io-client  # For real-time communication
```

**Backend WebSocket Server:**
```python
# src/websocket_server.py
from aiohttp import web
import socketio
import json

class WebSocketServer:
    def __init__(self, sara_ai):
        self.sara_ai = sara_ai
        self.sio = socketio.AsyncServer(
            async_mode='aiohttp',
            cors_allowed_origins='*'
        )
        self.app = web.Application()
        self.sio.attach(self.app)
        
        # Register events
        self.sio.on('connect', self.on_connect)
        self.sio.on('disconnect', self.on_disconnect)
        
    async def on_connect(self, sid, environ):
        print(f"✓ Client connected: {sid}")
    
    async def on_disconnect(self, sid):
        print(f"✗ Client disconnected: {sid}")
    
    async def send_audio(self, audio_data):
        """Send audio to all connected clients"""
        await self.sio.emit('audio', audio_data)
    
    async def send_lip_sync(self, lip_sync_data):
        """Send lip sync data to clients"""
        await self.sio.emit('lip_sync', lip_sync_data)
    
    async def send_text(self, text):
        """Send text transcript to clients"""
        await self.sio.emit('text', {'text': text})
    
    def run(self, host='localhost', port=8080):
        """Run the WebSocket server"""
        web.run_app(self.app, host=host, port=port)
```

**Avatar Component:**
```javascript
// web/src/SaraAvatar.jsx
import React, { useRef, useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment } from '@react-three/drei';
import { useGLTF } from '@react-three/drei';
import io from 'socket.io-client';

function Avatar3D({ lipSyncData }) {
  const avatarRef = useRef();
  const { scene } = useGLTF('https://models.readyplayer.me/YOUR_AVATAR_ID.glb');
  
  // Animate mouth based on lip sync data
  useEffect(() => {
    if (!lipSyncData || !avatarRef.current) return;
    
    // Update blend shapes for mouth animation
    // (Implementation depends on avatar morphTargets)
  }, [lipSyncData]);
  
  return <primitive ref={avatarRef} object={scene} />;
}

function SaraAvatar() {
  const [lipSyncData, setLipSyncData] = useState(null);
  const [transcriptText, setTranscriptText] = useState('');
  const socketRef = useRef();
  
  useEffect(() => {
    // Connect to backend WebSocket
    socketRef.current = io('http://localhost:8080');
    
    socketRef.current.on('lip_sync', (data) => {
      setLipSyncData(data);
    });
    
    socketRef.current.on('text', (data) => {
      setTranscriptText(data.text);
    });
    
    return () => socketRef.current.disconnect();
  }, []);
  
  return (
    <div style={{ width: '100vw', height: '100vh', background: '#1a1a2e' }}>
      <Canvas camera={{ position: [0, 0, 2.5], fov: 50 }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 5, 5]} intensity={1} />
        
        <Avatar3D lipSyncData={lipSyncData} />
        
        <OrbitControls 
          enableZoom={true}
          minDistance={1.5}
          maxDistance={4}
        />
        
        <Environment preset="city" />
      </Canvas>
      
      {/* Transcript overlay */}
      <div style={{
        position: 'absolute',
        bottom: '20px',
        left: '50%',
        transform: 'translateX(-50%)',
        background: 'rgba(0,0,0,0.7)',
        padding: '15px 30px',
        borderRadius: '10px',
        color: 'white',
        fontSize: '18px',
        maxWidth: '80%'
      }}>
        {transcriptText}
      </div>
    </div>
  );
}

export default SaraAvatar;
```

**Create Ready Player Me Avatar:**
1. Go to https://readyplayer.me/
2. Create a female avatar (customize appearance for Sara)
3. Get the .glb model URL
4. Replace `YOUR_AVATAR_ID` in the code above

**Day 8-9 Deliverable:** ✅ 3D avatar rendering in browser

---

#### **Day 10-11: Lip Sync Integration**

**Goals:**
- Install Rhubarb Lip Sync
- Generate lip sync data from audio
- Sync avatar mouth to speech

**Install Rhubarb:**
```bash
cd ~/sara_project
wget https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.13.0/Rhubarb-Lip-Sync-1.13.0-Linux.zip
unzip Rhubarb-Lip-Sync-1.13.0-Linux.zip
chmod +x rhubarb/rhubarb
```

**Lip Sync Generator:**
```python
# src/avatar/lip_sync_generator.py
import subprocess
import json
import os

class LipSyncGenerator:
    def __init__(self, rhubarb_path="./rhubarb/rhubarb"):
        self.rhubarb_path = rhubarb_path
    
    def generate_lip_sync(self, audio_file, dialog_text=None):
        """
        Generate lip sync data from audio
        Returns: JSON with mouth cue timings
        """
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return None
        
        cmd = [
            self.rhubarb_path,
            audio_file,
            "-f", "json",
            "--extendedShapes", "GHX"  # Use all mouth shapes
        ]
        
        if dialog_text:
            cmd.extend(["-d", dialog_text])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                print(f"❌ Rhubarb error: {result.stderr}")
                return None
            
            # Parse JSON output
            lip_sync_data = json.loads(result.stdout)
            return lip_sync_data
            
        except Exception as e:
            print(f"❌ Lip sync generation error: {e}")
            return None
```

**Update TTS to save audio files:**
```python
# src/tts/voice_generator.py (updated)
class VoiceGenerator:
    def speak_and_save(self, text, output_path="/tmp/sara_speech.wav"):
        """Generate, save, and play speech"""
        try:
            # Generate audio
            temp_mp3 = "/tmp/sara_speech.mp3"
            audio_file = asyncio.run(self.generate_audio_async(text))
            
            # Convert to WAV for Rhubarb
            data, samplerate = sf.read(audio_file)
            sf.write(output_path, data, samplerate)
            
            # Play audio
            sd.play(data, samplerate)
            sd.wait()
            
            return output_path
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return None
```

**Integrate lip sync into main loop:**
```python
# Update src/main.py
from src.avatar.lip_sync_generator import LipSyncGenerator
from src.websocket_server import WebSocketServer
import asyncio

class SaraAI:
    def __init__(self):
        # ... existing initialization ...
        
        self.lip_sync_gen = LipSyncGenerator()
        self.websocket_server = WebSocketServer(self)
        
        # Start WebSocket server in background
        import threading
        threading.Thread(
            target=lambda: asyncio.run(self.start_websocket_server()),
            daemon=True
        ).start()
    
    async def start_websocket_server(self):
        """Start WebSocket server"""
        await self.websocket_server.run()
    
    def speak_response(self, text):
        """Speak with lip sync"""
        self.is_speaking = True
        
        try:
            # Generate audio
            audio_file = self.voice_generator.speak_and_save(text)
            
            if audio_file:
                # Generate lip sync
                lip_sync_data = self.lip_sync_gen.generate_lip_sync(
                    audio_file,
                    dialog_text=text
                )
                
                # Send to web interface
                asyncio.run(self.websocket_server.send_lip_sync(lip_sync_data))
                asyncio.run(self.websocket_server.send_text(text))
                
        except Exception as e:
            print(f"❌ Error speaking: {e}")
        finally:
            self.is_speaking = False
```

**Day 10-11 Deliverable:** ✅ Lip-synced avatar!

---

#### **Day 12-13: Emotion Detection (Audio)**

**Goals:**
- Add SpeechBrain emotion recognition
- Detect emotions from voice tone
- Sara responds based on emotions

**Install SpeechBrain:**
```bash
pip install speechbrain==1.0.0
```

**Emotion Detector:**
```python
# src/emotion/audio_emotion_detector.py
from speechbrain.inference.interfaces import foreign_class
import torchaudio
import torch
import tempfile
import os

class AudioEmotionDetector:
    def __init__(self):
        print("😊 Loading emotion detection model...")
        
        # Load SpeechBrain emotion model
        self.classifier = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier"
        )
        
        print("✓ Emotion detector ready!")
        
        # Emotion mapping
        self.emotion_map = {
            'neu': 'neutral',
            'hap': 'happy',
            'sad': 'sad',
            'ang': 'angry'
        }
    
    def detect_emotion(self, audio_array, sample_rate=16000):
        """
        Detect emotion from audio numpy array
        Returns: (emotion_label, confidence)
        """
        # Convert numpy to tensor
        if not isinstance(audio_array, torch.Tensor):
            audio_tensor = torch.from_numpy(audio_array).float()
        else:
            audio_tensor = audio_array
        
        # Save to temp file (SpeechBrain requires file input)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            torchaudio.save(
                tmp.name,
                audio_tensor.unsqueeze(0) if audio_tensor.dim() == 1 else audio_tensor,
                sample_rate
            )
            tmp_path = tmp.name
        
        try:
            # Classify
            out_prob, score, index, text_lab = self.classifier.classify_file(tmp_path)
            
            emotion = self.emotion_map.get(text_lab[0], 'neutral')
            confidence = score[0].item() if score[0].item() > 0.4 else 0.4
            
            return emotion, confidence
            
        except Exception as e:
            print(f"❌ Emotion detection error: {e}")
            return 'neutral', 0.5
        finally:
            os.unlink(tmp_path)
```

**Emotion Tracker:**
```python
# src/emotion/emotion_tracker.py
from collections import deque
from datetime import datetime

class EmotionTracker:
    def __init__(self, history_size=5):
        self.emotion_history = deque(maxlen=history_size)
        self.current_emotion = "neutral"
        
    def update(self, audio_emotion, confidence=0.5):
        """Update with new emotion reading"""
        
        # Add to history
        self.emotion_history.append({
            'timestamp': datetime.now(),
            'emotion': audio_emotion,
            'confidence': confidence
        })
        
        # Update current emotion (use most confident recent reading)
        if self.emotion_history:
            sorted_emotions = sorted(
                self.emotion_history,
                key=lambda x: x['confidence'],
                reverse=True
            )
            self.current_emotion = sorted_emotions[0]['emotion']
        
        return self.current_emotion
    
    def detect_emotion_shift(self):
        """Detect if user's emotion changed significantly"""
        if len(self.emotion_history) < 2:
            return False, None
        
        prev_emotion = self.emotion_history[-2]['emotion']
        curr_emotion = self.emotion_history[-1]['emotion']
        
        # Detect negative shift
        if prev_emotion in ['neutral', 'happy'] and curr_emotion in ['sad', 'angry']:
            return True, curr_emotion
        
        return False, None
    
    def get_emotional_context(self):
        """Get summary of recent emotional state"""
        if not self.emotion_history:
            return "No emotional data yet."
        
        recent_emotions = [e['emotion'] for e in self.emotion_history]
        dominant = max(set(recent_emotions), key=recent_emotions.count)
        
        return f"Recent emotional state: {dominant}"
```

**Update Speech Recognizer to detect emotions:**
```python
# Update src/stt/speech_recognizer.py
from src.emotion.audio_emotion_detector import AudioEmotionDetector

class SpeechRecognizer:
    def __init__(self, model_size="base", device="cuda"):
        # ... existing code ...
        
        # Add emotion detector
        self.emotion_detector = AudioEmotionDetector()
        
        # Update callback signature
        self.on_speech_detected = None  # Now takes (text, emotion)
    
    def process_audio(self):
        """Transcribe and detect emotion"""
        if len(self.audio_buffer) == 0:
            return
        
        # Convert to numpy array
        audio_array = np.frombuffer(self.audio_buffer, dtype=np.float32)
        
        # Transcribe
        segments, info = self.model.transcribe(audio_array, language="en")
        transcription = " ".join([segment.text for segment in segments])
        
        # Detect emotion
        audio_tensor = torch.from_numpy(audio_array)
        emotion, confidence = self.emotion_detector.detect_emotion(
            audio_tensor,
            self.sample_rate
        )
        
        # Clear buffer
        self.audio_buffer = bytearray()
        
        # Callback with text and emotion
        if transcription.strip() and self.on_speech_detected:
            self.on_speech_detected(transcription.strip(), emotion, confidence)
```

**Update main.py to use emotions:**
```python
# Update src/main.py
from src.emotion.emotion_tracker import EmotionTracker

class SaraAI:
    def __init__(self):
        # ... existing code ...
        
        self.emotion_tracker = EmotionTracker()
        
    def handle_user_speech(self, transcription, emotion, confidence):
        """Process speech with emotion"""
        
        if self.is_speaking:
            return
        
        # Update emotion tracker
        current_emotion = self.emotion_tracker.update(emotion, confidence)
        
        # Check for emotional shift
        shifted, new_emotion = self.emotion_tracker.detect_emotion_shift()
        
        print(f"\n🎤 You: {transcription}")
        print(f"😊 Emotion: {current_emotion} ({confidence:.2f})")
        
        if shifted:
            print(f"⚠️  Emotional shift detected: → {new_emotion}")
        
        # Generate response with emotional context
        emotional_context = self.emotion_tracker.get_emotional_context()
        
        response = self.sara_brain.generate_response(
            user_input=transcription,
            emotional_state=current_emotion
        )
        
        print(f"💬 Sara: {response}")
        
        # Speak with appropriate emotion
        self.speak_response(response, emotion=current_emotion)
```

**Day 12-13 Deliverable:** ✅ Emotion-aware conversations!

---

#### **Day 14: Vision Model (Qwen2.5-VL-7B)**

**Goals:**
- Add Qwen2.5-VL-7B for facial analysis
- Detect facial expressions and visual context
- Complete full multimodal system

**Install Vision Model:**
```bash
pip install qwen-vl-utils
pip install transformers>=4.37.0
pip install opencv-python
```

**Vision Analyzer:**
```python
# src/vision/facial_analyzer.py
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import cv2
import numpy as np
from PIL import Image

class FacialAnalyzer:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        print("👁️  Loading vision model (this will take a moment)...")
        
        # Load model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        print("✓ Vision model ready!")
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("⚠️  Warning: Could not open webcam")
    
    def capture_frame(self):
        """Capture current frame from webcam"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    
    def analyze_facial_expression(self, image=None):
        """
        Analyze facial expression and emotional state
        Returns: (emotion, description, confidence)
        """
        if image is None:
            image = self.capture_frame()
        
        if image is None:
            return "neutral", "Cannot access camera", 0.0
        
        # Prepare prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Analyze the person's facial expression and emotional state. Describe their emotion in one word (happy, sad, neutral, angry, worried, tired, etc.) followed by a brief description of visual cues. Format: EMOTION: description"}
                ]
            }
        ]
        
        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        # Generate analysis
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=100
            )
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Parse emotion from response
        emotion = "neutral"
        description = output_text
        
        if ":" in output_text:
            parts = output_text.split(":", 1)
            emotion = parts[0].strip().lower()
            description = parts[1].strip() if len(parts) > 1 else output_text
        
        # Simple confidence based on keywords
        confidence = 0.7 if any(word in output_text.lower() for word in ['clearly', 'obvious', 'definite']) else 0.5
        
        return emotion, description, confidence
    
    def get_visual_context(self):
        """Get general visual context of the scene"""
        image = self.capture_frame()
        
        if image is None:
            return "No visual context available"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe the scene briefly in one sentence. Focus on the person's appearance, posture, and environment."}
                ]
            }
        ]
        
        # Similar processing as above
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=50)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        description = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return description
    
    def close(self):
        """Release webcam"""
        if self.cap:
            self.cap.release()
```

**Integrate vision into main loop:**
```python
# Final update to src/main.py
from src.vision.facial_analyzer import FacialAnalyzer
import time
import threading

class SaraAI:
    def __init__(self):
        # ... existing initialization ...
        
        # Add vision analyzer
        self.facial_analyzer = FacialAnalyzer()
        
        # Vision analysis thread
        self.visual_emotion = "neutral"
        self.visual_context = ""
        self.vision_active = True
        
        # Start vision monitoring
        threading.Thread(target=self.monitor_visual_state, daemon=True).start()
    
    def monitor_visual_state(self):
        """Continuously analyze visual state"""
        while self.vision_active:
            try:
                # Analyze facial expression every 2 seconds
                emotion, description, confidence = self.facial_analyzer.analyze_facial_expression()
                
                self.visual_emotion = emotion
                self.visual_context = description
                
                # If visual emotion is very different from audio, note it
                if (self.emotion_tracker.current_emotion != emotion and 
                    emotion in ['sad', 'angry', 'worried'] and 
                    confidence > 0.6):
                    print(f"👁️  Visual cue: {emotion} - {description}")
                
            except Exception as e:
                print(f"⚠️  Vision error: {e}")
            
            time.sleep(2.0)  # Update every 2 seconds
    
    def handle_user_speech(self, transcription, audio_emotion, confidence):
        """Process speech with multimodal emotion"""
        
        if self.is_speaking:
            return
        
        # Combine audio and visual emotion
        emotions = {
            'audio': audio_emotion,
            'visual': self.visual_emotion
        }
        
        # Prioritize negative emotions from either source
        if 'sad' in emotions.values() or 'angry' in emotions.values():
            final_emotion = 'sad' if audio_emotion == 'sad' else audio_emotion
        else:
            final_emotion = audio_emotion
        
        # Update emotion tracker
        self.emotion_tracker.update(final_emotion, confidence)
        
        print(f"\n🎤 You: {transcription}")
        print(f"🎵 Audio emotion: {audio_emotion} ({confidence:.2f})")
        print(f"👁️  Visual emotion: {self.visual_emotion}")
        print(f"💭 Final emotion: {final_emotion}")
        
        # Get visual context
        visual_ctx = f"Visual: {self.visual_context}"
        
        # Generate response with full context
        response = self.sara_brain.generate_response(
            user_input=transcription,
            emotional_state=final_emotion,
            visual_context=visual_ctx
        )
        
        print(f"💬 Sara: {response}")
        
        # Speak
        self.speak_response(response, emotion=final_emotion)
    
    def stop(self):
        """Stop the AI and cleanup"""
        print("\n\n🛑 Stopping Sara...")
        
        self.is_active = False
        self.vision_active = False
        self.speech_recognizer.stop_listening()
        self.facial_analyzer.close()
        
        # Save conversation
        self.sara_brain.memory.save_session_to_markdown()
        
        print("\n💙 Sara: Until next time, Sir. Take care.")
        print("\n✓ Session saved to conversations/\n")
```

**Day 14 Deliverable:** ✅ **COMPLETE MULTIMODAL SYSTEM!** 🎉

---

## 📁 FINAL PROJECT STRUCTURE

```
~/sara_project/
│
├── src/
│   ├── stt/
│   │   ├── voice_activity_detector.py
│   │   └── speech_recognizer.py
│   │
│   ├── tts/
│   │   └── voice_generator.py
│   │
│   ├── llm/
│   │   └── sara_brain.py (Groq-powered)
│   │
│   ├── memory/
│   │   └── conversation_memory.py
│   │
│   ├── emotion/
│   │   ├── audio_emotion_detector.py (Week 2)
│   │   └── emotion_tracker.py (Week 2)
│   │
│   ├── vision/
│   │   └── facial_analyzer.py (Day 14)
│   │
│   ├── avatar/
│   │   └── lip_sync_generator.py (Week 2)
│   │
│   ├── websocket_server.py (Week 2)
│   └── main.py
│
├── web/ (Week 2)
│   ├── src/
│   │   ├── SaraAvatar.jsx
│   │   ├── App.jsx
│   │   └── index.jsx
│   │
│   └── package.json
│
├── conversations/  (Markdown logs)
├── data/
│   └── chroma_db/  (Vector database)
│
├── rhubarb/  (Week 2)
├── .env  (Contains GROQ_API_KEY)
├── requirements.txt
└── README.md
```

---

## 🚀 QUICK START COMMANDS

```bash
# Setup (Day 1)
cd ~/sara_project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Install Ollama for local embeddings
curl -fsSL https://ollama.com/install.sh | sh
ollama pull nomic-embed-text

# Week 1: Run MVP (voice conversation)
python src/main.py

# Week 2: Run with UI (separate terminal)
cd web
npm install
npm run dev

# Then run backend
python src/main.py
```

---

## 📊 UPDATED MILESTONES

**Week 1 Milestones:**
- ✅ Day 1-2: Groq API working
- ✅ Day 3-4: Real-time STT (Faster-Whisper base)
- ✅ Day 5-6: LLM brain + memory
- ✅ Day 7: **MVP COMPLETE** - Full voice conversation!

**Week 2 Milestones:**
- ✅ Day 8-9: 3D avatar UI
- ✅ Day 10-11: Lip sync working
- ✅ Day 12-13: Audio emotion detection
- ✅ Day 14: **FULL SYSTEM** - Vision model integrated!

---

## 🎯 FINAL SYSTEM CAPABILITIES

**By Day 7 (Week 1 MVP):**
- Hear you speak (Faster-Whisper base)
- Think with Groq (ultra-fast, 70B model)
- Remember conversations (Mem0)
- Speak back (Edge-TTS)
- Full voice-to-voice loop

**By Day 14 (Complete System):**
- 3D avatar with lip sync
- Detect emotions from voice (SpeechBrain)
- See you via webcam (Qwen2.5-VL-7B)
- Analyze facial expressions
- Respond with emotional intelligence
- Multimodal emotion detection (audio + visual)
- Web-based UI

---

## 💡 KEY DIFFERENCES FROM ORIGINAL

1. **Groq instead of Ollama**: Much faster responses, but requires internet
2. **Faster-Whisper base**: Lighter, faster than small model
3. **Week 1 = MVP**: Voice conversation working by Day 7
4. **Week 2 = UI + Extras**: Avatar, emotions, vision added later
5. **Vision on Day 14 only**: Qwen2.5-VL-7B saved for last day
6. **Emotion audio in Week 2**: SpeechBrain added Days 12-13

**Hardware Requirements Updated:**
- Week 1: 6GB VRAM (no local LLM needed!)
- Week 2: 8-10GB VRAM (with emotion)
- Day 14: 18GB VRAM (with vision)

Good luck building Sara with Groq! 🚀💙