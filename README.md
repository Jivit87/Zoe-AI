<p align="center">
  <h1 align="center">✨ SARA AI</h1>
  <p align="center"><b>Sentient AI Response Assistant</b></p>
  <p align="center">
    A real-time, voice-to-voice AI companion with emotional intelligence, long-term memory, and natural conversation flow.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/LLM-Groq%20%7C%20Llama%203.3%2070B-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/STT-Faster--Whisper-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/TTS-Kokoro--82M-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/RAG-ChromaDB%20%2B%20BM25-red?style=for-the-badge" />
</p>

---

## 🎯 What is SARA?

SARA is a fully local, real-time voice AI companion that listens, understands, remembers, and speaks — just like talking to a real person. She detects your emotions, remembers past conversations through a production-grade RAG pipeline, and responds with natural, empathetic speech.

**The full pipeline:**

```
🎤 Microphone → VAD (Silero) → STT (Faster-Whisper) → Emotion Detection
     → LLM Brain (Groq/Llama-3.3-70B) + RAG Memory (ChromaDB + BM25)
         → Streaming TTS (Kokoro-82M / Edge-TTS) → 🔊 Speaker
```

---

## ✨ Features

### 🗣️ Natural Voice Conversation
- **Real-time STT** — Faster-Whisper with Silero-VAD for precise speech boundary detection
- **Streaming TTS** — First sentence plays while the rest is still generating
- **Barge-in support** — Interrupt Sara mid-speech; she stops and listens immediately
- **Backchannel filtering** — "Yeah", "mm-hmm" don't trigger new turns

### 🧠 Emotional Intelligence
- **Text + audio analysis** — Keywords, speech pace (words/sec), and hedging detection
- **Adaptive responses** — LLM temperature, max tokens, and response style adjust to detected emotion
- **Context-aware silence** — Progressive check-ins at 15s, 30s, 60s of silence, adapted to emotional state

### 💾 Long-Term Memory (RAG)
- **Hybrid retrieval** — Dense (ChromaDB + sentence-transformers) + Sparse (BM25) with Reciprocal Rank Fusion
- **Anthropic's Contextual Retrieval** — Context prefixes reduce retrieval failures by ~67%
- **Cross-encoder re-ranking** — Joint query-document scoring for high precision
- **Corrective-RAG (CRAG)** — Filters low-confidence results to prevent noise injection
- **Adaptive gating** — Skips RAG for greetings/backchannels, saving ~200ms on ~40% of turns
- **Time-decay scoring** — Recent memories boosted ~30% with exponential decay
- **MMR diversity** — Prevents redundant memory retrieval

### 🎭 Conversation Management
- **State machine** — IDLE → LISTENING → PROCESSING → SPEAKING → INTERRUPTED
- **Hysteresis VAD** — High threshold (0.85) to start, low (0.30) to stop — eliminates flickering
- **Thinking sounds** — Plays "Hmm...", "Well..." fillers while LLM generates

---

## 🏗️ Architecture

```
src/
├── main.py                          # SaraAI orchestrator — full voice loop
├── llm/
│   └── sara_brain.py                # Groq LLM with streaming + personality
├── stt/
│   ├── speech_recognizer.py         # Faster-Whisper real-time transcription
│   └── voice_activity_detector.py   # Silero-VAD with hysteresis
├── tts/
│   └── voice_generator.py           # Kokoro-82M (ONNX) + Edge-TTS fallback
├── emotion/
│   └── emotion_detector.py          # Text + speech timing emotion analysis
├── memory/
│   └── conversation_memory.py       # Session memory + markdown logging
├── conversation/
│   ├── state_machine.py             # Thread-safe conversation state manager
│   ├── barge_in.py                  # Interruption detection during TTS
│   └── backchannel.py               # "mm-hmm" classifier + Sara's backchannels
├── rag/
│   ├── rag_pipeline.py              # Main RAG orchestrator
│   ├── indexer.py                   # Contextual Retrieval multi-chunk indexer
│   ├── retriever.py                 # Hybrid Dense+Sparse with RRF + time-decay
│   ├── query_processor.py           # Adaptive gating + re-contextualization
│   └── reranker.py                  # Cross-encoder re-ranking + CRAG
└── models/
    └── kokoro/                      # Kokoro-82M ONNX model files
        ├── kokoro-v0_19.onnx        # 325MB TTS model
        └── voices.bin               # Voice embeddings
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **LLM** | Groq Cloud API (Llama-3.3-70B) | Conversational intelligence with deep personality |
| **STT** | Faster-Whisper (int8) | Real-time speech-to-text on CPU |
| **VAD** | Silero-VAD | Voice activity detection with hysteresis |
| **TTS (primary)** | Kokoro-82M (ONNX) | Ultra-fast, high-quality local TTS |
| **TTS (fallback)** | Edge-TTS (Microsoft) | Cloud fallback if Kokoro unavailable |
| **Vector DB** | ChromaDB | Persistent dense vector storage |
| **Sparse Search** | BM25 (rank-bm25) | Keyword-based retrieval |
| **Embeddings** | all-MiniLM-L6-v2 | Sentence embeddings for dense retrieval |
| **Re-ranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 | Cross-encoder re-scoring |
| **Audio I/O** | sounddevice + soundfile | Microphone input & speaker output |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **macOS** (optimized for Apple Silicon — CPU/MPS)
- **Groq API key** — Free at [console.groq.com](https://console.groq.com)
- **Microphone + speakers**

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/SARA.git
cd SARA

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```env
# Get your free API key at: https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here
```

### Run Sara

```bash
python -m src.main
```

Sara will greet you and start listening. Speak naturally — she responds when you pause. Press `Ctrl+C` to stop.

---

## 🧪 Test Scripts

Test individual components without running the full pipeline:

| Script | Purpose | Command |
|---|---|---|
| `test_audio.py` | Verify microphone + speaker I/O | `python test_audio.py` |
| `test_groq.py` | Test Groq API connection + latency | `python test_groq.py` |
| `test_stt.py` | Real-time speech-to-text test | `python test_stt.py` |
| `test_brain.py` | Text chat with Sara (no mic needed) | `python test_brain.py` |
| `test_streaming.py` | Streaming TTS pipeline test | `python test_streaming.py` |

---

## 📁 Data & Persistence

| Path | Purpose |
|---|---|
| `data/chroma_db/` | ChromaDB vector store (RAG long-term memory) |
| `conversations/` | Markdown conversation logs (auto-saved per session) |
| `.env` | API keys (gitignored) |

---

## 🔧 Configuration

Key parameters can be tuned in the source files:

| Parameter | Location | Default | Description |
|---|---|---|---|
| LLM Model | `sara_brain.py` | `llama-3.3-70b-versatile` | Groq model for conversation |
| Whisper Model | `main.py` | `small` | STT model size (`tiny`/`base`/`small`/`medium`) |
| TTS Voice | `voice_generator.py` | `af_bella` | Kokoro voice ID |
| VAD Start Threshold | `voice_activity_detector.py` | `0.85` | Speech detection sensitivity |
| Silence Tiers | `main.py` | `15s/30s/60s/120s` | Progressive silence check-in timings |
| RAG Top-K | `sara_brain.py` | `5` | Number of memories retrieved per query |

---

## 📄 License

This project is for personal/educational use.
