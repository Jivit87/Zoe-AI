# Sara AI - Project Information

## Project Overview

Sara is an emotionally intelligent AI companion designed for natural, human-like voice conversations. Unlike typical AI assistants, Sara is built to be a genuine conversational presence - not a tool, but a companion who listens, remembers, and responds with emotional depth and authenticity.

## Core Philosophy

Sara doesn't position herself as an "AI assistant" - she's just Sara. The design philosophy emphasizes:
- Genuine emotional connection over transactional interactions
- Natural conversation flow with human speech patterns
- Proactive engagement rather than passive waiting
- Long-term memory and contextual awareness
- Authentic personality with quirks and warmth

## Current Status: Week 1 MVP (Completed)

The project has successfully implemented a full voice-to-voice conversation loop with the following capabilities:

### Implemented Features

#### 1. Voice Input (Speech-to-Text)
- **Technology**: Faster-Whisper (small model) running on CPU
- **Voice Activity Detection**: Silero-VAD for intelligent speech boundary detection
- **Features**:
  - Real-time microphone listening
  - Pre-roll audio buffering (captures audio just before speech starts)
  - Silence detection to determine end of utterance
  - Pause/Resume capability to prevent echo (Sara doesn't hear herself)
  - Minimum audio length filtering to avoid processing noise

#### 2. Conversational Intelligence (Brain)
- **LLM**: Groq API with llama-3.3-70b-versatile
- **Personality System**: Deeply human conversational patterns
- **Features**:
  - Natural speech patterns (fillers, contractions, incomplete thoughts)
  - Emotional intelligence and context awareness
  - Dynamic response adaptation based on user state
  - Conversation state tracking (distress, excitement, silence patterns)
  - Temperature adjustment based on emotional context
  - Response length optimization (brief during distress, normal otherwise)
  - Avoids repetitive questioning patterns

#### 3. Memory System
- **Storage**: Local JSON/Markdown files
- **Features**:
  - Conversation history logging with timestamps
  - Recent context buffer (last 10 turns)
  - Emotional state tracking
  - Session persistence to markdown files
  - Fast retrieval without heavy embeddings (optimized for speed)

#### 4. Voice Output (Text-to-Speech)
- **Primary**: Kokoro-82M ONNX model (ultra-fast, high-quality)
- **Fallback**: Edge-TTS
- **Voice**: af_bella (natural, conversational female voice)
- **Features**:
  - Local inference (no cloud dependency)
  - Real-time audio playback
  - Temporary file management
  - Automatic cleanup

#### 5. Proactive Engagement
- **Silence Monitoring**: Background thread tracks user silence
- **Check-ins**: Sara proactively reaches out after 30 seconds of silence
- **Greeting**: Initiates conversation with "Hello Sir, I'm online. How are you feeling right now?"

## Technical Architecture

### Core Components

```
src/
├── main.py                    # Main conversation loop orchestrator
├── llm/
│   └── sara_brain.py         # Groq-powered conversational intelligence
├── memory/
│   └── conversation_memory.py # Conversation history & context
├── stt/
│   ├── speech_recognizer.py  # Faster-Whisper integration
│   └── voice_activity_detector.py # Silero-VAD for speech detection
├── tts/
│   └── voice_generator.py    # Kokoro-82M ONNX TTS
└── models/
    └── kokoro/               # TTS model files
```

### Data Flow

1. **Listen**: Microphone → VAD → Speech Detection → Audio Buffer
2. **Transcribe**: Audio Buffer → Faster-Whisper → Text
3. **Think**: Text → Sara Brain (Groq) → Response
4. **Remember**: Conversation → Memory System → Markdown Log
5. **Speak**: Response → Kokoro TTS → Audio Playback

### Key Design Decisions

- **CPU-only inference**: Optimized for macOS without GPU requirements
- **Lightweight memory**: Avoids heavy embedding models (Ollama) for speed
- **Pause/Resume audio**: Prevents Sara from hearing herself speak
- **Markdown logging**: Human-readable conversation history
- **Proactive engagement**: Background thread for silence monitoring

## Conversation Personality

Sara's personality is defined by:

### Voice Characteristics
- Natural speech patterns with fillers ("Hmm...", "Well...", "I mean...")
- Contractions always ("I'm", "you're", "didn't")
- Incomplete thoughts when emotional
- Natural reactions ("Oh!", "Wait, what?", "Seriously?")
- Sentence fragments for emphasis

### Emotional Intelligence
- Reads subtext and emotional undertones
- Notices conversation patterns
- Validates feelings before offering solutions
- Matches user's energy level
- Remembers emotional context from previous conversations

### Interaction Style
- Short, powerful responses (1-3 sentences typically)
- Varies rhythm like natural conversation
- Doesn't always ask questions (avoids interview feel)
- Builds on previous conversation threads
- Shows genuine curiosity and care

## Testing Infrastructure

The project includes comprehensive test files:

- `test_audio.py`: Verifies microphone and speaker functionality
- `test_stt.py`: Tests real-time speech-to-text transcription
- `test_groq.py`: Validates Groq API connection and Sara's personality
- `test_brain.py`: Text-based chat interface for testing conversational logic

## Planned Features (Not Yet Implemented)

### Week 2+ Roadmap
- **Vision**: Camera integration for visual context awareness
- **Emotion Detection**: Real-time emotional state analysis from voice
- **Avatar**: Visual representation with emotional expressions
- **WhatsApp Integration**: Bridge for text-based conversations (directory exists but not implemented)
- **Advanced Memory**: Vector embeddings for semantic memory search
- **Multi-modal Context**: Combining voice, vision, and emotional cues

## Technology Stack

### Core Dependencies
- **PyTorch**: Deep learning framework (CPU/MPS for macOS)
- **Faster-Whisper**: Speech-to-text transcription
- **Silero-VAD**: Voice activity detection
- **Groq API**: LLM inference (llama-3.3-70b-versatile)
- **Kokoro-ONNX**: Text-to-speech synthesis
- **Edge-TTS**: TTS fallback
- **Mem0ai**: Memory framework (installed but minimal usage)
- **ChromaDB**: Vector database (installed but not actively used yet)
- **Sounddevice/Soundfile**: Audio I/O
- **Librosa**: Audio processing utilities

### Environment
- **Platform**: macOS (darwin)
- **Python**: 3.x
- **Compute**: CPU-optimized (int8 quantization for models)

## Data Storage

### Conversations
- Location: `conversations/conversation_history.md`
- Format: Markdown with timestamps
- Content: User input, Sara's responses, emotional states

### Vector Database
- Location: `data/chroma_db/`
- Status: Initialized but not actively used (optimized for speed)

### Models
- Kokoro TTS: `src/models/kokoro/` (kokoro-v0_19.onnx, voices.bin)

## Usage

### Starting Sara
```bash
python -m src.main
```

### Interaction Flow
1. Sara greets: "Hello Sir, I'm online. How are you feeling right now?"
2. Speak naturally - Sara listens continuously
3. Pause when done speaking - Sara detects silence and responds
4. Sara speaks her response (audio input paused during this)
5. Cycle continues until Ctrl+C

### Stopping Sara
- Press Ctrl+C
- Sara saves conversation to markdown
- Farewell message: "Until next time, Sir. Take care."

## Current Limitations

1. **No visual input**: Camera/vision not yet integrated
2. **Basic emotion detection**: Currently uses "neutral" placeholder
3. **Simple memory**: No semantic search, just recent context
4. **English only**: Whisper configured for English transcription
5. **Single user**: No multi-user support or user identification
6. **No WhatsApp bridge**: Directory exists but not implemented

## Performance Characteristics

- **STT Latency**: ~500ms for transcription (Faster-Whisper small model)
- **LLM Latency**: ~200-500ms (Groq API, depends on network)
- **TTS Latency**: <100ms (Kokoro-82M local inference)
- **Total Response Time**: ~1-2 seconds from speech end to Sara's voice

## Development History

Based on conversation logs, Sara has been tested with:
- Real-time voice conversations
- Proactive silence detection and check-ins
- Natural conversation flow with emotional awareness
- Memory persistence across sessions
- User name recognition and correction (Jivit Rana)

The system successfully handles:
- Interruptions and corrections
- Emotional responses (frustration, excitement)
- Silence and pauses
- Natural conversation patterns

## Project Structure Insights

- **Modular design**: Each component (STT, TTS, Brain, Memory) is independent
- **Callback-based**: Speech recognition uses callbacks for non-blocking operation
- **Thread-safe**: Background threads for silence monitoring and transcription
- **Graceful degradation**: TTS falls back to Edge-TTS if Kokoro fails
- **Clean shutdown**: Saves conversations before exit

## Future Vision

Sara is designed to evolve into a fully multi-modal AI companion with:
- Visual awareness through camera
- Emotional intelligence through voice analysis
- Persistent long-term memory with semantic understanding
- Multi-platform presence (voice, text, WhatsApp)
- Avatar-based visual representation
- Proactive emotional support and companionship

The current Week 1 MVP successfully demonstrates the core voice conversation loop and establishes Sara's unique personality foundation.
