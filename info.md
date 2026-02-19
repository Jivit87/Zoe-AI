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

## Current Status: Enhanced Conversation Flow (Completed)

The project implements a full voice-to-voice conversation loop with human-quality turn-taking, barge-in support, emotion detection, and progressive engagement:

### Implemented Features

#### 1. Voice Input (Speech-to-Text)
- **Technology**: Faster-Whisper (small model) running on CPU
- **Voice Activity Detection**: Silero-VAD with **hysteresis thresholding** (0.85 start / 0.30 stop)
- **Features**:
  - Real-time microphone listening
  - Pre-roll audio buffering (captures audio just before speech starts)
  - Silence detection to determine end of utterance
  - **Mic stays active during TTS** for barge-in detection
  - **Backchannel filtering**: "yeah"/"mm-hmm" don't trigger new turns
  - Minimum audio length filtering to avoid processing noise
  - Audio duration tracking for emotion analysis

#### 2. Conversational Intelligence (Brain)
- **LLM**: Groq API with llama-3.3-70b-versatile
- **Personality System**: Deeply human conversational patterns
- **Streaming**: `generate_response_streaming()` yields sentence-sized chunks to TTS as tokens arrive — first audio starts in ~0.65s instead of waiting for the full response
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
  - **Streaming playback**: `speak_stream()` plays each sentence chunk as it arrives
  - **Interruptible**: checks cancel event between chunks and during playback (50ms polling)
  - **Thinking sounds**: plays filler ("Hmm...", "Well...") before LLM responds to eliminate dead silence
  - Returns spoken/remaining text on interruption for context

#### 5. Conversation Management
- **State Machine**: `ConversationState` enum (IDLE → LISTENING → PROCESSING → SPEAKING → INTERRUPTED)
- **Barge-In Detection**: 4-condition filter (grace period + energy + VAD + time-gating)
- **Backchannel Classification**: filters short acknowledgments from real turns
- **Emotion Detection**: text keywords + speech timing (WPS) → dynamic LLM adaptation
- **Progressive Silence**: multi-tier check-ins at 15s/30s/60s/120s with context-aware responses

## Technical Architecture

### Core Components

```
src/
├── main.py                         # Conversation orchestrator + state machine
├── conversation/
│   ├── state_machine.py           # ConversationState enum + StateManager
│   ├── barge_in.py                # BargeInDetector (4-condition filter)
│   └── backchannel.py             # BackchannelClassifier + BackchannelManager
├── emotion/
│   └── emotion_detector.py        # Text + audio timing emotion analysis
├── llm/
│   └── sara_brain.py              # Groq-powered conversational intelligence
├── memory/
│   └── conversation_memory.py     # Conversation history & context
├── stt/
│   ├── speech_recognizer.py       # Faster-Whisper + barge-in routing
│   └── voice_activity_detector.py # Silero-VAD with hysteresis
├── tts/
│   └── voice_generator.py         # Kokoro-82M ONNX TTS + streaming
└── models/
    └── kokoro/                    # TTS model files
```

### Data Flow

1. **Listen**: Microphone → Hysteresis VAD → Speech Detection → Audio Buffer
2. **Barge-In Check**: During SPEAKING → BargeInDetector monitors mic → cancel TTS if real
3. **Transcribe**: Audio Buffer → Faster-Whisper → Text (backchannels filtered)
4. **Detect Emotion**: Text + Audio Duration → EmotionDetector → state
5. **Think**: Thinking sound plays → Groq streams tokens → sentence chunks
6. **Speak**: Chunks → Kokoro TTS → interruptible playback
7. **Remember**: Conversation → Memory System → Markdown Log

### Key Design Decisions

- **CPU-only inference**: Optimized for macOS without GPU requirements
- **Lightweight memory**: Avoids heavy embedding models for speed
- **Hysteresis VAD**: Dual thresholds prevent false-positive speech detection
- **Mic active during TTS**: Enables barge-in without echo cancellation hardware
- **State machine**: Thread-safe `StateManager` replaces boolean flags
- **Progressive silence**: Context-aware multi-tier check-ins (not fixed interval)

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
- `test_streaming.py`: Tests streaming LLM→TTS pipeline (no mic needed) — shows time-to-first-chunk vs total time

## Planned Features (Not Yet Implemented)

### Future Roadmap
- **Vision**: Camera integration for visual context awareness
- **Avatar**: Visual representation with emotional expressions
- **WhatsApp Integration**: Bridge for text-based conversations
- **Advanced Memory**: Vector embeddings for semantic memory search
- **Semantic EoT**: BERT/Groq classifier for end-of-turn detection
- **Speaker-adapted VAD**: Learn user voice profile during session
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
2. Speak naturally — Sara listens continuously
3. Pause when done — Sara detects silence and responds
4. Sara plays a thinking sound, then streams her response as chunks
5. **Interrupt anytime** — speak while Sara is talking and she'll stop
6. If silent for 15s/30s/60s, Sara checks in with context-aware responses
7. Cycle continues until Ctrl+C

### Stopping Sara
- Press Ctrl+C
- Sara saves conversation to markdown
- Farewell message: "Until next time, Sir. Take care."

## Current Limitations

1. **No visual input**: Camera/vision not yet integrated
2. **Simple memory**: No semantic search, just recent context
3. **English only**: Whisper configured for English transcription
4. **Single user**: No multi-user support or user identification
5. **No WhatsApp bridge**: Directory exists but not implemented
6. **No semantic EoT**: End-of-turn is still silence-based, not semantic

## Performance Characteristics

- **STT Latency**: ~500ms for transcription (Faster-Whisper small model)
- **LLM Time-to-First-Token**: ~200-400ms (Groq streaming API)
- **TTS Latency**: <100ms per chunk (Kokoro-82M local inference)
- **Time to First Audio**: ~0.65s from speech end (streaming — Sara speaks the first sentence before the full response is generated)
- **Total Response Time**: ~1-2 seconds end-to-end (perceived as much shorter due to streaming)

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
