# Voice AI Conversation Flow — Sara Implementation Guide

> How ChatGPT Advanced Voice, Google Gemini Live, ElevenLabs, and others handle real-time voice interactions — and how Sara matches or surpasses them.

## Implementation Status

| Feature | Status | Section |
|---|---|---|
| Cascaded pipeline (STT → LLM → TTS) | ✅ Done | 1A |
| VAD with pre-roll buffer (Silero) | ✅ Done | 3.1 |
| Silence monitoring + proactive check-ins | ✅ Done | 3.5 |
| **Streaming TTS** (LLM tokens → chunked playback) | ✅ Done | 5 |
| **Hysteresis VAD** (dual start/stop thresholds) | ✅ Done | 3.3 |
| **Thinking sounds** (filler during LLM wait) | ✅ Done | 7 |
| **Barge-in / interruption handling** | ✅ Done | 3.2 |
| **Backchannel classification** | ✅ Done | 3.3 |
| **Backchannel responses** (mm-hmm mid-speech) | ✅ Done | 3.4 |
| **Progressive silence responses** | ✅ Done | 3.5 |
| **Emotion detection** (text + audio timing) | ✅ Done | 6 |
| **Conversation state machine** | ✅ Done | 4 |
| Semantic end-of-turn detection | ⬜ Planned | 3.1 |
| Speaker-adapted VAD | ⬜ Planned | 3.3 |


---

## 1. The Big Picture: Two Fundamental Architectures

Before diving into features, you need to understand that all voice AI systems are built on one of two architectures. This choice determines *everything* about how interruptions, latency, and naturalness work.

### A) Cascaded Pipeline ✅ (What Sara uses)
```
Mic → VAD → STT (Whisper) → LLM (Groq) → TTS (Kokoro) → Speaker
```
Each stage is independent. Modular and flexible — you can swap any piece. But latency is additive: each hop adds delay.

**Sara's current latency breakdown:**
- STT: ~500ms (Faster-Whisper small, CPU)
- LLM time-to-first-token: ~200–400ms (Groq streaming)
- TTS: <100ms per chunk (Kokoro-82M local)
- **Time to first audio: ~0.65s** (thanks to streaming — see Section 5)

### B) Speech-to-Speech / Full-Duplex (What ChatGPT AVM & Gemini Live use)
```
Mic ←→ End-to-end neural model ←→ Speaker (simultaneously)
```
A single model listens AND speaks at the same time. It encodes audio to latent vectors and generates output audio directly. **Typical latency: 200–300ms.** But it's cloud-dependent and much harder to build.

**Sara's reality:** She's on the cascaded path. That's fine — with smart engineering, cascaded systems can feel just as natural. The tricks are in the *details* described below.

---

## 2. The Core Problem: Turn-Taking

Human conversation is a dance. Both sides instinctively know when to speak and when to listen. Voice AI needs to replicate these 5 sub-problems:

| Problem | Description | Sara Status |
|---|---|---|
| **End-of-Turn Detection** | Knowing when the user has *finished* speaking | ⬜ Naive (silence threshold only) |
| **Barge-In / Interruption** | Stopping Sara mid-speech when user speaks | ⬜ Not implemented |
| **False Positive Prevention** | Not treating "mm-hmm" or background noise as interruptions | ⬜ Not implemented |
| **Backchannel Responses** | Sara making small sounds while user speaks | ⬜ Not implemented |
| **Silence Handling** | Proactive check-ins vs awkward dead air | ✅ Basic (30s/120s tiers) |

---

## 3. How the Big Players Handle Each Problem

---

### 3.1 End-of-Turn Detection

**The naive approach (what Sara currently uses):**
Wait for N milliseconds of silence (~480ms / 15 chunks at 32ms each), then assume the user is done.

**The problem:** Humans pause mid-sentence all the time ("I was thinking... that maybe we should..."). A silence threshold that's too short causes Sara to interrupt. Too long and it feels sluggish.

**What the pros do — Multi-Signal End-of-Turn (EoT):**

ChatGPT AVM and Gemini Live use a combination of:

1. **Acoustic VAD** — Is audio energy above threshold? (basic)
2. **Semantic EoT** — Did the sentence feel *complete*? (advanced)
   - A fine-tuned BERT/language model reads the transcript so far and classifies: "Is this utterance finished or in-progress?"
   - Example: "Can you tell me—" → classifier outputs "UNFINISHED"
   - Example: "Can you tell me what time it is?" → classifier outputs "FINISHED"
3. **Prosodic cues** — Intonation patterns (falling pitch usually = end of sentence)

**For Sara — Pragmatic Implementation:**
```python
# Two-layer EoT detection
# Layer 1: Silero-VAD (already implemented) — acoustic signal
# Layer 2: Simple heuristic semantic check

SENTENCE_ENDINGS = ['.', '?', '!', '...']

def is_turn_complete(transcript: str, silence_ms: int) -> bool:
    stripped = transcript.strip()
    
    # Short silence + sentence-ending punctuation = probably done
    if silence_ms > 400 and any(stripped.endswith(e) for e in SENTENCE_ENDINGS):
        return True
    
    # Longer silence = done regardless
    if silence_ms > 800:
        return True
    
    # Very short utterances with silence = done
    if silence_ms > 500 and len(stripped.split()) < 4:
        return True
    
    return False
```

You can level this up by asking Groq (ultra-fast, cheap) to classify completeness:
```python
TURN_CHECK_PROMPT = """
Is this utterance complete or does it seem cut off mid-thought?
Reply ONLY with: COMPLETE or INCOMPLETE

Utterance: "{text}"
"""
```
Run this in parallel with STT transcription. It adds ~100ms but dramatically improves accuracy.

---

### 3.2 Barge-In (Interruption Handling)

⚠️ **Not yet implemented in Sara.**

This is the most technically challenging piece. When Sara is speaking, the user should be able to interrupt her and Sara should **immediately stop**.

**How ChatGPT AVM does it:**
The system runs VAD continuously on the microphone *even while audio is playing*. The moment user speech is detected during Sara's speech → TTS is cancelled instantly → pipeline resets to listen mode.

**The critical challenge: Echo Cancellation**
Sara's voice comes out of the speaker → gets picked up by the microphone → VAD detects it → false barge-in triggered.

Sara's current code does `pause/resume` which **completely blocks listening** during TTS. A better approach is **Acoustic Echo Cancellation (AEC)**.

**How AEC works:**
```
Speaker Output (known signal) ──┐
                                ├──► AEC Filter ──► Clean Mic Input
Microphone Input (dirty) ───────┘
```
The system knows what audio it's playing, so it can subtract that signal from the microphone input — leaving only the user's voice.

**For Sara — Implementation levels:**

**Level 1 (Quick win — improve current approach):**
```python
class BargeInDetector:
    def __init__(self, grace_period_ms=200):
        self.grace_period_ms = grace_period_ms  # Ignore first 200ms of TTS (echo)
        self.tts_start_time = None
        self.speech_energy_threshold = 0.03
        
    def on_tts_start(self):
        self.tts_start_time = time.time()
    
    def check_barge_in(self, audio_chunk, vad_probability) -> bool:
        if self.tts_start_time is None:
            return False
        
        elapsed_ms = (time.time() - self.tts_start_time) * 1000
        
        # Ignore during grace period (echo suppression)
        if elapsed_ms < self.grace_period_ms:
            return False
        
        # Require both: VAD probability AND energy threshold
        energy = np.sqrt(np.mean(audio_chunk**2))
        return vad_probability > 0.7 and energy > self.speech_energy_threshold
```

**Level 2 (Best for Sara — sounddevice AEC):**
```python
# Use sounddevice's built-in loopback + numpy to do basic echo cancellation
# Or use WebRTC's AEC via the py-webrtc library
pip install webrtcvad  # Has built-in AEC capabilities
```

**Level 3 (Production grade):**
Use `pyaudio` with platform AEC (macOS CoreAudio has built-in AEC). On macOS:
```python
# Enable Voice Isolation via CoreAudio API
# Use system-level mic modes
```

**TTS Playback Position Tracking (for smart interruption resume):**
When barge-in happens, Sara should know *where she was* in her response. With streaming, Sara knows exactly which chunk was interrupted:
```python
class TTSPlaybackTracker:
    def __init__(self, full_text: str):
        self.words = full_text.split()
        self.current_word_index = 0
        self.chars_per_second = 3.5  # Approximate speaking rate
        
    def get_remaining_text(self) -> str:
        return ' '.join(self.words[self.current_word_index:])
    
    def get_spoken_text(self) -> str:
        return ' '.join(self.words[:self.current_word_index])
```

---

### 3.3 False Positive Prevention (Avoiding Unnecessary Interruptions)

⚠️ **Not yet implemented in Sara.** Sara currently uses a single VAD threshold (0.5).

One of the biggest complaints about early ChatGPT AVM was it interrupted users too much. OpenAI spent months fixing this. The March 2025 update specifically addressed: "The chatbot will interrupt you much less."

**Sources of false positives:**
1. Background noise (TV, traffic, keyboard)
2. Backchannels — user says "yeah", "mm-hmm", "okay" *while Sara is speaking* — this is acknowledgment, NOT an interruption
3. Sara's own voice echo
4. Short coughs, sneezes, mouth sounds

**Solutions:**

**1. Hysteresis Thresholding** ← Best quick win for Sara:
```python
SPEECH_START_THRESHOLD = 0.85  # High — require strong confidence to start detecting
SPEECH_STOP_THRESHOLD = 0.30   # Low — keep detecting as long as slight probability

# Current Sara uses: 0.5 for both start and stop (no hysteresis)
# Change VoiceActivityDetector.threshold → split into two values

is_speaking = False
for prob in vad_probs:
    if not is_speaking and prob > SPEECH_START_THRESHOLD:
        is_speaking = True
    elif is_speaking and prob < SPEECH_STOP_THRESHOLD:
        is_speaking = False
```

**2. Time-Gating** — Require sustained speech before committing:
```python
MIN_SPEECH_DURATION_MS = 300  # Must speak for at least 300ms before it's "real"
speech_start_time = None

if vad_detects_speech:
    if speech_start_time is None:
        speech_start_time = time.time()
    elif (time.time() - speech_start_time) * 1000 > MIN_SPEECH_DURATION_MS:
        trigger_barge_in()
else:
    speech_start_time = None  # Reset if speech stops
```

**3. Backchannel Classification** — Distinguish "mm-hmm" from a real interruption:
```python
BACKCHANNELS = {'yeah', 'yes', 'mm', 'hmm', 'okay', 'ok', 'right', 'sure', 'uh huh', 'yep'}

def is_backchannel(transcript: str) -> bool:
    words = transcript.lower().strip().split()
    return len(words) <= 2 and all(w in BACKCHANNELS for w in words)
```
If it's a backchannel, Sara continues speaking but can acknowledge with a slight vocal shift.

---

### 3.4 Backchannel Responses (Sara reacts while user speaks)

⚠️ **Not yet implemented in Sara.**

This is a sophisticated feature that makes conversations feel deeply human. While the user is speaking, a human listener says "mm-hmm", "yeah", "I see" — subconsciously signaling they're paying attention.

**How to detect the right moment to backchannel:**
- When user pauses mid-sentence (200–400ms pause, not an end-of-turn)
- After emotionally heavy content
- After a list item or natural breath point

**Sara implementation:**
```python
class BackchannelManager:
    def __init__(self, tts):
        self.backchannels = [
            "Mm-hmm...",
            "I see.",
            "Yeah...",
            "Oh, really?",
            "Mmm.",
            "Go on.",
        ]
        self.last_backchannel = 0
        self.min_gap_seconds = 20  # Don't backchannel too frequently
    
    def maybe_backchannel(self, pause_duration_ms: int, user_is_mid_sentence: bool):
        if not user_is_mid_sentence:
            return
        if pause_duration_ms < 300 or pause_duration_ms > 600:
            return  # Only on natural mid-sentence pauses
        if time.time() - self.last_backchannel < self.min_gap_seconds:
            return
        
        response = random.choice(self.backchannels)
        self.tts.speak_softly(response)  # Lower volume for backchannels
        self.last_backchannel = time.time()
```

---

### 3.5 Silence & Proactive Engagement

✅ **Basic version implemented** — Sara checks in after 30s, then 120s.

Here's how to make it much more intelligent:

**The Problem:** Silence has many meanings:
- User is thinking (don't interrupt)
- User is distracted (gentle nudge okay)
- User is emotionally overwhelmed (be careful)
- User fell asleep / stepped away

**Progressive silence strategy (current Sara is a simplified version of this):**
```
0–5s silence    → Normal, user is thinking. Say nothing.
5–15s silence   → Still okay. Brief soft sound ("hmm...") if appropriate.
15–30s silence  → Light check-in: "Take your time."
30–60s silence  → Warmer check-in: "Hey, still here with you."   ← Sara does this
60–120s silence → "Whenever you're ready, I'm listening."
120s+ silence   → Assume session pause, reduce background activity. ← Sara does longer cooldown
```

**Context-aware silence responses** — Sara should know WHY there might be silence:
```python
def get_silence_response(context: dict, silence_duration: int) -> str:
    emotional_state = context.get('last_emotional_state', 'neutral')
    last_topic = context.get('last_topic', '')
    
    if emotional_state == 'distressed':
        return "It's okay to take a moment."  # Gentle
    elif emotional_state == 'excited':
        return "What are you thinking?"       # Engaging
    elif 'decision' in last_topic:
        return "No rush at all."              # Supportive
    else:
        return random.choice([
            "Still here...",
            "Take your time.",
            "Whenever you're ready."
        ])
```

---

## 4. The Full Conversation State Machine

Here is the state machine that underlies all professional voice AI systems:

```
┌─────────────────────────────────────────────────┐
│                   STATES                        │
│                                                 │
│  IDLE ──────────► LISTENING ──────────► PROCESSING
│    ▲                  │   ▲                    │
│    │   (silence       │   │ (barge-in          │
│    │    timeout)      │   │  detected)         ▼
│    │                  │              SPEAKING ──┘
│    │                  │                  │
│    └──────────────────┘◄─────────────────┘
│              (speech done)
│
│  PAUSED ◄──────── (Ctrl or session break)
└─────────────────────────────────────────────────┘
```

**Sara's current implementation** uses boolean flags (`is_speaking`, `is_paused`) rather than a formal state machine. A proper `StateManager` class would make barge-in and backchannel logic cleaner:

```python
from enum import Enum

class ConversationState(Enum):
    IDLE = "idle"                     # Waiting, nothing happening
    LISTENING = "listening"           # User is speaking
    PROCESSING = "processing"         # STT + LLM running
    SPEAKING = "speaking"             # TTS playing (streaming chunks)
    INTERRUPTED = "interrupted"       # Barge-in received during SPEAKING
    BACKCHANNEL = "backchannel"       # Brief Sara response during user speech
    SILENCE_MONITORING = "silence"    # Proactive engagement countdown

class StateManager:
    def __init__(self):
        self.state = ConversationState.IDLE
        self.previous_state = None
        
    def transition(self, new_state: ConversationState):
        self.previous_state = self.state
        self.state = new_state
        self._on_transition(self.previous_state, new_state)
    
    def _on_transition(self, from_state, to_state):
        if to_state == ConversationState.INTERRUPTED:
            self._cancel_tts()
            self._cancel_llm_if_possible()
        elif to_state == ConversationState.SPEAKING:
            self._start_barge_in_monitor()
        elif to_state == ConversationState.LISTENING:
            self._start_silence_timer()
```

---

## 5. ✅ Streaming Responses — IMPLEMENTED

**The biggest latency win in Sara's pipeline, now done.**

Sara's streaming pipeline (`generate_response_streaming()` + `speak_stream()`):

```python
# In sara_brain.py
async def generate_response_streaming(user_input, emotional_state="neutral"):
    """Yields sentence-sized chunks as Groq tokens stream in."""
    stream = groq_client.chat.completions.create(..., stream=True)
    
    text_buffer = ""
    SENTENCE_ENDINGS = {'.', '!', '?'}
    MIN_CHUNK_WORDS = 3
    
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token is None:
            continue
        text_buffer += token
        stripped = text_buffer.rstrip()
        if stripped and stripped[-1] in SENTENCE_ENDINGS and len(stripped.split()) >= MIN_CHUNK_WORDS:
            yield text_buffer
            text_buffer = ""
    
    if text_buffer.strip():
        yield text_buffer
    # Memory saved here after full response assembled

# In voice_generator.py
def speak_stream(chunks: Iterator[str]) -> str:
    """Play each chunk immediately as it arrives."""
    full_parts = []
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk:
            full_parts.append(chunk)
            audio_file = self.generate_audio(chunk)      # Kokoro synthesis
            data, sr = sf.read(audio_file)
            sd.play(data, sr)
            sd.wait()                                     # Play, then get next chunk
    return " ".join(full_parts)

# In main.py — handle_user_speech()
chunks = self.brain.generate_response_streaming(transcription)
full_response = self.tts.speak_stream(chunks)   # Starts speaking on chunk 1
```

**Measured results:**
- ⏱ Time to first audio: **0.65s** (vs ~1.5–2s before)
- 4 sentence chunks played sequentially and seamlessly
- Full response assembled correctly in memory

**The result:** Instead of waiting 500ms+ for the full LLM response, Sara starts speaking within **0.65s** of the first complete phrase arriving. Total perceived latency dropped by ~60%.

---

## 6. Emotion-Aware Response Adaptation

✅ **Text-based emotion analysis is implemented.** Audio-based analysis is not.

**What ChatGPT AVM does:** It uses the raw audio (not just text) to detect emotional cues — tone, pace, tremor. Since Sara uses Whisper (text only), she approximates this via `_analyze_user_state()` in `sara_brain.py`:

```python
# Current Sara implementation (text-based):
analysis = {
    "seems_distressed": any(word in input_lower for word in 
        ["sad", "depressed", "hurt", "hate", "awful", "terrible", "worst"]),
    "seems_excited": any(word in input_lower for word in 
        ["amazing", "awesome", "great", "excited", "love", "best", "!!!", "yes!"]),
    "is_defensive": any(phrase in input_lower for phrase in 
        ["whatever", "fine", "nothing", "doesn't matter"]),
    "is_opening_up": len(input_lower.split()) > 30,
}
```

Temperature then adjusts dynamically: 0.7 (distressed) → 0.85 (normal) → 0.95 (excited).

**Upgrade path** — add audio-based signals from speech timing:
```python
class EmotionDetector:
    def analyze(self, transcript: str, audio_duration: float, word_count: int) -> dict:
        words_per_second = word_count / audio_duration if audio_duration > 0 else 0
        
        # Fast speech = excitement or anxiety
        speech_pace = "fast" if words_per_second > 2.5 else "normal"
        # Many filler words = uncertainty
        hedging = any(w in transcript.lower() for w in ['i think', 'maybe', 'i guess'])
        
        distress_score = sum(1 for k in DISTRESS_KEYWORDS if k in transcript.lower())
        
        if distress_score >= 2 or (distress_score >= 1 and speech_pace == "fast"):
            return {"state": "distressed", "llm_temp": 0.6, "max_tokens": 80}
        # ...
```

---

## 7. ElevenLabs-Style Voice Expressiveness

⚠️ **Not yet implemented in Sara.** Kokoro-82M produces natural but flat audio.

**Ways to add expressiveness:**

**1. Thinking sounds — eliminate dead silence during LLM wait** (high value, easy):
```python
THINKING_SOUNDS = ["Hmm...", "Well...", "Let me think...", "Mmm.", "Yeah..."]

# In handle_user_speech() — play BEFORE streaming starts:
thinking = random.choice(THINKING_SOUNDS)
self.tts.speak(thinking)   # ~200ms — plays while Groq processes first tokens
# Then stream the actual response
chunks = self.brain.generate_response_streaming(transcription)
full_response = self.tts.speak_stream(chunks)
```
This eliminates the ~200–400ms dead silence gap between user finishing and Sara's first word. Pairs perfectly with the existing streaming implementation.

**2. SSML-style markers in TTS input:**
```python
def add_expressiveness(text: str, emotion: str) -> str:
    text = text.replace(',', ', ...')   # Slight pause
    text = text.replace('...', '... ')  # Longer pause
    if emotion == 'distressed':
        text = '. '.join(s.strip() for s in text.split('.'))
    return text
```

---

## 8. Practical Upgrade Roadmap for Sara

### Phase 1 — ✅ Done

| Feature | Status | Impact |
|---|---|---|
| ~~Streaming TTS~~ | ✅ **Done** | -60% perceived latency |
| Thinking sounds | ⬜ Next | Eliminates dead silence gap |
| Hysteresis VAD | ⬜ Next | -60% false barge-ins |
| Semantic EoT | ⬜ Next | Better turn detection |

### Phase 2 — Natural Feel (3–5 days)

| Feature | Implementation | Expected Impact |
|---|---|---|
| Barge-in with grace period | Monitor mic during TTS, ignore first 200ms | Natural interruptions |
| TTS position tracking | Know which streaming chunk was interrupted | Cleaner recovery |
| Backchannel detection | Classify short utterances, don't interrupt for them | Much less annoying |
| Progressive silence | Multi-tier proactive check-ins | Human-feeling pauses |

### Phase 3 — Production Polish (1–2 weeks)

| Feature | Implementation | Expected Impact |
|---|---|---|
| Emotion-aware pacing | Audio timing → adjust TTS rate + LLM params | Emotional intelligence |
| Speaker-adapted VAD | Learn user voice profile during session | Near-zero false positives |
| Backchannel responses | Sara says "mm-hmm" mid-user-speech | Feels deeply human |
| Semantic turn detection | Groq mini-call in parallel with STT | Much smarter endpointing |

---

## 9. Sara's Current Pipeline Diagram

```
                    ┌─────────────────────────────────────┐
                    │         SARA CONVERSATION LOOP      │
                    └─────────────────────────────────────┘

MICROPHONE INPUT
       │
       ▼
┌─────────────┐     ┌───────────────────┐     ┌─────────────────────┐
│ Silero VAD  │────►│ Single threshold  │────►│ Pause/Resume        │
│ (streaming  │     │ (0.5) ← upgrade   │     │ (crude AEC)         │
│  32ms chunk)│     │  to hysteresis    │     │ ← upgrade to real   │
└─────────────┘     └───────────────────┘     │   AEC + barge-in    │
                                              └─────────────────────┘
                                                       │
                                           480ms silence → END OF TURN
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │ Faster-Whisper  │
                                              │ (small, int8)   │
                                              │  ~500ms         │
                                              └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────────┐
                                              │ _analyze_user_state │
                                              │ (text-based emotion)│
                                              └─────────────────────┘
                           IMMEDIATE:                  │
                       ┌──────────────┐               ▼
                       │ [⬜ Planned]  │    ┌─────────────────────┐
                       │  Play        │◄───│ Groq Streaming      │
                       │  Thinking    │    │ (llama-3.3-70b)     │
                       │  Sound       │    │ ~200-400ms to first │
                       └──────────────┘    │  token              │
                                           └─────────────────────┘
                                                       │
                                           (sentence chunks stream in)
                                                       │
                                                       ▼
                                           ┌────────────────────────┐
                                           │  ✅ Sentence Chunker   │
                                           │  yield on . ! ?        │
                                           │  min 3 words           │
                                           └────────────────────────┘
                                                       │
                                                       ▼
                                           ┌────────────────────────┐
                                           │  ✅ Kokoro TTS          │
                                           │  speak_stream()        │
                                           │  play chunk → wait     │
                                           │  → next chunk arrives  │
                                           └────────────────────────┘
                                                       │
                                                SPEAKER OUTPUT

Legend: ✅ Implemented  ⬜ Planned
```

---

## 10. What Makes ChatGPT AVM Feel *Different*

After all this research, the key advantage of ChatGPT's Advanced Voice Mode isn't any single feature — it's the **end-to-end audio model** (GPT-4o). The model literally hears the user's voice, processes audio tokens directly, and generates audio tokens in response. It picks up on:

- **Hesitation** (the *hhh* sound before speaking)
- **Emotional tone** (shakiness, brightness in voice)
- **Laughter, sighs, "umms"** — all processed natively
- **Overlapping speech** — handled gracefully in the neural network itself

Sara, using the cascaded STT→LLM→TTS approach, cannot replicate this *directly*. But she can *compensate* with:
1. ✅ Faster, smarter turn-taking (streaming done — more to come)
2. ✅ Emotional depth in personality (her strong suit already)
3. ✅ True persistent memory (which ChatGPT currently lacks between sessions)
4. ⬜ Thinking sounds + barge-in (next up — see Phase 1/2 roadmap)
5. Personalization that ChatGPT's one-size-fits-all approach can't match

**Sara's advantage:** She can remember Jivit across sessions. ChatGPT AVM doesn't. Lean into that.

---

## References

- OpenAI Advanced Voice Mode documentation & release notes
- Softcery: Real-Time vs Turn-Based Voice Agent Architecture (2025)
- FireRedChat: Full-Duplex Voice Interaction System (arxiv 2509.06502)
- Notch: Turn Detection in Voice AI (2025)
- VoiceInfra: Voice AI Prompt Engineering Guide (2025)
- NVIDIA PersonaPlex-7B (Full-Duplex Model Card)
- Gnani.ai: Real-Time Barge-In AI for Voice Conversations