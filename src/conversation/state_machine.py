"""
Conversation State Machine
===========================
Central state management for Sara's conversation flow.
All components check and transition state through this manager.

States:
    IDLE         → Waiting, nothing happening
    LISTENING    → User is speaking, audio buffering
    PROCESSING   → STT + LLM running
    SPEAKING     → TTS streaming chunks
    INTERRUPTED  → Barge-in during SPEAKING
    BACKCHANNEL  → Brief Sara "mm-hmm" during user speech
"""

import threading
import time
from enum import Enum
from typing import Optional, Callable


class ConversationState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    BACKCHANNEL = "backchannel"


class StateManager:
    """Thread-safe conversation state manager with cancel support."""

    def __init__(self):
        self.state = ConversationState.IDLE
        self.previous_state: Optional[ConversationState] = None
        self._lock = threading.Lock()

        # Cancel event — checked by speak_stream() between chunks
        self.cancel_event = threading.Event()

        # Callbacks for state transitions
        self._on_transition_callbacks: list[Callable] = []

        # Track timing
        self.state_entered_at: float = time.time()
        self.last_user_interaction: float = time.time()

        # Track what was spoken before interruption (for context)
        self.interrupted_spoken_text: Optional[str] = None
        self.interrupted_remaining_text: Optional[str] = None

    def transition(self, new_state: ConversationState):
        """Thread-safe state transition."""
        with self._lock:
            if new_state == self.state:
                return

            old_state = self.state
            self.previous_state = old_state
            self.state = new_state
            self.state_entered_at = time.time()

            # Handle transition side-effects
            if new_state == ConversationState.INTERRUPTED:
                self.cancel_event.set()  # Signal TTS to stop
            elif new_state == ConversationState.SPEAKING:
                self.cancel_event.clear()  # Reset cancel for new speech
            elif new_state == ConversationState.LISTENING:
                self.last_user_interaction = time.time()
            elif new_state == ConversationState.IDLE:
                self.last_user_interaction = time.time()

            # Fire callbacks
            for cb in self._on_transition_callbacks:
                try:
                    cb(old_state, new_state)
                except Exception as e:
                    print(f"⚠️  State transition callback error: {e}")

    def on_transition(self, callback: Callable):
        """Register a callback for state transitions."""
        self._on_transition_callbacks.append(callback)

    @property
    def is_speaking(self) -> bool:
        return self.state == ConversationState.SPEAKING

    @property
    def is_listening(self) -> bool:
        return self.state == ConversationState.LISTENING

    @property
    def is_idle(self) -> bool:
        return self.state in (ConversationState.IDLE, ConversationState.LISTENING)

    @property
    def should_cancel(self) -> bool:
        """Check if current operation (TTS) should be cancelled."""
        return self.cancel_event.is_set()

    def get_silence_duration(self) -> float:
        """Seconds since last user interaction."""
        return time.time() - self.last_user_interaction

    def touch(self):
        """Update last interaction timestamp."""
        self.last_user_interaction = time.time()

    def store_interruption_context(self, spoken: str, remaining: str):
        """Store what was said before interruption for context."""
        self.interrupted_spoken_text = spoken
        self.interrupted_remaining_text = remaining

    def clear_interruption_context(self):
        self.interrupted_spoken_text = None
        self.interrupted_remaining_text = None

    def __repr__(self):
        return f"StateManager(state={self.state.value}, silence={self.get_silence_duration():.1f}s)"
