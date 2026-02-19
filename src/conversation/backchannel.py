"""
Backchannel Detection & Response
==================================
Two-part module:

1. BackchannelClassifier — identifies short user utterances ("yeah",
   "mm-hmm", "okay") as acknowledgments, NOT new turns or interruptions.

2. BackchannelManager — Sara's own backchannel responses ("Mm-hmm...",
   "I see.") during user mid-sentence pauses, signaling she's listening.
"""

import time
import random


# Words that signal acknowledgment, not a new conversational turn
BACKCHANNEL_WORDS = {
    "yeah", "yes", "yep", "yup",
    "mm", "hmm", "mmm", "mhm", "mm-hmm", "uh-huh", "uh huh",
    "okay", "ok", "alright",
    "right", "sure", "got it",
    "ah", "oh", "huh",
    "cool", "nice", "wow",
    "I see", "i see",
}


class BackchannelClassifier:
    """Classifies short utterances as backchannels vs real speech."""

    def __init__(self, max_words: int = 3):
        self.max_words = max_words

    def is_backchannel(self, transcript: str) -> bool:
        """
        Check if a transcript is a backchannel acknowledgment.

        Args:
            transcript: The transcribed text from STT

        Returns:
            True if this is a backchannel (not a real turn)
        """
        cleaned = transcript.lower().strip().rstrip(".,!?")

        # Must be short
        words = cleaned.split()
        if len(words) > self.max_words:
            return False

        # Check against known backchannel patterns
        # Check full phrase first, then individual words
        if cleaned in BACKCHANNEL_WORDS:
            return True

        # Check if all individual words are backchannels
        if all(w in BACKCHANNEL_WORDS for w in words):
            return True

        return False


class BackchannelManager:
    """
    Generates Sara's backchannel responses during user speech.

    While the user is speaking, Sara can say "Mm-hmm" or "I see"
    during natural mid-sentence pauses (300–600ms) to signal
    she's actively listening.
    """

    RESPONSES = [
        "Mm-hmm...",
        "I see.",
        "Yeah...",
        "Mmm.",
        "Go on.",
        "Right.",
    ]

    def __init__(self, min_gap_seconds: float = 20.0):
        self.min_gap_seconds = min_gap_seconds
        self._last_backchannel_time: float = 0

    def should_backchannel(
        self,
        pause_duration_ms: float,
        user_is_mid_sentence: bool,
    ) -> bool:
        """
        Check if Sara should produce a backchannel response.

        Args:
            pause_duration_ms: Duration of the current user pause
            user_is_mid_sentence: Whether the pause is mid-sentence (not end-of-turn)

        Returns:
            True if Sara should backchannel now
        """
        if not user_is_mid_sentence:
            return False

        # Only on natural mid-sentence pauses (300–600ms)
        if pause_duration_ms < 300 or pause_duration_ms > 600:
            return False

        # Rate-limit: don't backchannel too frequently
        if time.time() - self._last_backchannel_time < self.min_gap_seconds:
            return False

        return True

    def get_response(self) -> str:
        """Get a random backchannel response and update timing."""
        self._last_backchannel_time = time.time()
        return random.choice(self.RESPONSES)

    def reset(self):
        """Reset timing state."""
        self._last_backchannel_time = 0
