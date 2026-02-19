"""
Emotion Detection from Text + Audio Timing
=============================================
Combines keyword-based text analysis with speech timing signals
(words-per-second, hedging phrases) to estimate the user's emotional
state. Returns parameters that the LLM brain and TTS use to adapt
response style.

This replaces the hardcoded "neutral" emotional_state in main.py.
"""

from typing import Dict


# Keyword sets for text-based emotion analysis
DISTRESS_KEYWORDS = [
    "sad", "depressed", "hurt", "hate", "awful", "terrible", "worst",
    "stressed", "anxious", "scared", "worried", "can't", "help me",
    "please", "horrible", "lonely", "hopeless", "crying", "breakdown",
    "overwhelmed", "exhausted", "miserable", "angry", "furious",
]

EXCITEMENT_KEYWORDS = [
    "amazing", "awesome", "great", "excited", "love", "best",
    "incredible", "fantastic", "wonderful", "perfect", "brilliant",
    "can't wait", "wow", "yes!", "!!!", "omg",
]

HEDGING_PHRASES = [
    "i think", "maybe", "i guess", "sort of", "kind of",
    "i don't know", "not sure", "possibly", "perhaps",
]


class EmotionDetector:
    """
    Detect emotion from transcript text and speech characteristics.

    Returns a dict with:
        state:      "distressed" | "excited" | "uncertain" | "neutral"
        llm_temp:   Recommended LLM temperature
        max_tokens: Recommended max tokens for response
    """

    def analyze(
        self,
        transcript: str,
        audio_duration: float = 0.0,
        word_count: int = 0,
    ) -> Dict:
        """
        Analyze user emotion from text content and speech timing.

        Args:
            transcript:     Transcribed user speech
            audio_duration: Duration of the audio in seconds (0 if unavailable)
            word_count:     Number of words in the transcript (0 = auto-count)

        Returns:
            Dict with "state", "llm_temp", and "max_tokens"
        """
        text_lower = transcript.lower()

        if word_count == 0:
            word_count = len(transcript.split())

        # Text-based scoring
        distress_score = sum(1 for k in DISTRESS_KEYWORDS if k in text_lower)
        excitement_score = sum(1 for k in EXCITEMENT_KEYWORDS if k in text_lower)
        hedging = any(phrase in text_lower for phrase in HEDGING_PHRASES)

        # Audio timing signals (when available)
        speech_pace = "normal"
        if audio_duration > 0 and word_count > 0:
            words_per_second = word_count / audio_duration
            if words_per_second > 2.5:
                speech_pace = "fast"  # Excitement or anxiety
            elif words_per_second < 1.0:
                speech_pace = "slow"  # Sadness or deliberation

        # Decision logic — prioritize distress
        if distress_score >= 2 or (distress_score >= 1 and speech_pace == "fast"):
            return {
                "state": "distressed",
                "llm_temp": 0.6,
                "max_tokens": 80,
            }
        elif excitement_score >= 2 or (excitement_score >= 1 and speech_pace == "fast"):
            return {
                "state": "excited",
                "llm_temp": 0.9,
                "max_tokens": 150,
            }
        elif hedging or speech_pace == "slow":
            return {
                "state": "uncertain",
                "llm_temp": 0.7,
                "max_tokens": 100,
            }
        else:
            return {
                "state": "neutral",
                "llm_temp": 0.85,
                "max_tokens": 150,
            }
