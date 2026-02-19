"""
Memory Indexer: Anthropic's Contextual Retrieval + Multi-Chunk Indexing
========================================================================
Each conversation turn is stored as MULTIPLE overlapping representations:
  1. Contextual chunk — verbatim text WITH context prefix (Anthropic technique)
  2. Extracted facts/entities — for precision retrieval
  3. Summary — 1-sentence distillation for broad queries
  4. Session digest — end-of-session overview for long-term memory

The contextual prefix (50-100 tokens prepended to each chunk) reduces
retrieval failures by up to 67% compared to raw chunking. (Anthropic, Sep 2024)
"""

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from groq import Groq


@dataclass
class ConversationTurn:
    """A single conversation turn ready for indexing."""

    speaker: str  # "user" or "sara"
    text: str
    timestamp: float = None
    emotional_state: str = "neutral"
    session_id: str = ""

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class MemoryIndexer:
    """
    Transforms conversation turns into rich, searchable memory chunks.

    Chunk types:
      - "contextual"  : Verbatim text with Anthropic-style context prefix
      - "facts"       : Extracted factual claims / entities
      - "summary"     : LLM-generated brief summary
      - "session_summary" : End-of-session digest
    """

    CHUNK_SIZE = 400  # chars
    CHUNK_OVERLAP = 80  # chars overlap between splits

    def __init__(self, groq_client: Groq, model: str = "llama-3.3-70b-versatile"):
        self.client = groq_client
        self.model = model

    def index_turn(
        self, turn: ConversationTurn, recent_context: str = ""
    ) -> List[Dict]:
        """
        Convert a single conversation turn into indexable chunks.

        Args:
            turn: The conversation turn to index
            recent_context: Last few exchanges for contextual prefix generation

        Returns:
            List of dicts: {"id": str, "text": str, "metadata": dict}
        """
        chunks = []
        base_id = str(uuid.uuid4())[:8]
        base_meta = {
            "speaker": turn.speaker,
            "timestamp": turn.timestamp,
            "session_id": turn.session_id,
            "emotional_state": turn.emotional_state,
            "source_turn_id": base_id,
        }

        # 1. Contextual chunks (Anthropic's Contextual Retrieval)
        # — prepend context prefix to each verbatim chunk
        text_splits = self._split_text(turn.text)
        context_prefix = self._generate_context_prefix(
            turn.text, turn.speaker, recent_context
        )

        for i, text_chunk in enumerate(text_splits):
            # Prepend context to the chunk for richer embeddings
            contextualized = f"{context_prefix}\n{text_chunk}" if context_prefix else text_chunk
            chunks.append(
                {
                    "id": f"{base_id}_ctx{i}",
                    "text": contextualized,
                    "metadata": {
                        **base_meta,
                        "chunk_type": "contextual",
                        "chunk_index": i,
                        "raw_text": text_chunk,  # Keep original for display
                    },
                }
            )

        # 2. Extract facts and entities via Groq
        extracted = self._extract_metadata(turn.text, turn.speaker)

        if extracted.get("facts"):
            facts_text = (
                f"Facts from {turn.speaker}: " + " | ".join(extracted["facts"])
            )
            chunks.append(
                {
                    "id": f"{base_id}_facts",
                    "text": facts_text,
                    "metadata": {
                        **base_meta,
                        "chunk_type": "facts",
                        "entities": json.dumps(
                            extracted.get("entities", [])
                        ),
                    },
                }
            )

        if extracted.get("summary"):
            chunks.append(
                {
                    "id": f"{base_id}_summary",
                    "text": extracted["summary"],
                    "metadata": {
                        **base_meta,
                        "chunk_type": "summary",
                        "emotion_detected": extracted.get(
                            "emotion_detected", "neutral"
                        ),
                    },
                }
            )

        return chunks

    def index_session(self, turns: List[ConversationTurn]) -> List[Dict]:
        """Index a full session — individual turns + session-level summary."""
        all_chunks = []

        # Build running context for contextual retrieval
        running_context = ""
        for turn in turns:
            turn_chunks = self.index_turn(turn, recent_context=running_context)
            all_chunks.extend(turn_chunks)
            # Update running context (keep last 3 exchanges)
            running_context += f"\n{turn.speaker}: {turn.text}"
            lines = running_context.strip().split("\n")
            if len(lines) > 6:
                running_context = "\n".join(lines[-6:])

        # Session-level summary
        if len(turns) >= 3:
            session_summary = self._summarize_session(turns)
            if session_summary:
                session_id = turns[0].session_id or str(uuid.uuid4())[:8]
                all_chunks.append(
                    {
                        "id": f"session_{session_id}_summary",
                        "text": session_summary,
                        "metadata": {
                            "chunk_type": "session_summary",
                            "session_id": session_id,
                            "timestamp": turns[-1].timestamp,
                            "turn_count": len(turns),
                        },
                    }
                )

        return all_chunks

    # ------------------------------------------------------------------ #
    # CONTEXTUAL RETRIEVAL (Anthropic technique)                           #
    # ------------------------------------------------------------------ #

    def _generate_context_prefix(
        self, text: str, speaker: str, recent_context: str
    ) -> str:
        """
        Generate a short context prefix for Anthropic-style Contextual Retrieval.

        This 50-100 token prefix situates the chunk within the broader
        conversation, so it remains meaningful even when retrieved in isolation.
        """
        if not recent_context.strip():
            # First turn — minimal prefix
            return f"[{speaker} speaking at the start of the conversation]"

        prompt = f"""You are creating a brief context prefix for a conversation chunk to help with memory retrieval.

Recent conversation:
{recent_context}

Current chunk from {speaker}: "{text[:300]}"

Write a concise 1-2 sentence prefix that situates this chunk in context. Include:
- Who is speaking and the conversation topic
- Any emotional context or key references

Return ONLY the prefix text, wrapped in square brackets. Example:
[User discussing job interview anxiety after mentioning they have one tomorrow. Emotional state: stressed.]"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1,
            )
            prefix = response.choices[0].message.content.strip()
            # Ensure it starts with [ and ends with ]
            if not prefix.startswith("["):
                prefix = f"[{prefix}"
            if not prefix.endswith("]"):
                prefix = f"{prefix}]"
            return prefix
        except Exception as e:
            print(f"[Indexer] Context prefix generation failed: {e}")
            return f"[{speaker} speaking]"

    # ------------------------------------------------------------------ #
    # METADATA EXTRACTION                                                  #
    # ------------------------------------------------------------------ #

    def _extract_metadata(self, text: str, speaker: str) -> Dict[str, Any]:
        """
        Use LLM to extract structured metadata from a conversation turn.
        Returns: {facts, entities, summary, emotion_detected}
        """
        prompt = f"""Analyze this conversation turn from "{speaker}" and extract:
1. Key facts/statements (list of short strings)
2. Named entities (people, places, things mentioned)
3. One-sentence summary
4. Primary emotion detected

Text: "{text}"

Return ONLY valid JSON:
{{
  "facts": ["fact1", "fact2"],
  "entities": ["entity1"],
  "summary": "one sentence summary",
  "emotion_detected": "emotion"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            print(f"[Indexer] Metadata extraction failed: {e}")

        return {}

    # ------------------------------------------------------------------ #
    # TEXT SPLITTING                                                        #
    # ------------------------------------------------------------------ #

    def _split_text(self, text: str) -> List[str]:
        """Split long text into overlapping chunks at sentence boundaries."""
        if len(text) <= self.CHUNK_SIZE:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.CHUNK_SIZE
            chunk = text[start:end]
            # Try to split at sentence boundary
            last_period = max(
                chunk.rfind(". "), chunk.rfind("! "), chunk.rfind("? ")
            )
            if last_period > self.CHUNK_SIZE // 2:
                chunk = chunk[: last_period + 1]
                end = start + last_period + 1

            chunks.append(chunk.strip())
            start = end - self.CHUNK_OVERLAP

        return [c for c in chunks if c]

    # ------------------------------------------------------------------ #
    # SESSION SUMMARY                                                      #
    # ------------------------------------------------------------------ #

    def _summarize_session(
        self, turns: List[ConversationTurn]
    ) -> Optional[str]:
        """Generate a high-level summary of an entire session."""
        conversation = "\n".join(
            f"{t.speaker.upper()}: {t.text}" for t in turns[-20:]
        )
        prompt = f"""Summarize this conversation between a user and Sara (AI companion) in 3-4 sentences.
Focus on: topics discussed, emotional arc, key facts about the user revealed.

Conversation:
{conversation}

Return ONLY the summary paragraph."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return None
