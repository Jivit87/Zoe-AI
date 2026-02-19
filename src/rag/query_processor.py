"""
Query Processor: Adaptive Gating + Conversational Re-Context + HyDE
=====================================================================
Transforms raw queries into retrieval-optimized forms:

  1. Adaptive Gating    — Should we even search memory? (skip for greetings)
  2. Conversational Re-Context — Resolve pronouns using recent conversation
  3. Query Rewriting    — Rephrase for better semantic match
  4. HyDE (optional)    — Hypothetical document for embedding search
  5. Decomposition (opt) — Break complex queries into sub-queries
"""

import json
import re
from typing import Dict, List, Optional

from groq import Groq


# Words/phrases that indicate no memory retrieval is needed
_SKIP_PATTERNS = {
    "hello", "hi", "hey", "howdy", "sup", "yo",
    "bye", "goodbye", "see you", "goodnight", "good night",
    "yeah", "yes", "yep", "yup", "no", "nah", "nope",
    "okay", "ok", "alright", "sure", "right",
    "mm", "hmm", "mmm", "mhm", "mm-hmm", "uh-huh", "uh huh",
    "ah", "oh", "huh", "uh", "um",
    "cool", "nice", "wow",
    "thanks", "thank you", "ty",
}


class QueryProcessor:
    """
    Multi-stage query processing for optimal retrieval:

    1. Adaptive gate — skip RAG for greetings/backchannels
    2. Conversational re-context — resolve "that", "he", "it"
    3. Semantic rewriting — enrich query for embedding search
    4. HyDE — hypothetical memory generation (optional, slow)
    5. Decomposition — break complex queries (optional, slow)
    """

    def __init__(
        self, groq_client: Groq, model: str = "llama-3.3-70b-versatile"
    ):
        self.client = groq_client
        self.model = model

    # ------------------------------------------------------------------ #
    # ADAPTIVE RETRIEVAL GATING                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def should_retrieve(query: str) -> bool:
        """
        Adaptive gate: Should we search memory for this query?

        Returns False for greetings, backchannels, and trivially short
        messages that won't benefit from memory retrieval.
        Saves ~200ms on ~40% of conversation turns.
        """
        cleaned = query.lower().strip().rstrip(".,!?")

        # Very short messages are usually backchannels
        if len(cleaned) < 3:
            return False

        # Check against known skip patterns
        words = cleaned.split()
        if len(words) <= 2 and cleaned in _SKIP_PATTERNS:
            return False
        if len(words) == 1 and words[0] in _SKIP_PATTERNS:
            return False

        # Single-word acknowledgments
        if all(w in _SKIP_PATTERNS for w in words) and len(words) <= 3:
            return False

        return True

    # ------------------------------------------------------------------ #
    # CONVERSATIONAL RE-CONTEXTUALIZATION                                  #
    # ------------------------------------------------------------------ #

    def recontextualize_query(
        self, query: str, conversation_context: str
    ) -> str:
        """
        Resolve pronouns and ambiguous references using recent conversation.

        "How's that going?" → "How is the user's job interview preparation going?"

        This is ALWAYS applied (unlike HyDE/decomposition which are optional)
        because multi-turn conversations constantly produce ambiguous queries.
        """
        if not conversation_context.strip():
            return query

        # Quick heuristic: if query already has specific nouns, skip LLM call
        contains_pronoun = any(
            p in query.lower().split()
            for p in ["it", "that", "this", "he", "she", "they", "them", "there"]
        )
        is_short = len(query.split()) < 6

        if not contains_pronoun and not is_short:
            return query  # Already specific enough

        prompt = f"""Rewrite this query by resolving any pronouns or ambiguous references using the conversation context.

Recent conversation:
{conversation_context}

User's query: "{query}"

Rewrite the query to be self-contained and specific. If the query is already clear, return it unchanged.
Return ONLY the rewritten query, nothing else."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1,
            )
            rewritten = response.choices[0].message.content.strip()
            # Strip quotes if LLM wraps it
            return rewritten.strip('"').strip("'")
        except Exception:
            return query

    # ------------------------------------------------------------------ #
    # QUERY REWRITING                                                      #
    # ------------------------------------------------------------------ #

    def rewrite_query(
        self, query: str, conversation_context: str = ""
    ) -> str:
        """Rewrite query to be semantically richer for embedding search."""
        prompt = f"""You are helping retrieve relevant memories from Sara, an AI companion.

User message: "{query}"
Recent conversation: {conversation_context or "None"}

Rewrite this as a semantic search query for memory retrieval.
Expand abbreviations, add relevant context, make it more explicit.
Return ONLY the rewritten query."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip().strip('"')
        except Exception:
            return query

    # ------------------------------------------------------------------ #
    # HyDE (Optional — disabled for voice by default)                      #
    # ------------------------------------------------------------------ #

    def generate_hyde_document(
        self, query: str, context: str = ""
    ) -> str:
        """
        HyDE: Generate a hypothetical memory that would answer this query.
        The hypothetical doc is then embedded for search — bridges vocabulary gap.
        """
        prompt = f"""Sara is an AI companion. Generate a hypothetical conversation memory relevant to this query.

Query: "{query}"
Context: {context or "general conversation"}

Write a realistic 2-3 sentence memory snippet. Return ONLY the snippet."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return query

    # ------------------------------------------------------------------ #
    # QUERY DECOMPOSITION (Optional — disabled for voice by default)       #
    # ------------------------------------------------------------------ #

    def decompose_query(self, query: str) -> List[str]:
        """Break a complex query into 2-4 simpler sub-queries."""
        prompt = f"""Break this query into 2-4 simple sub-queries for memory retrieval.
Query: "{query}"

Return a JSON array of strings. Example: ["sub-query 1", "sub-query 2"]
If already simple, return just: ["{query}"]
Return ONLY valid JSON."""

        try:
            result = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
            )
            raw = result.choices[0].message.content.strip()
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                sub_queries = json.loads(match.group())
                if query not in sub_queries:
                    sub_queries.insert(0, query)
                return sub_queries[:4]
        except Exception:
            pass
        return [query]

    # ------------------------------------------------------------------ #
    # FULL PROCESSING PIPELINE                                             #
    # ------------------------------------------------------------------ #

    def process(
        self,
        query: str,
        conversation_context: str = "",
        use_recontextualization: bool = True,
        use_hyde: bool = False,
        use_decomposition: bool = False,
    ) -> Dict:
        """
        Full query processing pipeline.

        Returns dict with all query variants for retrieval:
          {original, rewritten, sub_queries, hyde_document, should_retrieve}
        """
        result = {
            "original": query,
            "rewritten": query,
            "sub_queries": [query],
            "hyde_document": None,
            "should_retrieve": True,
        }

        # Step 0: Adaptive gate
        if not self.should_retrieve(query):
            result["should_retrieve"] = False
            return result

        # Step 1: Conversational re-contextualization (optional — adds ~600ms)
        if use_recontextualization:
            recontextualized = self.recontextualize_query(
                query, conversation_context
            )
            result["rewritten"] = recontextualized
        else:
            result["rewritten"] = query

        # Step 2: Decomposition (optional — adds ~200ms)
        if use_decomposition:
            result["sub_queries"] = self.decompose_query(
                result["rewritten"]
            )

        # Step 3: HyDE (optional — adds ~300ms)
        if use_hyde:
            result["hyde_document"] = self.generate_hyde_document(
                result["rewritten"], conversation_context
            )

        return result
