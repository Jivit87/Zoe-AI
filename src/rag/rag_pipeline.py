"""
Sara RAG Pipeline — Main Orchestrator
=======================================
Full pipeline: Ingest → Index → Retrieve → Rerank → Verify → Format

Usage:
    from src.rag.rag_pipeline import SaraRAG

    rag = SaraRAG(groq_client=groq)

    # Index a conversation turn
    rag.remember(speaker="user", text="I'm stressed about my job interview")

    # Retrieve relevant memories
    context = rag.recall(query="How is the user feeling?")

    # End of session
    rag.flush_session()

Implements 2025 SOTA:
  - Anthropic's Contextual Retrieval (67% fewer retrieval failures)
  - Hybrid Dense+Sparse retrieval with RRF fusion
  - Cross-encoder re-ranking with CRAG filtering
  - Adaptive retrieval gating (skip RAG for greetings)
  - Conversational query re-contextualization
  - Time-decay + MMR diversity
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Union

from groq import Groq

from src.rag.indexer import ConversationTurn, MemoryIndexer
from src.rag.query_processor import QueryProcessor
from src.rag.reranker import CrossEncoderReranker
from src.rag.retriever import HybridRetriever, MemoryChunk


class SaraRAG:
    """
    Production RAG pipeline for Sara's memory system.

    Architecture:
      INGESTION:  Turn → Contextual Chunks → Dual Index (ChromaDB + BM25)
      RETRIEVAL:  Gate → Rewrite → Hybrid Search → RRF → Rerank → CRAG → MMR → Format
    """

    def __init__(
        self,
        groq_client: Groq,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "sara_memories_v2",
        session_id: Optional[str] = None,
        use_reranker: bool = True,
        use_hyde: bool = False,  # Disabled for voice speed
        use_decomposition: bool = False,  # Disabled for voice speed
        use_recontextualization: bool = True, # Adds ~600ms latency
        use_mmr: bool = True,
        top_k_final: int = 5,
    ):
        self.groq = groq_client
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.use_reranker = use_reranker
        self.use_hyde = use_hyde
        self.use_decomposition = use_decomposition
        self.use_recontextualization = use_recontextualization
        self.use_mmr = use_mmr
        self.top_k_final = top_k_final

        # Component initialization
        print("[RAG] Initializing components...")

        self.indexer = MemoryIndexer(groq_client=groq_client)
        print("[RAG] ✓ Indexer ready (Contextual Retrieval)")

        self.retriever = HybridRetriever(
            collection_name=collection_name,
            persist_directory=persist_directory,
            top_k_dense=20,
            top_k_sparse=20,
            top_k_rerank=15,
        )
        print("[RAG] ✓ Hybrid retriever ready (Dense + BM25)")

        self.query_processor = QueryProcessor(groq_client=groq_client)
        print("[RAG] ✓ Query processor ready (Adaptive Gate + Re-context)")

        self.reranker = CrossEncoderReranker() if use_reranker else None
        if not use_reranker:
            print("[RAG] ✓ Reranker disabled")

        # Session turn buffer
        self._session_turns: List[ConversationTurn] = []
        self._recent_context: str = ""  # Running conversation context

        print(f"[RAG] ✓ Pipeline ready | session={self.session_id}")

    # ------------------------------------------------------------------ #
    # INGESTION API                                                        #
    # ------------------------------------------------------------------ #

    def remember(
        self,
        speaker: str,
        text: str,
        emotional_state: str = "neutral",
        index_immediately: bool = True,
    ) -> None:
        """
        Store a conversation turn in memory.
        Call after every user message AND Sara response.
        """
        turn = ConversationTurn(
            speaker=speaker,
            text=text,
            timestamp=time.time(),
            emotional_state=emotional_state,
            session_id=self.session_id,
        )
        self._session_turns.append(turn)

        # Update running context (for contextual retrieval prefix)
        self._recent_context += f"\n{speaker}: {text}"
        ctx_lines = self._recent_context.strip().split("\n")
        if len(ctx_lines) > 6:
            self._recent_context = "\n".join(ctx_lines[-6:])

        if index_immediately:
            chunks = self.indexer.index_turn(
                turn, recent_context=self._recent_context
            )
            if chunks:
                self.retriever.add_memories_batch(chunks)

    def flush_session(self) -> None:
        """
        End of session — create session-level summary chunk
        for high-level long-term memory retrieval.
        """
        if len(self._session_turns) >= 3:
            print(
                f"[RAG] Flushing session ({len(self._session_turns)} turns)..."
            )
            session_chunks = self.indexer.index_session(self._session_turns)
            # Only add the session summary (avoid re-indexing individual turns)
            summary_chunks = [
                c
                for c in session_chunks
                if c["metadata"].get("chunk_type") == "session_summary"
            ]
            if summary_chunks:
                self.retriever.add_memories_batch(summary_chunks)
                print("[RAG] ✓ Session summary indexed")
        self._session_turns = []
        self._recent_context = ""

    # ------------------------------------------------------------------ #
    # RETRIEVAL API                                                        #
    # ------------------------------------------------------------------ #

    def recall(
        self,
        query: str,
        conversation_context: str = "",
        top_k: Optional[int] = None,
        return_chunks: bool = False,
    ) -> Union[str, List[MemoryChunk]]:
        """
        Full RAG retrieval pipeline:
          1. Adaptive gate — skip if greeting/backchannel
          2. Conversational re-contextualization
          3. Hybrid search (Dense + Sparse + RRF + Time-decay)
          4. Cross-encoder re-ranking + CRAG filtering
          5. MMR diversity selection
          6. Format for Sara's prompt

        Args:
            query: Current user message
            conversation_context: Recent conversation for re-contextualization
            top_k: Number of memories to return
            return_chunks: Return raw MemoryChunk objects instead of string

        Returns:
            Formatted memory context string (or List[MemoryChunk])
        """
        top_k = top_k or self.top_k_final
        t0 = time.time()

        # Use internal running context if none provided
        if not conversation_context:
            conversation_context = self._recent_context

        # Stage 1: Query Processing (includes adaptive gate)
        processed = self.query_processor.process(
            query=query,
            conversation_context=conversation_context,
            use_recontextualization=self.use_recontextualization,
            use_hyde=self.use_hyde,
            use_decomposition=self.use_decomposition,
        )

        # Adaptive gate check
        if not processed["should_retrieve"]:
            return [] if return_chunks else ""

        # Collect all query variants
        all_queries = [processed["rewritten"]] + processed["sub_queries"]
        if processed["hyde_document"]:
            all_queries.append(processed["hyde_document"])

        # Deduplicate
        seen = set()
        unique_queries = []
        for q in all_queries:
            if q and q not in seen:
                seen.add(q)
                unique_queries.append(q)

        # Stage 2: Hybrid Retrieval + RRF + Time-Decay
        candidates = self.retriever.retrieve(
            queries=unique_queries, top_k=15
        )

        if not candidates:
            return [] if return_chunks else ""

        # Stage 3: Cross-Encoder Re-ranking + CRAG
        if self.reranker and len(candidates) > top_k:
            candidates = self.reranker.rerank(
                query=processed["rewritten"],
                chunks=candidates,
                top_k=top_k + 2,  # Buffer for MMR
            )

        # Stage 4: MMR Diversity
        if self.use_mmr and len(candidates) > top_k:
            candidates = self.retriever.maximal_marginal_relevance(
                chunks=candidates, lambda_param=0.7, top_k=top_k
            )
        else:
            candidates = candidates[:top_k]

        elapsed = time.time() - t0
        print(f"[RAG] Recalled {len(candidates)} memories in {elapsed:.2f}s")

        if return_chunks:
            return candidates

        return self._format_context(candidates)

    # ------------------------------------------------------------------ #
    # CONTEXT FORMATTING                                                   #
    # ------------------------------------------------------------------ #

    def _format_context(self, chunks: List[MemoryChunk]) -> str:
        """
        Format retrieved chunks into structured context for Sara's prompt.
        Groups by type, adds temporal markers.
        """
        if not chunks:
            return ""

        session_summaries = []
        facts = []
        verbatim = []

        for chunk in chunks:
            chunk_type = chunk.metadata.get("chunk_type", "contextual")
            speaker = chunk.metadata.get("speaker", "unknown")
            ts = chunk.metadata.get("timestamp", 0)
            age = self._age_description(ts)

            # Use raw_text for display if available (strip context prefix)
            display_text = chunk.metadata.get("raw_text", chunk.text)

            if chunk_type == "session_summary":
                session_summaries.append(f"  [{age}] {display_text}")
            elif chunk_type in ("facts", "summary"):
                facts.append(f"  [{age}] {display_text}")
            else:
                verbatim.append(f"  [{age}] {speaker}: {display_text}")

        sections = []
        if session_summaries:
            sections.append(
                "PAST SESSION HIGHLIGHTS:\n" + "\n".join(session_summaries)
            )
        if facts:
            sections.append("RELEVANT FACTS:\n" + "\n".join(facts))
        if verbatim:
            sections.append("RELEVANT EXCHANGES:\n" + "\n".join(verbatim))

        if not sections:
            return ""

        return (
            "=== SARA'S MEMORY ===\n"
            + "\n\n".join(sections)
            + "\n====================="
        )

    @staticmethod
    def _age_description(timestamp: float) -> str:
        """Human-readable age of a memory."""
        age = time.time() - timestamp
        if age < 120:
            return "just now"
        elif age < 3600:
            return f"{int(age / 60)}m ago"
        elif age < 86400:
            return f"{int(age / 3600)}h ago"
        elif age < 7 * 86400:
            return f"{int(age / 86400)}d ago"
        else:
            return f"{int(age / (7 * 86400))}w ago"

    # ------------------------------------------------------------------ #
    # DIAGNOSTICS                                                          #
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        total = self.retriever.collection.count()
        return {
            "total_chunks": total,
            "bm25_docs": len(self.retriever._bm25_docs),
            "session_id": self.session_id,
            "session_turns": len(self._session_turns),
        }
