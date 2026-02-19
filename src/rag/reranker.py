"""
Cross-Encoder Re-Ranker + Corrective-RAG Filtering
====================================================
Stage 2 of the retrieval pipeline:
  1. Cross-encoder re-ranking (joint query-document scoring)
  2. CRAG relevance filtering (discard low-confidence results)

Cross-encoders jointly encode query+document giving much higher precision
than bi-encoder cosine similarity. Too slow for the full index (~100ms for
15 pairs) but perfect for re-ranking top-N candidates.

Corrective-RAG (CRAG): After re-ranking, chunks below a minimum relevance
threshold are discarded to prevent injecting noise into Sara's context.
"""

from typing import List, Optional

from src.rag.retriever import MemoryChunk


class CrossEncoderReranker:
    """
    Re-scores retrieved candidates using a cross-encoder model.
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, high quality)
    Falls back to no-op if sentence-transformers unavailable.

    Includes Corrective-RAG: chunks below min_relevance are discarded.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        min_relevance: float = 0.15,  # CRAG threshold (normalized 0-1)
    ):
        self.model = None
        self.model_name = model_name
        self.min_relevance = min_relevance
        self._load_model()

    def _load_model(self):
        """Load cross-encoder model with graceful fallback."""
        try:
            from sentence_transformers.cross_encoder import CrossEncoder

            self.model = CrossEncoder(self.model_name)
            print(f"[Reranker] ✓ Cross-encoder loaded: {self.model_name}")
        except Exception as e:
            print(f"[Reranker] Cross-encoder unavailable, will skip rerank: {e}")

    def rerank(
        self,
        query: str,
        chunks: List[MemoryChunk],
        top_k: Optional[int] = None,
    ) -> List[MemoryChunk]:
        """
        Re-score chunks using cross-encoder + CRAG filtering.

        1. Score each (query, chunk) pair with cross-encoder
        2. Blend: 40% original RRF score + 60% cross-encoder score
        3. CRAG: Discard chunks below min_relevance threshold
        4. Return sorted top-k
        """
        if not chunks:
            return []

        top_k = top_k or len(chunks)

        if self.model is None:
            return chunks[:top_k]

        # Cross-encoder scoring: (query, document) pairs
        pairs = [[query, chunk.text] for chunk in chunks]

        try:
            scores = self.model.predict(pairs)

            for chunk, score in zip(chunks, scores):
                chunk.rerank_score = float(score)
                normalized = self._normalize(float(score), scores)
                # Blend: 40% original RRF/time-decay + 60% cross-encoder
                chunk.final_score = (
                    0.4 * chunk.final_score + 0.6 * normalized
                )

            # CRAG: Discard chunks below relevance threshold
            chunks = [c for c in chunks if c.final_score >= self.min_relevance]

            chunks.sort(key=lambda c: c.final_score, reverse=True)

        except Exception as e:
            print(f"[Reranker] Reranking failed: {e}")

        return chunks[:top_k]

    @staticmethod
    def _normalize(score: float, all_scores) -> float:
        """Min-max normalize a score within the batch."""
        min_s = float(min(all_scores))
        max_s = float(max(all_scores))
        if max_s == min_s:
            return 0.5
        return (score - min_s) / (max_s - min_s)
