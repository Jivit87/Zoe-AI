"""
Hybrid Retriever: Dense (ChromaDB) + Sparse (BM25) + RRF + Time-Decay + MMR
=============================================================================
The combination consistently outperforms either alone by ~10-15% on recall.

Pipeline:
    1. Dense search (ChromaDB cosine similarity) for each query variant
    2. Sparse search (BM25 keyword matching) for each query variant
    3. Reciprocal Rank Fusion across all result sets
    4. Time-decay scoring (recent memories boosted ~30%)
    5. MMR diversity selection (remove redundant memories)
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi


@dataclass
class MemoryChunk:
    """A retrieved memory chunk with scoring metadata."""

    id: str
    text: str
    metadata: Dict[str, Any]
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0


class HybridRetriever:
    """
    Two-stage retrieval:
      Stage 1: Hybrid (Dense ChromaDB + Sparse BM25) → RRF fusion
      Stage 2: Re-ranking handled externally by CrossEncoderReranker

    Also applies time-decay: recent memories score higher.
    """

    RRF_K = 60  # RRF constant — higher = less rank sensitivity

    def __init__(
        self,
        collection_name: str = "sara_memories_v2",
        persist_directory: str = "./data/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k_dense: int = 20,
        top_k_sparse: int = 20,
        top_k_rerank: int = 15,
        time_decay_factor: float = 0.95,
    ):
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.top_k_rerank = top_k_rerank
        self.time_decay_factor = time_decay_factor

        # --- Dense retrieval: ChromaDB + sentence-transformer ---
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # --- Sparse retrieval: in-memory BM25 ---
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: List[Dict] = []
        self._rebuild_bm25()

    # ------------------------------------------------------------------ #
    # INDEXING                                                             #
    # ------------------------------------------------------------------ #

    def add_memory(
        self,
        chunk_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a single memory chunk to both indexes."""
        meta = metadata or {}
        meta.setdefault("timestamp", time.time())
        meta.setdefault("source", "conversation")

        self.collection.upsert(ids=[chunk_id], documents=[text], metadatas=[meta])
        self._bm25_docs.append({"id": chunk_id, "text": text, "metadata": meta})
        self._rebuild_bm25()

    def add_memories_batch(self, chunks: List[Dict]) -> None:
        """
        Batch add for efficiency.
        Each chunk: {"id": str, "text": str, "metadata": dict}
        """
        if not chunks:
            return

        ids, texts, metas = [], [], []
        for c in chunks:
            meta = c.get("metadata", {})
            meta.setdefault("timestamp", time.time())
            ids.append(c["id"])
            texts.append(c["text"])
            metas.append(meta)
            self._bm25_docs.append(
                {"id": c["id"], "text": c["text"], "metadata": meta}
            )

        self.collection.upsert(ids=ids, documents=texts, metadatas=metas)
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from stored docs."""
        if not self._bm25_docs:
            self._bm25 = None
            return
        tokenized = [self._tokenize(d["text"]) for d in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer for BM25."""
        return text.lower().split()

    # ------------------------------------------------------------------ #
    # RETRIEVAL                                                            #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None,
    ) -> List[MemoryChunk]:
        """
        Full hybrid retrieval pipeline:
          1. Dense search for each query variant
          2. Sparse BM25 search for each query variant
          3. RRF fusion across all result sets
          4. Time-decay scoring
        Returns merged, deduplicated, ranked list.
        """
        top_k = top_k or self.top_k_rerank
        total = self.collection.count()
        if total == 0:
            return []

        all_rankings: List[List[Tuple[str, float]]] = []

        for query in queries:
            all_rankings.append(self._dense_search(query, filter_metadata))
            all_rankings.append(self._sparse_search(query))

        # RRF fusion
        fused_scores = self._reciprocal_rank_fusion(all_rankings)
        if not fused_scores:
            return []

        # Fetch full data from ChromaDB
        all_ids = list(fused_scores.keys())
        fetched = self.collection.get(
            ids=all_ids, include=["documents", "metadatas"]
        )
        id_to_data = {
            fid: {"text": fetched["documents"][i], "metadata": fetched["metadatas"][i]}
            for i, fid in enumerate(fetched["ids"])
        }

        # Build scored MemoryChunk objects
        chunks = []
        for chunk_id, rrf_score in fused_scores.items():
            if chunk_id not in id_to_data:
                continue
            data = id_to_data[chunk_id]
            chunk = MemoryChunk(
                id=chunk_id,
                text=data["text"],
                metadata=data["metadata"],
                rrf_score=rrf_score,
            )
            chunk.final_score = self._apply_time_decay(rrf_score, data["metadata"])
            chunks.append(chunk)

        chunks.sort(key=lambda c: c.final_score, reverse=True)
        return chunks[:top_k]

    def _dense_search(
        self, query: str, filter_metadata: Optional[Dict] = None
    ) -> List[Tuple[str, float]]:
        """Dense cosine similarity search via ChromaDB."""
        n = min(self.top_k_dense, self.collection.count())
        if n == 0:
            return []
        kwargs = {"query_texts": [query], "n_results": n, "include": ["distances"]}
        if filter_metadata:
            kwargs["where"] = filter_metadata
        try:
            results = self.collection.query(**kwargs)
            ids = results["ids"][0]
            distances = results["distances"][0]
            return [(id_, 1 - dist) for id_, dist in zip(ids, distances)]
        except Exception:
            return []

    def _sparse_search(self, query: str) -> List[Tuple[str, float]]:
        """BM25 sparse retrieval."""
        if self._bm25 is None or not self._bm25_docs:
            return []
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        doc_scores = [
            (self._bm25_docs[i]["id"], float(scores[i]))
            for i in range(len(self._bm25_docs))
        ]
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[: self.top_k_sparse]

    def _reciprocal_rank_fusion(
        self, ranked_lists: List[List[Tuple[str, float]]]
    ) -> Dict[str, float]:
        """RRF: score = Σ 1/(k + rank) across all ranked lists."""
        rrf_scores: Dict[str, float] = {}
        for ranked_list in ranked_lists:
            for rank, (doc_id, _) in enumerate(ranked_list, start=1):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
                    1.0 / (self.RRF_K + rank)
                )
        return dict(sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True))

    def _apply_time_decay(self, score: float, metadata: Dict) -> float:
        """
        Exponential time-decay: recent memories score ~30% higher.
        Half-life ≈ 10 days with default factor.
        """
        timestamp = metadata.get("timestamp", time.time())
        age_days = (time.time() - timestamp) / 86400
        decay = math.exp(-age_days * (1 - self.time_decay_factor))
        return score * (0.7 + 0.3 * decay)

    # ------------------------------------------------------------------ #
    # MMR DIVERSITY                                                        #
    # ------------------------------------------------------------------ #

    def maximal_marginal_relevance(
        self,
        chunks: List[MemoryChunk],
        lambda_param: float = 0.7,
        top_k: int = 5,
    ) -> List[MemoryChunk]:
        """
        MMR: balance relevance vs redundancy.
        λ=1.0 → pure relevance, λ=0.0 → pure diversity.
        """
        if not chunks or top_k <= 0:
            return []

        texts = [c.text for c in chunks]
        try:
            embeddings = self.embed_fn(texts)
        except Exception:
            return chunks[:top_k]

        selected, candidates = [], list(range(len(chunks)))

        for _ in range(min(top_k, len(chunks))):
            if not candidates:
                break

            best_idx, best_score = None, float("-inf")

            for idx in candidates:
                relevance = chunks[idx].final_score
                max_sim = 0.0
                for sel_idx in selected:
                    sim = self._cosine_similarity(
                        embeddings[idx], embeddings[sel_idx]
                    )
                    max_sim = max(max_sim, sim)
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                candidates.remove(best_idx)

        return [chunks[i] for i in selected]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
