"""
Sara RAG — Retrieval-Augmented Generation Memory System
========================================================
Production-grade memory pipeline with SOTA 2025 techniques:
- Hybrid retrieval (Dense + Sparse)
- Anthropic's Contextual Retrieval
- Cross-encoder re-ranking + Corrective-RAG
- Adaptive retrieval gating
- Conversational query re-contextualization
"""

from src.rag.rag_pipeline import SaraRAG

__all__ = ["SaraRAG"]
