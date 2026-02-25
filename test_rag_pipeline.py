"""
Test: RAG Architecture Verification
====================================
This script tests the entire RAG pipeline end-to-end to ensure
ChromaDB, BM25, the Indexer, and the Retriever are working.

It simulates a short session, checks memory retrieval, and then
flushes the session to verify that the LLM API does not hit
rate limits during the summary/metadata extraction phase.
"""

from src.rag.rag_pipeline import SaraRAG
from src.llm.sara_brain import SaraBrain
import time


def test_rag():
    print("====================================")
    print("  Testing RAG Architecture End-to-End")
    print("====================================\n")

    # Initialize SaraBrain (which spins up the SaraRAG pipeline automatically)
    print("🧠 Initializing SaraBrain and RAG pipeline...")
    brain = SaraBrain()
    rag = brain.rag

    print(f"✓ RAG initialized. Session ID: {rag.session_id}")
    print()

    # 1. Simulate a conversation
    print(f"[{rag.session_id}] 📝 Simulating conversation inputs...")
    rag.remember("user", "My name is Jivit Rana, I have trouble sleeping.")
    time.sleep(0.5)
    rag.remember("sara", "I'm sorry to hear that, Jivit. Have you tried listening to anything calming?")
    time.sleep(0.5)
    rag.remember("user", "Yes, I really enjoy listening to Kamgana Ganesh at 3am. It helps me find peace.")
    time.sleep(0.5)
    rag.remember("sara", "That sounds like a beautiful way to spend the late hours.")
    print("✓ Conversation turns recorded in session memory.")
    print()

    # 2. Test Retrieval Pipeline
    print("🔍 Testing RAG Recall (Query Processor + Dense/Sparse Hybrid Retrieval)...")
    query = "What kind of music does the user like?"
    print(f"   Query: '{query}'")
    context_string = rag.recall(query)

    print()
    print("   [Retrieved Context]")
    print("   -------------------")
    print(context_string)
    print("   -------------------")

    if "Kamgana Ganesh" in context_string:
        print("✓ RAG successfully retrieved the relevant memory!")
    else:
        print("❌ RAG failed to retrieve the relevant memory.")
    print()

    # 3. Test Session Flush (This verifies Rate Limiting is fixed)
    print("💾 Testing Session Flush (Metadata extraction & Summarization)...")
    try:
        rag.flush_session()
        print("✓ Session successfully flushed to ChromaDB!")
    except Exception as e:
        print(f"❌ Session flush failed: {e}")

    print("\n✅ RAG Architecture Verification Complete.")

if __name__ == "__main__":
    test_rag()
