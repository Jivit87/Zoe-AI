"""
Day 5-6 Test: Text Chat with Sara
====================================
Chat with Sara via text to test her brain (Groq LLM)
and memory system (Mem0). Conversations are saved to
the conversations/ directory as markdown files.

Usage:
    python test_brain.py

Type 'quit', 'exit', or 'bye' to end.
"""

from src.llm.sara_brain import SaraBrain


if __name__ == "__main__":
    print("=" * 50)
    print("  💬 Chat with Sara (text mode)")
    print("=" * 50)
    print()

    sara = SaraBrain()

    print()
    print("💡 Type a message and press Enter.")
    print("   Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            user_input = "quit"

        if user_input.strip().lower() in ("quit", "exit", "bye"):
            # Save conversation to markdown
            sara.memory.save_session_to_markdown()
            print("\nSara: Goodbye, Sir. Take care. 💙")
            print("\n✓ Session saved to conversations/\n")
            break

        if not user_input.strip():
            continue

        response = sara.generate_response(user_input)
        print(f"Sara: {response}\n")
