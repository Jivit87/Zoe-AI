"""
Conversation Memory System (Optimized for Speed)
===================================================
Provides long-term memory for Sara.
Optimized to avoid heavy CPU embeddings (Ollama) which cause latency.
Uses local JSON/Markdown storage for now.
"""

from datetime import datetime
import os
import json


class ConversationMemory:
    def __init__(self, user_id="main_user"):
        self.user_id = user_id

        print("🧠 Initializing memory system (Light Mode)...")
        
        # Markdown conversation log directory
        self.conversation_dir = "./conversations"
        os.makedirs(self.conversation_dir, exist_ok=True)

        # Current session buffer
        self.current_session = []
        
        # Simple recent history buffer for direct context injection
        self.recent_history = [] 

        print("✓ Memory system ready!")

    def add_conversation_turn(
        self, user_text, assistant_text, emotional_state=None, context=None
    ):
        """Add a conversation turn to logs and recent history."""
        timestamp = datetime.now().isoformat()

        # Update recent history (keep last 10 turns)
        self.recent_history.append(f"User: {user_text}")
        self.recent_history.append(f"Sara: {assistant_text}")
        if len(self.recent_history) > 20:
            self.recent_history = self.recent_history[-20:]

        # Add to current session log buffer
        self.current_session.append(
            {
                "timestamp": timestamp,
                "user": user_text,
                "sara": assistant_text,
                "emotion": emotional_state,
            }
        )
        
        # Auto-save to file immediately so we never lose data
        self.save_session_to_markdown()

    def retrieve_relevant_memories(self, query, limit=5):
        """
        Fast retrieval of recent context.
        Skipping vector search (Ollama) for now to minimize latency.
        """
        # Return last 3 turns as immediate context
        # This is surprisingly effective and zero-latency compared to embedding search
        recent = self.recent_history[-6:] # last 3 exchanges
        return recent

    def save_session_to_markdown(self):
        """Append to single conversation_history.md."""
        if not self.current_session:
            return

        filename = "conversation_history.md"
        filepath = os.path.join(self.conversation_dir, filename)

        # Check if file exists
        file_exists = os.path.exists(filepath)
        mode = "a" if file_exists else "w"

        with open(filepath, mode, encoding="utf-8") as f:
            if not file_exists:
                 f.write(f"# Sara Conversation History\n\n")

            for turn in self.current_session:
                time_str = datetime.fromisoformat(turn["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"## [{time_str}]\n")
                f.write(f"**You:** {turn['user']}\n")
                f.write(f"**Sara:** {turn['sara']}\n")
                f.write("\n")

        # Clear buffer after writing
        self.current_session = []
