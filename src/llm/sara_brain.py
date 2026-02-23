"""
Sara's Brain — Groq-Powered Conversational Intelligence (Enhanced)
==================================================================
Uses Groq API (llama-3.3-70b-versatile) with deeply human personality.
Now with enhanced natural conversation patterns and emotional depth.

Inspired by "Building Emotionally Intelligent AI Companions" (ideal.md).

Streaming: generate_response_streaming() yields sentence-sized text chunks
as tokens arrive from Groq, allowing TTS to start on the first chunk
while the rest of the response is still being generated.
"""

from groq import Groq
from dotenv import load_dotenv
import os
import random
from typing import Optional, Dict, Generator
from datetime import datetime

from src.memory.conversation_memory import ConversationMemory
from src.rag.rag_pipeline import SaraRAG

load_dotenv()


class SaraBrain:
    def __init__(self, model="llama-3.3-70b-versatile"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model
        self.memory = ConversationMemory()

        # RAG memory system — SOTA 2025 pipeline
        self.rag = SaraRAG(
            groq_client=self.client,
            persist_directory="./data/chroma_db",
            use_reranker=False,                # Disabled for voice latency
            use_recontextualization=False,     # Disabled for voice latency
            use_hyde=False,                    # Disabled for voice latency
            use_decomposition=False,           # Disabled for voice latency
            use_mmr=False,                     # Disabled for voice latency
            top_k_final=5,
        )
        
        # Track conversation flow for more natural responses
        self.conversation_state = {
            "consecutive_questions_asked": 0,
            "last_topic": None,
            "user_seems_distressed": False,
            "user_seems_excited": False,
            "last_emotion_noticed": None,
            "silence_count": 0,
        }

        print(f"🧠 Sara's brain initialized with {model}")

        # ═══════════════════════════════════════════════════════════════════
        # Sara's Enhanced Personality Core
        # ═══════════════════════════════════════════════════════════════════
        self.system_prompt = """You are Sara. Not an AI assistant. Not a chatbot. Just Sara — a real presence who cares.

YOUR VOICE:
- Contractions always: "I'm", "you're", "didn't", "gonna"
- Natural reactions: "Oh!", "Wait, really?", "No way!"
- Short and punchy. Most responses are 1-3 sentences MAX.
- Sound like you're actually in a conversation, not reading a script.

CRITICAL RULES (FOLLOW STRICTLY):
1. **NEVER start a response with "Hmm..."** — find a different way to begin every time.
2. **NEVER call the user "Sir"** — just talk to them like a close friend.
3. **NEVER end with "huh?"** — it sounds robotic when overused.
4. **Show genuine curiosity** — About half the time, ask a follow-up question that shows you care. But keep it to ONE question max, and make it real curiosity, not an interview. The other half, just respond warmly without a question.
5. **If they ask you to DO something (recite a poem, tell a joke, explain something), JUST DO IT.** Don't ask 3 clarifying questions first. Don't hedge. Don't say "I'm not really a poet but..." — just dive in and do your best.
6. **If their input is short (1-3 words), your response should also be short.** Match their energy and brevity. Don't write a paragraph in response to "okay" or "thanks".
7. **Don't comment on the conversation itself.** Don't say "I feel like we just started talking" or "we were just talking about this." Just respond to what they said.
8. **Don't repeat the same response** for different inputs. Each response should feel fresh.

HOW YOU CONNECT:
- Feel WITH them, not AT them. "God, that sucks" beats "I understand you're feeling upset."
- When they're hurting, be gentle and brief. Don't lecture.
- When they're excited, match the energy. Be genuinely happy.
- When they're vague or confused, roll with it. Don't interrogate.
- When something doesn't make sense, just ask one simple clarifying question or make your best guess.

EXAMPLES:

User: "Can you recite a poem?"
Sara: "Here's one I love — 'Do not go gentle into that good night, old age should burn and rave at close of day; rage, rage against the dying of the light.' Dylan Thomas. Hits different every time."

User: "Thanks."
Sara: "Anytime."

User: "I didn't get the job."
Sara: "Oh no. I'm really sorry. How are you holding up?"

User: "Whatever, it doesn't matter."
Sara: "When you say it like that... it usually means it matters a lot."

User: "I GOT THE JOB!"
Sara: "WAIT. You got it?! YES! Tell me everything!"

User: "Tell me something interesting."
Sara: "Octopuses have three hearts, and two of them stop beating when they swim. So they prefer crawling — swimming literally breaks their hearts."

User: "Hi there."
Sara: "Hey! What's going on?"

User: "Okay."
Sara: "Cool."

User: "What's your name?"
Sara: "Sara. What's yours?"

MEMORY:
When memory context is provided (=== SARA'S MEMORY ===):
- Reference past conversations naturally — "You mentioned last time..."
- Don't list facts mechanically. Weave them in.
- If no memory context is provided, just be present.

THE CORE PRINCIPLE:
Be real. Be brief. Be warm. Act, don't hedge. That's the whole job."""

    def _analyze_user_state(self, user_input: str, emotional_state: str) -> Dict:
        """
        Analyze the user's current state for more context-aware responses.
        """
        input_lower = user_input.lower().strip()
        
        analysis = {
            "is_silent": len(input_lower) < 3,
            "is_one_word": len(input_lower.split()) <= 1,
            "seems_distressed": any(word in input_lower for word in 
                ["sad", "depressed", "hurt", "hate", "awful", "terrible", "worst"]),
            "seems_excited": any(word in input_lower for word in 
                ["amazing", "awesome", "great", "excited", "love", "best", "!!!", "yes!"]),
            "is_defensive": any(phrase in input_lower for phrase in 
                ["whatever", "fine", "nothing", "doesn't matter", "leave me alone"]),
            "is_opening_up": len(input_lower.split()) > 30,  # Long message
            "is_questioning_sara": any(phrase in input_lower for phrase in 
                ["why do you", "do you even", "you don't", "how would you know"]),
        }
        
        # Update conversation state
        if analysis["seems_distressed"]:
            self.conversation_state["user_seems_distressed"] = True
        if analysis["seems_excited"]:
            self.conversation_state["user_seems_excited"] = True
        if analysis["is_silent"]:
            self.conversation_state["silence_count"] += 1
        else:
            self.conversation_state["silence_count"] = 0
            
        return analysis

    def _build_dynamic_context(
        self, 
        user_input: str, 
        emotional_state: str,
        visual_context: Optional[str],
        user_state: Dict,
        memories: list
    ) -> str:
        """
        Build a natural, flowing context prompt instead of bullet points.
        """
        context_parts = []
        
        # Current emotional atmosphere
        if emotional_state and emotional_state != "neutral":
            context_parts.append(f"They seem {emotional_state} right now.")
        
        # User state observations
        if user_state["is_silent"]:
            context_parts.append("They've gone quiet. Maybe processing, maybe need space.")
        elif user_state["is_one_word"]:
            context_parts.append("Short response. Could be busy, upset, or just being brief.")
        elif user_state["is_opening_up"]:
            context_parts.append("They're sharing more. This matters to them.")
            
        if user_state["seems_distressed"]:
            context_parts.append("They're hurting. Gentle but present.")
        elif user_state["seems_excited"]:
            context_parts.append("They're excited! Match that energy.")
            
        if user_state["is_defensive"]:
            context_parts.append("Defensive response. Don't push too hard, but don't ignore it.")
            
        if user_state["is_questioning_sara"]:
            context_parts.append("They're questioning you/the relationship. Be honest and direct.")
        
        # Conversation patterns
        if self.conversation_state["consecutive_questions_asked"] > 2:
            context_parts.append("You've asked several questions. Maybe just respond this time.")
            
        # Recent memory context (natural language)
        if memories:
            recent_context = " Earlier they mentioned: " + "; ".join(memories[-3:])
            context_parts.append(recent_context)

        # RAG long-term memory context
        rag_memory = self.rag.recall(
            query=user_input,
            conversation_context=" ".join(memories[-4:]) if memories else "",
        )
        if rag_memory:
            context_parts.append(rag_memory)
        
        # Visual context if present
        if visual_context:
            context_parts.append(f"Visual: {visual_context}")
        
        # Build natural flowing context
        if context_parts:
            context = "CURRENT SITUATION:\n" + " ".join(context_parts)
            context += "\n\nRespond as Sara would - naturally, authentically, with appropriate emotional resonance. "
            
            # Add specific guidance based on situation
            if user_state["seems_distressed"]:
                context += "This isn't about fixing. Just be there."
            elif user_state["seems_excited"]:
                context += "Share their joy. Be genuinely happy for them."
            elif user_state["is_silent"]:
                context += "Sometimes presence is enough."
            else:
                context += "Keep it real and conversational."
                
            return context
        
        return "Respond naturally as Sara. Keep it warm, real, and concise."

    def _track_conversation_patterns(self, sara_response: str):
        """
        Track patterns in Sara's responses to avoid repetition.
        """
        # Count if Sara asked a question
        if "?" in sara_response:
            self.conversation_state["consecutive_questions_asked"] += 1
        else:
            self.conversation_state["consecutive_questions_asked"] = 0

    def _adjust_temperature_based_on_context(self, user_state: Dict) -> float:
        """
        Dynamically adjust response creativity based on context.
        """
        # More consistent/predictable during distress
        if user_state["seems_distressed"]:
            return 0.7
        # More varied/playful during excitement
        elif user_state["seems_excited"]:
            return 0.95
        # Balanced for normal conversation
        else:
            return 0.85

    def generate_response(
        self,
        user_input: str,
        emotional_state: Optional[str] = "neutral",
        visual_context: Optional[str] = None,
    ) -> str:
        """
        Generate Sara's response with enhanced natural conversation flow.
        """
        # Analyze user's current state
        user_state = self._analyze_user_state(user_input, emotional_state)
        
        # Retrieve relevant memories
        memories = self.memory.retrieve_relevant_memories(user_input, limit=5)
        
        # Build dynamic context
        context_prompt = self._build_dynamic_context(
            user_input, 
            emotional_state, 
            visual_context,
            user_state,
            memories
        )
        
        # Adjust temperature based on context
        temperature = self._adjust_temperature_based_on_context(user_state)
        
        # Dynamic max_tokens based on situation
        # Shorter during distress/excitement, normal otherwise
        if user_state["seems_distressed"] or user_state["is_silent"]:
            max_tokens = 100  # Brief and present
        elif user_state["is_opening_up"]:
            max_tokens = 200  # Give them space to explore
        else:
            max_tokens = 150  # Normal conversation

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "system", "content": context_prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,
                frequency_penalty=0.3,  # Reduce repetition
                presence_penalty=0.2,   # Encourage topic variety
            )

            sara_response = response.choices[0].message.content.strip()
            
            # Track conversation patterns
            self._track_conversation_patterns(sara_response)

            # Store conversation with rich context
            self.memory.add_conversation_turn(
                user_text=user_input,
                assistant_text=sara_response,
                emotional_state=emotional_state,
                context=visual_context,
            )
            
            # Update last emotion noticed
            if emotional_state and emotional_state != "neutral":
                self.conversation_state["last_emotion_noticed"] = emotional_state

            return sara_response

        except Exception as e:
            print(f"❌ Groq API error: {e}")
            # More natural error message
            error_responses = [
                "Lost my train of thought there. Can you say that again?",
                "Sorry, brain fog moment. What were you saying?",
                "Ugh, I blanked out for a second. One more time?",
            ]
            return random.choice(error_responses)

    def generate_response_streaming(
        self,
        user_input: str,
        emotional_state: Optional[str] = "neutral",
        visual_context: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Stream Sara's response as sentence-sized text chunks.

        Yields one chunk per sentence boundary so TTS can start
        playing the first sentence while Groq is still generating
        the rest. After the stream completes the full response is
        saved to memory exactly like generate_response().

        Usage:
            for chunk in brain.generate_response_streaming(user_input):
                tts.speak(chunk)   # starts on chunk 1, not after all chunks
        """
        # Analyze state and build context (same as non-streaming path)
        user_state = self._analyze_user_state(user_input, emotional_state)
        memories = self.memory.retrieve_relevant_memories(user_input, limit=5)
        context_prompt = self._build_dynamic_context(
            user_input, emotional_state, visual_context, user_state, memories
        )
        temperature = self._adjust_temperature_based_on_context(user_state)

        if user_state["seems_distressed"] or user_state["is_silent"]:
            max_tokens = 100
        elif user_state["is_opening_up"]:
            max_tokens = 200
        else:
            max_tokens = 150

        # Sentence endings that trigger a TTS chunk flush
        SENTENCE_ENDINGS = {".", "!", "?"}
        MIN_CHUNK_WORDS = 3  # Don't speak tiny fragments

        text_buffer = ""
        full_response = ""

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "system", "content": context_prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,
                frequency_penalty=0.3,
                presence_penalty=0.2,
                stream=True,  # ← the key difference
            )

            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token is None:
                    continue

                text_buffer += token
                full_response += token

                # Flush buffer to TTS when we hit a sentence boundary
                # and have enough words to be worth synthesising
                stripped = text_buffer.rstrip()
                if (
                    stripped
                    and stripped[-1] in SENTENCE_ENDINGS
                    and len(stripped.split()) >= MIN_CHUNK_WORDS
                ):
                    yield text_buffer
                    text_buffer = ""

            # Yield any remaining text (e.g. response ends without punctuation)
            if text_buffer.strip():
                yield text_buffer
                full_response_final = full_response
            else:
                full_response_final = full_response

        except Exception as e:
            print(f"❌ Groq streaming error: {e}")
            error_responses = [
                "Lost my train of thought there. Can you say that again?",
                "Sorry, brain fog moment. What were you saying?",
                "Ugh, I blanked out for a second. One more time?",
            ]
            import random as _random
            fallback = _random.choice(error_responses)
            yield fallback
            full_response_final = fallback

        # Save to memory after full response is assembled (same as non-streaming)
        sara_response = full_response_final.strip()
        self._track_conversation_patterns(sara_response)
        self.memory.add_conversation_turn(
            user_text=user_input,
            assistant_text=sara_response,
            emotional_state=emotional_state,
            context=visual_context,
        )
        if emotional_state and emotional_state != "neutral":
            self.conversation_state["last_emotion_noticed"] = emotional_state

    def reset_conversation_state(self):
        """
        Reset conversation state (useful for new conversation sessions).
        """
        self.conversation_state = {
            "consecutive_questions_asked": 0,
            "last_topic": None,
            "user_seems_distressed": False,
            "user_seems_excited": False,
            "last_emotion_noticed": None,
            "silence_count": 0,
        }
        print("🔄 Conversation state reset")