from typing import Optional
import logging
from ..tools.llm.ollama_client import OllamaClient 

logger = logging.getLogger(__name__)

class AIProcessor:
    def __init__(self):
        self.client = OllamaClient()
        self.model = "llama3.1:8b" 

    def summarize(self, text: str, max_length: int = 200) -> str:
        """Summarizes the given text using the local LLM."""
        if not text:
            return ""
            
        prompt = f"Summarize the following text in less than {max_length} characters. Keep it concise and factual:\n\n{text}"
        try:
            response = self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model
            )
            return response.get("content", "").strip()
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return text[:max_length] + "..."

    def calculate_urgency(self, text: str) -> int:
        """Calculates an urgency score (0-100) for the text."""
        prompt = (
            "Analyze the urgency of the following text. "
            "Return ONLY a single integer between 0 and 100, where 100 is critical/immediate action required and 0 is irrelevant. "
            "Do not explain.\n\n"
            f"{text}"
        )
        try:
            response = self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model
            )
            content = response.get("content", "").strip()
            import re
            match = re.search(r'\d+', content)
            if match:
                return int(match.group())
            return 50
        except Exception as e:
            logger.error(f"Error calculating urgency: {e}")
            return 0

    def chat(self, query: str, provider: str = "Ollama", model: str = "llama3.1:8b", 
             context: str = "", use_rag: bool = False, rag_engine=None) -> str:
        """
        Chat with the AI using the selected provider and model.
        
        Args:
            query: User's question
            provider: LLM provider (Ollama or OpenAI)
            model: Model name
            context: Optional context string (fallback if RAG not used)
            use_rag: Whether to use RAG for semantic retrieval
            rag_engine: DashboardRAG instance for semantic search
        """
        try:
            # Use RAG for semantic retrieval if enabled
            if use_rag and rag_engine:
                try:
                    # Semantic search for top-5 relevant items
                    relevant_cards = rag_engine.search(query, top_k=5)
                    
                    if relevant_cards:
                        # Build context from retrieved cards
                        context = "Relevant Dashboard Items:\n\n"
                        for i, card in enumerate(relevant_cards):
                            context += f"{i+1}. [{card.source_type.value}] {card.title}\n"
                            context += f"   Summary: {card.summary}\n"
                            context += f"   Link: {card.source_link}\n"
                            context += f"   Urgency: {card.urgency_score}/100\n"
                            context += f"   Time: {card.timestamp.strftime('%Y-%m-%d %H:%M')}\n\n"
                        logger.info(f"RAG retrieved {len(relevant_cards)} relevant items")
                    else:
                        logger.warning("RAG returned no results, using fallback context")
                except Exception as e:
                    logger.error(f"RAG search failed: {e}, using fallback context")
            
            # Construct prompt with context if available
            if context:
                full_prompt = f"Context:\n{context}\n\nUser Query: {query}"
            else:
                full_prompt = query

            if provider.lower() == "openai":
                from openai import OpenAI
                import os
                
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    return "Error: OPENAI_API_KEY not found in environment variables."
                
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": full_prompt}],
                )
                return response.choices[0].message.content
            
            else: # Default to Ollama
                response = self.client.chat(
                    message=[{"role": "user", "content": full_prompt}],
                    model=model
                )
                return response if isinstance(response, str) else str(response)
                
        except Exception as e:
            logger.error(f"Error in chat ({provider}/{model}): {e}")
            return f"Error: {str(e)}"
