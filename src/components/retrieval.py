"""
RAG chain implementation combining retrieval and generation.
"""
import logging
import os
from typing import List, Optional
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline using Claude."""

    def __init__(
        self,
        vector_store,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        top_k: int = 4,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            vector_store: VectorStore instance for retrieval
            api_key: Anthropic API key (uses env var if not provided)
            model: Claude model to use
            top_k: Number of documents to retrieve
        """
        self.vector_store = vector_store
        self.top_k = top_k

        self.model = model or os.getenv('CLAUDE_MODEL', 'claude-3-5-sonnet-20241022')

        # Initialize Anthropic client
        api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        self.client = Anthropic(api_key=api_key)

        logger.info(f"Initialized RAG pipeline with model: {self.model}")

    def retrieve_context(self, query: str) -> tuple[List[dict], str]:
        """
        Retrieve relevant context for a query.

        Args:
            query: User question

        Returns:
            Tuple of (retrieved documents, formatted context string)
        """
        # Retrieve relevant chunks
        results = self.vector_store.search(query, top_k=self.top_k)

        if not results:
            return [], ""

        # Format context for the prompt
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Document {i}]\n{result['text']}")

        context = "\n\n".join(context_parts)
        return results, context

    def generate_answer(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate an answer using Claude based on retrieved context.

        Args:
            query: User question
            context: Retrieved context documents
            system_prompt: Optional custom system prompt

        Returns:
            Generated answer
        """
        if not system_prompt:
            system_prompt = (
                "You are a helpful AI assistant that answers questions "
                "based on the provided context documents.\n\n"
                "Rules:\n"
                "1. Answer ONLY based on the information in the provided "
                "documents\n"
                "2. If the answer is not in the documents, say \"I don't "
                "have enough information to answer that question\"\n"
                "3. Cite which document number(s) you used when answering\n"
                "4. Be concise but complete\n"
                "5. If multiple documents have relevant information, "
                "synthesize them"
            )

        user_message = f"""Context Documents:
{context}

Question: {query}

Please provide a clear, accurate answer based on the context above."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )

            answer = response.content[0].text
            logger.info(f"Generated answer for query: {query[:50]}...")
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    def query(self, question: str) -> dict:
        """
        Complete RAG query: retrieve context and generate answer.

        Args:
            question: User question

        Returns:
            Dictionary with answer and source documents
        """
        try:
            # Retrieve relevant documents
            source_docs, context = self.retrieve_context(question)

            if not source_docs:
                return {
                    'answer': (
                        "I don't have any documents to answer your "
                        "question. Please upload documents first."
                    ),
                    'sources': [],
                    'context': ""
                }

            # Generate answer
            answer = self.generate_answer(question, context)

            # Format response
            return {
                'answer': answer,
                'sources': source_docs,
                'context': context,
                'num_sources': len(source_docs),
            }

        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return {
                'answer': f"Error processing your question: {str(e)}",
                'sources': [],
                'error': str(e)
            }

    def query_with_chat_history(
        self,
        question: str,
        chat_history: List[dict],
    ) -> dict:
        """
        Query with chat history context (for future implementation).

        Args:
            question: User question
            chat_history: Previous conversation turns

        Returns:
            Dictionary with answer and sources
        """
        # For now, just call regular query
        # In future, we can reformulate the question based on chat history
        return self.query(question)
