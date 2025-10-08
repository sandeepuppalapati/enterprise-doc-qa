"""
Embeddings and vector store management using ChromaDB.
"""
import logging
import os
from typing import List, Optional
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages document embeddings and vector similarity search."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[any] = None,
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_function: Custom embedding function (uses default if None)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or os.getenv(
            'CHROMA_PERSIST_DIRECTORY', './chroma_db'
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        logger.info(f"Initialized vector store: {self.collection_name}")

    def add_documents(
        self,
        chunks: List[dict],
        text_key: str = 'text',
    ) -> int:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of chunk dictionaries containing text and metadata
            text_key: Key in chunk dict that contains the text

        Returns:
            Number of documents added
        """
        if not chunks:
            logger.warning("No chunks provided to add")
            return 0

        try:
            # Prepare data for ChromaDB
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            documents = [chunk[text_key] for chunk in chunks]
            metadatas = [chunk.get('metadata', {}) for chunk in chunks]

            # Convert non-string metadata values to strings
            for metadata in metadatas:
                for key, value in metadata.items():
                    if not isinstance(value, (str, int, float, bool)):
                        metadata[key] = str(value)

            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

            logger.info(f"Added {len(chunks)} documents to vector store")
            return len(chunks)

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise

    def search(
        self,
        query: str,
        top_k: int = 4,
        filter_metadata: Optional[dict] = None,
    ) -> List[dict]:
        """
        Search for similar documents using semantic similarity.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of dictionaries containing results with text, metadata, and scores
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filter_metadata,
            )

            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else None,
                        'id': results['ids'][0][i] if results['ids'] else None,
                    })

            logger.info(f"Retrieved {len(formatted_results)} results for query")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise

    def clear_collection(self):
        """Delete all documents from the collection."""
        try:
            # Delete the collection
            self.client.delete_collection(name=self.collection_name)

            # Recreate it
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            logger.info(f"Cleared collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise

    def get_collection_stats(self) -> dict:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'total_documents': count,
                'persist_directory': self.persist_directory,
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}

    def delete_collection(self):
        """Permanently delete the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
