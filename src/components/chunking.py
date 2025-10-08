"""
Text chunking utilities for splitting documents into manageable pieces.
"""
import logging
import os
from typing import List, Optional
import re

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Splits documents into chunks for embedding and retrieval."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """
        Initialize the document chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size or int(os.getenv('CHUNK_SIZE', '1000'))
        self.chunk_overlap = chunk_overlap or int(os.getenv('CHUNK_OVERLAP', '200'))
        self.separators = ["\n\n", "\n", ". ", " ", ""]

        logger.info(
            f"Initialized chunker with size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks using separators."""
        chunks = []

        def split_by_separator(text_to_split: str, separators: List[str]) -> List[str]:
            if not separators:
                return [text_to_split]

            separator = separators[0]
            remaining_separators = separators[1:]

            splits = text_to_split.split(separator) if separator else [text_to_split]

            result = []
            for split in splits:
                if len(split) > self.chunk_size:
                    result.extend(split_by_separator(split, remaining_separators))
                elif split:
                    result.append(split + (separator if separator and split != splits[-1] else ""))

            return result

        # First pass: split by separators
        initial_chunks = split_by_separator(text, self.separators)

        # Second pass: combine small chunks and handle overlap
        current_chunk = ""
        for chunk in initial_chunks:
            if len(current_chunk) + len(chunk) <= self.chunk_size:
                current_chunk += chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = chunk

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> List[dict]:
        """
        Split text into chunks with metadata.

        Args:
            text: Text content to split
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        try:
            chunks = self._split_text(text)

            # Attach metadata to each chunk
            chunked_documents = []
            for idx, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata.update({
                    'chunk_index': idx,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk),
                })

                chunked_documents.append({
                    'text': chunk,
                    'metadata': chunk_metadata,
                })

            logger.info(f"Split text into {len(chunks)} chunks")
            return chunked_documents

        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            raise

    def chunk_documents(
        self,
        documents: List[dict],
        text_key: str = 'text',
    ) -> List[dict]:
        """
        Chunk multiple documents.

        Args:
            documents: List of document dictionaries
            text_key: Key in document dict that contains the text

        Returns:
            List of chunked documents with metadata
        """
        all_chunks = []

        for doc_idx, doc in enumerate(documents):
            text = doc.get(text_key, '')
            metadata = doc.get('metadata', {})
            metadata['document_index'] = doc_idx

            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)

        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks

    def get_chunk_stats(self, chunks: List[dict]) -> dict:
        """
        Get statistics about the chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {'total_chunks': 0}

        chunk_sizes = [len(chunk['text']) for chunk in chunks]

        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes),
        }
