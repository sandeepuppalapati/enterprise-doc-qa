"""
Unit tests for the document chunking component.
"""
from src.components.chunking import DocumentChunker


class TestDocumentChunker:
    """Test cases for DocumentChunker class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        chunker = DocumentChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100

    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a test. " * 20  # Create text longer than chunk_size

        chunks = chunker.chunk_text(text)

        assert len(chunks) > 1
        assert all('text' in chunk for chunk in chunks)
        assert all('metadata' in chunk for chunk in chunks)

    def test_chunk_text_with_metadata(self):
        """Test chunking with custom metadata."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        text = "Sample text. " * 10
        metadata = {'filename': 'test.pdf', 'page': 1}

        chunks = chunker.chunk_text(text, metadata)

        assert all(chunk['metadata']['filename'] == 'test.pdf' for chunk in chunks)
        assert all(chunk['metadata']['page'] == 1 for chunk in chunks)
        assert all('chunk_index' in chunk['metadata'] for chunk in chunks)

    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunker = DocumentChunker()
        chunks = chunker.chunk_text("")

        assert chunks == []

    def test_chunk_text_short(self):
        """Test chunking text shorter than chunk size."""
        chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
        text = "Short text."

        chunks = chunker.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0]['text'] == text

    def test_chunk_documents(self):
        """Test chunking multiple documents."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        documents = [
            {'text': 'Document one. ' * 10, 'metadata': {'file': 'doc1.pdf'}},
            {'text': 'Document two. ' * 10, 'metadata': {'file': 'doc2.pdf'}},
        ]

        all_chunks = chunker.chunk_documents(documents)

        assert len(all_chunks) > 0
        # Check that document_index was added to metadata
        assert all('document_index' in chunk['metadata'] for chunk in all_chunks)

    def test_get_chunk_stats(self):
        """Test chunk statistics calculation."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        text = "Sample text. " * 20

        chunks = chunker.chunk_text(text)
        stats = chunker.get_chunk_stats(chunks)

        assert 'total_chunks' in stats
        assert 'avg_chunk_size' in stats
        assert 'min_chunk_size' in stats
        assert 'max_chunk_size' in stats
        assert 'total_characters' in stats
        assert stats['total_chunks'] == len(chunks)

    def test_get_chunk_stats_empty(self):
        """Test statistics for empty chunk list."""
        chunker = DocumentChunker()
        stats = chunker.get_chunk_stats([])

        assert stats == {'total_chunks': 0}
