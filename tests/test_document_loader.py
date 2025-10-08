"""
Unit tests for the document loader component.
"""
import pytest
from src.components.document_loader import DocumentLoader


class TestDocumentLoader:
    """Test cases for DocumentLoader class."""

    def test_init(self):
        """Test initialization."""
        loader = DocumentLoader()
        assert loader.supported_extensions == ['.pdf']

    def test_load_pdf_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        loader = DocumentLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_pdf('/nonexistent/file.pdf')

    def test_load_pdf_wrong_extension(self):
        """Test loading a file with wrong extension."""
        loader = DocumentLoader()

        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load_pdf('test.txt')

    def test_load_from_bytes_empty(self):
        """Test loading from empty bytes."""
        loader = DocumentLoader()

        with pytest.raises(Exception):
            loader.load_from_bytes(b'', 'test.pdf')

    # Note: Full PDF loading tests would require sample PDF files
    # These would be integration tests rather than unit tests
