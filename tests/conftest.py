"""
Pytest configuration and fixtures.
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_text():
    """Provide sample text for testing."""
    return """
    This is a sample document for testing.
    It contains multiple paragraphs and sentences.

    The second paragraph has more information.
    We can use this to test chunking and processing.

    Finally, a third paragraph to ensure we have enough content.
    This helps test various edge cases in our code.
    """


@pytest.fixture
def sample_metadata():
    """Provide sample metadata for testing."""
    return {
        'filename': 'test_document.pdf',
        'page': 1,
        'author': 'Test Author',
    }


@pytest.fixture(autouse=True)
def env_setup(monkeypatch):
    """Set up environment variables for testing."""
    monkeypatch.setenv('CHUNK_SIZE', '1000')
    monkeypatch.setenv('CHUNK_OVERLAP', '200')
    monkeypatch.setenv('CHROMA_PERSIST_DIRECTORY', './test_chroma_db')
