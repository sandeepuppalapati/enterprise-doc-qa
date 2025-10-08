"""
Document loader for processing PDF files.
"""
import logging
from pathlib import Path
from typing import List, Optional
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads and extracts text from PDF documents."""

    def __init__(self):
        self.supported_extensions = ['.pdf']

    def load_pdf(self, file_path: str) -> str:
        """
        Load text content from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text content from all pages

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a supported format
        """
        path = Path(file_path)

        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. "
                f"Supported types: {', '.join(self.supported_extensions)}"
            )

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            reader = PdfReader(file_path)
            text_content = []

            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
                    else:
                        logger.warning(f"Page {page_num} contains no text")
                except Exception as e:
                    logger.error(f"Error extracting text from page {page_num}: {e}")
                    continue

            if not text_content:
                raise ValueError(f"No text could be extracted from {file_path}")

            full_text = "\n\n".join(text_content)
            logger.info(f"Successfully loaded {len(reader.pages)} pages from {path.name}")

            return full_text

        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise

    def load_from_bytes(self, file_bytes: bytes, filename: str) -> str:
        """
        Load text content from PDF file bytes (for Streamlit upload).

        Args:
            file_bytes: PDF file content as bytes
            filename: Original filename for logging

        Returns:
            Extracted text content from all pages
        """
        try:
            from io import BytesIO

            pdf_file = BytesIO(file_bytes)
            reader = PdfReader(pdf_file)
            text_content = []

            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
                except Exception as e:
                    logger.error(f"Error extracting text from page {page_num}: {e}")
                    continue

            if not text_content:
                raise ValueError(f"No text could be extracted from {filename}")

            full_text = "\n\n".join(text_content)
            logger.info(f"Successfully loaded {len(reader.pages)} pages from {filename}")

            return full_text

        except Exception as e:
            logger.error(f"Error loading PDF from bytes: {e}")
            raise

    def get_metadata(self, file_path: str) -> dict:
        """
        Extract metadata from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary containing PDF metadata
        """
        try:
            reader = PdfReader(file_path)
            metadata = {
                'num_pages': len(reader.pages),
                'file_name': Path(file_path).name,
            }

            if reader.metadata:
                metadata.update({
                    'title': reader.metadata.get('/Title', 'Unknown'),
                    'author': reader.metadata.get('/Author', 'Unknown'),
                    'subject': reader.metadata.get('/Subject', 'Unknown'),
                    'creator': reader.metadata.get('/Creator', 'Unknown'),
                })

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {'error': str(e)}
