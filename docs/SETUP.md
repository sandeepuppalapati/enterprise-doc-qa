# Setup Guide

## Prerequisites

- Python 3.11 or higher
- Anthropic API key
- Git (for version control)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sandeepuppalapati/enterprise-doc-qa.git
cd enterprise-doc-qa
```

### 2. Create Virtual Environment

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your API key
# On macOS/Linux:
nano .env

# On Windows:
notepad .env
```

**Required:** Add your Anthropic API key:
```
ANTHROPIC_API_KEY=sk-ant-...your-key-here
```

**Optional:** Customize settings:
```
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=4
CLAUDE_MODEL=claude-3-5-sonnet-20241022
```

### 5. Run the Application

```bash
streamlit run src/ui/app.py
```

The application will open in your browser at `http://localhost:8501`

## Docker Setup

### Build and Run with Docker

```bash
# Build the image
docker build -t doc-qa-system .

# Run the container
docker run -p 8501:8501 --env-file .env doc-qa-system
```

### Docker Compose (Alternative)

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    env_file:
      - .env
    volumes:
      - ./chroma_db:/app/chroma_db
```

Run with:
```bash
docker-compose up
```

## Getting Your Anthropic API Key

1. Go to [https://console.anthropic.com/](https://console.anthropic.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (starts with `sk-ant-`)
6. Add it to your `.env` file

## Troubleshooting

### ImportError: No module named 'X'

**Solution:** Ensure virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### ChromaDB Persistence Issues

**Solution:** Ensure the `chroma_db` directory exists and has write permissions:
```bash
mkdir -p chroma_db
chmod 755 chroma_db  # On macOS/Linux
```

### Streamlit Port Already in Use

**Solution:** Run on a different port:
```bash
streamlit run src/ui/app.py --server.port=8502
```

### PDF Processing Errors

**Common causes:**
- PDF is password-protected
- PDF contains only images (no extractable text)
- PDF is corrupted

**Solution:** Try with a different PDF or use OCR preprocessing

## Development Setup

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_chunking.py
```

### Code Formatting

```bash
# Format code with black
black src/ tests/

# Check code style
flake8 src/ tests/

# Type checking
mypy src/
```

## Next Steps

1. Upload a PDF document using the sidebar
2. Click "Process Document"
3. Ask questions in the main panel
4. View sources and citations for each answer

## Support

If you encounter issues:
1. Check the [GitHub Issues](https://github.com/sandeepuppalapati/enterprise-doc-qa/issues)
2. Review the logs in the terminal
3. Ensure all dependencies are correctly installed
