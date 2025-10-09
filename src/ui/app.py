"""
Streamlit UI for Enterprise Document Q&A System.
"""
import streamlit as st
import logging
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path to import components
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.components.document_loader import DocumentLoader  # noqa: E402
from src.components.chunking import DocumentChunker  # noqa: E402
from src.components.embeddings import VectorStore  # noqa: E402
from src.components.retrieval import RAGPipeline  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Enterprise Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize session state variables."""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'doc_metadata' not in st.session_state:
        st.session_state.doc_metadata = []


def initialize_components():
    """Initialize the RAG components."""
    try:
        # Check for API key
        if not os.getenv('ANTHROPIC_API_KEY'):
            st.error(
                "⚠️ ANTHROPIC_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )
            st.stop()

        # Initialize vector store
        if st.session_state.vector_store is None:
            st.session_state.vector_store = VectorStore(
                collection_name="doc_qa_collection"
            )

        # Initialize RAG pipeline
        if st.session_state.rag_pipeline is None:
            st.session_state.rag_pipeline = RAGPipeline(
                vector_store=st.session_state.vector_store
            )

        logger.info("Components initialized successfully")

    except Exception as e:
        st.error(f"Error initializing components: {e}")
        logger.error(f"Initialization error: {e}")
        st.stop()


def process_uploaded_file(uploaded_file):
    """Process and index an uploaded PDF file."""
    try:
        with st.spinner("📄 Processing document..."):
            # Load document
            loader = DocumentLoader()
            text = loader.load_from_bytes(uploaded_file.read(), uploaded_file.name)

            # Chunk document
            chunker = DocumentChunker()
            metadata = {'filename': uploaded_file.name}
            chunks = chunker.chunk_text(text, metadata)

            # Get chunk statistics
            stats = chunker.get_chunk_stats(chunks)

            # Add to vector store
            st.session_state.vector_store.add_documents(chunks)

            # Update session state
            st.session_state.documents_loaded = True
            st.session_state.doc_metadata.append({
                'filename': uploaded_file.name,
                'chunks': stats['total_chunks'],
                'size': stats['total_characters']
            })

            st.success(f"✅ Successfully processed {uploaded_file.name}")
            st.info(
                f"📊 Created {stats['total_chunks']} chunks "
                f"(avg size: {int(stats['avg_chunk_size'])} chars)"
            )

            return True

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        logger.error(f"File processing error: {e}")
        return False


def display_answer(result):
    """Display the answer with sources."""
    # Display answer
    st.markdown("### 💡 Answer")
    st.markdown(result['answer'])

    # Display sources
    if result.get('sources'):
        st.markdown("---")
        st.markdown("### 📚 Sources")

        for i, source in enumerate(result['sources'], 1):
            relevance = 1 - source.get('distance', 0)
            with st.expander(
                f"Source {i} - Relevance Score: {relevance:.2%}"
            ):
                filename = source['metadata'].get('filename', 'Unknown')
                st.markdown(f"**Filename:** {filename}")
                chunk_idx = source['metadata'].get('chunk_index', 'N/A')
                total = source['metadata'].get('total_chunks', 'N/A')
                st.markdown(f"**Chunk:** {chunk_idx} / {total}")
                st.markdown("**Content:**")
                content = (
                    source['text'][:500] + "..."
                    if len(source['text']) > 500
                    else source['text']
                )
                st.text(content)


def main():
    """Main application function."""
    initialize_session_state()
    initialize_components()

    # Header
    st.title("📄 Enterprise Document Q&A System")
    st.markdown("Ask questions about your documents using AI-powered semantic search")

    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=['pdf'],
            help="Upload a PDF document to ask questions about"
        )

        if uploaded_file:
            if st.button("🔄 Process Document", type="primary"):
                process_uploaded_file(uploaded_file)

        # Display loaded documents
        if st.session_state.doc_metadata:
            st.markdown("---")
            st.subheader("📚 Loaded Documents")
            for doc in st.session_state.doc_metadata:
                st.text(f"📄 {doc['filename']}")
                st.caption(f"   {doc['chunks']} chunks")

        # Collection stats
        if st.session_state.vector_store:
            st.markdown("---")
            stats = st.session_state.vector_store.get_collection_stats()
            st.metric("Total Chunks", stats.get('total_documents', 0))

        # Clear collection
        if st.session_state.documents_loaded:
            st.markdown("---")
            if st.button("🗑️ Clear All Documents", type="secondary"):
                if st.session_state.vector_store:
                    st.session_state.vector_store.clear_collection()
                    st.session_state.documents_loaded = False
                    st.session_state.doc_metadata = []
                    st.session_state.chat_history = []
                    st.success("Cleared all documents")
                    st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Ask Questions")

        if not st.session_state.documents_loaded:
            st.info("👈 Please upload and process a document to get started")
        else:
            # Question input
            question = st.text_input(
                "Enter your question:",
                placeholder="What are the key terms of the contract?",
                help="Ask any question about your uploaded documents"
            )

            # Sample questions
            st.markdown("**Sample questions:**")
            col_q1, col_q2 = st.columns(2)
            with col_q1:
                if st.button("📋 Summarize the main points"):
                    question = "Summarize the main points of this document"
                if st.button("🔍 What are the key findings?"):
                    question = "What are the key findings or conclusions?"
            with col_q2:
                if st.button("📊 What data is presented?"):
                    question = "What data or statistics are presented?"
                if st.button("⚠️ Are there any risks mentioned?"):
                    question = "Are there any risks or limitations mentioned?"

            # Process question
            if question:
                with st.spinner("🤔 Thinking..."):
                    result = st.session_state.rag_pipeline.query(question)

                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': question,
                        'result': result
                    })

                # Display answer
                display_answer(result)

    with col2:
        st.header("📜 History")

        if st.session_state.chat_history:
            # Show recent questions
            recent = reversed(st.session_state.chat_history[-5:])
            for i, item in enumerate(recent, 1):
                q_num = len(st.session_state.chat_history) - i + 1
                q_preview = item['question'][:40]
                with st.expander(f"Q{q_num}: {q_preview}..."):
                    st.markdown(f"**Q:** {item['question']}")
                    st.markdown(f"**A:** {item['result']['answer'][:200]}...")
        else:
            st.info("No questions asked yet")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ChromaDB, PyPDF, and Claude AI • No LangChain • Custom RAG Pipeline</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
