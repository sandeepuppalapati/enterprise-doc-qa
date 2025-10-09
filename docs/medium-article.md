# Building a Production RAG System Without LangChain

*How I built an enterprise document Q&A system from scratch using Claude AI and ChromaDB*

---

## TL;DR

I built a production-ready Retrieval-Augmented Generation (RAG) system without using LangChain. By implementing the pipeline from scratch with direct API calls, I gained deeper understanding of RAG fundamentals, reduced dependencies from 50+ to 8, and maintained full control over the implementation.

**Tech Stack:** Python, Claude AI, ChromaDB, PyPDF, Streamlit
**GitHub:** [enterprise-doc-qa](https://github.com/sandeepuppalapati/enterprise-doc-qa)

---

## Why Skip LangChain?

LangChain is a powerful framework, but I chose to build without it for three reasons:

### 1. **Learning by Building**
Understanding what happens under the hood is crucial. When you use a framework, you're trusting its abstractions. Building from scratch forces you to understand:
- How chunking strategies affect retrieval quality
- Why embeddings matter for semantic search
- How to structure prompts for optimal LLM responses

### 2. **Dependency Management**
LangChain comes with 50+ dependencies. For a focused RAG application, this is overkill:

```python
# With LangChain
langchain==0.1.0
langchain-anthropic==0.1.1
langchain-community==0.0.10
# ... plus 47 more dependencies

# My approach
anthropic>=0.39.0
chromadb>=0.4.0
pypdf>=3.17.0
streamlit>=1.29.0
# Total: 8 dependencies
```

### 3. **Production Control**
Direct API integration means:
- Easier debugging (no abstraction layers)
- Faster updates (no waiting for framework support)
- Custom error handling
- Better performance tuning

---

## Architecture Overview

Here's the complete RAG pipeline I built:

```
┌─────────────┐
│   User      │
│   Query     │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────┐
│      Streamlit UI Layer              │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│   Custom RAG Pipeline                │
│                                      │
│  ┌──────────┐    ┌──────────────┐  │
│  │ PyPDF    │───▶│ Custom       │  │
│  │ Loader   │    │ Chunker      │  │
│  └──────────┘    └──────┬───────┘  │
│                         │           │
│                         ▼           │
│              ┌──────────────────┐  │
│              │  ChromaDB        │  │
│              │  (embeddings +   │  │
│              │   vector store)  │  │
│              └──────────┬───────┘  │
│                         │           │
│         ┌───────────────┴──────┐   │
│         ▼                      │   │
│  ┌──────────────┐             │   │
│  │  Semantic    │             │   │
│  │  Search      │             │   │
│  │  (Top-K)     │             │   │
│  └──────┬───────┘             │   │
│         │                     │   │
│         ▼                     │   │
│  ┌──────────────┐             │   │
│  │  Direct      │             │   │
│  │  Claude API  │◀────────────┘   │
│  └──────┬───────┘                 │
└─────────┼───────────────────────┘
          │
          ▼
     ┌─────────┐
     │ Answer  │
     │   +     │
     │ Sources │
     └─────────┘
```

---

## Implementation Deep Dive

### 1. Custom Document Chunking

The first challenge was splitting documents intelligently. Here's my approach:

```python
class DocumentChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def _split_text(self, text: str) -> List[str]:
        """Recursive splitting with smart separators."""
        def split_by_separator(text_to_split, separators):
            if not separators:
                return [text_to_split]

            separator = separators[0]
            remaining = separators[1:]

            splits = text_to_split.split(separator) if separator else [text_to_split]

            result = []
            for split in splits:
                if len(split) > self.chunk_size:
                    result.extend(split_by_separator(split, remaining))
                elif split:
                    result.append(split + separator if separator else split)

            return result

        return split_by_separator(text, self.separators)
```

**Key insights:**
- **Hierarchical splitting:** Try paragraphs first, then sentences, then words
- **Preserve context:** Overlap ensures no information lost at boundaries
- **Metadata tracking:** Attach source info to each chunk

### 2. Vector Store Integration

ChromaDB handles both embeddings and storage. Here's the clean interface:

```python
class VectorStore:
    def __init__(self, collection_name="documents"):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, chunks: List[dict]) -> int:
        """Add document chunks to vector store."""
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [chunk.get('metadata', {}) for chunk in chunks]

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        return len(chunks)

    def search(self, query: str, top_k=4) -> List[dict]:
        """Semantic search using cosine similarity."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        return [{
            'text': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i],
        } for i in range(len(results['documents'][0]))]
```

**Why ChromaDB:**
- Built-in embedding generation (no separate service)
- Persistent storage out of the box
- Fast cosine similarity search
- Lightweight and easy to deploy

### 3. RAG Pipeline with Claude

The retrieval + generation logic is straightforward with direct API calls:

```python
class RAGPipeline:
    def __init__(self, vector_store, model="claude-3-5-sonnet-20241022"):
        self.vector_store = vector_store
        self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.model = model

    def query(self, question: str) -> dict:
        # Step 1: Retrieve relevant chunks
        source_docs = self.vector_store.search(question, top_k=4)

        if not source_docs:
            return {'answer': "No documents found.", 'sources': []}

        # Step 2: Build context from retrieved chunks
        context = "\n\n".join([
            f"[Document {i+1}]\n{doc['text']}"
            for i, doc in enumerate(source_docs)
        ])

        # Step 3: Generate answer with Claude
        system_prompt = """You are a helpful AI assistant that answers
        questions based on provided context. Always cite which document
        number you used. If the answer isn't in the documents, say so."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }]
        )

        return {
            'answer': response.content[0].text,
            'sources': source_docs,
            'num_sources': len(source_docs)
        }
```

**Critical decisions:**
- **Top-K=4:** Sweet spot for context without overwhelming the LLM
- **Source citations:** System prompt ensures LLM references documents
- **Error handling:** Graceful degradation if no documents match

### 4. Source Citation Implementation

The killer feature: showing users *exactly* where answers came from.

```python
def display_answer(result):
    st.markdown("### 💡 Answer")
    st.markdown(result['answer'])

    st.markdown("### 📚 Sources")
    for i, source in enumerate(result['sources'], 1):
        relevance = 1 - source.get('distance', 0)
        with st.expander(f"Source {i} - Relevance: {relevance:.2%}"):
            st.markdown(f"**File:** {source['metadata']['filename']}")
            st.markdown(f"**Chunk:** {source['metadata']['chunk_index']}")
            st.text(source['text'][:500])
```

**Why this matters:**
- **Trust:** Users can verify AI responses
- **Debugging:** See if retrieval is working correctly
- **Transparency:** Critical for enterprise adoption

---

## Performance & Results

Testing with the "Attention is All You Need" paper (the Transformer paper):

**Metrics:**
- Document processing: ~15 seconds (15 pages → 42 chunks)
- Query response: 2-3 seconds
- Accuracy: High-quality answers with proper citations
- Cost: ~$0.01 per query (Claude API)

**Sample Query:**
```
Q: "What is the main contribution of this paper?"

A: The main contribution is the Transformer architecture,
   a novel neural network based entirely on attention mechanisms,
   dispensing with recurrence and convolutions entirely.
   [Document 1, Document 3]

Sources:
- Document 1: 93.2% relevance
- Document 2: 88.7% relevance
- Document 3: 85.4% relevance
```

---

## Lessons Learned

### What Worked Well

1. **Custom Chunking Strategy**
   - Hierarchical splitting preserved context better than fixed-size chunks
   - 200-character overlap prevented information loss

2. **Direct API Integration**
   - Easier to debug than framework abstractions
   - Full control over prompt engineering

3. **ChromaDB Choice**
   - Zero configuration for embeddings
   - Fast enough for production use

### What I'd Do Differently

1. **Add Hybrid Search**
   - Combine semantic search with keyword matching
   - Would improve precision for specific terms

2. **Implement Caching**
   - Cache frequent queries
   - Reduce API costs

3. **Better Chunking for Tables**
   - Current approach struggles with tabular data
   - Need specialized extraction logic

---

## When to Use LangChain vs Build Yourself

**Use LangChain when:**
- Rapid prototyping needed
- Building complex chains with multiple steps
- Need integrations with 10+ different services
- Team already familiar with the framework

**Build from scratch when:**
- Learning RAG fundamentals
- Need minimal dependencies
- Require custom logic or optimizations
- Production system with specific requirements

---

## Deployment Considerations

### Docker Setup

The app is containerized for easy deployment:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["streamlit", "run", "src/ui/app.py"]
```

### Cost Management

**Important:** Don't expose your API key publicly!

Options for deployment:
1. **Demo mode:** Pre-cached responses, no API calls
2. **Authentication:** Password-protect the app
3. **User-provided keys:** Let users bring their own API keys
4. **Screenshots only:** Skip live deployment

I chose screenshots + GitHub repo to avoid API cost exposure.

---

## Key Takeaways

1. **RAG isn't magic** - It's retrieval + generation, both of which you can control
2. **Frameworks are tools** - Know when to use them and when to build yourself
3. **Source citation is critical** - Enterprise users need to verify AI responses
4. **Chunking strategy matters** - Affects everything downstream
5. **Start simple** - MVP first, optimize later

---

## Try It Yourself

**GitHub Repository:** [sandeepuppalapati/enterprise-doc-qa](https://github.com/sandeepuppalapati/enterprise-doc-qa)

The full source code includes:
- Complete RAG implementation
- Unit tests (13/13 passing)
- Docker deployment
- Comprehensive documentation

Clone it, experiment with it, and let me know what you build!

---

## Resources

- [Anthropic Claude API](https://docs.anthropic.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [RAG Paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)

---

*Have questions? Found this helpful? Let me know in the comments!*

*Connect with me on [LinkedIn](https://linkedin.com/in/sandeep-uppalapati) or check out more projects on [GitHub](https://github.com/sandeepuppalapati).*
