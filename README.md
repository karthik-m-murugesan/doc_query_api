# Document Query API Walkthrough

Welcome to the **Document Query API**! This project implements a Retrieval-Augmented Generation (RAG) system, allowing users to upload documents (PDFs, TXT files) and ask questions about them. 

The interesting part of this workspace is that it contains **two different implementations** showcasing distinct approaches to the problem:
1. **A fully local, persistent approach** (`app/main.py`)
2. **A cloud-based, in-memory approach** (`app/rag_api.py`)

Here is a breakdown of how both systems are built and what they do.

---

## 1. Local RAG Implementation (`app/main.py`)

This file contains a production-ready, locally hosted solution. It is built to ensure privacy and relies on local models to process data without sending anything to paid third-party APIs.

### Stack:
- **Framework**: FastAPI (High performance, built-in Swagger UI)
- **Orchestration**: LangChain (LCEL)
- **Vector Database**: ChromaDB (Persistent storage saved to `./chroma_db`)
- **Embeddings**: Sentence-transformers (`all-MiniLM-L6-v2`) via HuggingFace
- **LLM Engine**: Ollama (e.g., Llama 3, Gemma 2)

### Flow & Endpoints:
- **`POST /upload`**: Takes `.pdf` or `.txt` files. Uses LangChain's loaders (`PyPDFLoader`, `TextLoader`) and splits the text into chunks of 1000 characters. These are embedded and stored persistently in Chroma. 
- **`POST /ask`**: Takes a JSON body like `{"question": "..."}`. Searches ChromaDB for the 6 most relevant chunks and passes them as context to the local LLM running in Ollama. Returns the answer along with source document references.
- **`GET /status`**: A quick utility to ensure the vector database and the models are initialized properly.

---

## 2. Cloud-based API Implementation (`app/rag_api.py`)

This file approaches the exact same problem using a lighter web framework and a heavy reliance on a powerful cloud LLM (Claude) while doing the math (cosine similarity) from scratch instead of using a vector database.

### Stack:
- **Framework**: Flask
- **Embeddings**: Sentence-transformers (`all-MiniLM-L6-v2`)
- **Retrieval Engine**: Custom Numpy / Scikit-learn Cosine Similarity
- **LLM Engine**: Anthropic Claude (`claude-sonnet-4-20250514`)
- **Storage**: In-memory Python Dictionaries (Data is wiped when the server restarts)

### Flow & Endpoints:
- **`POST /api/upload`**: Takes a file upload, parses the text (with `PyPDF2` for PDFs), splits it into 500-word chunks, calculates embeddings using sentence-transformers, and saves everything to in-memory dictionaries (`documents`, `document_chunks`, `chunk_embeddings`).
- **`POST /api/query`**: Looks up a specific `document_id`. Creates an embedding of your query, computes cosine similarity against all chunks matching that document via NumPy/Scikit-learn, and pulls the top 3 results. Finally, it sends these specific chunks as context to Anthropic's cloud API for Claude to generate the response.
- **`GET /api/documents`**: Lists all currently processed documents residing in memory.
- **`POST /api/search`**: A semantic search endpoint allowing users to see raw relevant text chunks without engaging the LLM. 

---

## Comparison Summary

| Feature | `main.py` (FastAPI) | `rag_api.py` (Flask) |
| --- | --- | --- |
| **Privacy / Cost** | 100% Local, Free (runs on your hardware via Ollama) | Paid Cloud API (Anthropic), Data sent over network |
| **State Storage** | VectorDB (Chroma) - Survives restarts | In-memory `dict()` - Data lost on restart |
| **Retrieval Query Scope**| Global (Searches across all uploaded docs combined) | Local (Queries are isolated to a single `document_id`) |
| **Simplicity** | Requires LangChain and maintaining Ollama background worker | Requires an `ANTHROPIC_API_KEY` but logic is largely standard Python |
