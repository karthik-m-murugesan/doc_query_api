# Document Query API Task List

Based on the current state of the workspace (a local FastAPI LangChain implementation and a cloud Flask custom implementation), here is a prioritized list of tasks to improve and expand the application.

## 📋 Phase 1: Core Stabilizing & Unification

- [ ] **Consolidate Codebases**: Decide whether to move forward with the fully local (FastAPI) or hybrid/cloud (Flask) approach, or create a unified backend where the user can toggle between models (Local Ollama vs Cloud Claude).
- [ ] **Data Persistence for Flask App**: If keeping `rag_api.py`, replace the in-memory Python dictionaries (`documents`, `document_chunks`, `chunk_embeddings`) with a formal vector database (like ChromaDB, Qdrant, or Pinecone) to ensure files aren't lost upon server restart.
- [ ] **Add Error Handling**: Improve exception handling for PDF parsing (e.g., handling scanned PDFs without OCR text or encrypted PDFs) and API rate limits.
- [ ] **Requirements Management**: Merge missing dependencies. Right now `rag_api.py` requires `flask`, `PyPDF2`, `sentence_transformers`, `scikit-learn`, `anthropic`, and `numpy` which are currently absent from `app/requirements.txt` representing only the FastAPI setup.

## 🎨 Phase 2: User Interface & Experience

- [ ] **Create a Frontend Web App**: Right now, interaction relies on API calls (Postman/cURL) or FastAPI Swagger. Build a simple and clean UI (using React, Vue, or Vanilla JS/HTML/CSS) to allow non-technical users to drag-and-drop PDFs to upload and see a chat interface to query the documents.
- [ ] **Loading States & Streaming**: For the LLM responses (FastAPI LCEL or Claude), implement response streaming so the user sees the answer being typed out in real-time instead of waiting 10-15 seconds for a complete block of text.
- [ ] **Citation UI**: Ensure that when a query responds, the UI explicitly highlights *which* source document (and which page, if available) provided the context.

## 🚀 Phase 3: Advanced Features

- [ ] **Multi-Document Support**: Expand the `rag_api.py` implementation to support querying across *all* uploaded documents instead of just an isolated `document_id`.
- [ ] **Web Scraping / URL Support**: Add an endpoint that takes a URL, scrapes the text content (using BeautifulSoup or a similar library), and applies the exact same chunking and RAG principles as a PDF.
- [ ] **OCR Support**: Integrate Tesseract or AWS Textract to natively read from scanned images (.png, .jpg) or non-searchable PDFs.

## 🛠 Phase 4: DevOps & Deployment

- [ ] **Dockerization**: Create a `Dockerfile` and `docker-compose.yml` to package the API cleanly alongside ChromaDB (and optionally Ollama) to guarantee the environment works universally.
- [ ] **Environment Configuration**: Extract hardcoded configuration (like chunk sizes, `k` retrieval amounts, local model names like `llama3`) into a `.env` file via `python-dotenv`.
