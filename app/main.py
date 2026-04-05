# main.py - A complete, production-ready REST API for document upload (PDF/TXT) + RAG Q&A
# Fully local - no paid API keys required
# Uses:
#   - FastAPI
#   - LangChain (modern LCEL style)
#   - sentence-transformers (embeddings - runs locally)
#   - Ollama (LLM - runs locally, e.g. llama3, llama3.1, gemma2, mistral, etc.)
#   - Chroma (persistent vector database)

# Installation
# pip install fastapi uvicorn langchain langchain-chroma langchain-huggingface langchain-ollama langchain-community pypdf
#
# Before running the API:
# 1. Install Ollama → https://ollama.com
# 2. ollama pull llama3     # or llama3.1, gemma2, mistral, phi3, etc.
# 3. (optional) change the model name below to whatever you pulled

import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

app = FastAPI(
    title="Local RAG API (PDF & TXT)",
    description="Upload documents and ask questions about them using Retrieval-Augmented Generation. Fully local (Ollama + sentence-transformers).",
    version="1.0.0" 
)

# -------------------------- Configuration --------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatOllama(
    model="llama3",          # ← change to llama3.1, gemma2, mistral, etc. if you prefer
    temperature=0.7,
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="rag_collection"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})  # top 6 most relevant chunks

# Modern LangChain LCEL RAG chain
system_prompt = (
    "You are an expert assistant. Answer the question using only the provided context. "
    "If the context does not contain enough information to answer, respond with "
    "'I don't know based on the uploaded documents.' Keep your answer concise but complete."
    "\n\nContext:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# -------------------------- Pydantic models --------------------------
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]  # unique source filenames / pages used

# -------------------------- Endpoints --------------------------
@app.post("/upload", response_model=dict)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDF or TXT files.
    They will be chunked, embedded, and added to the persistent vector database.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []
    os.makedirs("uploads", exist_ok=True)

    for file in files:
        if not file.filename.lower().endswith((".pdf", ".txt")):
            results.append({
                "filename": file.filename,
                "status": "error",
                "detail": "Only .pdf and .txt files are allowed"
            })
            continue

        try:
            contents = await file.read()
            file_path = os.path.join("uploads", file.filename)

            with open(file_path, "wb") as f:
                f.write(contents)

            # Load document
            if file.filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding="utf-8")

            docs = loader.load()

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                add_start_index=True,
            )
            splits = text_splitter.split_documents(docs)

            # Add to vectorstore (persistent)
            vectorstore.add_documents(splits)

            results.append({
                "filename": file.filename,
                "status": "success",
                "chunks_added": len(splits)
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "detail": str(e)
            })
        finally:
            # Clean up temporary file
            if os.path.exists(file_path):
                os.remove(file_path)

    return {"results": results}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(payload: QueryRequest):
    """
    Ask a question about the uploaded documents.
    Returns the answer + the sources used.
    """
    if vectorstore._collection.count() == 0:
        return QueryResponse(
            answer="No documents have been uploaded yet. Please upload some documents first.",
            sources=[]
        )

    response = rag_chain.invoke({"input": payload.question})

    # Extract unique sources
    sources = set()
    for doc in response["context"]:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        if page is not None:
            source = f"{os.path.basename(source)} (page {page + 1})"  # page is 0-indexed in metadata
        else:
            source = os.path.basename(source)
        sources.add(source)

    return QueryResponse(
        answer=response["answer"].strip(),
        sources=sorted(list(sources))
    )

@app.get("/status")
async def status():
    return {
        "status": "ready",
        "documents_in_db": vectorstore._collection.count(),
        "model": llm.model,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
    }

# Run with: uvicorn main:app --reload
# Swagger UI: http://127.0.0.1:8000/docs