from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic
from datetime import datetime
import hashlib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt'}

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# In-memory storage (use a database in production)
documents = {}
document_chunks = {}
chunk_embeddings = {}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(filepath):
    """Extract text from PDF file"""
    text = ""
    with open(filepath, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_txt(filepath):
    """Extract text from TXT file"""
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks

def generate_embeddings(chunks):
    """Generate embeddings for text chunks"""
    return embedding_model.encode(chunks)

def retrieve_relevant_chunks(query, doc_id, top_k=3):
    """Retrieve most relevant chunks for a query"""
    if doc_id not in chunk_embeddings:
        return []
    
    query_embedding = embedding_model.encode([query])[0]
    doc_embeddings = chunk_embeddings[doc_id]
    
    # Calculate cosine similarity
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    relevant_chunks = []
    for idx in top_indices:
        relevant_chunks.append({
            'text': document_chunks[doc_id][idx],
            'similarity': float(similarities[idx])
        })
    
    return relevant_chunks

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Upload and process a document"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PDF and TXT allowed'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text based on file type
        file_ext = filename.rsplit('.', 1)[1].lower()
        if file_ext == 'pdf':
            text = extract_text_from_pdf(filepath)
        else:
            text = extract_text_from_txt(filepath)
        
        # Generate document ID
        doc_id = hashlib.md5(f"{filename}{datetime.utcnow()}".encode()).hexdigest()
        
        # Chunk the text
        chunks = chunk_text(text)
        
        # Generate embeddings
        embeddings = generate_embeddings(chunks)
        
        # Store document information
        documents[doc_id] = {
            'id': doc_id,
            'filename': filename,
            'upload_time': datetime.utcnow().isoformat(),
            'num_chunks': len(chunks),
            'text_preview': text[:500]
        }
        
        document_chunks[doc_id] = chunks
        chunk_embeddings[doc_id] = embeddings
        
        return jsonify({
            'message': 'Document uploaded successfully',
            'document_id': doc_id,
            'filename': filename,
            'num_chunks': len(chunks)
        }), 201
        
    except Exception as e:
        return jsonify({'error': f'Failed to process document: {str(e)}'}), 500

@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List all uploaded documents"""
    return jsonify({
        'documents': list(documents.values()),
        'count': len(documents)
    })

@app.route('/api/documents/<doc_id>', methods=['GET'])
def get_document(doc_id):
    """Get document details"""
    if doc_id not in documents:
        return jsonify({'error': 'Document not found'}), 404
    
    return jsonify(documents[doc_id])

@app.route('/api/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a document"""
    if doc_id not in documents:
        return jsonify({'error': 'Document not found'}), 404
    
    # Remove from storage
    documents.pop(doc_id)
    document_chunks.pop(doc_id, None)
    chunk_embeddings.pop(doc_id, None)
    
    return jsonify({'message': 'Document deleted successfully'})

@app.route('/api/query', methods=['POST'])
def query_document():
    """Ask a question about a document"""
    data = request.get_json()
    
    if not data or 'document_id' not in data or 'question' not in data:
        return jsonify({'error': 'document_id and question are required'}), 400
    
    doc_id = data['document_id']
    question = data['question']
    
    if doc_id not in documents:
        return jsonify({'error': 'Document not found'}), 404
    
    try:
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(question, doc_id, top_k=3)
        
        # Build context from relevant chunks
        context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
        
        # Get Claude API key from environment
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            return jsonify({
                'answer': '[Claude API integration disabled - set ANTHROPIC_API_KEY]',
                'relevant_chunks': relevant_chunks,
                'context_used': context[:500] + '...'
            })
        
        # Call Claude API for answer generation
        client = anthropic.Anthropic(api_key=api_key)
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": f"""Based on the following context from a document, please answer the question.

Context:
{context}

Question: {question}

Please provide a clear and concise answer based only on the information provided in the context. If the context doesn't contain enough information to answer the question, please say so."""
            }]
        )
        
        answer = message.content[0].text
        
        return jsonify({
            'answer': answer,
            'document_id': doc_id,
            'document_name': documents[doc_id]['filename'],
            'relevant_chunks': relevant_chunks,
            'num_chunks_used': len(relevant_chunks)
        })
        
    except Exception as e:
        return jsonify({'error': f'Query failed: {str(e)}'}), 500

@app.route('/api/search', methods=['POST'])
def search_chunks():
    """Search for relevant chunks across a document"""
    data = request.get_json()
    
    if not data or 'document_id' not in data or 'query' not in data:
        return jsonify({'error': 'document_id and query are required'}), 400
    
    doc_id = data['document_id']
    query = data['query']
    top_k = data.get('top_k', 5)
    
    if doc_id not in documents:
        return jsonify({'error': 'Document not found'}), 404
    
    try:
        relevant_chunks = retrieve_relevant_chunks(query, doc_id, top_k=top_k)
        
        return jsonify({
            'query': query,
            'document_id': doc_id,
            'results': relevant_chunks
        })
        
    except Exception as e:
        return jsonify({'error': f'Search failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
