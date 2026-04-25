# embeddings.py
from sentence_transformers import SentenceTransformer

# Load a lightweight model locally
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts):
    """Generate embeddings for a list of texts."""
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()

def embed_query(query):
    """Generate embedding for a single query."""
    embedding = model.encode(query, convert_to_numpy=True)
    return embedding.tolist()

def embed_chunk(chunk):
    """Generate embedding for a single chunk."""
    embedding = model.encode(chunk, convert_to_numpy=True)
    return embedding.tolist()

def embed_chunks(chunks):
    """Generate embeddings for multiple chunks."""
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings.tolist()