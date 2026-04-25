# retrieval.py
import numpy as np

def search(index, query_emb, texts, k=3):
    query_emb = np.array(query_emb).reshape(1, -1).astype("float32")
    distances, indices = index.search(query_emb, k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append((texts[idx], distances[0][i]))
    return results

def search_chunk(index, query_emb, chunks, k=3):
    query_emb = np.array(query_emb).reshape(1, -1).astype("float32")
    distances, indices = index.search(query_emb, k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append((chunks[idx], distances[0][i]))
    return results