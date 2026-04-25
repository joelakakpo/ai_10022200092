# storage.py
import faiss
import numpy as np

def build_index(embeddings):
    emb_array = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(emb_array.shape[1])
    index.add(emb_array)
    return index
