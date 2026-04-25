import json

def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def save_chunks(chunks, path):
    with open(path, "w") as f:
        json.dump(chunks, f, indent=2)

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    # Load cleaned text
    text = load_text("data/processed/full_text.txt")

    # Chunk into different sizes
    chunks_small = chunk_text(text, 200, 20)
    chunks_medium = chunk_text(text, 400, 50)
    chunks_large = chunk_text(text, 800, 100)

    print("Chunks created:", len(chunks_small), len(chunks_medium), len(chunks_large))

    # Save medium chunks
    save_chunks(chunks_medium, "data/processed/chunks.json")

    # Example query chunking
    query = "Who won the election?"
    query_chunks = chunk_text(query, 20, 5)
    print("Query chunks:", len(query_chunks))
    save_chunks(query_chunks, "data/processed/query_chunks.json")