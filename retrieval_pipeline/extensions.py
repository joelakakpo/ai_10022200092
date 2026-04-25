# extensions.py
def expand_query(query):
    expansions = [query, query + " results", "winner of election"]
    return expansions

def expand_chunks(chunks):
    expanded_chunks = []
    for chunk in chunks:
        expanded_chunks.extend([chunk, chunk + " results", "winner of election"])
    return expanded_chunks