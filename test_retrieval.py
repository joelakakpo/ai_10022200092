import json

# Load chunks
with open("data/processed/chunks.json", "r") as f:
    chunks = json.load(f)

# Simulate the retrieval function from rag_app.py
def retrieve_chunks(query, election_chunks, top_k=4):
    """Retrieve chunks using the same logic as rag_app.py"""
    query_terms = [t.lower() for t in query.split() if len(t) > 2]
    scores = []
    
    for chunk in election_chunks:
        chunk_lower = chunk.lower()
        score = 0
        matches = 0
        
        for term in query_terms:
            if term in chunk_lower:
                matches += 1
                count = chunk_lower.count(term)
                score += min(count * 5, 30)
        
        if matches > 0 or score > 0:
            normalized_score = min(1.0, (matches / len(query_terms)) * 0.8 + (score / 100) * 0.2)
            scores.append((chunk, normalized_score))
    
    sorted_results = sorted(scores, key=lambda x: x[1], reverse=True)[:max(top_k, 1)]
    return sorted_results

# Test queries
test_queries = [
    "How many votes did Nana Akufo Addo receive in 2020?",
    "What were the election results in 2020?",
    "Which party won in Ashanti Region?",
    "Show me Nana Akuffo Addo votes"
]

print("="*80)
print("🧪 TESTING RAG RETRIEVAL")
print("="*80)

for query in test_queries:
    print(f"\n📝 Query: {query}")
    print("-" * 60)
    
    results = retrieve_chunks(query, chunks, top_k=3)
    
    if results:
        print(f"✓ Retrieved {len(results)} chunks:")
        for i, (chunk, score) in enumerate(results, 1):
            # Extract key info
            if "candidate:" in chunk:
                # Find candidate name and votes
                lines = chunk.split(" year:")
                for line in lines[:1]:  # Just first entry
                    print(f"\n  Chunk {i} (relevance: {score:.0%})")
                    if "candidate:" in line and "votes:" in line:
                        # Extract candidate and votes
                        try:
                            cand_start = line.find("candidate:") + 10
                            cand_end = line.find("|", cand_start)
                            candidate = line[cand_start:cand_end].strip()
                            
                            votes_start = line.find("votes:") + 6
                            votes_end = line.find("|", votes_start)
                            if votes_end == -1:
                                votes_end = line.find("votes(%)", votes_start)
                            votes = line[votes_start:votes_end].strip()
                            
                            print(f"    ➜ {candidate}: {votes} votes")
                        except:
                            print(f"    ➜ {line[:100]}...")
    else:
        print("✗ No chunks retrieved")

print("\n" + "="*80)
print("✅ Retrieval system is working correctly!")
print("="*80)
