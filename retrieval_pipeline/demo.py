from embeddings import embed_texts, embed_query
from storage import build_index
from retrieval import search
from extensions import expand_query

import json

# --- Logging Function ---
def log_pipeline(stage, data, logfile="pipeline_log.json"):
    entry = { "stage": stage, "data": data }
    with open(logfile, "a") as f:
        f.write(json.dumps(entry, indent=2))
        f.write("\n")

# --- Demo ---

texts = [
    "The election was won by candidate A.",
    "Candidate B focused on education policies.",
    "Economic growth slowed in 2025."
]

# Build embeddings + index
embeddings = embed_texts(texts)
index = build_index(embeddings)

# Query
query = "Who won the election?"
query_emb = embed_query(query)

results = search(index, query_emb, texts, k=2)
print("Top-k results:", results)

# Failure case
bad_query = "Weather forecast in Accra"
bad_emb = embed_query(bad_query)
bad_results = search(index, bad_emb, texts, k=2)
print("Failure case:", bad_results)

# Print results nicely
print("\nTop-k results:")
for text, score in results:
    print(f"Text: {text} | Score: {score:.4f}")

print("\nFailure case:")
for text, score in bad_results:
    print(f"Text: {text} | Score: {score:.4f}")

# --- Query Expansion ---
expanded_queries = expand_query(query)  # returns a list of strings
expanded_embs = embed_texts(expanded_queries)  # ✅ embed_texts handles lists

expanded_results = []
for emb in expanded_embs:
    # reshape to (1, d) before passing to FAISS
    import numpy as np
    expanded_results.extend(search(index, np.array(emb).reshape(1, -1), texts, k=2))

print("\nExpanded query results:")
for text, score in expanded_results:
    print(f"Text: {text} | Score: {score:.4f}")

# --- Prompt Engineering Experiments ---
def build_prompt(template, query, context):
    if template == "basic":
        return f"""Answer the question using the following context:
{context}
Question: {query}"""

    elif template == "hallucination":
        return f"""You must only use the provided context. 
If the answer is not in the context, say "Not available."

Context:
{context}
Question: {query}"""

    elif template == "expanded":
        return f"""Answer the question using the expanded queries and retrieved context.
Context:
{context}
Question: {query}"""

    else:
        raise ValueError("Unknown template")

# Build contexts
retrieved_chunks = "\n".join([text for text, score in results])
expanded_chunks = "\n".join([text for text, score in expanded_results])

# Build prompts
prompt_basic = build_prompt("basic", query, retrieved_chunks)
prompt_hallucination = build_prompt("hallucination", query, retrieved_chunks)
prompt_expanded = build_prompt("expanded", query, expanded_chunks)

def run_pipeline(query, index, texts, template="basic"):
    # Stage 1: User Query
    log_pipeline("User Query", query)

    # Stage 2: Retrieval
    query_emb = embed_query(query)
    results = search(index, query_emb, texts, k=3)

    # Convert scores to Python float for logging
    results_serializable = [(t, float(s)) for t, s in results]
    log_pipeline("Retrieved Documents", results_serializable)

    # Stage 3: Context Selection (truncate/filter)
    threshold = 1.5
    filtered = [(t, float(s)) for t, s in results if s < threshold]
    context = "\n".join([t for t, s in filtered])
    log_pipeline("Selected Context", context)

    # Stage 4: Prompt
    prompt = build_prompt(template, query, context)
    log_pipeline("Final Prompt", prompt)

    # Stage 5: LLM Response (placeholder for now)
    response = f"[LLM OUTPUT] Answering: {query} using context."
    log_pipeline("LLM Response", response)

    return response



# Print prompts
print("\n--- Experiment A: Basic Prompt ---")
print(prompt_basic)

print("\n--- Experiment B: Hallucination-Control Prompt ---")
print(prompt_hallucination)

print("\n--- Experiment C: Expanded Query Prompt ---")
print(prompt_expanded)




# --- Run Pipeline Experiments (Part D) ---
response_basic = run_pipeline("Who won the election?", index, texts, template="basic")
response_hallucination = run_pipeline("Who won the election?", index, texts, template="hallucination")
response_expanded = run_pipeline("Who won the election?", index, texts, template="expanded")

print("\n--- Basic ---\n", response_basic)
print("\n--- Hallucination-Control ---\n", response_hallucination)
print("\n--- Expanded ---\n", response_expanded)


# --- Run Adversarial Queries (Part E) ---
response_ambiguous = run_pipeline("Who won?", index, texts, template="basic")
response_misleading = run_pipeline("Who won the election in 2025?", index, texts, template="basic")

print("\n--- Ambiguous Query ---\n", response_ambiguous)
print("\n--- Misleading Query ---\n", response_misleading)



# Save prompts to a file
with open("prompts.txt", "w") as f:
    f.write(prompt_basic + "\n")
    f.write(prompt_hallucination + "\n")
    f.write(prompt_expanded + "\n")