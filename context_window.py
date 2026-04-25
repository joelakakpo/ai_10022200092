def manage_context(chunks, scores, k=3, threshold=0.5):
    # Rank by score
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1])
    # Filter by threshold
    filtered = [c for c, s in ranked if s < threshold]
    # Truncate to top-k
    return filtered[:k]
