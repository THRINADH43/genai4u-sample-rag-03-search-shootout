def hit_and_rr(ranked_chunks, answer):
    """Hit if any of the top chunks contains the answer text; reciprocal rank of the first one."""
    for rank, chunk in enumerate(ranked_chunks, start=1):
        if answer.lower() in chunk.lower():
            return 1, 1 / rank
    return 0, 0.0
