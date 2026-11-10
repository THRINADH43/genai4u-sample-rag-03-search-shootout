"""The four search approaches. Each has search(query, k) -> list of chunk indexes, best first."""

import math
import re
from collections import Counter, defaultdict

STOP = {"a", "an", "the", "is", "are", "of", "in", "on", "to", "and", "or", "for", "with", "it",
        "by", "as", "at", "be", "can", "what", "which", "who", "how", "does", "do", "where", "when"}


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


class BM25:
    name = "keyword"

    def __init__(self, chunks, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c)) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / len(self.lens)
        self.df = Counter(t for d in self.docs for t in d)
        self.n = len(chunks)

    def scores(self, query):
        out = defaultdict(float)
        for t in tokenize(query):
            if t not in self.df:
                continue
            idf = math.log((self.n - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1)
            for i, d in enumerate(self.docs):
                if t in d:
                    tf = d[t]
                    out[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
        return out

    def search(self, query, k):
        s = self.scores(query)
        return sorted(s, key=lambda i: -s[i])[:k]


class Vector:
    name = "vector"

    def __init__(self, chunks, embedder):
        self.embedder = embedder
        self.vecs = embedder.embed(chunks)

    def search(self, query, k):
        q = self.embedder.embed([query])[0]
        sims = [sum(a * b for a, b in zip(q, v)) for v in self.vecs]
        return sorted(range(len(sims)), key=lambda i: -sims[i])[:k]


class Hybrid:
    """Reciprocal rank fusion: score = sum of 1 / (60 + rank) over both lists."""
    name = "hybrid"

    def __init__(self, keyword, vector, depth=20):
        self.keyword, self.vector, self.depth = keyword, vector, depth

    def search(self, query, k):
        fused = defaultdict(float)
        for ranking in (self.keyword.search(query, self.depth), self.vector.search(query, self.depth)):
            for rank, i in enumerate(ranking, start=1):
                fused[i] += 1 / (60 + rank)
        return sorted(fused, key=lambda i: -fused[i])[:k]


class Reranked:
    """Take the hybrid top 20 and reorder them with a cross-encoder that reads query and chunk together."""
    name = "reranked"

    def __init__(self, hybrid, chunks, cross_encoder, depth=20):
        self.hybrid, self.chunks, self.ce, self.depth = hybrid, chunks, cross_encoder, depth

    def search(self, query, k):
        candidates = self.hybrid.search(query, self.depth)
        scores = self.ce.score([(query, self.chunks[i]) for i in candidates])
        order = sorted(range(len(candidates)), key=lambda j: -scores[j])
        return [candidates[j] for j in order][:k]
