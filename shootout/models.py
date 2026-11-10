"""Local models (free, CPU). Fakes are used in tests so nothing is downloaded there."""

import hashlib
import math
import re


class MiniLM:
    name = "all-MiniLM-L6-v2"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    def embed(self, texts):
        return self.m.encode(list(texts), normalize_embeddings=True).tolist()


class CrossEncoderModel:
    name = "ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        from sentence_transformers import CrossEncoder
        self.m = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    def score(self, pairs):
        return self.m.predict(pairs).tolist()


class FakeEmbedder:
    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 64
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
            n = math.sqrt(sum(x * x for x in v)) or 1
            out.append([x / n for x in v])
        return out


class FakeCrossEncoder:
    def score(self, pairs):
        return [len(set(q.lower().split()) & set(c.lower().split())) for q, c in pairs]
