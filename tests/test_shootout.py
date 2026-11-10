import unittest

from run_experiment import run
from shootout.chunking import fixed_chunks, sentence_chunks
from shootout.metrics import hit_and_rr
from shootout.models import FakeCrossEncoder, FakeEmbedder
from shootout.retrievers import BM25, Hybrid, Vector

CHUNKS = ["rice grows in paddies", "cricket has eleven players", "tea grows in assam hills"]


class ChunkingTests(unittest.TestCase):
    def test_fixed_overlap(self):
        text = " ".join(map(str, range(30)))
        c = fixed_chunks(text, size=25, overlap=5)
        self.assertEqual(len(c), 2)
        self.assertTrue(c[1].startswith("20 "))

    def test_sentences(self):
        self.assertEqual(sentence_chunks("One. Two? Three!"), ["One.", "Two?", "Three!"])


class RetrieverTests(unittest.TestCase):
    def test_bm25_finds_keyword(self):
        self.assertEqual(BM25(CHUNKS).search("eleven players", 1), [1])

    def test_hybrid_uses_both(self):
        kw, vec = BM25(CHUNKS), Vector(CHUNKS, FakeEmbedder())
        self.assertEqual(Hybrid(kw, vec).search("tea assam", 1), [2])


class MetricTests(unittest.TestCase):
    def test_hit_and_mrr(self):
        self.assertEqual(hit_and_rr(["x", "the answer"], "answer"), (1, 0.5))
        self.assertEqual(hit_and_rr(["x"], "answer"), (0, 0.0))


class ExperimentTests(unittest.TestCase):
    def test_every_combination_is_reported(self):
        docs = {"a.txt": "Rice grows in paddies. Cricket has eleven players."}
        qs = [{"id": "q1", "question": "how many cricket players", "answer": "eleven"}]
        rows = run(docs, qs, FakeEmbedder(), FakeCrossEncoder())
        self.assertEqual(len(rows), 8)
        for r in rows:
            for key in ("hit@5", "mrr", "latency_ms_median", "cost_usd"):
                self.assertIn(key, r)


if __name__ == "__main__":
    unittest.main()
