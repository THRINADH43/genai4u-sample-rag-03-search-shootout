"""One command repeats the whole experiment:  python run_experiment.py
Writes results/results.json and results/results.md. Uses only local models: no API key, $0."""

import json
import os
import statistics
import time

from shootout.chunking import STRATEGIES
from shootout.metrics import hit_and_rr
from shootout.retrievers import BM25, Hybrid, Reranked, Vector

K = 5


def load_docs(folder="docs"):
    return {n: open(os.path.join(folder, n), encoding="utf-8").read() for n in sorted(os.listdir(folder)) if n.endswith(".txt")}


def build(chunks, embedder, cross):
    kw = BM25(chunks)
    vec = Vector(chunks, embedder)
    hyb = Hybrid(kw, vec)
    return [kw, vec, hyb, Reranked(hyb, chunks, cross)]


def run(docs, questions, embedder, cross):
    rows = []
    for strat_name, chunker in STRATEGIES.items():
        chunks = [c for text in docs.values() for c in chunker(text)]
        for retriever in build(chunks, embedder, cross):
            hits, rrs, times = [], [], []
            for q in questions:
                t0 = time.perf_counter()
                top = retriever.search(q["question"], K)
                times.append((time.perf_counter() - t0) * 1000)
                h, rr = hit_and_rr([chunks[i] for i in top], q["answer"])
                hits.append(h)
                rrs.append(rr)
            rows.append({"chunking": strat_name, "chunks": len(chunks), "search": retriever.name,
                         f"hit@{K}": round(sum(hits) / len(hits), 3), "mrr": round(sum(rrs) / len(rrs), 3),
                         "latency_ms_median": round(statistics.median(times), 2),
                         "latency_ms_p95": round(sorted(times)[int(0.95 * len(times)) - 1], 2),
                         "cost_usd": 0.0,
                         "misses": [q["id"] for q, h in zip(questions, hits) if not h]})
    return rows


def to_markdown(rows):
    lines = [f"| chunking | search | hit@{K} | MRR | median ms | p95 ms | cost |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['chunking']} ({r['chunks']}) | {r['search']} | {r[f'hit@{K}']:.3f} | {r['mrr']:.3f} | "
                     f"{r['latency_ms_median']} | {r['latency_ms_p95']} | $0 |")
    return "\n".join(lines)


def main():
    from shootout.models import CrossEncoderModel, MiniLM
    questions = json.load(open("questions.json"))
    rows = run(load_docs(), questions, MiniLM(), CrossEncoderModel())
    os.makedirs("results", exist_ok=True)
    json.dump(rows, open("results/results.json", "w"), indent=1)
    open("results/results.md", "w").write(to_markdown(rows) + "\n")
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
