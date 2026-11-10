# Search strategy shootout

Which retrieval approach works best on my documents? I compared **4 search approaches** ×
**2 chunking strategies** on **32 questions** with known answers, measuring hit rate, mean
reciprocal rank (MRR), latency and cost. Everything runs locally on a CPU: no API key, $0.

## Repeat the whole experiment with one command

```
pip install -r requirements.txt
python run_experiment.py          # prints the table, writes results/results.json and results/results.md
python -m unittest discover -s tests -t .
```

## Setup

- **Documents:** 18 short `.txt` documents I wrote (monsoon, rice, cricket, UPI, ISRO ...).
- **Questions:** `questions.json`, 32 questions. Each has an answer string (for example
  "Sriharikota"). A search **hits** if one of its top 5 chunks contains the answer text.
- **Chunking:** `fixed25` = 25-word windows with 5 words overlap (30 chunks);
  `sentence` = one sentence per chunk (51 chunks).
- **Search approaches** (`shootout/retrievers.py`):
  - `keyword`: BM25 written by me (k1 1.5, b 0.75).
  - `vector`: `all-MiniLM-L6-v2` sentence embeddings, cosine similarity.
  - `hybrid`: reciprocal rank fusion of the keyword and vector top 20 (`1 / (60 + rank)`).
  - `reranked`: hybrid top 20 reordered by the cross-encoder `ms-marco-MiniLM-L-6-v2`.
- Every combination gets the same questions, the same k = 5 and the same scoring, so the
  comparison is fair.
- **Metrics:** hit@5, MRR (1 / rank of the first chunk containing the answer, 0 if none),
  median and 95th-percentile search latency per question (embedding the query included;
  building the index excluded), and cost (all models run locally, so $0).

## Results

| chunking | search | hit@5 | MRR | median ms | p95 ms | cost |
|---|---|---|---|---|---|---|
| fixed25 (30) | keyword | 0.938 | 0.906 | 0.05 | 0.07 | $0 |
| fixed25 (30) | vector | 1.000 | 0.953 | 20.4 | 36.49 | $0 |
| fixed25 (30) | hybrid | 1.000 | 0.964 | 21.92 | 32.65 | $0 |
| fixed25 (30) | reranked | 1.000 | 0.969 | 315.59 | 604.08 | $0 |
| sentence (51) | keyword | 0.938 | 0.878 | 0.04 | 0.06 | $0 |
| sentence (51) | vector | 1.000 | 0.964 | 19.85 | 23.96 | $0 |
| sentence (51) | hybrid | 1.000 | 0.923 | 21.92 | 70.18 | $0 |
| sentence (51) | reranked | 1.000 | 0.984 | 164.23 | 258.43 | $0 |

Keyword search missed q07 ("What is the IPL?": the document says "Indian Premier League",
never "IPL") and q18 / q08 (questions phrased with different words from the text, for
example "What goes into masala tea?"). Every other approach found every answer.

## Recommendation

**Hybrid search on fixed 25-word chunks.**

- It finds every answer (hit@5 = 1.000) with MRR 0.964, at about 22 ms per question.
- Reranking is only slightly better (MRR 0.969 on fixed chunks, 0.984 on sentences), but
  it is **8 to 15 times slower** (160 to 600 ms). For one or two more first-place results out
  of 32, that is not worth it unless answers must be in position 1.
- I prefer hybrid over pure vector because keyword search keeps exact names, numbers and
  codes (like "2015" or "IRCTC") safe when a document set has many of them, even though on
  this set vector alone did as well.
- Fixed chunks beat sentence chunks for hybrid (0.964 vs 0.923): single sentences are short,
  so BM25 scores on them are noisy and pull the fusion down.

## Where this fails, honestly

- **The test is too easy to separate the strong methods.** Three of four approaches hit
  100%. With 18 short documents, almost any decent method finds the answer in the top 5.
  A bigger, messier document set would be needed to rank vector, hybrid and reranked properly.
- 32 questions is small: one question changes hit@5 by 3 points.
- I wrote both the documents and the questions, so the questions may use the documents'
  own words more than real users would. That flatters keyword search.
- Latency was measured on one laptop CPU in Docker and varies run to run (the p95 numbers
  especially).
- Cost is $0 only because the models run locally; a hosted embedding API would add cost
  per query and per indexed chunk.
