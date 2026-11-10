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
  mean and 95th-percentile search latency per question (embedding the query included;
  building the index excluded), and cost per 1,000 questions. All models run locally, so no
  API is called and the cost is $0.00; on a hosted embedding API it would be the price of
  1,000 query embeddings (plus reranking calls).

## Results

| chunking | search | hit@5 | MRR | mean ms | p95 ms | cost per 1,000 questions |
|---|---|---|---|---|---|---|
| fixed25 (30) | keyword | 0.938 | 0.906 | 0.02 | 0.02 | $0.00 |
| fixed25 (30) | vector | 1.000 | 0.953 | 9.2 | 10.26 | $0.00 |
| fixed25 (30) | hybrid | 1.000 | 0.964 | 10.05 | 11.01 | $0.00 |
| fixed25 (30) | reranked | 1.000 | 0.969 | 131.15 | 175.6 | $0.00 |
| sentence (51) | keyword | 0.938 | 0.878 | 0.04 | 0.05 | $0.00 |
| sentence (51) | vector | 1.000 | 0.964 | 15.89 | 18.99 | $0.00 |
| sentence (51) | hybrid | 1.000 | 0.923 | 13.65 | 15.63 | $0.00 |
| sentence (51) | reranked | 1.000 | 0.984 | 143.12 | 192.61 | $0.00 |

Keyword search missed q07 ("What is the IPL?": the document says "Indian Premier League",
never "IPL") and q18 / q08 (questions phrased with different words from the text, for
example "What goes into masala tea?"). Every other approach found every answer.

## Recommendation

**Hybrid search on fixed 25-word chunks.**

- It finds every answer (hit@5 = 1.000) with MRR 0.964, at about 10 ms per question.
- Reranking is only slightly better (MRR 0.969 on fixed chunks, 0.984 on sentences), but
  it is **about 10 to 13 times slower** (131 to 143 ms on average). For one or two more first-place results out
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
- Latency was measured on one laptop CPU in Docker and varies run to run: a first run gave
  about twice these times. The ranking of the methods stayed the same.
- Cost is $0 only because the models run locally; a hosted embedding API would add cost
  per query and per indexed chunk.
