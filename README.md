# LegalRAG

LegalRAG is a domain-specific legal assistant built on Retrieval-Augmented Generation (RAG).
It retrieves and aggregates information from legal documents (judicial decisions, laws, contracts,
commentaries) and answers questions with references back to the source text.

> **Status:** Working prototype — document ingestion, multilingual retrieval and answer
> generation are implemented. Items under *Roadmap* are planned, not yet built.

## Features
- **Multilingual retrieval** — multilingual E5 embeddings, so queries and documents may be in different languages.
- **Source-grounded answers** — responses are generated from retrieved passages.
- **Streamlit UI** — upload documents, build the index, and ask questions.

## Architecture
- **Framework:** LlamaIndex
- **LLM:** Anthropic Claude 3.5 Sonnet
- **Embeddings:** `intfloat/multilingual-e5-large` (HuggingFace)
- **Vector store:** Qdrant (persisted to disk by default; `QDRANT_PATH` overrides the location, in-memory in the offline demo script)
- **UI:** Streamlit

## Measuring retrieval

Changing the embedding model, the chunk size or `similarity_top_k` used to
produce no number at all, so there was no way to know whether a change helped.
`scripts/demo_retrieval.py` prints passages for a human to eyeball, which is a
smoke test rather than a measurement.

```bash
python -m scripts.evaluate_retrieval
python -m scripts.evaluate_retrieval --fail-under recall@5=0.8   # gate a change
python -m scripts.evaluate_retrieval --json > baseline.json
python -m scripts.evaluate_retrieval --baseline baseline.json    # exit 1 on a drop
```

It runs the retrieval half of the stack over a committed labelled set of
`(query, relevant_ids)` rows and reports **recall@k** and **MRR**, plus the
queries that retrieved nothing relevant, so a drop names the query that broke
rather than only moving a mean. MRR is there because recall@10 cannot tell
"found it at rank 1" from "found it at rank 9", and only one of those is a
retriever you would ship.

No API key, no ~2 GB model, no torch, exactly like the demo.

**What it does and does not tell you.** Retrieval runs under a deterministic
bag-of-words embedder, because LlamaIndex's `MockEmbedding` returns the same
constant vector for every text and would make the ranking arbitrary. So these
numbers are not a verdict on multilingual-E5's semantics. They are a regression
gate on everything around it: chunking, `similarity_top_k`, the index
configuration, the retrieval wiring. Pass `--embed-model real` when the question
is about the model itself; that path loads the heavy dependencies.

The metric arithmetic lives in `src/evaluation.py`, free of LlamaIndex and of
any embedder, so it is tested against hand-computed values rather than against
whatever a model happened to do that day.

## Roadmap (planned)
- Legal entity extraction / term normalization (e.g. Legal-BERT)
- Richer document parsing (unstructured.io / LlamaParse)
- Automated quality evaluation with an LLM judge (faithfulness, answer
  relevance) on top of the retrieval measurement above

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your_key"
streamlit run app.py
```

## Try it offline
Smoke-test the retrieval pipeline with no Anthropic API key, no ~2 GB embedding
model and no torch. The demo indexes a tiny hardcoded legal corpus with a
lightweight `MockEmbedding` and prints the top passages for a sample query:
```bash
pip install llama-index-core llama-index-vector-stores-qdrant qdrant-client
python -m scripts.demo_retrieval
```
