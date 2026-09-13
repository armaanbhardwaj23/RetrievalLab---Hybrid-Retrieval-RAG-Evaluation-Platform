# RetrievalLab

RetrievalLab is a small, inspectable platform for learning how RAG retrieval
systems work and measuring dense, BM25, and hybrid retrieval quality. It is
designed to make retrieval decisions and failures visible before adding product
infrastructure.

## Evaluation status

On the 30-question, page-labeled set across both supplied PDFs (20 MScAC and 10 Accessibility Services questions), hybrid RRF reached `Recall@5 = 1.000` and `MRR = 0.950`. BM25 and dense retrieval each reached `Recall@5 = 0.967`; hybrid recovered the one missed case and ranked relevant evidence earlier overall.

| Strategy | Recall@5 | MRR | Mean retrieval latency |
| --- | ---: | ---: | ---: |
| BM25 | 0.967 | 0.900 | 4.2 ms |
| Dense | 0.967 | 0.808 | 356.9 ms |
| Hybrid RRF | 1.000 | 0.950 | 338.5 ms |

These are real measurements from one local run. The corpus and labeled set are small, and OpenRouter network latency varies, so they are not general performance claims. The results assess retrieval evidence only, not answer quality or citation correctness.

## Architecture

```text
PDF / Markdown / TXT
  → ingestion with source metadata
  → traceable chunks
  → dense vectors + BM25 term statistics
  → dense / BM25 / RRF hybrid retrieval
  → grounded answer + source labels
  → retrieval evaluation
```

## Setup

```zsh
uv sync
cp .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env`. The key is never committed.

## Usage

Place PDF, Markdown, and TXT source files under `data/raw/`.

```zsh
# Chunk and embed all supported files into data/processed/dense_index.json.
uv run retrievallab index

# Inspect a retrieval strategy.
uv run retrievallab search "What is CSC2701H?" --strategy hybrid

# Generate a source-labeled answer from the retrieved chunks.
uv run retrievallab answer "What courses are required in the MScAC program?"

# Evaluate one strategy against data/evaluation/questions.json.
uv run retrievallab evaluate --strategy bm25
uv run retrievallab evaluate --strategy dense
uv run retrievallab evaluate --strategy hybrid
```

## Retrieval approaches

- **Dense:** embeds chunks and questions with OpenRouter-hosted
  `openai/text-embedding-3-small`, then ranks cosine similarity exactly in
  memory.
- **BM25:** uses a hand-written lexical scorer with term rarity, term-frequency
  saturation, and length normalization.
- **Hybrid:** fuses dense and BM25 rank lists using Reciprocal Rank Fusion. It
  does not add their incompatible raw scores.

## Evaluation methodology

Each question has manually verified relevant evidence identified by source
filename and PDF page. The evaluation runner reports:

- **Recall@K:** the fraction of labeled evidence found in the top `K` results.
- **MRR:** the reciprocal rank of the first relevant result.
- **Mean retrieval latency:** wall-clock time for each retrieval call.

## Design decisions and current limitations

- Exact in-memory vector search keeps the early system inspectable; a vector
  database is deliberately postponed.
- Chunks are currently limited by characters, not model-token counts. This is a
  transparent baseline to revisit after evaluating more documents.
- The dense index is saved as readable JSON. It is easy to inspect but not
  storage-efficient for large corpora.
- The answer generator is source-constrained and returns source labels, but its
  answers still need grounding review.
- The Accessibility Services PDF emits a font-encoding warning during text
  extraction; it is retained so extraction quality can be examined later.

## Future work

- Expand the benchmark with more independently sourced UofT documents.
- Add an allowlisted official-website snapshot importer.
- Compare chunking configurations using the same evaluation set.
- Add reranking only if measured retrieval failures justify it.
