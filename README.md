# RetrievalLab

RetrievalLab is a small, inspectable platform for learning how RAG retrieval
systems work and measuring dense, BM25, and hybrid retrieval quality. It is
designed to make retrieval decisions and failures visible before adding product
infrastructure.

## Evaluation status

The evaluation set now contains **30 page-labeled questions** across both supplied PDFs: 20 from the MScAC handbook and 10 from the Accessibility Services handbook. The expanded comparison has not yet been run, so RetrievalLab deliberately makes no current performance claim for it.

The earlier 20-question, MScAC-only result is retained as historical baseline only: hybrid RRF reached `MRR = 1.000`, versus `0.925` for dense and BM25; all reached `Recall@5 = 1.000`.

Those figures are real measurements from one local run, but the corpus is small,
the labels come from one handbook, and OpenRouter network latency varies; this
is not a general performance claim.

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
