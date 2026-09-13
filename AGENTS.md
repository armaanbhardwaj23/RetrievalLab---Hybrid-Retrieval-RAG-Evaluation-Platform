# AGENTS.md

## Project

# RetrievalLab — Hybrid Retrieval & RAG Evaluation Platform

RetrievalLab is an AI engineering project focused on understanding and evaluating RAG retrieval systems.

The goal is NOT to build another generic RAG chatbot.

The goal is to progressively build a retrieval system and measure how different retrieval strategies affect retrieval quality, latency, and answer grounding.

---

## How Codex Should Work With Me

Act as a pair programmer and mentor.

I know the basic RAG concept, but I want to understand the engineering concepts while building the project.

Therefore:

- Do not blindly generate large amounts of code.
- Do not implement major features without explaining the design first.
- Prefer small, understandable changes.
- Explain important concepts before implementing them.
- Explain architectural decisions and trade-offs.
- Avoid unnecessary abstractions and dependencies.
- Do not over-engineer the project.

For significant changes, use this workflow:

1. Explain the concept.
2. Explain the proposed design.
3. Implement the smallest useful change.
4. Run tests/checks.
5. Explain what changed and why.

---

## Initial Goal

The first version should implement:

1. Document ingestion
2. PDF / Markdown / TXT support
3. Chunking
4. Chunk metadata
5. Dense/vector retrieval
6. BM25 retrieval
7. Hybrid retrieval using Reciprocal Rank Fusion (RRF)
8. Basic RAG answer generation
9. Source citations
10. Retrieval evaluation

The initial architecture should roughly be:

Documents
→ Ingestion
→ Chunking + Metadata
→ Dense Retrieval
              ↘
                Hybrid Retrieval → RAG → Answer + Sources
              ↗
        BM25 Retrieval

Evaluation should compare the retrieval approaches.

---

## Learning Goals

While building, help me understand:

- document ingestion
- chunking strategies
- embeddings
- vector similarity
- dense retrieval
- sparse retrieval
- BM25
- Reciprocal Rank Fusion
- top-K retrieval
- Recall@K
- MRR
- retrieval latency
- citation grounding
- RAG failure modes

Do not hide the core retrieval logic behind frameworks when a simple implementation would make the concept easier to understand.

---

## Two-Day Initial Scope

### Day 1

Build the baseline:

- repository/environment setup
- document ingestion
- chunking
- metadata
- embeddings
- vector retrieval
- basic RAG generation

### Day 2

Add:

- BM25
- RRF hybrid retrieval
- 20–30 question evaluation set
- Recall@K
- MRR
- comparison of dense vs BM25 vs hybrid retrieval
- citations

The important outcome is not just that the system works.

We should produce measurable results showing how the retrieval approaches compare.

---

## Postpone For Now

Do NOT implement these during the initial milestone unless explicitly requested:

- knowledge graphs / Neo4j
- multimodal retrieval
- image/audio retrieval
- RAG-Fusion
- adaptive retrieval
- sophisticated reranking
- agents
- LangGraph
- frontend
- distributed infrastructure
- Kubernetes
- cloud deployment
- complex observability

These are possible future extensions.

Only introduce additional complexity when there is a clear problem that requires it.

---

## Engineering Principles

Prefer:

- Python
- clear modules
- type hints where useful
- small functions
- simple interfaces
- tests for important logic
- configuration through environment variables
- minimal dependencies

Avoid:

- unnecessary frameworks
- premature optimization
- unnecessary abstractions
- hard-coded secrets
- giant files/functions

Do not commit `.env` files or secrets.

---

## Evaluation

Create a small evaluation dataset of approximately 20–30 questions.

Each question should have a known relevant source/chunk where possible.

Initially measure:

- Recall@K
- MRR
- retrieval latency

Do not invent benchmark numbers.

Only report measurements actually produced by the project.

If an approach performs poorly, document the failure and why we think it happened.

---

## README

The README should eventually explain:

1. Problem
2. Architecture
3. Setup
4. Usage
5. Retrieval approaches
6. Evaluation methodology
7. Results
8. Design decisions
9. Failure cases
10. Future work

Lead with measurable results where possible.

Example:

> Hybrid retrieval improved Recall@5 from X to Y on a 25-question evaluation set.

Only use real measured values.

---

## Current Task

For the first interaction after reading this file:

1. Inspect the repository.
2. Explain your understanding of RetrievalLab.
3. Propose the minimal project structure for the first milestone.
4. Explain why each component exists.
5. Explain the concepts I should understand before implementation.
6. Identify architectural decisions we need to make now.
7. Explicitly list what we are postponing.

Do NOT implement the entire project yet.

Wait for my confirmation before making significant changes.