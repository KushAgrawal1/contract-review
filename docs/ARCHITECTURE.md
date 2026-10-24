# Architecture

The assistant reviews a commercial contract for 40 clause types and answers
questions about it with citations.

## Flow

1. **Ingest**: contracts from the CUAD dataset are split into chunks and
   stored in Postgres with pgvector embeddings (`sql/init.sql`).
2. **Retrieve**: candidate chunks are found with vector search and fused with
   rank fusion. An optional BGE reranker can be installed from
   `requirements-rerank.txt`.
3. **Extract**: an OpenAI model, orchestrated with LangChain and LangGraph,
   decides per clause whether it is present and returns a supporting quote.
4. **Verify**: a quote is marked "verified" only if it appears word for word
   in the source contract.
5. **Serve**: FastAPI exposes the review and question-answering endpoints.
   Docker Compose runs the app and the database.

## Layout

| Path | Purpose |
| --- | --- |
| `app/` | API and review pipeline |
| `sql/` | Database schema |
| `evals/` | Evaluation against held-out CUAD contracts |
| `scripts/` | Helper scripts |
| `tests/` | Unit tests (no database or API key needed) |
