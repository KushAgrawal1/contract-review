CREATE EXTENSION IF NOT EXISTS vector;

-- One row per contract. The full text is kept so chunk offsets can always be checked against it.
CREATE TABLE IF NOT EXISTS contracts (
    id       SERIAL PRIMARY KEY,
    doc_name TEXT NOT NULL UNIQUE,
    n_chars  INT NOT NULL,
    text     TEXT NOT NULL
);

-- A chunk is a character range [char_start, char_end) of its contract.
-- `chunker` lets several chunking strategies live side by side so they can be compared ('fixed', 'clause').
CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    contract_id INT NOT NULL REFERENCES contracts(id) ON DELETE CASCADE,
    chunker     TEXT NOT NULL,
    chunk_index INT NOT NULL,
    char_start  INT NOT NULL,
    char_end    INT NOT NULL,
    heading     TEXT,
    content     TEXT NOT NULL,
    embedding   vector(1536),
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_contract_idx ON chunks (contract_id, chunker);
