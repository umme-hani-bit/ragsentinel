CREATE EXTENSION IF NOT EXISTS vector;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ragsvc') THEN
    CREATE ROLE ragsvc LOGIN PASSWORD 'ragsvc'
      NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
  END IF;
END$$;

CREATE TABLE IF NOT EXISTS documents (
  doc_id text PRIMARY KEY,
  tenant_id text NOT NULL,
  employee_id text,
  date_str text,
  subject text,
  raw_body text,
  clean_body text,
  augmented_body text,
  inserted_secrets_count int DEFAULT 0,
  inserted_injections_count int DEFAULT 0,
  metadata jsonb DEFAULT '{}'::jsonb,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id bigserial PRIMARY KEY,
  doc_id text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  tenant_id text NOT NULL,
  chunk_index int NOT NULL,
  text text NOT NULL,
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(text,''))) STORED,
  created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_tenant_doc ON chunks(tenant_id, doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv);

-- bge-large-en-v1.5 = 1024 dims; change if your embedding model differs
CREATE TABLE IF NOT EXISTS embeddings (
  chunk_id bigint PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
  tenant_id text NOT NULL,
  vec vector(1024) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_embeddings_tenant ON embeddings(tenant_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_vec_hnsw
  ON embeddings USING hnsw (vec vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS retrieval_logs (
  id bigserial PRIMARY KEY,
  tenant_id text NOT NULL,
  qhash text,
  query text,
  top_k int,
  timings_ms jsonb,
  defense_profile text,
  config_fingerprint text,
  created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tool_logs (
  id bigserial PRIMARY KEY,
  tenant_id text NOT NULL,
  tool_name text NOT NULL,
  allowed boolean NOT NULL,
  reason text,
  payload jsonb,
  created_at timestamptz DEFAULT now()
);

-- RLS: default DENY + FORCE
ALTER TABLE documents       ENABLE ROW LEVEL SECURITY; ALTER TABLE documents       FORCE ROW LEVEL SECURITY;
ALTER TABLE chunks          ENABLE ROW LEVEL SECURITY; ALTER TABLE chunks          FORCE ROW LEVEL SECURITY;
ALTER TABLE embeddings      ENABLE ROW LEVEL SECURITY; ALTER TABLE embeddings      FORCE ROW LEVEL SECURITY;
ALTER TABLE retrieval_logs  ENABLE ROW LEVEL SECURITY; ALTER TABLE retrieval_logs  FORCE ROW LEVEL SECURITY;
ALTER TABLE tool_logs       ENABLE ROW LEVEL SECURITY; ALTER TABLE tool_logs       FORCE ROW LEVEL SECURITY;

CREATE POLICY rls_tenant_documents ON documents
  USING (tenant_id = current_setting('app.tenant_id', true))
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

CREATE POLICY rls_tenant_chunks ON chunks
  USING (tenant_id = current_setting('app.tenant_id', true))
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

CREATE POLICY rls_tenant_embeddings ON embeddings
  USING (tenant_id = current_setting('app.tenant_id', true))
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

CREATE POLICY rls_tenant_retrieval_logs ON retrieval_logs
  USING (tenant_id = current_setting('app.tenant_id', true))
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

CREATE POLICY rls_tenant_tool_logs ON tool_logs
  USING (tenant_id = current_setting('app.tenant_id', true))
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

-- Least-privilege grants
GRANT USAGE ON SCHEMA public TO ragsvc;
GRANT SELECT, INSERT, UPDATE, DELETE ON documents, chunks, embeddings, retrieval_logs, tool_logs TO ragsvc;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO ragsvc;
