DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rageval') THEN
    EXECUTE 'CREATE ROLE rageval LOGIN PASSWORD ''rageval'' BYPASSRLS';
  END IF;
END $$;

GRANT USAGE ON SCHEMA public TO rageval;
GRANT SELECT ON documents, chunks, embeddings TO rageval;
GRANT INSERT ON retrieval_logs, tool_logs TO rageval;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO rageval;