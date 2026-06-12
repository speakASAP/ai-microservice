ALTER TABLE claude_code_jobs
  ADD COLUMN IF NOT EXISTS implementation_provider VARCHAR(32) DEFAULT 'claude-code',
  ADD COLUMN IF NOT EXISTS intent TEXT,
  ADD COLUMN IF NOT EXISTS intent_checksum VARCHAR(128);
