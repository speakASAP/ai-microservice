-- Print-mode jobs (planning / validation): claude --print, no worktree
ALTER TABLE claude_code_jobs
  ADD COLUMN IF NOT EXISTS execution_mode VARCHAR(16) NOT NULL DEFAULT 'code',
  ADD COLUMN IF NOT EXISTS model VARCHAR(128);
