-- Add retry support to claude_code_jobs table
ALTER TABLE claude_code_jobs
ADD COLUMN IF NOT EXISTS retry_count INTEGER NOT NULL DEFAULT 0,
ADD COLUMN IF NOT EXISTS max_retries INTEGER NOT NULL DEFAULT 3,
ADD COLUMN IF NOT EXISTS next_retry_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS last_error_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS error_history JSONB;

-- Update status constraint to include 'retrying'
ALTER TABLE claude_code_jobs
DROP CONSTRAINT IF EXISTS status_valid;

ALTER TABLE claude_code_jobs
ADD CONSTRAINT status_valid CHECK (status IN ('queued', 'executing', 'success', 'failed', 'timeout', 'retrying'));

-- Create index for startup recovery query
CREATE INDEX IF NOT EXISTS idx_claude_code_jobs_retry_recovery
ON claude_code_jobs(status, next_retry_at)
WHERE status = 'retrying' AND next_retry_at IS NOT NULL;
