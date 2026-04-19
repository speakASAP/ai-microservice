-- Create claude_code_jobs table
CREATE TABLE claude_code_jobs (
  job_id VARCHAR(255) PRIMARY KEY,
  task_id UUID NOT NULL,

  -- Input parameters
  repo_path TEXT NOT NULL,
  branch VARCHAR(255) NOT NULL,
  instructions TEXT NOT NULL,
  expected_outcome TEXT,
  timeout_seconds INTEGER DEFAULT 300,
  validation_script TEXT,

  -- Execution state
  status VARCHAR(50) DEFAULT 'queued',
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,

  -- Execution results
  exit_code INTEGER,
  stdout TEXT,
  stderr TEXT,
  git_diff TEXT,
  validation_passed BOOLEAN,
  validation_output TEXT,

  -- Metadata
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),

  CONSTRAINT status_valid CHECK (status IN ('queued', 'executing', 'success', 'failed', 'timeout'))
);

CREATE INDEX idx_claude_code_jobs_task_id ON claude_code_jobs(task_id);
CREATE INDEX idx_claude_code_jobs_status ON claude_code_jobs(status);
CREATE INDEX idx_claude_code_jobs_created_at ON claude_code_jobs(created_at DESC);
