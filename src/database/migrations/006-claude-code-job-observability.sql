ALTER TABLE claude_code_jobs
  ADD COLUMN IF NOT EXISTS "lifecycleStage" varchar(64),
  ADD COLUMN IF NOT EXISTS "statusDetail" text,
  ADD COLUMN IF NOT EXISTS "outputSummary" text,
  ADD COLUMN IF NOT EXISTS "failureSummary" text,
  ADD COLUMN IF NOT EXISTS "validationSummary" text,
  ADD COLUMN IF NOT EXISTS "auditSummary" text,
  ADD COLUMN IF NOT EXISTS "executionDurationMs" integer,
  ADD COLUMN IF NOT EXISTS "lastObservedAt" timestamptz;

CREATE INDEX IF NOT EXISTS idx_claude_code_jobs_lifecycle_stage
  ON claude_code_jobs ("lifecycleStage");

CREATE INDEX IF NOT EXISTS idx_claude_code_jobs_last_observed_at
  ON claude_code_jobs ("lastObservedAt");
