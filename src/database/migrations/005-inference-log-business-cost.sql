ALTER TABLE inference_logs
  ADD COLUMN IF NOT EXISTS business_id VARCHAR(128),
  ADD COLUMN IF NOT EXISTS model_used VARCHAR(128),
  ADD COLUMN IF NOT EXISTS input_tokens INTEGER,
  ADD COLUMN IF NOT EXISTS output_tokens INTEGER,
  ADD COLUMN IF NOT EXISTS token_usage_estimate INTEGER,
  ADD COLUMN IF NOT EXISTS estimated_cost_usd NUMERIC(12, 8);

CREATE INDEX IF NOT EXISTS idx_inference_business_id
ON inference_logs(business_id);
