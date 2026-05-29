export interface CompleteResponse {
  /** Raw text content from LLM (always present) */
  text: string;
  /** Model used, e.g. "claude-sonnet" */
  model_used: string;
  /** Input token count from Anthropic API usage */
  inputTokens: number;
  /** Output token count from Anthropic API usage */
  outputTokens: number;
  /** Sum of inputTokens + outputTokens — used by business-orchestrator for budget tracking */
  token_usage_estimate: number;
  /** Optional error code if the LLM response could not be parsed */
  error_code?: string;
  /** Parsed JSON fields spread at top level when output_schema was provided */
  [key: string]: unknown;
}
