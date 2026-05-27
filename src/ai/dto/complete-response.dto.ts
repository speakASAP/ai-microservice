export interface CompleteResponse {
  /** Parsed JSON value extracted from the LLM response, or raw text if no schema was requested */
  data: unknown;
  /** Raw text content from LLM (always present) */
  text: string;
  /** Model tier used for this request */
  model_used: string;
  /** Input token count from Anthropic API usage */
  inputTokens: number;
  /** Output token count from Anthropic API usage */
  outputTokens: number;
  /** Sum of inputTokens + outputTokens — used by business-orchestrator for budget tracking */
  token_usage_estimate: number;
  /** Optional error code if the LLM response could not be parsed */
  error_code?: string;
}
