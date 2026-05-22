export interface CompleteResponse {
  /** Parsed JSON value extracted from the LLM response, or raw text if no schema was requested */
  data: unknown;
  /** Raw text content from LLM (always present) */
  text: string;
  /** Model tier used for this request */
  model_used: string;
  /** Real token count from LiteLLM usage.prompt_tokens */
  inputTokens: number;
  /** Real token count from LiteLLM usage.completion_tokens */
  outputTokens: number;
  /** Sum of inputTokens + outputTokens — used by business-orchestrator for budget tracking */
  token_usage_estimate: number;
  /** Optional error code if the LLM response could not be parsed */
  error_code?: string;
}
