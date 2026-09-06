import { z } from 'zod';

export const ModelTierSchema = z.enum(['free', 'cheap', 'smart', 'premium']);

export const AiCompleteRequestSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  model_tier: ModelTierSchema,
  user_prompt: z.string().min(1),
  system_prompt: z.string().optional(),
  output_schema: z.record(z.string(), z.unknown()).optional(),
  max_tokens: z.number().int().positive().optional(),
  correlation_id: z.string().optional(),
  business_id: z.string().min(1).max(128).optional(),
  businessId: z.string().min(1).max(128).optional(),
  human_approval: z.boolean().optional(),
  humanApproval: z.boolean().optional(),
  agent_slug: z.string().min(1).max(160).optional(),
  agent_service_scope: z.string().min(1).max(120).optional(),
  /**
   * Per-request budget for the upstream LiteLLM call, in ms. Optional: a caller that omits it
   * keeps the global LITELLM_TIMEOUT_MS, which is pinned from above and cannot simply be
   * raised (education-service allows 180s and retries once, so 2x the global is the ceiling).
   * A caller whose own workload is measurably slower — cv-tuning's CV prompts run to 70.3s
   * against a 73s chain — asks for its own instead. Clamped server-side to
   * MAX_CALLER_TIMEOUT_MS: an unbounded value would let one service park a request and
   * starve the shared pool.
   */
  timeout_ms: z.number().int().positive().optional(),
});

export const AiCompleteResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  text: z.string(),
  /** The real upstream model id, e.g. "openrouter/google/gemma-4-31b-it:free". */
  model_used: z.string(),
  /** The tier that was requested. Never conflate with model_used: one is a routing
   *  intent, the other is what actually served the call. */
  tier_used: ModelTierSchema,
  /** False when the real upstream model could not be determined, so model_used holds the
   *  tier name as a stand-in. Callers relying on the served model (cv-tuning's
   *  anti-fabrication guard, spec 8.1) MUST treat false as degraded rather than
   *  string-sniffing whether model_used looks like a real id. */
  model_resolved: z.boolean(),
  /** True when a LiteLLM fallback deployment served the call instead of the tier's own
   *  model. LiteLLM echoes the alias either way, so this is the ONLY signal that the
   *  model silently changed — a quality change that still returns well-formed prose. */
  served_by_fallback: z.boolean(),
  inputTokens: z.number().int().nonnegative().optional(),
  outputTokens: z.number().int().nonnegative().optional(),
  token_usage_estimate: z.number().int().nonnegative().optional(),
  error_code: z.string().optional(),
  error_message: z.string().optional(),
  agent_id: z.string().optional(),
  agent_slug: z.string().optional(),
  agent_name: z.string().optional(),
  agent_service_scope: z.string().optional(),
}).passthrough();

export type ModelTier = z.infer<typeof ModelTierSchema>;
export type AiCompleteRequest = z.infer<typeof AiCompleteRequestSchema>;
export type AiCompleteRequestInput = z.input<typeof AiCompleteRequestSchema>;
export type AiCompleteResponse = z.infer<typeof AiCompleteResponseSchema>;
