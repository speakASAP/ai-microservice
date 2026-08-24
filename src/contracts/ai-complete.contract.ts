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
  agent_slug: z.string().min(1).max(160).optional(),
  agent_service_scope: z.string().min(1).max(120).optional(),
});

export const AiCompleteResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  text: z.string(),
  /** The real upstream model id, e.g. "openrouter/google/gemma-4-31b-it:free". */
  model_used: z.string(),
  /** The tier that was requested. Never conflate with model_used: one is a routing
   *  intent, the other is what actually served the call. */
  tier_used: ModelTierSchema,
  /** False when the upstream response carried no model id, so model_used is a tier
   *  name standing in for one. Callers relying on the served model (cv-tuning's
   *  anti-fabrication guard, spec 8.1) MUST treat false as degraded rather than
   *  string-sniffing whether model_used looks like a real id. */
  model_resolved: z.boolean(),
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
