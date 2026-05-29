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
});

export const AiCompleteResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  text: z.string(),
  model_used: z.string(),
  inputTokens: z.number().int().nonnegative().optional(),
  outputTokens: z.number().int().nonnegative().optional(),
  token_usage_estimate: z.number().int().nonnegative().optional(),
}).passthrough();

export type ModelTier = z.infer<typeof ModelTierSchema>;
export type AiCompleteRequest = z.infer<typeof AiCompleteRequestSchema>;
export type AiCompleteResponse = z.infer<typeof AiCompleteResponseSchema>;
