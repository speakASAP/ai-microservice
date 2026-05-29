import { z } from 'zod';

export const EmailIntentSchema = z.enum([
  'support', 'sales', 'contract', 'technical', 'billing',
  'spam', 'unknown', 'multi_intent',
]);

export const EmailActionSchema = z.enum(['auto_respond', 'route_to_queue', 'escalate']);

export const EmailIngestRequestSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  message_id: z.string().optional(),
  from: z.string().optional(),
  subject: z.string().optional(),
  body: z.string().optional(),
  headers: z.record(z.string(), z.unknown()).optional(),
}).passthrough();

export const EmailIngestResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  success: z.literal(true),
  payload: z.record(z.string(), z.unknown()),
  duration_ms: z.number().int().nonnegative(),
  model_used: z.string(),
});

export const EmailClassifyRequestSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  use_llm: z.boolean().optional(),
  payload: z.record(z.string(), z.unknown()).optional(),
  message_id: z.string().optional(),
}).passthrough();

export const EmailClassifyResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  success: z.literal(true),
  intent: EmailIntentSchema,
  confidence: z.number().min(0).max(1),
  raw_scores: z.record(z.string(), z.number()).nullable().optional(),
  model_used: z.string(),
  duration_ms: z.number().int().nonnegative(),
  llm_output: z.unknown().optional(),
  llm_fallback_reason: z.string().optional(),
});

export const EmailExtractRequestSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  intent: z.string().optional(),
  payload: z.record(z.string(), z.unknown()).optional(),
}).passthrough();

export const EmailExtractResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  success: z.literal(true),
  model_used: z.string(),
}).passthrough();

export const EmailDecideRequestSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  intent: z.string().min(1),
  confidence: z.union([z.number(), z.string()]),
  entities: z.record(z.string(), z.unknown()).optional(),
  use_llm: z.boolean().optional(),
});

export const EmailDecideResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  success: z.literal(true),
  action: EmailActionSchema,
  model_used: z.string(),
  escalation_reason: z.string().nullable().optional(),
  queue: z.string().nullable().optional(),
  llm_output: z.unknown().optional(),
  llm_fallback_reason: z.string().optional(),
}).passthrough();

export type EmailIntent = z.infer<typeof EmailIntentSchema>;
export type EmailAction = z.infer<typeof EmailActionSchema>;
export type EmailIngestRequest = z.infer<typeof EmailIngestRequestSchema>;
export type EmailClassifyRequest = z.infer<typeof EmailClassifyRequestSchema>;
export type EmailDecideRequest = z.infer<typeof EmailDecideRequestSchema>;
