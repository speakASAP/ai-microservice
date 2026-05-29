import { z } from 'zod';

export const HealthResponseSchema = z.object({
  status: z.enum(['ok', 'degraded', 'down']),
  service: z.string(),
});

export const NotFoundResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  error: z.string(),
}).passthrough();

export type HealthResponse = z.infer<typeof HealthResponseSchema>;
export type NotFoundResponse = z.infer<typeof NotFoundResponseSchema>;
