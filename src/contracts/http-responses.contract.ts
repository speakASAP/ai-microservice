import { z } from 'zod';

export const SuccessEnvelopeSchema = <T extends z.ZodTypeAny>(dataSchema: T) =>
  z.object({
    success: z.literal(true),
    data: dataSchema,
  });

export const ErrorEnvelopeSchema = z.object({
  error: z.string(),
  context: z.string().optional(),
  issues: z.array(z.unknown()).optional(),
});

export const HealthResponseSchema = z.object({
  status: z.enum(['ok', 'degraded', 'down']),
  service: z.string(),
});

export type ErrorEnvelope = z.infer<typeof ErrorEnvelopeSchema>;
export type HealthResponse = z.infer<typeof HealthResponseSchema>;
