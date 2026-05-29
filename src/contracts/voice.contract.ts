import { z } from 'zod';

export const TranscribeRequestSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  audio_url: z.string().url(),
  language: z.string().optional(),
});

export const TranscribeResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  transcript: z.string(),
  duration_ms: z.number().int().nonnegative().optional(),
  model_used: z.string().optional(),
});

export type TranscribeRequest = z.infer<typeof TranscribeRequestSchema>;
export type TranscribeResponse = z.infer<typeof TranscribeResponseSchema>;
