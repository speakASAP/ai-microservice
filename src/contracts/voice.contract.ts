import { z } from 'zod';

export const TranscribeRequestSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  fileKey: z.string().min(1),
  language: z.string().optional(),
});

export const TranscribeResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  transcript: z.string(),
});

export type TranscribeRequest = z.infer<typeof TranscribeRequestSchema>;
export type TranscribeRequestInput = z.input<typeof TranscribeRequestSchema>;
export type TranscribeResponse = z.infer<typeof TranscribeResponseSchema>;
