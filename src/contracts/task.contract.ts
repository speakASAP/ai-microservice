import { z } from 'zod';

export const TaskDraftRequestSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  transcript: z.string().optional(),
  textNote: z.string().optional(),
  language: z.string().optional(),
});

export const TaskDraftResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  title: z.string(),
  description: z.string(),
  priority: z.enum(['low', 'normal', 'high']),
  deadline: z.string().optional(),
  modelTier: z.string(),
});

export type TaskDraftRequest = z.infer<typeof TaskDraftRequestSchema>;
export type TaskDraftResponse = z.infer<typeof TaskDraftResponseSchema>;
