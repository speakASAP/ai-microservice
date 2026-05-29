import { z } from 'zod';

export const TaskDraftRequestSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  transcript: z.string().min(1),
  context: z.string().optional(),
});

export const TaskDraftResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  tasks: z.array(z.object({
    title: z.string(),
    description: z.string().optional(),
    priority: z.number().int().min(1).max(5).optional(),
  })),
  model_used: z.string(),
}).passthrough();

export type TaskDraftRequest = z.infer<typeof TaskDraftRequestSchema>;
export type TaskDraftResponse = z.infer<typeof TaskDraftResponseSchema>;
