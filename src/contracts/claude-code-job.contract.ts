import { z } from 'zod';

export const ClaudeCodeJobStatusSchema = z.enum(['queued', 'pending', 'running', 'done', 'completed', 'failed', 'cancelled']);

export const ExecuteCodeRequestSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  prompt: z.string().min(1),
  working_directory: z.string().optional(),
  timeout_ms: z.number().int().positive().optional(),
  correlation_id: z.string().optional(),
});

export const JobEnqueueResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  jobId: z.string(),
  status: ClaudeCodeJobStatusSchema,
  createdAt: z.unknown(),
}).passthrough();

export const JobStatusResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  jobId: z.string(),
  status: ClaudeCodeJobStatusSchema,
}).passthrough();

export type ClaudeCodeJobStatus = z.infer<typeof ClaudeCodeJobStatusSchema>;
export type ExecuteCodeRequest = z.infer<typeof ExecuteCodeRequestSchema>;
export type JobEnqueueResponse = z.infer<typeof JobEnqueueResponseSchema>;
export type JobStatusResponse = z.infer<typeof JobStatusResponseSchema>;
