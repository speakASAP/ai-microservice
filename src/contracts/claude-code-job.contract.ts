import { z } from 'zod';

export const ClaudeCodeJobStatusSchema = z.enum(['queued', 'executing', 'success', 'failed', 'timeout', 'retrying']);

export const ExecuteCodeRequestSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  taskId: z.string().uuid(),
  repoPath: z.string().min(1),
  branch: z.string().min(1),
  instructions: z.string().min(1),
  expectedOutcome: z.string().optional(),
  timeoutSeconds: z.number().int().min(10).max(3600).optional(),
  validationScript: z.string().optional(),
  executionMode: z.enum(['code', 'print']).optional(),
  model: z.string().optional(),
});

export const JobEnqueueResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  jobId: z.string().uuid(),
  taskId: z.string().uuid(),
  status: ClaudeCodeJobStatusSchema,
  createdAt: z.union([z.string(), z.date()]),
});

export const JobStatusResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  jobId: z.string().uuid(),
  taskId: z.string().uuid(),
  status: ClaudeCodeJobStatusSchema,
  startedAt: z.union([z.string(), z.date()]).optional(),
  completedAt: z.union([z.string(), z.date()]).optional(),
  exitCode: z.number().int().optional(),
  stdout: z.string().optional(),
  stderr: z.string().optional(),
  gitDiff: z.string().optional(),
  validationPassed: z.boolean().optional(),
  validationOutput: z.string().optional(),
});

export type ClaudeCodeJobStatus = z.infer<typeof ClaudeCodeJobStatusSchema>;
export type ExecuteCodeRequest = z.infer<typeof ExecuteCodeRequestSchema>;
export type JobEnqueueResponse = z.infer<typeof JobEnqueueResponseSchema>;
export type JobStatusResponse = z.infer<typeof JobStatusResponseSchema>;
