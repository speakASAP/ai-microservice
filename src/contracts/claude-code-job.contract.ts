import { z } from 'zod';

export const ClaudeCodeJobStatusSchema = z.enum(['queued', 'executing', 'success', 'failed', 'timeout', 'retrying']);
export const ImplementationProviderSchema = z.enum(['claude-code', 'codex', 'ollama', 'openrouter', 'litellm']);

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
  implementationProvider: ImplementationProviderSchema.default('claude-code'),
  intent: z.string().min(1).optional(),
  intentChecksum: z.string().min(1).optional(),
});

export const JobEnqueueResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  jobId: z.string().uuid(),
  taskId: z.string().uuid(),
  status: ClaudeCodeJobStatusSchema,
  createdAt: z.union([z.string(), z.date()]),
  implementationProvider: ImplementationProviderSchema.optional(),
  intentChecksum: z.string().optional(),
  lifecycleStage: z.string().optional(),
  statusDetail: z.string().optional(),
  auditSummary: z.string().optional(),
});

export const JobStatusResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  jobId: z.string().uuid(),
  taskId: z.string().uuid(),
  status: ClaudeCodeJobStatusSchema,
  implementationProvider: ImplementationProviderSchema.optional(),
  intent: z.string().optional(),
  intentChecksum: z.string().optional(),
  startedAt: z.union([z.string(), z.date()]).optional(),
  completedAt: z.union([z.string(), z.date()]).optional(),
  exitCode: z.number().int().optional(),
  stdout: z.string().optional(),
  stderr: z.string().optional(),
  gitDiff: z.string().optional(),
  validationPassed: z.boolean().optional(),
  validationOutput: z.string().optional(),
  lifecycleStage: z.string().optional(),
  statusDetail: z.string().optional(),
  outputSummary: z.string().optional(),
  failureSummary: z.string().optional(),
  validationSummary: z.string().optional(),
  auditSummary: z.string().optional(),
  executionDurationMs: z.number().int().optional(),
  lastObservedAt: z.union([z.string(), z.date()]).optional(),
  retryCount: z.number().int().optional(),
  maxRetries: z.number().int().optional(),
  nextRetryAt: z.union([z.string(), z.date()]).optional(),
  lastErrorAt: z.union([z.string(), z.date()]).optional(),
  errorHistory: z.array(z.object({
    attempt: z.number().int(),
    error: z.string(),
    timestamp: z.string(),
  })).optional(),
});

export type ClaudeCodeJobStatus = z.infer<typeof ClaudeCodeJobStatusSchema>;
export type ImplementationProvider = z.infer<typeof ImplementationProviderSchema>;
export type ExecuteCodeRequest = z.infer<typeof ExecuteCodeRequestSchema>;
export type ExecuteCodeRequestInput = z.input<typeof ExecuteCodeRequestSchema>;
export type JobEnqueueResponse = z.infer<typeof JobEnqueueResponseSchema>;
export type JobStatusResponse = z.infer<typeof JobStatusResponseSchema>;
