import { JobStatus } from '../job-status.enum';

/**
 * Response DTO for job enqueue operation.
 * Returned immediately after a job is submitted to the queue.
 */
export class JobEnqueueResponseDto {
  jobId!: string;
  taskId!: string;
  status!: JobStatus;
  createdAt!: Date;
  implementationProvider?: 'claude-code' | 'codex' | 'ollama' | 'openrouter' | 'litellm';
  intentChecksum?: string;
  lifecycleStage?: string;
  statusDetail?: string;
  auditSummary?: string;
}
