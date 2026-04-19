import { JobStatus } from '../job-status.enum';

/**
 * Response DTO for job enqueue operation.
 * Returned immediately after a job is submitted to the queue.
 */
export class JobEnqueueResponseDto {
  /** Unique job identifier for polling status */
  jobId: string;

  /** Parent task ID (links to orchestrator task) */
  taskId: string;

  /** Initial job status (always 'queued' on creation) */
  status: JobStatus;

  /** Timestamp when job was created */
  createdAt: Date;
}
