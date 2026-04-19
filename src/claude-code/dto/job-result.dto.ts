import { Type } from 'class-transformer';
import { JobStatus } from '../job-status.enum';

/**
 * Response DTO for completed job execution.
 * Contains execution results, logs, and validation outcome.
 */
export class JobResultDto {
  /** Unique identifier assigned by job queue */
  jobId: string;

  /** References parent task for tracing */
  taskId: string;

  /**
   * Job state: QUEUED (waiting), EXECUTING (in progress),
   * SUCCESS (completed), FAILED (error), TIMEOUT (exceeded limit)
   */
  status: JobStatus;

  /** ISO 8601 timestamp when execution started */
  @Type(() => Date)
  startedAt?: Date;

  /** ISO 8601 timestamp when execution completed */
  @Type(() => Date)
  completedAt?: Date;

  /** Unix exit code from the process (0 = success) */
  exitCode?: number;

  /** Standard output from execution */
  stdout?: string;

  /** Standard error from execution */
  stderr?: string;

  /** Git diff showing repository changes after execution (max 10MB) */
  gitDiff?: string;

  /** Whether optional validationScript passed */
  validationPassed?: boolean;

  /** Output from validation script if run */
  validationOutput?: string;
}
