import { JobStatus } from '../job-status.enum';

/**
 * Response DTO for job status polling.
 * Contains full job execution details including results, logs, and validation output.
 */
export class JobStatusResponseDto {
  jobId!: string;
  taskId!: string;
  status!: JobStatus;

  /** Timestamp when job execution started (null if not yet started) */
  startedAt?: Date;

  /** Timestamp when job execution completed (null if not yet completed) */
  completedAt?: Date;

  /** Process exit code (0 = success, non-zero = failure; null if still running) */
  exitCode?: number;

  /** Standard output from command execution (empty string if no output) */
  stdout?: string;

  /** Standard error from command execution (empty string if no errors) */
  stderr?: string;

  /** Git diff output showing code changes made (null if no validation script) */
  gitDiff?: string;

  /** Whether optional validation script passed (exit code 0) */
  validationPassed?: boolean;

  /** Output from validation script execution */
  validationOutput?: string;
}
