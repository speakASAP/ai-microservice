import { Injectable, Logger, BadRequestException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { createHash, randomUUID } from 'crypto';
import { ClaudeCodeJob } from '../database/entities/claude-code-job.entity';
import type { ExecuteCodeRequestInput, ImplementationProvider } from '../contracts';
import { JobEnqueueResponseDto } from './dto/job-enqueue-response.dto';
import { JobStatusResponseDto } from './dto/job-status-response.dto';
import { JobStatus } from './job-status.enum';
import { AmqpConnection } from '@golevelup/nestjs-rabbitmq';

const SUMMARY_LIMIT = 1200;
type JobStatusValue = JobStatus | ClaudeCodeJob['status'];

function redactSensitive(value = ''): string {
  return value
    .replace(/(authorization\s*[:=]\s*bearer\s+)[^\s'"`]+/gi, '$1[REDACTED]')
    .replace(/((?:api[_-]?key|token|secret|password|passwd|pwd)\s*[:=]\s*)[^\s'"`]+/gi, '$1[REDACTED]')
    .replace(/-----BEGIN [^-]+ PRIVATE KEY-----[\s\S]*?-----END [^-]+ PRIVATE KEY-----/g, '[REDACTED_PRIVATE_KEY]')
    .replace(/\b(?:sk|pk|rk|ghp|glpat|xox[baprs])_[A-Za-z0-9_=-]{16,}\b/g, '[REDACTED_TOKEN]')
    .replace(/\b[A-Za-z0-9+/]{32,}={0,2}\b/g, '[REDACTED_SECRET]');
}

function summarizeText(value?: string, limit = SUMMARY_LIMIT): string | undefined {
  const normalized = redactSensitive(value ?? '').replace(/\s+/g, ' ').trim();
  if (!normalized) return undefined;
  return normalized.length > limit ? `${normalized.slice(0, limit)}... [truncated]` : normalized;
}

function countLines(value?: string): number {
  if (!value) return 0;
  return value.split(/\r?\n/).filter((line) => line.length > 0).length;
}

/**
 * ClaudeCodeService manages Claude Code job execution lifecycle.
 * Handles job enqueueing, status polling, and result persistence.
 */
@Injectable()
export class ClaudeCodeService {
  private logger = new Logger(ClaudeCodeService.name);

  constructor(
    @InjectRepository(ClaudeCodeJob)
    private jobRepository: Repository<ClaudeCodeJob>,
    private amqpConnection: AmqpConnection,
  ) {}

  /**
   * Enqueue a new Claude Code job for execution.
   * Creates a job record with status=queued and publishes to RabbitMQ.
   */
  async enqueueJob(dto: ExecuteCodeRequestInput): Promise<JobEnqueueResponseDto> {
    const jobId = randomUUID();
    const timeoutSeconds = dto.timeoutSeconds || 300;

    const executionMode = dto.executionMode ?? 'code';
    const model = dto.model ?? (executionMode === 'print' ? process.env.CC_PRINT_MODEL : undefined);
    const implementationProvider = dto.implementationProvider ?? 'claude-code';
    const intentChecksum = dto.intentChecksum ?? this.hashIntent(dto.intent);

    const job = this.jobRepository.create({
      jobId,
      taskId: dto.taskId,
      repoPath: dto.repoPath,
      branch: dto.branch,
      instructions: dto.instructions,
      expectedOutcome: dto.expectedOutcome,
      timeoutSeconds,
      validationScript: dto.validationScript,
      executionMode,
      model,
      implementationProvider,
      intent: dto.intent,
      intentChecksum,
      status: 'queued' as JobStatus,
      lifecycleStage: 'queued',
      statusDetail: 'Job accepted and queued for implementation execution.',
      auditSummary: this.buildAuditSummary({
        status: JobStatus.QUEUED,
        implementationProvider,
        intentChecksum,
        validationPassed: undefined,
      }),
      lastObservedAt: new Date(),
    });

    const saved = await this.jobRepository.save(job);

    // Enqueue to RabbitMQ
    try {
      await this.amqpConnection.publish(
        'claude-code-exchange',
        'claude-code.execute',
        {
          jobId,
          taskId: dto.taskId,
          repoPath: dto.repoPath,
          branch: dto.branch,
          instructions: dto.instructions,
          expectedOutcome: dto.expectedOutcome,
          timeoutSeconds,
          validationScript: dto.validationScript,
          executionMode,
          model,
          implementationProvider,
          intent: dto.intent,
          intentChecksum,
        },
      );
      this.logger.log(JSON.stringify({
        event: 'Claude Code Job Enqueued',
        jobId,
        taskId: dto.taskId,
        repoPath: dto.repoPath,
        branch: dto.branch,
        implementationProvider,
      }));
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      this.logger.warn(`Failed to publish job ${jobId} to RabbitMQ: ${errorMsg}`);
      // Job is persisted; consumer will process when RabbitMQ recovers
    }

    return {
      jobId: saved.jobId,
      taskId: saved.taskId,
      status: saved.status as JobStatus,
      createdAt: saved.createdAt,
      implementationProvider: saved.implementationProvider,
      intentChecksum: saved.intentChecksum ?? undefined,
      lifecycleStage: saved.lifecycleStage ?? 'queued',
      statusDetail: saved.statusDetail ?? undefined,
      auditSummary: saved.auditSummary ?? undefined,
    };
  }

  /**
   * Get job status and results by job ID.
   * Returns null if job not found.
   */
  async getJobStatus(jobId: string): Promise<JobStatusResponseDto | null> {
    const job = await this.jobRepository.findOne({ where: { jobId } });
    if (!job) {
      return null;
    }

    return {
      jobId: job.jobId,
      taskId: job.taskId,
      status: job.status as JobStatus,
      implementationProvider: job.implementationProvider,
      intent: job.intent ?? undefined,
      intentChecksum: job.intentChecksum ?? undefined,
      startedAt: job.startedAt ?? undefined,
      completedAt: job.completedAt ?? undefined,
      exitCode: job.exitCode ?? undefined,
      stdout: job.stdout ?? undefined,
      stderr: job.stderr ?? undefined,
      gitDiff: job.gitDiff ?? undefined,
      validationPassed: job.validationPassed ?? undefined,
      validationOutput: job.validationOutput ?? undefined,
      lifecycleStage: job.lifecycleStage ?? this.lifecycleForStatus(job.status as JobStatus),
      statusDetail: job.statusDetail ?? this.defaultStatusDetail(job),
      outputSummary: job.outputSummary ?? this.buildOutputSummary(job.stdout, job.stderr, job.gitDiff),
      failureSummary: job.failureSummary ?? this.buildFailureSummary(job.status as JobStatus, job.stderr, job.exitCode),
      validationSummary: job.validationSummary ?? this.buildValidationSummary(job.validationPassed, job.validationOutput),
      auditSummary: job.auditSummary ?? this.buildAuditSummary({
        status: job.status as JobStatus,
        implementationProvider: job.implementationProvider,
        intentChecksum: job.intentChecksum,
        validationPassed: job.validationPassed,
      }),
      executionDurationMs: job.executionDurationMs ?? this.durationMs(job.startedAt, job.completedAt),
      lastObservedAt: job.lastObservedAt ?? job.updatedAt ?? undefined,
      retryCount: job.retryCount,
      maxRetries: job.maxRetries,
      nextRetryAt: job.nextRetryAt ?? undefined,
      lastErrorAt: job.lastErrorAt ?? undefined,
      errorHistory: job.errorHistory?.map((entry) => ({
        ...entry,
        error: summarizeText(entry.error, 400) ?? 'No error detail recorded.',
      })),
    };
  }

  /**
   * Update job execution results (called by RabbitMQ consumer).
   * Validates status transitions to prevent invalid state changes.
   * Valid transitions:
   * - queued → executing | failed
   * - executing → success | failed | timeout
   * - success/failed/timeout → terminal (no updates allowed)
   */
  async updateJobExecution(
    jobId: string,
    data: Partial<{
      status: JobStatus;
      startedAt: Date;
      completedAt: Date;
      exitCode: number;
      stdout: string;
      stderr: string;
      gitDiff: string;
      validationPassed: boolean;
      validationOutput: string;
      implementationProvider: ImplementationProvider;
      lifecycleStage: string;
      statusDetail: string;
      outputSummary: string;
      failureSummary: string;
      validationSummary: string;
      auditSummary: string;
      executionDurationMs: number;
      lastObservedAt: Date;
    }>,
  ): Promise<void> {
    const job = await this.jobRepository.findOne({ where: { jobId } });

    // Validate status transitions if status is being updated
    if (data.status) {
      if (job && !this.isValidStatusTransition(job.status as JobStatus, data.status)) {
        throw new BadRequestException(
          `Invalid status transition from ${job.status} to ${data.status}`,
        );
      }
    }

    const status = data.status ?? (job?.status as JobStatus | undefined);
    const implementationProvider = data.implementationProvider ?? job?.implementationProvider;
    const startedAt = data.startedAt ?? job?.startedAt;
    const completedAt = data.completedAt ?? job?.completedAt;
    const validationPassed = data.validationPassed ?? job?.validationPassed;
    const updateData = {
      ...data,
      lifecycleStage: data.lifecycleStage ?? (status ? this.lifecycleForStatus(status) : undefined),
      statusDetail: data.statusDetail ?? this.defaultStatusDetail({
        ...job,
        ...data,
        status,
        implementationProvider,
      } as ClaudeCodeJob),
      outputSummary: data.outputSummary ?? this.buildOutputSummary(
        data.stdout ?? job?.stdout,
        data.stderr ?? job?.stderr,
        data.gitDiff ?? job?.gitDiff,
      ),
      failureSummary: data.failureSummary ?? this.buildFailureSummary(
        status,
        data.stderr ?? job?.stderr,
        data.exitCode ?? job?.exitCode,
      ),
      validationSummary: data.validationSummary ?? this.buildValidationSummary(
        validationPassed,
        data.validationOutput ?? job?.validationOutput,
      ),
      auditSummary: data.auditSummary ?? this.buildAuditSummary({
        status,
        implementationProvider,
        intentChecksum: job?.intentChecksum,
        validationPassed,
      }),
      executionDurationMs: data.executionDurationMs ?? this.durationMs(startedAt, completedAt),
      lastObservedAt: data.lastObservedAt ?? new Date(),
    };

    await this.jobRepository.update({ jobId }, updateData);
    this.logger.log(JSON.stringify({
      event: 'Claude Code Job Updated',
      jobId,
      status: data.status || 'unchanged',
    }));
  }

  /**
   * Check if a status transition is valid according to job state machine.
   */
  private isValidStatusTransition(from: JobStatus, to: JobStatus): boolean {
    const validTransitions: Record<JobStatus, JobStatus[]> = {
      [JobStatus.QUEUED]: [JobStatus.EXECUTING, JobStatus.FAILED],
      [JobStatus.EXECUTING]: [JobStatus.SUCCESS, JobStatus.FAILED, JobStatus.TIMEOUT, JobStatus.RETRYING],
      [JobStatus.RETRYING]: [JobStatus.EXECUTING, JobStatus.FAILED],
      [JobStatus.SUCCESS]: [],
      [JobStatus.FAILED]: [],
      [JobStatus.TIMEOUT]: [],
    };
    return validTransitions[from]?.includes(to) ?? false;
  }

  /**
   * Get a job by ID (helper for consumers to check retry status).
   */
  async getJobById(jobId: string): Promise<ClaudeCodeJob | null> {
    return this.jobRepository.findOne({ where: { jobId } });
  }

  /**
   * Mark a job as retrying with error tracking.
   * Updates job to RETRYING status, increments retry count, schedules next retry.
   */
  async markJobRetrying(
    jobId: string,
    data: {
      retryCount: number;
      nextRetryAt: Date;
      lastError: string;
    },
  ): Promise<void> {
    const job = await this.jobRepository.findOne({ where: { jobId } });
    if (!job) return;

    const errorEntry = {
      attempt: data.retryCount,
      error: summarizeText(data.lastError, 400) ?? 'No error detail recorded.',
      timestamp: new Date().toISOString(),
    };
    const errorHistory = [...(job.errorHistory ?? []), errorEntry];

    await this.jobRepository.update(
      { jobId },
      {
        status: JobStatus.RETRYING,
        retryCount: data.retryCount,
        nextRetryAt: data.nextRetryAt,
        lastErrorAt: new Date(),
        errorHistory,
        lifecycleStage: 'retrying',
        statusDetail: `Retry ${data.retryCount} scheduled after transient execution error.`,
        failureSummary: summarizeText(data.lastError, 800),
        auditSummary: this.buildAuditSummary({
          status: JobStatus.RETRYING,
          implementationProvider: job.implementationProvider,
          intentChecksum: job.intentChecksum,
          validationPassed: job.validationPassed,
        }),
        lastObservedAt: new Date(),
      },
    );
    this.logger.log(JSON.stringify({
      event: 'Claude Code Job Retry Scheduled',
      jobId,
      retryCount: data.retryCount,
      nextRetryAt: data.nextRetryAt,
    }));
  }

  /**
   * Get all retrying jobs that are due for re-execution.
   * Used by consumer's OnApplicationBootstrap for recovery.
   */
  async getRetryingJobsDue(): Promise<ClaudeCodeJob[]> {
    return this.jobRepository
      .createQueryBuilder('job')
      .where('job.status = :status', { status: JobStatus.RETRYING })
      .andWhere('job.nextRetryAt <= :now', { now: new Date() })
      .getMany();
  }

  /**
   * Get queued jobs for brokerless direct execution.
   */
  async getQueuedJobs(limit = 1): Promise<ClaudeCodeJob[]> {
    return this.jobRepository
      .createQueryBuilder('job')
      .where('job.status = :status', { status: JobStatus.QUEUED })
      .orderBy('job.createdAt', 'ASC')
      .limit(limit)
      .getMany();
  }

  private hashIntent(intent?: string): string | undefined {
    const normalized = intent?.trim();
    if (!normalized) return undefined;
    return createHash('sha256').update(normalized).digest('hex');
  }

  private lifecycleForStatus(status?: JobStatusValue): string | undefined {
    if (!status) return undefined;
    const lifecycle: Record<JobStatus, string> = {
      [JobStatus.QUEUED]: 'queued',
      [JobStatus.EXECUTING]: 'executing',
      [JobStatus.RETRYING]: 'retrying',
      [JobStatus.SUCCESS]: 'completed',
      [JobStatus.FAILED]: 'failed',
      [JobStatus.TIMEOUT]: 'timed_out',
    };
    return lifecycle[status];
  }

  private defaultStatusDetail(job: Partial<ClaudeCodeJob> & { status?: JobStatusValue }): string | undefined {
    const provider = job.implementationProvider ?? 'claude-code';
    switch (job.status) {
      case JobStatus.QUEUED:
        return `Job is queued for ${provider} execution.`;
      case JobStatus.EXECUTING:
        return `Job is executing with ${provider}.`;
      case JobStatus.RETRYING:
        return `Job is waiting for retry ${job.retryCount ?? 0} of ${job.maxRetries ?? 3}.`;
      case JobStatus.SUCCESS:
        return `Job completed successfully with ${provider}.`;
      case JobStatus.FAILED:
        return `Job failed with ${provider}${typeof job.exitCode === 'number' ? ` exit code ${job.exitCode}` : ''}.`;
      case JobStatus.TIMEOUT:
        return `Job timed out with ${provider}.`;
      default:
        return undefined;
    }
  }

  private buildOutputSummary(stdout?: string, stderr?: string, gitDiff?: string): string | undefined {
    const parts = [
      `stdout_lines=${countLines(stdout)}`,
      `stderr_lines=${countLines(stderr)}`,
      `git_diff_lines=${countLines(gitDiff)}`,
    ];
    const excerpt = summarizeText([stdout, stderr].filter(Boolean).join('\n'), 900);
    return excerpt ? `${parts.join(' ')} excerpt="${excerpt}"` : parts.join(' ');
  }

  private buildFailureSummary(status?: JobStatusValue, stderr?: string, exitCode?: number): string | undefined {
    if (![JobStatus.FAILED, JobStatus.TIMEOUT].includes(status as JobStatus) && exitCode !== undefined && exitCode === 0) {
      return undefined;
    }
    if (![JobStatus.FAILED, JobStatus.TIMEOUT].includes(status as JobStatus) && !stderr) {
      return undefined;
    }
    const detail = summarizeText(stderr, 900) ?? 'No stderr detail recorded.';
    const code = typeof exitCode === 'number' ? `exit_code=${exitCode} ` : '';
    return `${code}${detail}`;
  }

  private buildValidationSummary(validationPassed?: boolean, validationOutput?: string): string | undefined {
    if (validationPassed === undefined && !validationOutput) return undefined;
    const state = validationPassed ? 'passed' : 'failed';
    const output = summarizeText(validationOutput, 900);
    return output ? `validation=${state} ${output}` : `validation=${state}`;
  }

  private buildAuditSummary(data: {
    status?: JobStatusValue;
    implementationProvider?: ImplementationProvider;
    intentChecksum?: string;
    validationPassed?: boolean;
  }): string | undefined {
    if (!data.status && !data.implementationProvider && !data.intentChecksum) return undefined;
    return [
      `provider=${data.implementationProvider ?? 'unknown'}`,
      `status=${data.status ?? 'unknown'}`,
      `intent_checksum=${data.intentChecksum ?? 'missing'}`,
      `validation=${data.validationPassed === undefined ? 'not_run' : data.validationPassed ? 'passed' : 'failed'}`,
    ].join(' ');
  }

  private durationMs(startedAt?: Date, completedAt?: Date): number | undefined {
    if (!startedAt || !completedAt) return undefined;
    const started = new Date(startedAt).getTime();
    const completed = new Date(completedAt).getTime();
    if (!Number.isFinite(started) || !Number.isFinite(completed) || completed < started) {
      return undefined;
    }
    return completed - started;
  }
}
