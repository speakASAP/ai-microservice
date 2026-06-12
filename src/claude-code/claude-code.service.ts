import { Injectable, Logger, BadRequestException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { createHash, randomUUID } from 'crypto';
import { ClaudeCodeJob } from '../database/entities/claude-code-job.entity';
import { ExecuteCodeRequestInput } from '../contracts';
import { JobEnqueueResponseDto } from './dto/job-enqueue-response.dto';
import { JobStatusResponseDto } from './dto/job-status-response.dto';
import { JobStatus } from './job-status.enum';
import { AmqpConnection } from '@golevelup/nestjs-rabbitmq';

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
      intentChecksum: saved.intentChecksum,
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
      implementationProvider: 'claude-code' | 'codex' | 'ollama' | 'openrouter' | 'litellm';
    }>,
  ): Promise<void> {
    // Validate status transitions if status is being updated
    if (data.status) {
      const job = await this.jobRepository.findOne({ where: { jobId } });
      if (job && !this.isValidStatusTransition(job.status as JobStatus, data.status)) {
        throw new BadRequestException(
          `Invalid status transition from ${job.status} to ${data.status}`,
        );
      }
    }

    await this.jobRepository.update({ jobId }, data);
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
      error: data.lastError,
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
}
