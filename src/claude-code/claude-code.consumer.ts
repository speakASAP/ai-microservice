import {
  Injectable,
  Logger,
  OnApplicationBootstrap,
} from '@nestjs/common';
import { RabbitSubscribe, AmqpConnection } from '@golevelup/nestjs-rabbitmq';
import { ClaudeCodeService } from './claude-code.service';
import { LoggingClient } from './logging.client';
import { JobStatus } from './job-status.enum';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs';
import * as path from 'path';

const execAsync = promisify(exec);
const CC_CLI = () => process.env.CC_CLI_PATH?.trim() || 'claude';
const CC_PRINT_MODEL = () => process.env.CC_PRINT_MODEL?.trim() || 'claude-sonnet-4-6';
const RETRY_BACKOFF_MS = [30_000, 90_000, 270_000];

/**
 * Determines if an error is retryable (transient) vs permanent.
 * Transient: timeout, connection reset, spawn failures, process signals
 * Permanent: non-zero exit code from completed process
 */
function isRetryableError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  const msg = error.message.toLowerCase();
  return (
    msg.includes('etimedout') ||
    msg.includes('econnreset') ||
    msg.includes('enoent') ||
    msg.includes('spawn') ||
    msg.includes('sigkill') ||
    msg.includes('sigterm') ||
    msg.includes('timeout') ||
    msg.includes('socket hang up')
  );
}

/**
 * RabbitMQ consumer that processes Claude Code job execution requests.
 * Implements smart retry with exponential backoff for transient failures.
 * Logs job lifecycle events to logging-microservice.
 */
@Injectable()
export class ClaudeCodeConsumer implements OnApplicationBootstrap {
  private readonly logger = new Logger(ClaudeCodeConsumer.name);
  private directQueueTimer?: NodeJS.Timeout;
  private directQueueActive = false;

  constructor(
    private readonly service: ClaudeCodeService,
    private readonly loggingClient: LoggingClient,
    private readonly amqpConnection: AmqpConnection,
  ) {}

  /**
   * OnApplicationBootstrap: Recover jobs stuck in retrying state.
   * After process restart, re-queue jobs that were waiting for retry.
   */
  async onApplicationBootstrap(): Promise<void> {
    try {
      const dueJobs = await this.service.getRetryingJobsDue();
      for (const job of dueJobs) {
        this.logger.warn(
          JSON.stringify({
            event: 'Claude Code Job Retry Recovery',
            jobId: job.jobId,
            retryCount: job.retryCount,
          }),
        );
        await this.loggingClient.log('warn', 'Claude Code Job Retry Recovery', {
          jobId: job.jobId,
          retryCount: job.retryCount,
        });
        // Re-schedule immediately (0 ms delay)
        this.scheduleRetry(
          job.jobId,
          job.repoPath,
          job.branch,
          job.instructions,
          job.timeoutSeconds,
          job.validationScript,
          job.executionMode ?? 'code',
          job.model,
          0,
          job.maxRetries ?? 3,
        );
      }
    } catch (error) {
      this.logger.error('Failed to recover retrying jobs', error);
    }

    if (this.isDirectExecutionEnabled()) {
      this.startDirectQueuePolling();
    }
  }

  /**
   * Handle job execution from RabbitMQ message.
   * Creates worktree, executes claude code CLI, captures results, validates.
   * On transient error: schedules retry with exponential backoff.
   * On permanent error: marks job as failed.
   */
  @RabbitSubscribe({
    exchange: 'claude-code-exchange',
    routingKey: 'claude-code.execute',
    queue: 'claude-code-execute-queue',
  })
  async handleJobExecution(msg: any, amqpChannel: any) {
    const {
      jobId,
      repoPath,
      branch,
      instructions,
      timeoutSeconds,
      validationScript,
      executionMode = 'code',
      model,
    } = msg;

    this.logger.log(
      JSON.stringify({
        event: 'Claude Code Job Executing',
        jobId,
        repoPath,
      }),
    );
    await this.loggingClient.log('info', 'Claude Code Job Executing', {
      jobId,
      repoPath,
      branch,
    });

    try {
      // Update status to executing
      const startedAt = new Date();
      await this.service.updateJobExecution(jobId, {
        status: JobStatus.EXECUTING as JobStatus,
        startedAt,
      });

      if (executionMode === 'print') {
        await this.executePrintJob(jobId, repoPath, instructions, timeoutSeconds, model ?? CC_PRINT_MODEL(), startedAt);
        return;
      }

      // Create worktree
      const worktreePath = `/tmp/worktree-${jobId}`;
      this.logger.debug(`Creating worktree at ${worktreePath}`);

      try {
        // Clean up any existing worktree
        await execAsync(`git worktree remove ${worktreePath} --force || true`, {
          cwd: repoPath,
        });
      } catch (e) {
        // Ignore cleanup errors
      }

      const { stdout: remoteOutput } = await execAsync(
        `git remote get-url origin`,
        { cwd: repoPath },
      );
      const remoteUrl = remoteOutput.trim();

      await execAsync(
        `git worktree add ${worktreePath} origin/${branch}`,
        { cwd: repoPath, timeout: timeoutSeconds * 1000 },
      );

      // Execute claude code
      this.logger.debug(
        `Executing claude code in ${worktreePath} with branch ${branch}`,
      );

      let stdout = '';
      let stderr = '';
      let exitCode = 0;

      try {
        const { stdout: cmdOutput } = await execAsync(
          `claude code --repo-path ${worktreePath} --instructions "${instructions.replace(/"/g, '\\"')}" --max-tokens 4000`,
          { timeout: timeoutSeconds * 1000 },
        );
        stdout = cmdOutput;
      } catch (error: any) {
        exitCode = error.code || 1;
        stdout = error.stdout || '';
        stderr = error.stderr || '';
      }

      // Get git diff
      let gitDiff = '';
      try {
        const { stdout: diffOutput } = await execAsync(`git diff`, {
          cwd: worktreePath,
        });
        gitDiff = diffOutput;
      } catch (e) {
        this.logger.warn(`Failed to get git diff: ${e}`);
      }

      // Run optional validation script
      let validationPassed = true;
      let validationOutput = '';

      if (validationScript) {
        try {
          const scriptPath = path.join(worktreePath, validationScript);
          const { stdout: valOutput } = await execAsync(`bash ${scriptPath}`, {
            cwd: worktreePath,
            timeout: 60000, // 1 min timeout for validation
          });
          validationOutput = valOutput;
          validationPassed = true;
        } catch (error: any) {
          validationPassed = false;
          validationOutput = error.stderr || error.stdout || error.message;
          this.logger.warn(
            `Validation failed for job ${jobId}: ${validationOutput}`,
          );
        }
      }

      // Update job with results
      const finalStatus = exitCode === 0 ? JobStatus.SUCCESS : JobStatus.FAILED;
      await this.service.updateJobExecution(jobId, {
        status: finalStatus as JobStatus,
        exitCode,
        stdout,
        stderr,
        gitDiff,
        validationPassed,
        validationOutput,
        completedAt: new Date(),
      });

      this.logger.log(
        JSON.stringify({
          event: 'Claude Code Job Completed',
          jobId,
          status: finalStatus,
          exitCode,
          durationMs: new Date().getTime() - startedAt.getTime(),
        }),
      );
      await this.loggingClient.log('info', 'Claude Code Job Completed', {
        jobId,
        status: finalStatus,
        exitCode,
      });

      // Clean up worktree
      try {
        await execAsync(`git worktree remove ${worktreePath} --force`, {
          cwd: repoPath,
        });
      } catch (e) {
        this.logger.warn(`Failed to clean up worktree ${worktreePath}: ${e}`);
      }
    } catch (error: any) {
      // Outer catch: determine if error is retryable
      const job = await this.service.getJobById(jobId);
      const retryCount = job?.retryCount ?? 0;
      const maxRetries = job?.maxRetries ?? 3;

      if (isRetryableError(error) && retryCount < maxRetries && job) {
        // Transient error: schedule retry
        const nextRetry = retryCount;
        const delayMs = RETRY_BACKOFF_MS[nextRetry] ?? RETRY_BACKOFF_MS[2];

        this.logger.warn(
          JSON.stringify({
            event: 'Claude Code Job Retry Scheduled',
            jobId,
            retryCount: nextRetry + 1,
            delayMs,
            error: error.message,
          }),
        );
        await this.loggingClient.log(
          'warn',
          'Claude Code Job Retry Scheduled',
          {
            jobId,
            retryCount: nextRetry + 1,
            delayMs,
            error: error.message,
          },
        );

        const nextRetryAt = new Date(Date.now() + delayMs);
        await this.service.markJobRetrying(jobId, {
          retryCount: nextRetry + 1,
          nextRetryAt,
          lastError: error.message,
        });

        // Schedule retry after delay
        this.scheduleRetry(
          jobId,
          job.repoPath,
          job.branch,
          job.instructions,
          job.timeoutSeconds,
          job.validationScript,
          job.executionMode ?? 'code',
          job.model,
          delayMs,
          maxRetries,
        );
      } else {
        // Permanent error or max retries exceeded
        this.logger.error(
          JSON.stringify({
            event: 'Claude Code Job Failed',
            jobId,
            retryCount,
            maxRetries,
            error: error.message,
          }),
        );
        await this.loggingClient.log('error', 'Claude Code Job Failed', {
          jobId,
          retryCount,
          maxRetries,
          error: error.message,
        });

        await this.service.updateJobExecution(jobId, {
          status: JobStatus.FAILED as JobStatus,
          stderr: error.message,
          completedAt: new Date(),
        });
      }
    }
  }

  /**
   * Planning / validation: claude --print in repo (no worktree, no file edits).
   */
  private async executePrintJob(
    jobId: string,
    repoPath: string,
    instructions: string,
    timeoutSeconds: number,
    model: string,
    startedAt: Date,
  ): Promise<void> {
    const tmpFile = path.join('/tmp', `cc-print-${jobId}.txt`);
    let stdout = '';
    let stderr = '';
    let exitCode = 0;

    try {
      fs.writeFileSync(tmpFile, instructions, 'utf-8');
      const modelFlag = model ? `--model ${model}` : '';
      const { stdout: out } = await execAsync(
        `cd "${repoPath.replace(/"/g, '\\"')}" && ${CC_CLI()} --print ${modelFlag} < "${tmpFile}"`,
        { timeout: timeoutSeconds * 1000, maxBuffer: 10 * 1024 * 1024 },
      );
      stdout = out;
    } catch (error: any) {
      exitCode = error.code || 1;
      stdout = error.stdout || '';
      stderr = error.stderr || error.message || '';
    } finally {
      try { fs.unlinkSync(tmpFile); } catch { /* ignore */ }
    }

    const finalStatus = exitCode === 0 ? JobStatus.SUCCESS : JobStatus.FAILED;
    await this.service.updateJobExecution(jobId, {
      status: finalStatus as JobStatus,
      exitCode,
      stdout,
      stderr,
      completedAt: new Date(),
    });

    await this.loggingClient.log('info', 'Claude Code Print Job Completed', {
      jobId,
      status: finalStatus,
      exitCode,
      durationMs: Date.now() - startedAt.getTime(),
      model,
    });
  }

  /**
   * Schedule a job retry after the specified delay.
   * Uses setTimeout to re-publish to RabbitMQ after backoff.
   */
  private scheduleRetry(
    jobId: string,
    repoPath: string,
    branch: string,
    instructions: string,
    timeoutSeconds: number,
    validationScript: string | undefined,
    executionMode: 'code' | 'print',
    model: string | undefined,
    delayMs: number,
    maxRetries: number,
  ): void {
    const message = {
      jobId,
      repoPath,
      branch,
      instructions,
      timeoutSeconds,
      validationScript,
      executionMode,
      model,
    };

    setTimeout(async () => {
      if (this.isDirectExecutionEnabled()) {
        await this.handleJobExecution(message, null);
        return;
      }

      try {
        await this.amqpConnection.publish('claude-code-exchange', 'claude-code.execute', message);
        this.logger.log(`Retry published for job ${jobId} after ${delayMs}ms delay`);
      } catch (error) {
        this.logger.error(`Failed to re-publish retry for job ${jobId}: ${error}`);
      }
    }, delayMs);
  }

  private isDirectExecutionEnabled(): boolean {
    return process.env.CLAUDE_CODE_DIRECT_EXECUTION === 'true';
  }

  private startDirectQueuePolling(): void {
    if (this.directQueueTimer) {
      return;
    }

    this.logger.warn('Claude Code direct execution enabled; polling queued jobs without RabbitMQ');
    this.directQueueTimer = setInterval(() => {
      void this.processDirectQueueOnce();
    }, Number(process.env.CLAUDE_CODE_DIRECT_POLL_MS ?? 5000));
    void this.processDirectQueueOnce();
  }

  private async processDirectQueueOnce(): Promise<void> {
    if (this.directQueueActive) {
      return;
    }
    this.directQueueActive = true;

    try {
      const jobs = await this.service.getQueuedJobs(1);
      for (const job of jobs) {
        await this.handleJobExecution({
          jobId: job.jobId,
          repoPath: job.repoPath,
          branch: job.branch,
          instructions: job.instructions,
          timeoutSeconds: job.timeoutSeconds,
          validationScript: job.validationScript,
          executionMode: job.executionMode ?? 'code',
          model: job.model,
        }, null);
      }
    } catch (error) {
      this.logger.error('Claude Code direct queue polling failed', error);
    } finally {
      this.directQueueActive = false;
    }
  }
}
