import { Injectable, Logger } from '@nestjs/common';
import { RabbitSubscribe } from '@golevelup/nestjs-rabbitmq';
import { ClaudeCodeService } from './claude-code.service';
import { JobStatus } from './job-status.enum';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as path from 'path';

const execAsync = promisify(exec);

/**
 * RabbitMQ consumer that processes Claude Code job execution requests.
 * Subscribes to claude-code.execute messages and runs code in isolated git worktrees.
 */
@Injectable()
export class ClaudeCodeConsumer {
  private logger = new Logger(ClaudeCodeConsumer.name);

  constructor(private service: ClaudeCodeService) {}

  /**
   * Handle job execution from RabbitMQ message.
   * Creates worktree, executes claude code CLI, captures results, validates, and updates job.
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
    } = msg;

    this.logger.log(JSON.stringify({
      event: 'Claude Code Job Executing',
      jobId,
      repoPath,
    }));

    try {
      // Update status to executing
      const startedAt = new Date();
      await this.service.updateJobExecution(jobId, {
        status: JobStatus.EXECUTING as JobStatus,
        startedAt,
      });

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

      this.logger.log(JSON.stringify({
        event: 'Claude Code Job Completed',
        jobId,
        status: finalStatus,
        exitCode,
        durationMs: new Date().getTime() - startedAt.getTime(),
      }));

      // Clean up worktree
      try {
        await execAsync(`git worktree remove ${worktreePath} --force`, {
          cwd: repoPath,
        });
      } catch (e) {
        this.logger.warn(`Failed to clean up worktree ${worktreePath}: ${e}`);
      }
    } catch (error: any) {
      this.logger.error(`Job execution failed: ${jobId}`, error);

      await this.service.updateJobExecution(jobId, {
        status: JobStatus.FAILED as JobStatus,
        stderr: error.message,
        completedAt: new Date(),
      });
    }
  }
}
