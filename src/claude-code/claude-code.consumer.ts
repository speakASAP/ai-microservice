import {
  Injectable,
  Logger,
  OnApplicationBootstrap,
} from '@nestjs/common';
import { RabbitSubscribe, AmqpConnection } from '@golevelup/nestjs-rabbitmq';
import { ClaudeCodeService } from './claude-code.service';
import { LoggingClient } from './logging.client';
import { JobStatus } from './job-status.enum';
import type { ImplementationProvider } from '../contracts';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs';
import * as path from 'path';

const execAsync = promisify(exec);
const CC_CLI = () => process.env.CC_CLI_PATH?.trim() || 'claude';
const CC_PRINT_MODEL = () => process.env.CC_PRINT_MODEL?.trim() || 'claude-sonnet-4-6';
const CODEX_CLI = () => process.env.CODEX_CLI_PATH?.trim() || 'codex';
const CODEX_MODEL = () => process.env.CODEX_MODEL?.trim() || undefined;
const CODEX_SANDBOX = () => process.env.CODEX_SANDBOX?.trim() || 'workspace-write';
const CODEX_PRINT_SANDBOX = () => process.env.CODEX_PRINT_SANDBOX?.trim() || 'read-only';
const CODEX_APPROVAL_POLICY = () => process.env.CODEX_APPROVAL_POLICY?.trim() || 'never';
const RATE_LIMIT_FALLBACK_PROVIDER = () => process.env.CLAUDE_CODE_RATE_LIMIT_FALLBACK_PROVIDER?.trim() || 'litellm';
const LITELLM_BASE_URL = () => process.env.LITELLM_BASE_URL?.replace(/\/$/, '') || '';
const LITELLM_MASTER_KEY = () => process.env.LITELLM_MASTER_KEY || '';
const LITELLM_FALLBACK_MODELS = () => (process.env.CLAUDE_CODE_LITELLM_FALLBACK_MODELS || 'free,cheap')
  .split(',')
  .map((model) => model.trim())
  .filter(Boolean);
const LITELLM_TIMEOUT_MS = () => Number(process.env.CLAUDE_CODE_LITELLM_TIMEOUT_MS || 120_000);
const RETRY_BACKOFF_MS = [30_000, 90_000, 270_000];

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

function resolveProvider(provider?: string): ImplementationProvider {
  return provider === 'codex' ? 'codex' : 'claude-code';
}

function providerLabel(provider: ImplementationProvider): string {
  if (provider === 'codex') return 'Codex';
  if (provider === 'ollama') return 'Ollama';
  if (provider === 'openrouter') return 'OpenRouter';
  if (provider === 'litellm') return 'LiteLLM';
  return 'Claude Code';
}

function providerForLiteLlmModel(model: string): ImplementationProvider {
  const normalized = model.toLowerCase();
  if (normalized === 'cheap') return 'openrouter';
  if (normalized.includes('openrouter')) return 'openrouter';
  if (normalized === 'free' || normalized.endsWith('-fallback') || normalized.includes('ollama')) return 'ollama';
  return 'litellm';
}

function resolveModel(
  provider: ImplementationProvider,
  executionMode: 'code' | 'print',
  model?: string,
): string | undefined {
  if (model) return model;
  if (provider === 'codex') return CODEX_MODEL();
  return executionMode === 'print' ? CC_PRINT_MODEL() : undefined;
}

function modelFlag(model?: string): string {
  return model ? `--model ${shellQuote(model)}` : '';
}

function buildCodeCommand(
  provider: ImplementationProvider,
  worktreePath: string,
  instructionsFile: string,
  model?: string,
): string {
  if (provider === 'codex') {
    return [
      shellQuote(CODEX_CLI()),
      'exec',
      '-C',
      shellQuote(worktreePath),
      '--sandbox',
      shellQuote(CODEX_SANDBOX()),
      modelFlag(model),
      '-',
      '<',
      shellQuote(instructionsFile),
    ].filter(Boolean).join(' ');
  }

  return [
    'cd',
    shellQuote(worktreePath),
    '&&',
    shellQuote(CC_CLI()),
    '--print',
    '--dangerously-skip-permissions',
    '--permission-mode',
    'bypassPermissions',
    '--add-dir',
    shellQuote(worktreePath),
    modelFlag(model),
    '<',
    shellQuote(instructionsFile),
  ].filter(Boolean).join(' ');
}

function buildPrintCommand(
  provider: ImplementationProvider,
  repoPath: string,
  instructionsFile: string,
  model?: string,
): string {
  if (provider === 'codex') {
    return [
      shellQuote(CODEX_CLI()),
      'exec',
      '-C',
      shellQuote(repoPath),
      '--sandbox',
      shellQuote(CODEX_PRINT_SANDBOX()),
      modelFlag(model),
      '-',
      '<',
      shellQuote(instructionsFile),
    ].filter(Boolean).join(' ');
  }

  return [
    'cd',
    shellQuote(repoPath),
    '&&',
    shellQuote(CC_CLI()),
    '--print',
    modelFlag(model),
    '<',
    shellQuote(instructionsFile),
  ].filter(Boolean).join(' ');
}

function isRateLimitOutput(stdout = '', stderr = ''): boolean {
  const combined = `${stdout}\n${stderr}`.toLowerCase();
  return (
    combined.includes('session limit') ||
    combined.includes('rate limit') ||
    combined.includes('rate_limit') ||
    combined.includes('too many requests') ||
    combined.includes('429')
  );
}

function shouldFallbackAfterClaudeLimit(provider: ImplementationProvider, exitCode: number, stdout = '', stderr = ''): boolean {
  return provider === 'claude-code'
    && exitCode !== 0
    && isRateLimitOutput(stdout, stderr);
}

function shouldFallbackToCodex(provider: ImplementationProvider, exitCode: number, stdout = '', stderr = ''): boolean {
  return shouldFallbackAfterClaudeLimit(provider, exitCode, stdout, stderr)
    && RATE_LIMIT_FALLBACK_PROVIDER() === 'codex';
}

function shouldFallbackToLiteLlm(provider: ImplementationProvider, exitCode: number, stdout = '', stderr = ''): boolean {
  return shouldFallbackAfterClaudeLimit(provider, exitCode, stdout, stderr)
    && ['litellm', 'ollama', 'openrouter'].includes(RATE_LIMIT_FALLBACK_PROVIDER());
}

function extractPatch(text: string): string {
  const fenced = text.match(/```(?:diff|patch)?\s*([\s\S]*?)```/i);
  let candidate = (fenced?.[1] ?? text).trim();
  if (candidate.startsWith('--git ')) return `diff ${candidate}`.trim();
  candidate = candidate.replace(/^```(?:diff|patch)?\s*/i, '').trim();
  if (candidate.startsWith('--git ')) return `diff ${candidate}`.trim();
  const diffIndex = candidate.indexOf('diff --git ');
  if (diffIndex >= 0) return candidate.slice(diffIndex).trim();
  try {
    const parsed = JSON.parse(candidate);
    if (typeof parsed.patch === 'string') return parsed.patch.trim();
    if (typeof parsed.diff === 'string') return parsed.diff.trim();
  } catch {
    // Not JSON; handled below.
  }
  return '';
}

function normalizeRepoPath(file: string): string {
  return file.replace(/^\.\//, '').replace(/^[ab]\//, '').trim();
}

function extractStrictOnlyEditFiles(instructions: string): string[] {
  const allowed = new Set<string>();
  const filePattern = '[A-Za-z0-9_./-]+\\.(?:json|ts|tsx|js|jsx|md|yml|yaml)';
  const patterns = [
    new RegExp(`only\\s+(?:edit|modify|change)\\s+(${filePattern})`, 'gi'),
    new RegExp(`(?:edit|modify|change)\\s+only\\s+(${filePattern})`, 'gi'),
  ];
  for (const pattern of patterns) {
    for (const match of instructions.matchAll(pattern)) {
      allowed.add(normalizeRepoPath(match[1]));
    }
  }
  return Array.from(allowed);
}

function validatePatchScope(patch: string, allowedFiles: string[]): void {
  if (allowedFiles.length === 0) return;
  const allowed = new Set(allowedFiles.map(normalizeRepoPath));
  const touched = Array.from(patch.matchAll(/^diff --git a\/(.+?) b\/(.+)$/gm))
    .flatMap((match) => [normalizeRepoPath(match[1]), normalizeRepoPath(match[2])]);
  const uniqueTouched = Array.from(new Set(touched));
  if (uniqueTouched.length === 0) {
    throw new Error(`patch does not contain diff headers for required file(s): ${allowedFiles.join(', ')}`);
  }
  const unexpected = uniqueTouched.filter((file) => !allowed.has(file));
  if (unexpected.length > 0) {
    throw new Error(`patch touches unexpected file(s): ${unexpected.join(', ')}; allowed file(s): ${allowedFiles.join(', ')}`);
  }
  const missing = allowedFiles.filter((file) => !uniqueTouched.includes(normalizeRepoPath(file)));
  if (missing.length > 0) {
    throw new Error(`patch does not touch required file(s): ${missing.join(', ')}`);
  }
}

function extractValidatorFeedback(instructions: string): string {
  const payloadMatch = instructions.match(/Task payload:\s*(\{[\s\S]*?\})\s*Acceptance criteria:/);
  if (!payloadMatch) return '';
  try {
    const payload = JSON.parse(payloadMatch[1]);
    return typeof payload.user_rejection_feedback === 'string'
      ? payload.user_rejection_feedback.trim()
      : '';
  } catch {
    return '';
  }
}

function applyStructuredValidatorFallback(
  worktreePath: string,
  instructions: string,
  strictOnlyEditFiles: string[],
): string | null {
  const feedback = extractValidatorFeedback(instructions);
  const normalizedFeedback = feedback.toLowerCase();
  if (
    strictOnlyEditFiles.length !== 1
    || !normalizedFeedback.includes('compileroptions.types')
    || !normalizedFeedback.includes('jest')
  ) {
    return null;
  }

  const targetFile = normalizeRepoPath(strictOnlyEditFiles[0]);
  if (!targetFile.endsWith('.json')) return null;
  const fullPath = path.resolve(worktreePath, targetFile);
  if (!fullPath.startsWith(path.resolve(worktreePath) + path.sep)) {
    throw new Error(`structured fallback target escapes worktree: ${targetFile}`);
  }

  const current = fs.readFileSync(fullPath, 'utf8');
  const compactTypes = /("types"\s*:\s*\[\s*"node"\s*)\]/;
  if (compactTypes.test(current)) {
    fs.writeFileSync(fullPath, current.replace(compactTypes, '$1, "jest"]'), 'utf8');
    return targetFile;
  }

  const parsed = JSON.parse(current);
  parsed.compilerOptions = parsed.compilerOptions ?? {};
  const types = Array.isArray(parsed.compilerOptions.types)
    ? parsed.compilerOptions.types
    : [];
  if (!types.includes('jest')) types.push('jest');
  parsed.compilerOptions.types = types;
  fs.writeFileSync(fullPath, `${JSON.stringify(parsed, null, 2)}\n`, 'utf8');
  return targetFile;
}

async function readRepoContext(worktreePath: string, instructions: string): Promise<string> {
  const chunks: string[] = [];
  try {
    const { stdout } = await execAsync("find . -maxdepth 4 -type f ! -path './node_modules/*' ! -path './.git/*' ! -path './dist/*' | sort | head -180", { cwd: worktreePath });
    chunks.push(`File tree:\n${stdout.trim()}`);
  } catch (error) {
    chunks.push(`File tree unavailable: ${String(error).slice(0, 200)}`);
  }

  const candidateFiles = Array.from(new Set([
    ...Array.from(instructions.matchAll(/[A-Za-z0-9_./-]+\.(?:json|ts|tsx|js|jsx|md|yml|yaml)/g)).map((match) => match[0]),
    'package.json',
  ])).filter((file) => !file.includes('node_modules') && !file.startsWith('/'));

  for (const file of candidateFiles.slice(0, 12)) {
    const normalized = file.replace(/^\.\//, '');
    const fullPath = path.join(worktreePath, normalized);
    if (!fullPath.startsWith(worktreePath)) continue;
    try {
      const stat = fs.statSync(fullPath);
      if (!stat.isFile() || stat.size > 25_000) continue;
      chunks.push(`\n--- ${normalized} ---\n${fs.readFileSync(fullPath, 'utf8')}`);
    } catch {
      // File may not exist; keep going.
    }
  }

  return chunks.join('\n');
}

async function requestLiteLlmPatch(model: string, prompt: string, timeoutMs: number): Promise<string> {
  const baseUrl = LITELLM_BASE_URL();
  const token = LITELLM_MASTER_KEY();
  if (!baseUrl || !token) {
    throw new Error('LiteLLM fallback is not configured: LITELLM_BASE_URL and LITELLM_MASTER_KEY are required');
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${baseUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        temperature: 0,
        messages: [
          {
            role: 'system',
            content: 'You are a deterministic patch generator. Return ONLY a unified git diff starting with diff --git. Do not use markdown, JSON, explanations, or prose.',
          },
          { role: 'user', content: prompt },
        ],
      }),
      signal: controller.signal,
    });
    const raw = await response.text();
    if (!response.ok) {
      throw new Error(`LiteLLM ${model} HTTP ${response.status}: ${raw.slice(0, 500)}`);
    }
    const json = JSON.parse(raw);
    return String(json.choices?.[0]?.message?.content ?? json.choices?.[0]?.text ?? '').trim();
  } finally {
    clearTimeout(timer);
  }
}

async function applyLiteLlmFallbackPatch(
  worktreePath: string,
  instructions: string,
  claudeLimitOutput: string,
): Promise<{ provider: ImplementationProvider; stdout: string; stderr: string; exitCode: number }> {
  const context = await readRepoContext(worktreePath, instructions);
  const validatorFeedback = extractValidatorFeedback(instructions);
  const strictOnlyEditFiles = extractStrictOnlyEditFiles(instructions);
  const strictDiffHeaders = strictOnlyEditFiles.map((file) => `diff --git a/${file} b/${file}`).join(', ');
  const strictScopeRule = strictOnlyEditFiles.length > 0
    ? `Required patch file scope: modify only ${strictOnlyEditFiles.join(', ')}. The only valid diff header(s): ${strictDiffHeaders}.`
    : '';
  const taskScope = validatorFeedback
    ? [
        'Validator retry feedback is authoritative and overrides the broader original task.',
        'Implement only this validator retry feedback.',
        validatorFeedback,
      ].join('\n')
    : `Original task instructions:\n${instructions}`;
  const prompt = [
    'Claude Code is rate-limited. Implement the requested coding task by returning a patch only.',
    'The patch will be applied with git apply in the repository root.',
    'Do not change unrelated files. If validator feedback names exact files, edit only those files.',
    strictScopeRule,
    '',
    `Task scope:\n${taskScope}`,
    '',
    `Claude rate-limit output:\n${claudeLimitOutput}`,
    '',
    `Repository context:\n${context}`,
  ].join('\n');

  const errors: string[] = [];
  for (const model of LITELLM_FALLBACK_MODELS()) {
    try {
      const raw = await requestLiteLlmPatch(model, prompt, LITELLM_TIMEOUT_MS());
      const patch = extractPatch(raw);
      if (!patch) {
        errors.push(`${model}: model did not return a unified diff; output=${raw.slice(0, 500)}`);
        continue;
      }
      validatePatchScope(patch, strictOnlyEditFiles);
      const patchPath = path.join('/tmp', `litellm-fallback-${Date.now()}-${Math.random().toString(16).slice(2)}.diff`);
      fs.writeFileSync(patchPath, patch, 'utf8');
      try {
        await execAsync(`git apply --recount --whitespace=nowarn ${shellQuote(patchPath)}`, { cwd: worktreePath, timeout: 30_000, maxBuffer: 2 * 1024 * 1024 });
      } finally {
        try { fs.unlinkSync(patchPath); } catch { /* ignore */ }
      }
      return {
        provider: providerForLiteLlmModel(model),
        stdout: [`[RunLayer] Claude Code rate-limited; applied LiteLLM fallback patch via ${model}.`, raw].join('\n\n'),
        stderr: '',
        exitCode: 0,
      };
    } catch (error) {
      errors.push(`${model}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  try {
    const structuredFile = applyStructuredValidatorFallback(worktreePath, instructions, extractStrictOnlyEditFiles(instructions));
    if (structuredFile) {
      return {
        provider: 'litellm',
        stdout: `[RunLayer] Claude Code rate-limited; Ollama/OpenRouter patches were unusable, so structured validator fallback updated ${structuredFile}.`,
        stderr: errors.join('\n'),
        exitCode: 0,
      };
    }
  } catch (error) {
    errors.push(`structured-validator-fallback: ${error instanceof Error ? error.message : String(error)}`);
  }

  return {
    provider: 'litellm',
    stdout: '[RunLayer] Claude Code rate-limited; LiteLLM fallback attempted but did not apply a patch.',
    stderr: errors.join('\n'),
    exitCode: 1,
  };
}

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
          job.implementationProvider,
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
      implementationProvider,
    } = msg;
    const provider = resolveProvider(implementationProvider);
    const resolvedModel = resolveModel(provider, executionMode, model);
    let effectiveProvider = provider;
    let effectiveModel = resolvedModel;

    this.logger.log(
      JSON.stringify({
        event: 'Claude Code Job Executing',
        jobId,
        repoPath,
        implementationProvider: provider,
      }),
    );
    await this.loggingClient.log('info', 'Claude Code Job Executing', {
      jobId,
      repoPath,
      branch,
      implementationProvider: provider,
    });

    try {
      // Update status to executing
      const startedAt = new Date();
      await this.service.updateJobExecution(jobId, {
        status: JobStatus.EXECUTING as JobStatus,
        startedAt,
      });

      if (executionMode === 'print') {
        await this.executePrintJob(jobId, repoPath, instructions, timeoutSeconds, provider, resolvedModel, startedAt);
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

      // Execute implementation provider
      this.logger.debug(
        `Executing ${providerLabel(provider)} in ${worktreePath} with branch ${branch}`,
      );

      let stdout = '';
      let stderr = '';
      let exitCode = 0;

      const instructionsFile = path.join('/tmp', `cc-code-${jobId}.txt`);
      try {
        fs.writeFileSync(instructionsFile, instructions, 'utf-8');
        const { stdout: cmdOutput } = await execAsync(
          buildCodeCommand(provider, worktreePath, instructionsFile, resolvedModel),
          { timeout: timeoutSeconds * 1000, maxBuffer: 10 * 1024 * 1024 },
        );
        stdout = cmdOutput;
      } catch (error: any) {
        exitCode = error.code || 1;
        stdout = error.stdout || '';
        stderr = error.stderr || '';
      } finally {
        try { fs.unlinkSync(instructionsFile); } catch { /* ignore */ }
      }

      if (shouldFallbackToLiteLlm(effectiveProvider, exitCode, stdout, stderr)) {
        const claudeLimitOutput = [stdout, stderr].filter(Boolean).join('\n').trim();
        const fallback = await applyLiteLlmFallbackPatch(worktreePath, instructions, claudeLimitOutput);
        effectiveProvider = fallback.provider;
        stdout = fallback.stdout;
        stderr = fallback.stderr;
        exitCode = fallback.exitCode;

        await this.service.updateJobExecution(jobId, {
          implementationProvider: effectiveProvider,
          stdout,
          stderr,
        });
        await this.loggingClient.log(exitCode === 0 ? 'warn' : 'error', 'Claude Code rate-limited; used LiteLLM fallback', {
          jobId,
          repoPath,
          branch,
          implementationProvider: effectiveProvider,
          exitCode,
        });
      } else if (shouldFallbackToCodex(effectiveProvider, exitCode, stdout, stderr)) {
        const claudeLimitOutput = [stdout, stderr].filter(Boolean).join('\n').trim();
        effectiveProvider = 'codex';
        effectiveModel = resolveModel(effectiveProvider, executionMode, undefined);
        stdout = [
          '[RunLayer] Claude Code returned a rate/session limit; retrying the same job with Codex.',
          claudeLimitOutput,
        ].filter(Boolean).join('\n\n');
        stderr = '';
        exitCode = 0;

        await this.service.updateJobExecution(jobId, {
          implementationProvider: effectiveProvider,
          stdout,
          stderr,
        });
        await this.loggingClient.log('warn', 'Claude Code rate-limited; falling back to Codex', {
          jobId,
          repoPath,
          branch,
          implementationProvider: effectiveProvider,
        });

        try {
          fs.writeFileSync(instructionsFile, instructions, 'utf-8');
          const { stdout: codexOutput } = await execAsync(
            buildCodeCommand(effectiveProvider, worktreePath, instructionsFile, effectiveModel),
            { timeout: timeoutSeconds * 1000, maxBuffer: 10 * 1024 * 1024 },
          );
          stdout = [stdout, codexOutput].filter(Boolean).join('\n\n');
        } catch (error: any) {
          exitCode = error.code || 1;
          stdout = [stdout, error.stdout || ''].filter(Boolean).join('\n\n');
          stderr = error.stderr || error.message || '';
        } finally {
          try { fs.unlinkSync(instructionsFile); } catch { /* ignore */ }
        }
      }

      // Get git diff
      let gitDiff = '';
      try {
        await execAsync('git add -N .', { cwd: worktreePath });
      } catch (e) {
        this.logger.warn(`Failed to mark untracked files for diff: ${e}`);
      }
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
        implementationProvider: effectiveProvider,
        completedAt: new Date(),
      });

      this.logger.log(
        JSON.stringify({
          event: 'Claude Code Job Completed',
          jobId,
          status: finalStatus,
          exitCode,
          durationMs: new Date().getTime() - startedAt.getTime(),
          implementationProvider: effectiveProvider,
        }),
      );
      await this.loggingClient.log('info', 'Claude Code Job Completed', {
        jobId,
        status: finalStatus,
        exitCode,
        implementationProvider: effectiveProvider,
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
          job.implementationProvider,
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
    implementationProvider: ImplementationProvider,
    model: string | undefined,
    startedAt: Date,
  ): Promise<void> {
    const tmpFile = path.join('/tmp', `cc-print-${jobId}.txt`);
    let stdout = '';
    let stderr = '';
    let exitCode = 0;

    try {
      fs.writeFileSync(tmpFile, instructions, 'utf-8');
      const { stdout: out } = await execAsync(
        buildPrintCommand(implementationProvider, repoPath, tmpFile, model),
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

    await this.loggingClient.log('info', `${providerLabel(implementationProvider)} Print Job Completed`, {
      jobId,
      status: finalStatus,
      exitCode,
      durationMs: Date.now() - startedAt.getTime(),
      model,
      implementationProvider,
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
    implementationProvider: ImplementationProvider | undefined,
    delayMs: number,
    maxRetries: number,
  ): void {
    const provider = resolveProvider(implementationProvider);
    const message = {
      jobId,
      repoPath,
      branch,
      instructions,
      timeoutSeconds,
      validationScript,
      executionMode,
      model,
      implementationProvider: provider,
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
          implementationProvider: job.implementationProvider,
        }, null);
      }
    } catch (error) {
      this.logger.error('Claude Code direct queue polling failed', error);
    } finally {
      this.directQueueActive = false;
    }
  }
}
