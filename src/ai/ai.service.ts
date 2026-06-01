import { Injectable, Logger } from '@nestjs/common';
import { spawn } from 'child_process';
import { writeFileSync, unlinkSync, existsSync, openSync, closeSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { randomUUID } from 'crypto';
import type { AiCompleteRequestInput } from '../contracts';
import { LoggingClient } from '../claude-code/logging.client';

// model_tier field kept for API compat but ignored — all calls route through CC CLI with sonnet
const DEFAULT_MODEL = 'sonnet';
const ELEVATED_TIERS = new Set(['smart', 'premium']);

const CC_CLI = process.env.CC_CLI_PATH || '/home/ssf/.local/bin/claude';
const CC_TIMEOUT_MS = Number(process.env.CC_CLI_TIMEOUT_MS || 120_000);
// Max concurrent claude processes to prevent OOM under the pod memory limit
const CC_MAX_CONCURRENT = Number(process.env.CC_MAX_CONCURRENT || 2);

interface CcJsonResult {
  result?: string;
  is_error?: boolean;
  api_error_status?: number;
  usage?: { input_tokens?: number; output_tokens?: number };
  modelUsage?: Record<string, { inputTokens?: number; outputTokens?: number }>;
}

function parseCcStdout(stdout: string): CcJsonResult | null {
  const trimmed = stdout.trim();
  if (!trimmed) return null;
  try {
    return JSON.parse(trimmed) as CcJsonResult;
  } catch {
    return null;
  }
}

function isCcEnvelope(cc: CcJsonResult): boolean {
  return typeof cc.result === 'string'
    || cc.is_error === true
    || typeof cc.api_error_status === 'number'
    || cc.usage !== undefined
    || cc.modelUsage !== undefined;
}

function ccApiErrorResult(model: string, cc: CcJsonResult): AiCompleteResult {
  const message = (cc.result ?? '').trim()
    || `Claude API error (HTTP ${cc.api_error_status ?? 'unknown'})`;
  let error_code = 'CLI_FAILED';
  if (cc.api_error_status === 429) {
    error_code = 'RATE_LIMIT';
  } else if (cc.is_error) {
    error_code = 'MODEL_ERROR';
  }
  return {
    text: '',
    model_used: `claude-${model}`,
    inputTokens: 0,
    outputTokens: 0,
    token_usage_estimate: 0,
    error_code,
    error_message: message.slice(0, 500),
  };
}

export type AiCompleteResult = Record<string, unknown> & {
  text: string;
  model_used: string;
  inputTokens: number;
  outputTokens: number;
  token_usage_estimate: number;
  error_code?: string;
  error_message?: string;
};

function cliFailureResult(model: string, detail: string): AiCompleteResult {
  return {
    text: '',
    model_used: `claude-${model}`,
    inputTokens: 0,
    outputTokens: 0,
    token_usage_estimate: 0,
    error_code: 'CLI_FAILED',
    error_message: detail.slice(0, 500),
  };
}

@Injectable()
export class AiService {
  private readonly logger = new Logger(AiService.name);
  private activeProcesses = 0;

  constructor(private readonly loggingClient: LoggingClient) {}

  async complete(dto: AiCompleteRequestInput): Promise<AiCompleteResult> {
    const model = DEFAULT_MODEL;

    if (dto.model_tier && ELEVATED_TIERS.has(dto.model_tier)) {
      this.logger.warn(`model_tier '${dto.model_tier}' requested — routing to sonnet via CC CLI (elevated tier ignored)`);
    }

    // Build full prompt — merge system_prompt into user message
    let fullPrompt = '';
    if (dto.system_prompt) {
      fullPrompt += dto.system_prompt.trim() + '\n\n';
    }
    if (dto.output_schema) {
      fullPrompt += 'Respond with valid JSON only. No markdown fences.\n\n';
    }
    fullPrompt += dto.user_prompt;

    if (this.activeProcesses >= CC_MAX_CONCURRENT) {
      this.logger.warn(`claude CLI concurrency limit reached (${CC_MAX_CONCURRENT} active); rejecting request`);
      this.emitTelemetry(dto.correlation_id, `claude-${model}`, 0, 0);
      return cliFailureResult(model, `claude CLI concurrency limit reached (${CC_MAX_CONCURRENT} active)`);
    }

    const tmpFile = join(tmpdir(), `ai-complete-${randomUUID()}.txt`);
    writeFileSync(tmpFile, fullPrompt, 'utf-8');

    // Open the tmp file as a readable fd to pass as stdin via spawn (avoids shell injection)
    let stdinFd: number | undefined;
    try {
      stdinFd = openSync(tmpFile, 'r');
    } catch {
      stdinFd = undefined;
    }

    this.activeProcesses++;
    let stdout = '';
    try {
      stdout = await this.spawnCcCli(stdinFd);
    } catch (err: unknown) {
      const detail = err instanceof Error ? err.message : String(err);
      if (detail.startsWith('AI_HTTP_TIMEOUT')) {
        throw new Error(detail);
      }
      this.logger.error(`claude CLI failed: ${detail.slice(0, 300)}`);
      this.emitTelemetry(dto.correlation_id, `claude-${model}`, 0, 0);
      return cliFailureResult(model, `claude CLI failed: ${detail}`);
    } finally {
      this.activeProcesses--;
      if (stdinFd !== undefined) { try { closeSync(stdinFd); } catch { /* ignore */ } }
      try { if (existsSync(tmpFile)) unlinkSync(tmpFile); } catch { /* ignore */ }
    }

    // Parse CC JSON envelope to extract text and token counts
    let rawText = '';
    let inputTokens = 0;
    let outputTokens = 0;
    try {
      const ccResult = parseCcStdout(stdout) as CcJsonResult;
      if (!ccResult) {
        throw new Error('not json');
      }
      if (ccResult.is_error || ccResult.api_error_status) {
        this.emitTelemetry(dto.correlation_id, `claude-${model}`, 0, 0);
        return ccApiErrorResult(model, ccResult);
      }
      rawText = isCcEnvelope(ccResult) ? (ccResult.result ?? '') : stdout.trim();
      if (ccResult.usage) {
        inputTokens = ccResult.usage.input_tokens ?? 0;
        outputTokens = ccResult.usage.output_tokens ?? 0;
      } else if (ccResult.modelUsage) {
        const first = Object.values(ccResult.modelUsage)[0];
        if (first) {
          inputTokens = first.inputTokens ?? 0;
          outputTokens = first.outputTokens ?? 0;
        }
      }
    } catch {
      // CC CLI returned plain text (older version fallback)
      rawText = stdout.trim();
    }

    let parsedData: Record<string, unknown> = {};
    const trimmed = rawText;
    if (trimmed.startsWith('{') || trimmed.startsWith('[') || dto.output_schema) {
      try {
        const cleaned = trimmed.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim();
        const parsed = JSON.parse(cleaned) as unknown;
        if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
          parsedData = parsed as Record<string, unknown>;
        } else if (Array.isArray(parsed)) {
          parsedData = { data: parsed };
        }
      } catch {
        // Not JSON — callers read .text
      }
    }

    this.emitTelemetry(dto.correlation_id, `claude-${model}`, inputTokens, outputTokens);
    return {
      ...parsedData,
      text: rawText,
      model_used: `claude-${model}`,
      inputTokens,
      outputTokens,
      token_usage_estimate: inputTokens + outputTokens,
    };
  }

  private spawnCcCli(stdinFd: number | undefined): Promise<string> {
    return new Promise<string>((resolve, reject) => {
      const child = spawn(
        CC_CLI,
        ['--print', '--output-format', 'json', '--model', DEFAULT_MODEL],
        {
          stdio: [stdinFd !== undefined ? stdinFd : 'pipe', 'pipe', 'pipe'],
          env: { ...process.env, CLAUDE_CONFIG_DIR: process.env.CLAUDE_CONFIG_DIR || '/home/ssf/.claude' },
        },
      );

      const stdoutChunks: Buffer[] = [];
      const stderrChunks: Buffer[] = [];
      child.stdout?.on('data', (chunk: Buffer) => stdoutChunks.push(chunk));
      child.stderr?.on('data', (chunk: Buffer) => stderrChunks.push(chunk));

      const timer = setTimeout(() => {
        child.kill('SIGTERM');
        const killTimer = setTimeout(() => { try { child.kill('SIGKILL'); } catch { /* already dead */ } }, 5000);
        child.once('close', () => clearTimeout(killTimer));  // clear SIGKILL if process exits cleanly after SIGTERM
        reject(new Error(`AI_HTTP_TIMEOUT: claude CLI did not respond within ${CC_TIMEOUT_MS}ms`));
      }, CC_TIMEOUT_MS);

      child.on('close', (code, signal) => {
        clearTimeout(timer);
        const out = Buffer.concat(stdoutChunks).toString('utf-8');
        const err = Buffer.concat(stderrChunks).toString('utf-8');
        if (signal === 'SIGKILL' || signal === 'SIGTERM') {
          reject(new Error(`claude CLI killed by signal ${signal}: ${err.slice(0, 200)}`));
        } else if (code !== 0) {
          // Non-zero exit — check if stdout has a valid CC error envelope before rejecting
          const ccFromStdout = parseCcStdout(out);
          if (ccFromStdout && (ccFromStdout.is_error || ccFromStdout.api_error_status)) {
            resolve(out);
          } else {
            reject(new Error(err.trim() || out.trim() || `claude CLI exited with code ${code}`));
          }
        } else {
          resolve(out);
        }
      });

      child.on('error', (err) => {
        clearTimeout(timer);
        reject(err);
      });
    });
  }

  private emitTelemetry(
    correlationId: string | undefined,
    modelUsed: string,
    inputTokens: number,
    outputTokens: number,
  ): void {
    this.loggingClient
      .log('info', 'ai_complete', {
        correlation_id: correlationId,
        model_used: modelUsed,
        inputTokens,
        outputTokens,
        token_usage_estimate: inputTokens + outputTokens,
        compression: { rtk: true, caveman: 'lite' },
      })
      .catch(() => { /* logging must never crash the service */ });
  }
}
