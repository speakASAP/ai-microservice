import { Injectable, Logger } from '@nestjs/common';
import { exec } from 'child_process';
import { promisify } from 'util';
import { writeFileSync, unlinkSync, existsSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { randomUUID } from 'crypto';
import type { AiCompleteRequestInput } from '../contracts';

const execAsync = promisify(exec);

// model_tier field kept for API compat but ignored — all calls route through CC CLI with sonnet
const DEFAULT_MODEL = 'sonnet';
const ELEVATED_TIERS = new Set(['smart', 'premium']);

const CC_CLI = process.env.CC_CLI_PATH || '/home/ssf/.local/bin/claude';
const CC_TIMEOUT_MS = Number(process.env.CC_CLI_TIMEOUT_MS || 120_000);

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

    const tmpFile = join(tmpdir(), `ai-complete-${randomUUID()}.txt`);
    writeFileSync(tmpFile, fullPrompt, 'utf-8');

    let stdout = '';
    try {
      const result = await execAsync(
        `${CC_CLI} --print --output-format json --model ${model} < ${tmpFile}`,
        {
          timeout: CC_TIMEOUT_MS,
          env: { ...process.env, CLAUDE_CONFIG_DIR: process.env.CLAUDE_CONFIG_DIR || '/home/ssf/.claude' },
        },
      );
      stdout = result.stdout;
    } catch (err: unknown) {
      const e = err as { stdout?: string; stderr?: string; message?: string };
      const ccFromStdout = typeof e.stdout === 'string' ? parseCcStdout(e.stdout) : null;
      if (ccFromStdout && (ccFromStdout.is_error || ccFromStdout.api_error_status)) {
        this.logger.error(
          `claude CLI API error: status=${ccFromStdout.api_error_status ?? 'n/a'} ${(ccFromStdout.result ?? '').slice(0, 200)}`,
        );
        return ccApiErrorResult(model, ccFromStdout);
      }
      const detail = (e.stderr || e.message || 'unknown CLI error').trim();
      this.logger.error(`claude CLI failed: ${detail.slice(0, 300)}`);
      return cliFailureResult(model, `claude CLI failed: ${detail}`);
    } finally {
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

    return {
      ...parsedData,
      text: rawText,
      model_used: `claude-${model}`,
      inputTokens,
      outputTokens,
      token_usage_estimate: inputTokens + outputTokens,
    };
  }
}
