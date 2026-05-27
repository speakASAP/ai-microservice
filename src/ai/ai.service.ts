import { Injectable, InternalServerErrorException } from '@nestjs/common';
import { exec } from 'child_process';
import { promisify } from 'util';
import { writeFileSync, unlinkSync, existsSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { randomUUID } from 'crypto';
import type { CompleteRequestDto } from './dto/complete-request.dto';

const execAsync = promisify(exec);

// model_tier field kept for API compat but ignored — all calls use claude-sonnet-4-6
const DEFAULT_MODEL = 'sonnet';

const CC_CLI = process.env.CC_CLI_PATH || '/home/ssf/.local/bin/claude';
const CC_TIMEOUT_MS = Number(process.env.CC_CLI_TIMEOUT_MS || 120_000);

export type AiCompleteResult = Record<string, unknown> & {
  text: string;
  model_used: string;
  inputTokens: number;
  outputTokens: number;
  token_usage_estimate: number;
};

@Injectable()
export class AiService {
  async complete(dto: CompleteRequestDto): Promise<AiCompleteResult> {
    const model = DEFAULT_MODEL;

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
        `${CC_CLI} --print --model ${model} < ${tmpFile}`,
        {
          timeout: CC_TIMEOUT_MS,
          env: { ...process.env, CLAUDE_CONFIG_DIR: process.env.CLAUDE_CONFIG_DIR || '/home/ssf/.claude' },
        },
      );
      stdout = result.stdout;
    } catch (err: unknown) {
      const e = err as { stdout?: string; stderr?: string; message?: string };
      throw new InternalServerErrorException(
        `claude CLI failed: ${(e.stderr || e.message || '').slice(0, 300)}`,
      );
    } finally {
      try { if (existsSync(tmpFile)) unlinkSync(tmpFile); } catch { /* ignore */ }
    }

    const rawText = stdout.trim();

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
      inputTokens: 0,
      outputTokens: 0,
      token_usage_estimate: 0,
    };
  }
}
