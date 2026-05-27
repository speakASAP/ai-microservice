import { Injectable, InternalServerErrorException } from '@nestjs/common';
import type { CompleteRequestDto } from './dto/complete-request.dto';

const CLAUDE_MODEL = 'claude-sonnet-4-6-20251001';
const ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages';
const ANTHROPIC_API_VERSION = '2023-06-01';

interface AnthropicResponse {
  content: Array<{ type: string; text: string }>;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
  };
  model?: string;
}

/**
 * The response from /ai/complete is a flat object that merges:
 *   - metadata fields (model_used, inputTokens, outputTokens, token_usage_estimate, text)
 *   - the parsed JSON payload fields spread at the top level so callers can access
 *     e.g. response.output_ref, response.passed, response.new_tasks directly.
 */
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
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      throw new InternalServerErrorException('ANTHROPIC_API_KEY is not configured');
    }

    // Build messages — system_prompt goes in the Anthropic `system` field
    const messages: Array<{ role: 'user' | 'assistant'; content: string }> = [
      { role: 'user', content: dto.user_prompt },
    ];

    const requestBody: Record<string, unknown> = {
      model: CLAUDE_MODEL,
      max_tokens: dto.max_tokens ?? 1024,
      messages,
    };

    if (dto.system_prompt) {
      requestBody['system'] = dto.system_prompt;
    }

    // Request JSON output when output_schema is provided
    if (dto.output_schema) {
      const currentSystem = (requestBody['system'] as string | undefined) ?? '';
      requestBody['system'] = `${currentSystem}\nRespond with valid JSON only. No markdown fences.`.trim();
    }

    const res = await fetch(ANTHROPIC_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': ANTHROPIC_API_VERSION,
      },
      body: JSON.stringify(requestBody),
      signal: AbortSignal.timeout(120_000),
    });

    if (!res.ok) {
      throw new InternalServerErrorException(`Anthropic API error ${res.status}`);
    }

    const body = (await res.json()) as AnthropicResponse;

    const rawText = body.content?.find((c) => c.type === 'text')?.text ?? '';
    const inputTokens = body.usage?.input_tokens ?? 0;
    const outputTokens = body.usage?.output_tokens ?? 0;
    const token_usage_estimate = inputTokens + outputTokens;

    // Attempt JSON parse and spread fields at top level (same contract as before)
    let parsedData: Record<string, unknown> = {};
    const trimmed = rawText.trim();
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
      model_used: 'claude-sonnet-4-6',
      inputTokens,
      outputTokens,
      token_usage_estimate,
    };
  }
}
