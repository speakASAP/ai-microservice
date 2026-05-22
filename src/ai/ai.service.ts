import { Injectable, InternalServerErrorException } from '@nestjs/common';
import type { CompleteRequestDto } from './dto/complete-request.dto';

interface LiteLLMResponse {
  choices: Array<{ message: { content: string } }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
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
  /** Raw text content from LLM (always present) */
  text: string;
  /** Model tier used for this request */
  model_used: string;
  /** Real token count from LiteLLM usage.prompt_tokens */
  inputTokens: number;
  /** Real token count from LiteLLM usage.completion_tokens */
  outputTokens: number;
  /** Sum of inputTokens + outputTokens — used by business-orchestrator for budget tracking */
  token_usage_estimate: number;
};

@Injectable()
export class AiService {
  async complete(dto: CompleteRequestDto): Promise<AiCompleteResult> {
    const litellmBaseUrl = process.env.LITELLM_BASE_URL;
    const litellmApiKey = process.env.LITELLM_MASTER_KEY ?? '';

    if (!litellmBaseUrl) {
      throw new InternalServerErrorException('LITELLM_BASE_URL is not configured');
    }

    // Build messages array — free/cheap tiers reject system role; merge into user_prompt when system_prompt provided
    const messages: Array<{ role: string; content: string }> = [];
    if (dto.system_prompt) {
      messages.push({ role: 'user', content: `${dto.system_prompt}\n\n${dto.user_prompt}` });
    } else {
      messages.push({ role: 'user', content: dto.user_prompt });
    }

    const requestBody: Record<string, unknown> = {
      model: dto.model_tier,
      messages,
      temperature: 0.2,
    };

    if (dto.max_tokens) {
      requestBody['max_tokens'] = dto.max_tokens;
    }

    // Ask LiteLLM to return JSON when an output_schema is provided
    if (dto.output_schema) {
      requestBody['response_format'] = { type: 'json_object' };
    }

    const res = await fetch(`${litellmBaseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${litellmApiKey}`,
      },
      body: JSON.stringify(requestBody),
      signal: AbortSignal.timeout(120_000),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new InternalServerErrorException(`LiteLLM error ${res.status}: ${errText}`);
    }

    const body = (await res.json()) as LiteLLMResponse;

    const rawText = body.choices?.[0]?.message?.content ?? '';
    const inputTokens = body.usage?.prompt_tokens ?? 0;
    const outputTokens = body.usage?.completion_tokens ?? 0;
    const token_usage_estimate = inputTokens + outputTokens;
    const model_used = dto.model_tier;

    // Attempt JSON parse to extract structured data and spread at top level
    let parsedData: Record<string, unknown> = {};
    const trimmed = rawText.trim();
    if (trimmed.startsWith('{') || trimmed.startsWith('[') || dto.output_schema) {
      try {
        // Strip markdown code fences if present
        const cleaned = trimmed.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim();
        const parsed = JSON.parse(cleaned) as unknown;
        if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
          parsedData = parsed as Record<string, unknown>;
        } else if (Array.isArray(parsed)) {
          // Array response — store as top-level key for coordinators that expect raw arrays
          parsedData = { data: parsed };
        }
      } catch {
        // Not JSON or parse failed — leave parsedData empty; callers can read .text
      }
    }

    return {
      // Spread parsed fields first so metadata fields below always win on conflict
      ...parsedData,
      text: rawText,
      model_used,
      inputTokens,
      outputTokens,
      token_usage_estimate,
    };
  }
}
