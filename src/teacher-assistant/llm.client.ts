import { Injectable, Logger, ServiceUnavailableException } from '@nestjs/common';

export interface LlmMeta {
  model: string;
  tier: string;
  promptTokens: number;
  completionTokens: number;
}

export interface CompleteJsonArgs {
  systemPrompt: string;
  userPrompt: string;
  outputSchema: unknown;
  correlationId: string;
  maxTokens?: number;
}

const FENCE = /^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$/;

@Injectable()
export class LlmClient {
  private readonly logger = new Logger(LlmClient.name);

  async completeJson<T>(args: CompleteJsonArgs): Promise<{ data: T; meta: LlmMeta }> {
    const base = (process.env.AI_ORCHESTRATOR_URL || 'http://localhost:3380').replace(/\/$/, '');
    const tier = process.env.DRILL_GENERATION_MODEL_TIER || 'smart';

    const res = await fetch(`${base}/ai/complete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_tier: tier,
        system_prompt: args.systemPrompt,
        user_prompt: args.userPrompt,
        output_schema: args.outputSchema,
        max_tokens: args.maxTokens ?? 8000,
        correlation_id: args.correlationId,
      }),
    });

    if (!res.ok) {
      const body = await res.text();
      throw new ServiceUnavailableException(`ai/complete failed: ${res.status} ${body.slice(0, 200)}`);
    }

    const payload = (await res.json()) as { content: string; model?: string; usage?: Record<string, number> };
    const raw = payload.content ?? '';
    const unfenced = FENCE.exec(raw)?.[1] ?? raw;

    let data: T;
    try {
      data = JSON.parse(unfenced) as T;
    } catch {
      this.logger.warn(`ai/complete returned content that is not valid JSON (${raw.slice(0, 120)})`);
      throw new ServiceUnavailableException('ai/complete content is not valid JSON');
    }

    return {
      data,
      meta: {
        model: payload.model ?? 'unknown',
        tier,
        promptTokens: payload.usage?.prompt_tokens ?? 0,
        completionTokens: payload.usage?.completion_tokens ?? 0,
      },
    };
  }
}
