import { Injectable, Logger, ServiceUnavailableException } from '@nestjs/common';
import { JwtUtil } from '../service-identity/jwt.util';
import { AiCompleteResponse } from '../contracts/ai-complete.contract';

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

/** `/ai/complete` runs behind ServiceAuthGuard (registered as APP_GUARD in
 *  ServiceIdentityModule) and `AiController` carries no `@Public()`. The guard
 *  verifies an HS256 token against `JWT_SECRET` and requires `iss` to be
 *  exactly `ai-microservice` (see JwtUtil.verify), so this service mints its
 *  own short-lived token with the same in-repo utility. */
const SELF_SERVICE_ID = 'ai-microservice';
/** Short enough that a leaked token is near-worthless, long enough to outlive
 *  a slow CC-CLI completion that started just before the token was minted. */
const SERVICE_TOKEN_TTL_SECONDS = 900;

/** Generous by design: a 50-item generate on the claude-CLI path is minutes,
 *  not seconds. With no bound at all the request rides undici's ~300s default
 *  and the caller (and the eval harness) simply appears hung. */
const DEFAULT_TIMEOUT_MS = 300_000;

/** Keys that belong to the `/ai/complete` envelope itself. Everything else at
 *  the top level came from `AiService`'s `{ ...parsedData }` spread of the
 *  model's own JSON, which is the fallback used when `text` is empty. */
const ENVELOPE_KEYS = new Set([
  'schemaVersion',
  'text',
  'model_used',
  'inputTokens',
  'outputTokens',
  'token_usage_estimate',
  'error_code',
  'error_message',
  'agent_id',
  'agent_slug',
  'agent_name',
  'agent_service_scope',
]);

@Injectable()
export class LlmClient {
  private readonly logger = new Logger(LlmClient.name);

  async completeJson<T>(args: CompleteJsonArgs): Promise<{ data: T; meta: LlmMeta }> {
    const base = (process.env.AI_ORCHESTRATOR_URL || 'http://localhost:3380').replace(/\/$/, '');
    const tier = process.env.DRILL_GENERATION_MODEL_TIER || 'smart';
    const timeoutMs = this.resolveTimeoutMs();

    const res = await fetch(`${base}/ai/complete`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.mintServiceToken()}`,
      },
      signal: AbortSignal.timeout(timeoutMs),
      body: JSON.stringify({
        model_tier: tier,
        system_prompt: args.systemPrompt,
        // `output_schema` is only a boolean flag upstream: AiService uses its
        // presence to prepend a "JSON only" instruction and to set
        // `response_format: { type: 'json_object' }`. The schema object itself
        // never reaches the provider, so serialize it into the prompt here —
        // otherwise the model never learns the field names it must produce.
        user_prompt: this.withSchema(args.userPrompt, args.outputSchema),
        // Still sent: its presence is what triggers JSON mode upstream.
        output_schema: args.outputSchema,
        max_tokens: args.maxTokens ?? 8000,
        correlation_id: args.correlationId,
      }),
    });

    if (!res.ok) {
      // The upstream body may echo prompt fragments or provider detail; log it,
      // never hand it back to the caller.
      const body = await res.text().catch(() => '');
      this.logger.error(`ai/complete returned ${res.status}: ${body.slice(0, 500)}`);
      throw new ServiceUnavailableException(`ai/complete failed with status ${res.status}`);
    }

    const payload = (await res.json()) as AiCompleteResponse;

    // A provider failure comes back as HTTP 200 with an empty `text` and an
    // `error_code` (RATE_LIMIT / AI_AUTH_ERROR / CLI_FAILED / ...). Treating
    // that as success surfaces it as "not valid JSON", which sends whoever is
    // debugging to entirely the wrong place.
    if (payload.error_code) {
      this.logger.error(
        `ai/complete reported ${payload.error_code}: ${(payload.error_message ?? '').slice(0, 500)}`,
      );
      throw new ServiceUnavailableException(`ai/complete failed: ${payload.error_code}`);
    }

    const meta: LlmMeta = {
      model: payload.model_used ?? 'unknown',
      tier,
      promptTokens: payload.inputTokens ?? 0,
      completionTokens: payload.outputTokens ?? 0,
    };

    const raw = payload.text ?? '';
    if (raw.trim() === '') {
      // `AiService` spreads the model's parsed JSON across the top level of the
      // response (`{ ...parsedData, text, ... }`) on both the LiteLLM and the
      // CC-CLI path. When `text` is empty but those keys are present the answer
      // is still recoverable, so prefer that over failing the whole request.
      const spread = this.extractSpreadPayload(payload);
      if (spread) return { data: spread as T, meta };
    }

    const unfenced = FENCE.exec(raw)?.[1] ?? raw;

    let data: T;
    try {
      data = JSON.parse(unfenced) as T;
    } catch {
      this.logger.warn(`ai/complete returned text that is not valid JSON (${raw.slice(0, 120)})`);
      throw new ServiceUnavailableException('ai/complete text is not valid JSON');
    }

    return { data, meta };
  }

  /** Appends the JSON Schema to the user prompt so the model actually sees the
   *  field names it must produce. */
  private withSchema(userPrompt: string, outputSchema: unknown): string {
    if (outputSchema === undefined || outputSchema === null) return userPrompt;
    return `${userPrompt}\n\nReturn JSON matching exactly this schema:\n${JSON.stringify(outputSchema)}`;
  }

  /** Mints a self-issued service token. The secret is read here and never
   *  logged, stored on the instance, or included in any thrown message. */
  private mintServiceToken(): string {
    const secret = process.env.JWT_SECRET;
    if (!secret) {
      throw new ServiceUnavailableException('ai/complete auth is not configured');
    }
    return JwtUtil.sign(SELF_SERVICE_ID, secret, SERVICE_TOKEN_TTL_SECONDS);
  }

  private resolveTimeoutMs(): number {
    const raw = Number(process.env.DRILL_LLM_TIMEOUT_MS);
    return Number.isFinite(raw) && raw > 0 ? raw : DEFAULT_TIMEOUT_MS;
  }

  private extractSpreadPayload(payload: AiCompleteResponse): Record<string, unknown> | null {
    const extra: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(payload as Record<string, unknown>)) {
      if (!ENVELOPE_KEYS.has(key)) extra[key] = value;
    }
    return Object.keys(extra).length > 0 ? extra : null;
  }
}
