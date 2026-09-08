import { Injectable, Logger, ServiceUnavailableException } from '@nestjs/common';
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

/** `/ai/complete` runs behind ServiceAuthGuard. Auth-minted RS256 only —
 *  deliver via Vault → ExternalSecret as AI_SERVICE_TOKEN
 *  S2S: auth-microservice/docs/SERVICE_IDENTITY_CONSUMER_STANDARD.md — no in-process self-mint. */

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

/**
 * Upstream statuses worth a second attempt.
 *
 * 503 is what a LiteLLM timeout becomes by the time it reaches here; 500 and 502 are the
 * same class of "the far side broke, it may not next time". 429 is included because the
 * rate limit that caused it is usually another caller's burst, not this request being
 * inherently too large.
 *
 * 4xx is deliberately absent apart from 429: a malformed request or a rejected token
 * fails identically on the second attempt, so retrying only doubles the teacher's wait
 * and pays for a model call that teaches nothing.
 */
const RETRYABLE_STATUSES = new Set([429, 500, 502, 503, 504]);

/**
 * `error_code` values that mean "try again", as opposed to "this will never work".
 * AI_AUTH_ERROR and a schema the model cannot satisfy are permanent; a rate limit or a
 * CLI that fell over are not.
 */
const RETRYABLE_ERROR_CODES = new Set(['RATE_LIMIT', 'CLI_FAILED', 'AI_HTTP_TIMEOUT', 'TIMEOUT']);

/** Thrown for an upstream failure, carrying whether a second attempt is worth making. */
class UpstreamFailure extends ServiceUnavailableException {
  constructor(
    message: string,
    readonly retryable: boolean,
  ) {
    super(message);
  }
}

/**
 * A client-side abort (`AbortSignal.timeout`) surfaces as a TimeoutError/AbortError from
 * fetch rather than as a response, so it never reaches the status check above. It is the
 * most literal form of "the upstream was too slow", so it retries.
 */
function isTransientUpstreamFailure(err: unknown): boolean {
  if (err instanceof UpstreamFailure) {
    return err.retryable;
  }
  const name = (err as { name?: string } | null)?.name ?? '';
  const message = (err as Error | null)?.message ?? '';
  return (
    name === 'TimeoutError' ||
    name === 'AbortError' ||
    /aborted|timeout|ECONNRESET|ECONNREFUSED|EAI_AGAIN|socket hang up/i.test(message)
  );
}

@Injectable()
export class LlmClient {
  private readonly logger = new Logger(LlmClient.name);

  /**
   * ONE retry for transient upstream failures, then give up.
   *
   * A LiteLLM timeout reached the teacher as a red banner mid-generation, after they had
   * already waited: `AI_HTTP_TIMEOUT` -> `ai/complete returned 500` -> 503 (2026-08-09,
   * once in 24h, and a manual retry succeeded). Absorbing that beats surfacing it.
   *
   * Deliberately ONE, and deliberately only for transient causes. Each retry costs a
   * full model call and doubles the wait, so a deterministic failure — 400, 401, a
   * schema the model cannot satisfy — must fail on the first attempt rather than
   * charging twice to learn the same thing.
   */
  async completeJson<T>(args: CompleteJsonArgs): Promise<{ data: T; meta: LlmMeta }> {
    try {
      return await this.attemptCompleteJson<T>(args);
    } catch (err) {
      if (!isTransientUpstreamFailure(err)) {
        throw err;
      }
      this.logger.warn(
        `ai/complete failed transiently (${(err as Error).message.slice(0, 120)}); retrying once ` +
          `correlationId=${args.correlationId}`,
      );
      return this.attemptCompleteJson<T>(args);
    }
  }

  private async attemptCompleteJson<T>(
    args: CompleteJsonArgs,
  ): Promise<{ data: T; meta: LlmMeta }> {
    const base = (process.env.AI_ORCHESTRATOR_URL || 'http://localhost:3380').replace(/\/$/, '');
    const tier = process.env.DRILL_GENERATION_MODEL_TIER || 'smart';
    const timeoutMs = this.resolveTimeoutMs();

    const res = await fetch(`${base}/ai/complete`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.resolveServiceToken()}`,
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
      throw new UpstreamFailure(
        `ai/complete failed with status ${res.status}`,
        RETRYABLE_STATUSES.has(res.status),
      );
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
      throw new UpstreamFailure(
        `ai/complete failed: ${payload.error_code}`,
        RETRYABLE_ERROR_CODES.has(payload.error_code),
      );
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

  /** Auth-provisioned bearer for this (caller → ai-microservice) pair. Never
   *  logged or included in thrown messages. */
  private resolveServiceToken(): string {
    const token = process.env.AI_SERVICE_TOKEN?.trim();
    if (!token) {
      throw new ServiceUnavailableException('AI_SERVICE_TOKEN is not configured');
    }
    return token;
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
