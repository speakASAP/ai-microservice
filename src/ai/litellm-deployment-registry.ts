import { Logger } from '@nestjs/common';

/**
 * Resolves LiteLLM deployment ids to the real upstream model they route to.
 *
 * WHY THIS EXISTS
 * LiteLLM's chat/completions response echoes the *alias* from `model_list[].model_name`
 * back in its `model` field — a request for `smart` returns `model: "smart"`, never
 * `openrouter/google/gemma-4-31b-it:free` (verified against LiteLLM 1.82.6 on 2026-08-24).
 * There is no proxy setting that changes this. Downstream anti-fabrication guards need
 * the model that ACTUALLY served the call, so the alias is useless to them: a silent
 * fallback from `smart` to `smart-fallback` still reports `"smart"`.
 *
 * The real id is recoverable, just not from the body. Every response carries an
 * `x-litellm-model-id` header holding the deployment hash, and `/model/info` maps that
 * hash to `litellm_params.model`. This registry is that map, cached.
 */

/** Header carrying the deployment hash of whichever deployment served the request. */
export const LITELLM_MODEL_ID_HEADER = 'x-litellm-model-id';

/** Header carrying the model_list group (the alias) the request was routed under. */
export const LITELLM_MODEL_GROUP_HEADER = 'x-litellm-model-group';

/**
 * Header counting fallbacks LiteLLM used. Non-zero means a different model than the one
 * requested served the call — the exact event the alias echo hides.
 */
export const LITELLM_ATTEMPTED_FALLBACKS_HEADER = 'x-litellm-attempted-fallbacks';

/** How long a fetched deployment map stays usable before it is refetched. */
const CACHE_TTL_MS = 5 * 60_000;

interface ModelInfoEntry {
  model_name?: string;
  litellm_params?: { model?: string };
  model_info?: { id?: string };
}

export class LitellmDeploymentRegistry {
  private readonly logger = new Logger(LitellmDeploymentRegistry.name);

  /** deployment id (hash) -> real upstream model string */
  private cache = new Map<string, string>();
  private fetchedAt = 0;
  private inFlight: Promise<void> | null = null;

  constructor(
    private readonly baseUrl: string,
    private readonly masterKey: string,
    private readonly fetchImpl: typeof fetch = fetch,
  ) {}

  /**
   * Maps a deployment id to its real upstream model. Returns undefined when the id is
   * genuinely unknown AND when the lookup itself failed — the caller distinguishes the
   * two through `model_resolved`, and both mean "we do not know which model served this",
   * which is the only thing a grounding guard can safely act on.
   */
  async resolveModel(deploymentId: string): Promise<string | undefined> {
    if (!deploymentId) return undefined;

    const cached = this.cache.get(deploymentId);
    if (cached && Date.now() - this.fetchedAt < CACHE_TTL_MS) return cached;

    await this.refresh();
    return this.cache.get(deploymentId);
  }

  /**
   * Refetches the deployment map. Concurrent callers share one in-flight request so a
   * burst of completions cannot stampede /model/info.
   */
  private async refresh(): Promise<void> {
    if (this.inFlight) return this.inFlight;

    this.inFlight = this.doRefresh().finally(() => {
      this.inFlight = null;
    });
    return this.inFlight;
  }

  private async doRefresh(): Promise<void> {
    const url = `${this.baseUrl}/model/info`;
    try {
      const res = await this.fetchImpl(url, {
        headers: { authorization: `Bearer ${this.masterKey}` },
        signal: AbortSignal.timeout(10_000),
      });

      if (!res.ok) {
        // Logged, not thrown: a failed lookup must not fail the completion that already
        // succeeded. It degrades model_resolved to false, which callers treat as "unknown
        // model" — loud downstream, never silent.
        this.logger.error(
          `LiteLLM /model/info returned ${res.status}; deployment ids cannot be resolved to ` +
            'real model names, so completions will report model_resolved=false',
        );
        return;
      }

      const body = (await res.json()) as { data?: ModelInfoEntry[] };
      const next = new Map<string, string>();
      for (const entry of body.data ?? []) {
        const id = entry.model_info?.id;
        const model = entry.litellm_params?.model;
        if (id && model) next.set(id, model);
      }

      if (next.size === 0) {
        this.logger.error(`LiteLLM /model/info returned no usable deployments from ${url}`);
        return;
      }

      this.cache = next;
      this.fetchedAt = Date.now();
      this.logger.log(`resolved ${next.size} LiteLLM deployments from /model/info`);
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      this.logger.error(`LiteLLM /model/info unreachable at ${url}: ${message}`);
    }
  }
}
