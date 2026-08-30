import { AiService } from './ai.service';
import { LoggingClient } from '../claude-code/logging.client';
import { AiCompleteRequestSchema } from '../contracts';

/** Deployment ids as LiteLLM 1.82.6 reports them in x-litellm-model-id. */
const FREE_ID = 'free-deployment-hash';
const SMART_ID = 'smart-deployment-hash';
const SMART_FALLBACK_ID = 'smart-fallback-deployment-hash';

/**
 * Mocks the two LiteLLM calls the service makes: /chat/completions (whose body echoes the
 * ALIAS, never the served model) and /model/info (which maps deployment ids to real models).
 */
function litellmFetchMock(opts: { deploymentId: string; attemptedFallbacks?: string }) {
  return jest.fn().mockImplementation(async (url: string) => {
    if (String(url).includes('/model/info')) {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          data: [
            { model_name: 'free', litellm_params: { model: 'ollama/qwen2.5-coder:0.5b' }, model_info: { id: FREE_ID } },
            { model_name: 'smart', litellm_params: { model: 'openrouter/google/gemma-4-31b-it:free' }, model_info: { id: SMART_ID } },
            { model_name: 'smart-fallback', litellm_params: { model: 'openrouter/nvidia/nemotron-3-super-120b-a12b:free' }, model_info: { id: SMART_FALLBACK_ID } },
          ],
        }),
      } as unknown as Response;
    }
    return {
      ok: true,
      status: 200,
      headers: new Headers({
        'x-litellm-model-id': opts.deploymentId,
        'x-litellm-attempted-fallbacks': opts.attemptedFallbacks ?? '0',
      }),
      json: async () => ({
        choices: [{ message: { content: '{"output_ref":{"summary":"ok"}}' } }],
        usage: { prompt_tokens: 10, completion_tokens: 5 },
        model: 'smart',
      }),
    } as unknown as Response;
  });
}


describe('AiService - LiteLLM routing', () => {
  let service: AiService;
  let loggingClient: jest.Mocked<LoggingClient>;
  let agents: { findOne: jest.Mock };
  const originalFetch = global.fetch;
  const originalLitellmUrl = process.env.LITELLM_BASE_URL;
  const originalLitellmKey = process.env.LITELLM_MASTER_KEY;

  beforeEach(() => {
    process.env.LITELLM_BASE_URL = 'http://litellm.test:4000';
    process.env.LITELLM_MASTER_KEY = 'test-key';
    loggingClient = { log: jest.fn().mockResolvedValue(undefined) } as any;
    agents = { findOne: jest.fn() };
    service = new AiService(loggingClient, agents as any);
  });

  afterEach(() => {
    global.fetch = originalFetch;
    process.env.LITELLM_BASE_URL = originalLitellmUrl;
    process.env.LITELLM_MASTER_KEY = originalLitellmKey;
  });

  it('routes free tier to LiteLLM model name free', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{ message: { content: 'ok' } }],
        usage: { prompt_tokens: 5, completion_tokens: 2 },
        model: 'free',
      }),
    } as Response);

    const result = await service.complete({ model_tier: 'free', user_prompt: 'say ok' });

    expect(result.text).toBe('ok');
    expect(result.model_used).toBe('free');
    expect(global.fetch).toHaveBeenCalledWith(
      'http://litellm.test:4000/v1/chat/completions',
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('"model":"free"'),
      }),
    );
  });

  /**
   * A LiteLLM stall used to escape as a bare `throw`, which NestJS turned into a 500.
   * Callers classify by `error_code`, and a 500 carries none, so the teacher got a
   * generic "responded 503" banner for a condition their caller is willing to retry
   * (2026-08-14). The timeout must come back like every other failure on this path:
   * a resolved result carrying AI_HTTP_TIMEOUT.
   */
  it('returns AI_HTTP_TIMEOUT as a result rather than throwing when LiteLLM stalls', async () => {
    global.fetch = jest.fn().mockRejectedValue(
      Object.assign(new Error('The operation was aborted due to timeout'), { name: 'TimeoutError' }),
    );

    const result = await service.complete({ model_tier: 'smart', user_prompt: 'generate a drill' });

    expect(result.error_code).toBe('AI_HTTP_TIMEOUT');
    expect(result.text).toBe('');
    expect(result.model_used).toBe('smart');
  });

  it('keeps business_id optional in the request contract', () => {
    expect(() => AiCompleteRequestSchema.parse({ model_tier: 'free', user_prompt: 'say ok' })).not.toThrow();
  });

  it('keeps agent registry routing optional in the request contract', () => {
    expect(() => AiCompleteRequestSchema.parse({
      model_tier: 'free',
      user_prompt: 'summarize this',
      agent_slug: 'support-summary',
      agent_service_scope: 'shop-assistant',
    })).not.toThrow();
  });

  it('emits business_id in gateway telemetry when supplied', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{ message: { content: 'ok' } }],
        usage: { prompt_tokens: 7, completion_tokens: 3 },
        model: 'free',
      }),
    } as Response);

    await service.complete({
      model_tier: 'free',
      user_prompt: 'say ok',
      business_id: 'biz_test_001',
      correlation_id: 'corr-test-001',
    });

    expect(loggingClient.log).toHaveBeenCalledWith(
      'info',
      'ai_complete',
      expect.objectContaining({
        business_id: 'biz_test_001',
        correlation_id: 'corr-test-001',
        inputTokens: 7,
        outputTokens: 3,
        token_usage_estimate: 10,
      }),
    );
  });

  it('uses an active registry agent when agent_slug is supplied', async () => {
    agents.findOne.mockResolvedValue({
      id: 'agent-001',
      name: 'Support Summary',
      slug: 'support-summary',
      status: 'active',
      serviceScope: 'shop-assistant',
      routePath: '/ai/complete',
      modelTier: 'cheap',
      systemPrompt: 'You summarize support cases.',
      userPromptTemplate: 'Case: {{user_prompt}}',
      outputSchema: { type: 'object' },
      maxTokens: 321,
    });
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{ message: { content: '{"summary":"ok"}' } }],
        usage: { prompt_tokens: 11, completion_tokens: 4 },
        model: 'cheap',
      }),
    } as Response);

    const result = await service.complete({
      model_tier: 'free',
      user_prompt: 'customer needs help',
      agent_slug: 'support-summary',
      agent_service_scope: 'shop-assistant',
    });

    expect(result).toEqual(expect.objectContaining({
      text: '{"summary":"ok"}',
      summary: 'ok',
      model_used: 'cheap',
      agent_id: 'agent-001',
      agent_slug: 'support-summary',
      agent_name: 'Support Summary',
      agent_service_scope: 'shop-assistant',
    }));

    const requestBody = JSON.parse((global.fetch as jest.Mock).mock.calls[0][1].body);
    expect(requestBody).toEqual(expect.objectContaining({
      model: 'cheap',
      max_tokens: 321,
      response_format: { type: 'json_object' },
    }));
    expect(requestBody.messages[0].content).toContain('You summarize support cases.');
    expect(requestBody.messages[0].content).toContain('Case: customer needs help');
    expect(loggingClient.log).toHaveBeenCalledWith(
      'info',
      'ai_complete',
      expect.objectContaining({
        agent_id: 'agent-001',
        agent_slug: 'support-summary',
        agent_service_scope: 'shop-assistant',
      }),
    );
  });

  it('rejects draft or disabled registry agents before routing', async () => {
    agents.findOne.mockResolvedValue({
      id: 'agent-disabled',
      name: 'Disabled Agent',
      slug: 'disabled-agent',
      status: 'disabled',
      serviceScope: 'shop-assistant',
      routePath: '/ai/complete',
      modelTier: 'cheap',
      systemPrompt: '',
      userPromptTemplate: '{{user_prompt}}',
      maxTokens: 1000,
    });
    global.fetch = jest.fn();

    const result = await service.complete({
      model_tier: 'free',
      user_prompt: 'hi',
      agent_slug: 'disabled-agent',
    });

    expect(result).toEqual(expect.objectContaining({
      text: '',
      model_used: 'agent-registry',
      agent_slug: 'disabled-agent',
      error_code: 'AGENT_NOT_AVAILABLE',
    }));
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('keeps premium registry agents blocked by approval policy', async () => {
    agents.findOne.mockResolvedValue({
      id: 'agent-premium',
      name: 'Premium Agent',
      slug: 'premium-agent',
      status: 'active',
      serviceScope: 'ai-microservice',
      routePath: '/ai/complete',
      modelTier: 'premium',
      systemPrompt: '',
      userPromptTemplate: '{{user_prompt}}',
      maxTokens: 1000,
    });
    global.fetch = jest.fn();

    const result = await service.complete({
      model_tier: 'free',
      user_prompt: 'hi',
      agent_slug: 'premium-agent',
    });

    expect(result).toEqual(expect.objectContaining({
      agent_id: 'agent-premium',
      agent_slug: 'premium-agent',
      error_code: 'AI_AUTH_ERROR',
    }));
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('accepts camelCase businessId for compatibility', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{ message: { content: 'ok' } }],
        usage: { prompt_tokens: 1, completion_tokens: 1 },
        model: 'free',
      }),
    } as Response);

    await service.complete({ model_tier: 'free', user_prompt: 'say ok', businessId: 'biz_test_camel' });

    expect(loggingClient.log).toHaveBeenCalledWith(
      'info',
      'ai_complete',
      expect.objectContaining({ business_id: 'biz_test_camel' }),
    );
  });

  it('returns RATE_LIMIT when LiteLLM returns 429', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 429,
      text: async () => 'rate limited',
    } as Response);

    const result = await service.complete({ model_tier: 'cheap', user_prompt: 'hi' });

    expect(result.error_code).toBe('RATE_LIMIT');
    expect(result.text).toBe('');
  });

  it('accepts premium tier with explicit human approval', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{ message: { content: 'approved premium result' } }],
        usage: { prompt_tokens: 1, completion_tokens: 1 },
        model: 'openrouter/anthropic/claude-sonnet-4.6',
      }),
    } as Response);

    const result = await service.complete({ model_tier: 'premium', user_prompt: 'hi', human_approval: true });
    expect(result.error_code).toBeUndefined();
    expect(result.text).toContain('approved premium result');
  });

  it('blocks premium tier without human approval', async () => {
    const result = await service.complete({ model_tier: 'premium', user_prompt: 'hi' });
    expect(result.error_code).toBe('AI_AUTH_ERROR');
    expect(result.error_message).toContain('human approval');
  });
});

describe('AiService - Claude CLI fallback', () => {
  let service: AiService;
  let loggingClient: jest.Mocked<LoggingClient>;
  const originalLitellmUrl = process.env.LITELLM_BASE_URL;
  const originalLitellmKey = process.env.LITELLM_MASTER_KEY;
  const originalRouter = process.env.AI_COMPLETE_ROUTER;
  const originalFetch = global.fetch;

  beforeEach(() => {
    process.env.LITELLM_BASE_URL = '';
    process.env.LITELLM_MASTER_KEY = '';
    delete process.env.AI_COMPLETE_ROUTER;
    loggingClient = { log: jest.fn().mockResolvedValue(undefined) } as any;
    service = new AiService(loggingClient);
  });

  afterEach(() => {
    global.fetch = originalFetch;
    process.env.LITELLM_BASE_URL = originalLitellmUrl;
    process.env.LITELLM_MASTER_KEY = originalLitellmKey;
    if (originalRouter === undefined) {
      delete process.env.AI_COMPLETE_ROUTER;
    } else {
      process.env.AI_COMPLETE_ROUTER = originalRouter;
    }
  });

  it('routes to sonnet via CC CLI when LITELLM_BASE_URL unset', async () => {
    jest.spyOn(service as any, 'spawnCcCli').mockResolvedValue('ok\n');

    const result = await service.complete({ model_tier: 'free', user_prompt: 'say ok' });

    expect(result.text).toBe('ok');
    expect(result.model_used).toBe('claude-sonnet');
  });

  it('returns RATE_LIMIT when CLI returns 429 JSON envelope', async () => {
    const rateLimitJson = JSON.stringify({
      is_error: true,
      api_error_status: 429,
      result: "You've hit your session limit",
    });
    jest.spyOn(service as any, 'spawnCcCli').mockResolvedValue(rateLimitJson);

    const result = await service.complete({ model_tier: 'free', user_prompt: 'hi' });

    expect(result.error_code).toBe('RATE_LIMIT');
    expect(result.error_message).toContain('session limit');
  });

  it('falls back to LiteLLM when CC CLI fails and claude_cli_with_litellm_fallback router set', async () => {
    process.env.LITELLM_BASE_URL = 'http://litellm.test:4000';
    process.env.LITELLM_MASTER_KEY = 'test-key';
    process.env.AI_COMPLETE_ROUTER = 'claude_cli_with_litellm_fallback';

    jest.spyOn(service as any, 'spawnCcCli').mockRejectedValue(new Error('claude: command not found'));
    // `model` in the body is the ALIAS LiteLLM echoes back, deliberately different from
    // the real model here: the resolved id must come from the x-litellm-model-id header
    // via /model/info, never from the body.
    global.fetch = litellmFetchMock({ deploymentId: FREE_ID });

    const result = await service.complete({ model_tier: 'free', user_prompt: 'say ok' });

    expect(result.error_code).toBeUndefined();
    expect(result.model_used).toBe('ollama/qwen2.5-coder:0.5b');
    expect(result.model_resolved).toBe(true);
    expect(global.fetch).toHaveBeenCalled();
  });

  describe('served model resolution', () => {
    beforeEach(() => {
      process.env.LITELLM_BASE_URL = 'http://litellm.test:4000';
      process.env.LITELLM_MASTER_KEY = 'test-key';
      process.env.AI_COMPLETE_ROUTER = 'litellm';
    });

    it('reports the real upstream model, not the alias LiteLLM echoes', async () => {
      global.fetch = litellmFetchMock({ deploymentId: SMART_ID });

      const result = await service.complete({ model_tier: 'smart', user_prompt: 'hi' });

      // The whole point: a request for `smart` comes back as model: "smart" in the body.
      expect(result.model_used).toBe('openrouter/google/gemma-4-31b-it:free');
      expect(result.tier_used).toBe('smart');
      expect(result.model_resolved).toBe(true);
      expect(result.served_by_fallback).toBe(false);
    });

    it('flags a silent fallback that the echoed alias would hide', async () => {
      global.fetch = litellmFetchMock({ deploymentId: SMART_FALLBACK_ID, attemptedFallbacks: '1' });

      const result = await service.complete({ model_tier: 'smart', user_prompt: 'hi' });

      // Body still says "smart"; only the headers reveal a different model served it.
      expect(result.served_by_fallback).toBe(true);
      expect(result.model_used).toBe('openrouter/nvidia/nemotron-3-super-120b-a12b:free');
      expect(result.tier_used).toBe('smart');
    });

    it('reports model_resolved=false rather than passing the tier off as a model', async () => {
      global.fetch = litellmFetchMock({ deploymentId: '' });

      const result = await service.complete({ model_tier: 'smart', user_prompt: 'hi' });

      expect(result.model_resolved).toBe(false);
      expect(result.tier_used).toBe('smart');
    });
  });
});
