import { AiService } from './ai.service';
import { LoggingClient } from '../claude-code/logging.client';
import { AiCompleteRequestSchema } from '../contracts';

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
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{ message: { content: '{"output_ref":{"summary":"ok"}}' } }],
        usage: { prompt_tokens: 10, completion_tokens: 5 },
        model: 'ollama/qwen2.5-coder:0.5b',
      }),
    } as Response);

    const result = await service.complete({ model_tier: 'free', user_prompt: 'say ok' });

    expect(result.error_code).toBeUndefined();
    expect(result.model_used).toBe('ollama/qwen2.5-coder:0.5b');
    expect(global.fetch).toHaveBeenCalled();
  });
});
