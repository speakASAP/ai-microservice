import { AiService } from './ai.service';
import { LoggingClient } from '../claude-code/logging.client';

describe('AiService - LiteLLM routing', () => {
  let service: AiService;
  let loggingClient: jest.Mocked<LoggingClient>;
  const originalFetch = global.fetch;
  const originalLitellmUrl = process.env.LITELLM_BASE_URL;
  const originalLitellmKey = process.env.LITELLM_MASTER_KEY;

  beforeEach(() => {
    process.env.LITELLM_BASE_URL = 'http://litellm.test:4000';
    process.env.LITELLM_MASTER_KEY = 'test-key';
    loggingClient = { log: jest.fn().mockResolvedValue(undefined) } as any;
    service = new AiService(loggingClient);
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
