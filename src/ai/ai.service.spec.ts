import { AiService } from './ai.service';
import { LoggingClient } from '../claude-code/logging.client';

describe('AiService - Claude CLI', () => {
  let service: AiService;
  let loggingClient: jest.Mocked<LoggingClient>;

  beforeEach(() => {
    loggingClient = { log: jest.fn().mockResolvedValue(undefined) } as any;
    service = new AiService(loggingClient);
  });

  it('routes all model_tier values to sonnet model via CC CLI', async () => {
    for (const tier of ['free', 'cheap', 'smart', 'unknown']) {
      jest.spyOn(service as any, 'spawnCcCli').mockResolvedValue('ok\n');

      const result = await service.complete({ model_tier: tier as 'free', user_prompt: 'say ok' });

      expect(result.text).toBe('ok');
      expect(result.model_used).toBe('claude-sonnet');
    }
  });

  it('returns error_code CLI_FAILED when CLI fails', async () => {
    jest.spyOn(service as any, 'spawnCcCli').mockRejectedValue(new Error('auth failed'));

    const result = await service.complete({ model_tier: 'free', user_prompt: 'hi' });

    expect(result.error_code).toBe('CLI_FAILED');
    expect(result.error_message).toContain('auth failed');
    expect(result.text).toBe('');
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
    expect(result.text).toBe('');
  });

  it('parses JSON response and spreads fields at top level', async () => {
    jest.spyOn(service as any, 'spawnCcCli').mockResolvedValue('{"status":"pass","score":10}\n');

    const result = await service.complete({ model_tier: 'free', user_prompt: 'evaluate' });

    expect(result.status).toBe('pass');
    expect(result.score).toBe(10);
    expect(result.text).toBe('{"status":"pass","score":10}');
  });
});

describe('AiService telemetry', () => {
  let service: AiService;
  let loggingClient: jest.Mocked<LoggingClient>;

  beforeEach(() => {
    loggingClient = { log: jest.fn().mockResolvedValue(undefined) } as any;
    service = new AiService(loggingClient);
  });

  it('emits ai_complete log with compression metadata after successful completion', async () => {
    jest.spyOn(service as any, 'spawnCcCli').mockResolvedValue(
      JSON.stringify({
        result: 'hello world',
        usage: { input_tokens: 100, output_tokens: 20 },
      }),
    );

    await service.complete({
      model_tier: 'smart',
      user_prompt: 'say hello',
      correlation_id: 'test-corr-123',
    });

    expect(loggingClient.log).toHaveBeenCalledWith(
      'info',
      'ai_complete',
      expect.objectContaining({
        correlation_id: 'test-corr-123',
        inputTokens: 100,
        outputTokens: 20,
        compression: { rtk: true, caveman: 'lite' },
      }),
    );
  });

  it('emits ai_complete log with zero tokens on CLI failure', async () => {
    jest.spyOn(service as any, 'spawnCcCli').mockRejectedValue(
      new Error('claude CLI failed: timeout'),
    );

    await service.complete({
      model_tier: 'free',
      user_prompt: 'say hello',
      correlation_id: 'test-corr-456',
    });

    expect(loggingClient.log).toHaveBeenCalledWith(
      'info',
      'ai_complete',
      expect.objectContaining({
        correlation_id: 'test-corr-456',
        inputTokens: 0,
        outputTokens: 0,
        compression: { rtk: true, caveman: 'lite' },
      }),
    );
  });
});
