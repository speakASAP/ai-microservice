import { AiService } from './ai.service';
import { Test } from '@nestjs/testing';
import { exec } from 'child_process';

jest.mock('child_process', () => ({
  exec: jest.fn(),
}));

const mockExec = exec as jest.MockedFunction<typeof exec>;

describe('AiService - Claude CLI', () => {
  let service: AiService;

  beforeEach(async () => {
    jest.clearAllMocks();
    const module = await Test.createTestingModule({ providers: [AiService] }).compile();
    service = module.get(AiService);
  });

  it('invokes claude CLI with sonnet model regardless of model_tier', async () => {
    mockExec.mockImplementation((_cmd: string, _opts: unknown, cb: unknown) => {
      (cb as (err: null, result: { stdout: string; stderr: string }) => void)(
        null, { stdout: 'ok\n', stderr: '' }
      );
      return {} as ReturnType<typeof exec>;
    });

    for (const tier of ['free', 'cheap', 'smart', 'unknown']) {
      jest.clearAllMocks();
      mockExec.mockImplementation((_cmd: string, _opts: unknown, cb: unknown) => {
        (cb as (err: null, result: { stdout: string; stderr: string }) => void)(
          null, { stdout: 'ok\n', stderr: '' }
        );
        return {} as ReturnType<typeof exec>;
      });

      const result = await service.complete({ model_tier: tier as 'free', user_prompt: 'say ok' });

      expect(mockExec).toHaveBeenCalledWith(
        expect.stringContaining('--model sonnet'),
        expect.any(Object),
        expect.any(Function),
      );
      expect(result.text).toBe('ok');
      expect(result.model_used).toBe('claude-sonnet');
    }
  });

  it('returns error_code CLI_FAILED when CLI fails', async () => {
    mockExec.mockImplementation((_cmd: string, _opts: unknown, cb: unknown) => {
      (cb as (err: Error) => void)(Object.assign(new Error('CLI error'), { stderr: 'auth failed' }));
      return {} as ReturnType<typeof exec>;
    });

    const result = await service.complete({ model_tier: 'free', user_prompt: 'hi' });

    expect(result.error_code).toBe('CLI_FAILED');
    expect(result.error_message).toContain('auth failed');
    expect(result.text).toBe('');
  });

  it('returns RATE_LIMIT when CLI exits non-zero but stdout has 429 JSON', async () => {
    const rateLimitJson = JSON.stringify({
      is_error: true,
      api_error_status: 429,
      result: "You've hit your session limit",
    });
    mockExec.mockImplementation((_cmd: string, _opts: unknown, cb: unknown) => {
      (cb as (err: Error & { stdout?: string }) => void)(
        Object.assign(new Error('Command failed'), { stdout: rateLimitJson, stderr: '' }),
      );
      return {} as ReturnType<typeof exec>;
    });

    const result = await service.complete({ model_tier: 'free', user_prompt: 'hi' });

    expect(result.error_code).toBe('RATE_LIMIT');
    expect(result.error_message).toContain('session limit');
    expect(result.text).toBe('');
  });

  it('parses JSON response and spreads fields at top level', async () => {
    mockExec.mockImplementation((_cmd: string, _opts: unknown, cb: unknown) => {
      (cb as (err: null, result: { stdout: string; stderr: string }) => void)(
        null, { stdout: '{"status":"pass","score":10}\n', stderr: '' }
      );
      return {} as ReturnType<typeof exec>;
    });

    const result = await service.complete({ model_tier: 'free', user_prompt: 'evaluate' });

    expect(result.status).toBe('pass');
    expect(result.score).toBe(10);
    expect(result.text).toBe('{"status":"pass","score":10}');
  });
});
