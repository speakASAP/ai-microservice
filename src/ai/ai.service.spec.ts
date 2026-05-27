import { AiService } from './ai.service';
import { Test } from '@nestjs/testing';

describe('AiService - Claude direct', () => {
  let service: AiService;
  let originalFetch: typeof global.fetch;

  beforeEach(async () => {
    originalFetch = global.fetch;
    const module = await Test.createTestingModule({ providers: [AiService] }).compile();
    service = module.get(AiService);
  });

  afterEach(() => {
    global.fetch = originalFetch;
    delete process.env.ANTHROPIC_API_KEY;
  });

  it('calls Anthropic API when ANTHROPIC_API_KEY is set', async () => {
    const mockFetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        content: [{ type: 'text', text: '{"result":"ok"}' }],
        usage: { input_tokens: 10, output_tokens: 5 },
        model: 'claude-sonnet-4-6-20251001',
      }),
    });
    global.fetch = mockFetch as any;
    process.env.ANTHROPIC_API_KEY = 'test-key';

    const result = await service.complete({
      model_tier: 'free', // ignored — always uses Claude
      user_prompt: 'hello',
    });

    expect(mockFetch).toHaveBeenCalledWith(
      'https://api.anthropic.com/v1/messages',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({ 'x-api-key': 'test-key' }),
      }),
    );
    expect(result.inputTokens).toBe(10);
    expect(result.outputTokens).toBe(5);
    expect(result.token_usage_estimate).toBe(15);
    expect(result.model_used).toBe('claude-sonnet-4-6');
  });

  it('throws when ANTHROPIC_API_KEY is missing', async () => {
    delete process.env.ANTHROPIC_API_KEY;
    await expect(service.complete({ model_tier: 'free', user_prompt: 'hi' }))
      .rejects.toThrow('ANTHROPIC_API_KEY');
  });
});
