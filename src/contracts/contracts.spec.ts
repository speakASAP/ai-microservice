import { AiCompleteRequestSchema, AiCompleteResponseSchema } from './ai-complete.contract';

describe('AiCompleteRequestSchema', () => {
  it('accepts valid request', () => {
    const result = AiCompleteRequestSchema.safeParse({
      model_tier: 'free',
      user_prompt: 'hello',
    });
    expect(result.success).toBe(true);
    expect(result.data!.schemaVersion).toBe('1.0');
  });

  it('rejects unknown model_tier', () => {
    const result = AiCompleteRequestSchema.safeParse({ model_tier: 'turbo', user_prompt: 'hi' });
    expect(result.success).toBe(false);
  });

  it('rejects empty user_prompt', () => {
    const result = AiCompleteRequestSchema.safeParse({ model_tier: 'free', user_prompt: '' });
    expect(result.success).toBe(false);
  });
});

describe('AiCompleteResponseSchema', () => {
  it('accepts valid response', () => {
    const result = AiCompleteResponseSchema.safeParse({ text: 'hi', model_used: 'sonnet' });
    expect(result.success).toBe(true);
    expect(result.data!.schemaVersion).toBe('1.0');
  });
});
