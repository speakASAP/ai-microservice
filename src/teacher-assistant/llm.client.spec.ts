import { LlmClient } from './llm.client';

describe('LlmClient.completeJson', () => {
  const fetchMock = jest.fn();
  beforeEach(() => {
    jest.resetAllMocks();
    global.fetch = fetchMock as any;
    process.env.AI_ORCHESTRATOR_URL = 'http://ai-microservice:3380';
    process.env.DRILL_GENERATION_MODEL_TIER = 'smart';
  });

  it('sends the configured tier and the output schema', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ content: '{"items":[]}', model: 'm', usage: { prompt_tokens: 1, completion_tokens: 2 } }),
    });
    const client = new LlmClient();
    await client.completeJson({
      systemPrompt: 'sys', userPrompt: 'user',
      outputSchema: { type: 'object' }, correlationId: 'c-1',
    });
    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(body.model_tier).toBe('smart');
    expect(body.output_schema).toEqual({ type: 'object' });
    expect(body.correlation_id).toBe('c-1');
  });

  it('parses the JSON payload out of the content field', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ content: '{"items":[{"template":"a"}]}', model: 'm', usage: {} }),
    });
    const client = new LlmClient();
    const { data } = await client.completeJson<{ items: { template: string }[] }>({
      systemPrompt: 's', userPrompt: 'u', outputSchema: {}, correlationId: 'c',
    });
    expect(data.items[0].template).toBe('a');
  });

  it('strips markdown fences the model sometimes wraps JSON in', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ content: '```json\n{"items":[]}\n```', model: 'm', usage: {} }),
    });
    const client = new LlmClient();
    const { data } = await client.completeJson<{ items: unknown[] }>({
      systemPrompt: 's', userPrompt: 'u', outputSchema: {}, correlationId: 'c',
    });
    expect(data.items).toEqual([]);
  });

  it('throws a typed error on unparseable content rather than returning garbage', async () => {
    fetchMock.mockResolvedValue({
      ok: true, json: async () => ({ content: 'I cannot do that.', model: 'm', usage: {} }),
    });
    const client = new LlmClient();
    await expect(client.completeJson({
      systemPrompt: 's', userPrompt: 'u', outputSchema: {}, correlationId: 'c',
    })).rejects.toThrow(/not valid JSON/i);
  });

  it('throws on a non-ok upstream response', async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 502, text: async () => 'bad gateway' });
    const client = new LlmClient();
    await expect(client.completeJson({
      systemPrompt: 's', userPrompt: 'u', outputSchema: {}, correlationId: 'c',
    })).rejects.toThrow(/502/);
  });
});
