import { LlmClient } from './llm.client';
import { JwtUtil } from '../service-identity/jwt.util';
import {
  AiCompleteResponse,
  AiCompleteResponseSchema,
} from '../contracts/ai-complete.contract';

/**
 * Every mocked response in this file goes through the REAL response schema
 * from `src/contracts/ai-complete.contract.ts`. That is deliberate and is the
 * whole point of this rewrite: the previous version of this spec mocked
 * `{ content, model, usage }` — a shape `/ai/complete` has never returned —
 * so five green tests certified a client that failed 100% of real requests.
 * Building fixtures with `AiCompleteResponseSchema.parse` means a contract
 * change now breaks these tests instead of silently passing them.
 */
function aiCompleteResponse(overrides: Record<string, unknown> = {}): AiCompleteResponse {
  return AiCompleteResponseSchema.parse({
    text: '',
    model_used: 'test-model',
    ...overrides,
  });
}

function okResponse(overrides: Record<string, unknown> = {}) {
  return { ok: true, status: 200, json: async () => aiCompleteResponse(overrides) };
}

const TEST_SECRET = 'test-jwt-secret-not-a-real-credential';

describe('LlmClient.completeJson', () => {
  const fetchMock = jest.fn();

  beforeEach(() => {
    jest.resetAllMocks();
    global.fetch = fetchMock as unknown as typeof fetch;
    process.env.AI_ORCHESTRATOR_URL = 'http://ai-microservice:3380';
    process.env.DRILL_GENERATION_MODEL_TIER = 'smart';
    process.env.JWT_SECRET = TEST_SECRET;
  });

  const call = (client: LlmClient, outputSchema: unknown = { type: 'object' }) =>
    client.completeJson<Record<string, unknown>>({
      systemPrompt: 'sys',
      userPrompt: 'user',
      outputSchema,
      correlationId: 'c-1',
    });

  const lastInit = () => fetchMock.mock.calls[0][1] as RequestInit;
  const lastBody = () => JSON.parse(lastInit().body as string) as Record<string, any>;

  // --- C1: service authentication -----------------------------------------

  it('sends a service token that ServiceAuthGuard would accept', async () => {
    fetchMock.mockResolvedValue(okResponse({ text: '{"items":[]}' }));
    await call(new LlmClient());

    const headers = lastInit().headers as Record<string, string>;
    expect(headers.Authorization).toMatch(/^Bearer \S+$/);

    // Verify exactly the way ServiceAuthGuard does: HS256 over JWT_SECRET,
    // issuer pinned to 'ai-microservice' inside JwtUtil.verify.
    const token = headers.Authorization.slice('Bearer '.length);
    const payload = JwtUtil.verify(token, TEST_SECRET);
    expect(payload.serviceId).toBe('ai-microservice');
    expect(payload.exp).toBeGreaterThan(Math.floor(Date.now() / 1000));
  });

  it('fails closed rather than calling unauthenticated when JWT_SECRET is absent', async () => {
    delete process.env.JWT_SECRET;
    await expect(call(new LlmClient())).rejects.toThrow(/auth is not configured/i);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  // --- C2: the real response shape ----------------------------------------

  it('parses the JSON body out of the contract `text` field', async () => {
    fetchMock.mockResolvedValue(okResponse({ text: '{"items":[{"template":"a"}]}' }));
    const { data } = await new LlmClient().completeJson<{ items: { template: string }[] }>({
      systemPrompt: 's',
      userPrompt: 'u',
      outputSchema: {},
      correlationId: 'c',
    });
    expect(data.items[0].template).toBe('a');
  });

  it('maps model_used / inputTokens / outputTokens into meta', async () => {
    fetchMock.mockResolvedValue(
      okResponse({
        text: '{"items":[]}',
        model_used: 'anthropic/claude-sonnet-4',
        inputTokens: 1234,
        outputTokens: 567,
      }),
    );
    const { meta } = await call(new LlmClient());
    expect(meta).toEqual({
      model: 'anthropic/claude-sonnet-4',
      tier: 'smart',
      promptTokens: 1234,
      completionTokens: 567,
    });
  });

  it('falls back to AiService top-level spread keys when text is empty', async () => {
    // AiService returns `{ ...parsedData, text, model_used, ... }` on both the
    // LiteLLM and the CC-CLI path, so the answer survives an empty `text`.
    fetchMock.mockResolvedValue(okResponse({ text: '', items: [{ template: 'b' }] }));
    const { data } = await new LlmClient().completeJson<{ items: { template: string }[] }>({
      systemPrompt: 's',
      userPrompt: 'u',
      outputSchema: {},
      correlationId: 'c',
    });
    expect(data.items[0].template).toBe('b');
  });

  it('strips markdown fences the model sometimes wraps JSON in', async () => {
    fetchMock.mockResolvedValue(okResponse({ text: '```json\n{"items":[]}\n```' }));
    const { data } = await new LlmClient().completeJson<{ items: unknown[] }>({
      systemPrompt: 's',
      userPrompt: 'u',
      outputSchema: {},
      correlationId: 'c',
    });
    expect(data.items).toEqual([]);
  });

  it('throws a typed error on unparseable text rather than returning garbage', async () => {
    fetchMock.mockResolvedValue(okResponse({ text: 'I cannot do that.' }));
    await expect(call(new LlmClient())).rejects.toThrow(/not valid JSON/i);
  });

  // --- C3: the schema must reach the model --------------------------------

  it('serializes the output schema into the outgoing user_prompt', async () => {
    fetchMock.mockResolvedValue(okResponse({ text: '{"items":[]}' }));
    const schema = {
      type: 'object',
      required: ['items'],
      properties: { items: { type: 'array' } },
    };
    await call(new LlmClient(), schema);

    const body = lastBody();
    expect(body.user_prompt).toContain('user');
    expect(body.user_prompt).toContain(JSON.stringify(schema));
    // Still sent in the body: upstream uses its presence to turn on JSON mode.
    expect(body.output_schema).toEqual(schema);
  });

  it('sends the configured tier and correlation id', async () => {
    fetchMock.mockResolvedValue(okResponse({ text: '{"items":[]}' }));
    await call(new LlmClient());
    const body = lastBody();
    expect(body.model_tier).toBe('smart');
    expect(body.correlation_id).toBe('c-1');
    expect(body.system_prompt).toBe('sys');
  });

  // --- I3: a 200 carrying error_code is a failure -------------------------

  it('throws when a 200 response carries an error_code', async () => {
    fetchMock.mockResolvedValue(
      okResponse({ text: '', error_code: 'RATE_LIMIT', error_message: 'upstream 429' }),
    );
    await expect(call(new LlmClient())).rejects.toThrow(/RATE_LIMIT/);
  });

  it('does not mislabel a provider error as a JSON parse failure', async () => {
    fetchMock.mockResolvedValue(
      okResponse({ text: '', error_code: 'CLI_FAILED', error_message: 'claude exited 1' }),
    );
    await expect(call(new LlmClient())).rejects.not.toThrow(/not valid JSON/i);
  });

  // --- I4: bounded call ----------------------------------------------------

  it('bounds the upstream call with an abort signal', async () => {
    fetchMock.mockResolvedValue(okResponse({ text: '{"items":[]}' }));
    await call(new LlmClient());
    expect(lastInit().signal).toBeInstanceOf(AbortSignal);
  });

  // --- M8: no upstream body echoed back to the caller ---------------------

  it('throws on a non-ok upstream response without echoing the body', async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 502,
      text: async () => 'bad gateway: leaky-detail-from-provider',
    });
    await expect(call(new LlmClient())).rejects.toThrow(/502/);
    await expect(call(new LlmClient())).rejects.not.toThrow(/leaky-detail/);
  });
});

/**
 * A transient LiteLLM timeout used to reach the teacher as a red banner they had to
 * click Retry on, mid-way through a drill generation they had already waited for.
 *
 * Reported 2026-08-09: `AI_HTTP_TIMEOUT: LiteLLM did not respond within 120000ms` →
 * `ai/complete returned 500` → `ai-microservice responded 503`. It happened once in 24h,
 * and a retry succeeded, so the failure was worth absorbing rather than surfacing.
 *
 * ONE retry, and only for transient upstream failures. A retry is not free — it costs
 * another full model call and doubles the teacher's wait — so a deterministic failure
 * (bad request, auth, a schema the model cannot satisfy) must fail immediately.
 */
describe('LlmClient.completeJson — transient failure retry', () => {
  const fetchMock = jest.fn();

  beforeEach(() => {
    jest.resetAllMocks();
    global.fetch = fetchMock as unknown as typeof fetch;
    process.env.AI_ORCHESTRATOR_URL = 'http://ai-microservice:3380';
    process.env.DRILL_GENERATION_MODEL_TIER = 'smart';
    process.env.JWT_SECRET = 'test-jwt-secret-not-a-real-credential';
  });

  const call = (client: LlmClient) =>
    client.completeJson<Record<string, unknown>>({
      systemPrompt: 'sys',
      userPrompt: 'user',
      outputSchema: { type: 'object' },
      correlationId: 'c-1',
    });

  const ok = () => ({
    ok: true,
    status: 200,
    json: async () => ({ text: '{"items":[]}', model_used: 'smart', usage: {} }),
  });

  it('retries once after a 503 and returns the second response', async () => {
    fetchMock
      .mockResolvedValueOnce({ ok: false, status: 503, text: async () => 'upstream timeout' })
      .mockResolvedValueOnce(ok());

    const result = await call(new LlmClient());

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(result.data).toEqual({ items: [] });
  });

  it('retries once after an aborted request (the client-side timeout)', async () => {
    const abort = Object.assign(new Error('The operation was aborted'), { name: 'TimeoutError' });
    fetchMock.mockRejectedValueOnce(abort).mockResolvedValueOnce(ok());

    const result = await call(new LlmClient());

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(result.data).toEqual({ items: [] });
  });

  it('retries once when the upstream reports a transient error_code', async () => {
    fetchMock
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ text: '', error_code: 'RATE_LIMIT', error_message: 'slow down' }),
      })
      .mockResolvedValueOnce(ok());

    const result = await call(new LlmClient());

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(result.data).toEqual({ items: [] });
  });

  it('gives up after the second failure rather than retrying forever', async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 503, text: async () => 'still down' });

    await expect(call(new LlmClient())).rejects.toThrow();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('does NOT retry a 400 — a bad request fails the same way twice', async () => {
    // Retrying a deterministic failure just doubles the teacher's wait before the same
    // error, and pays for a second model call to learn nothing.
    fetchMock.mockResolvedValue({ ok: false, status: 400, text: async () => 'bad request' });

    await expect(call(new LlmClient())).rejects.toThrow();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('does NOT retry a 401', async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 401, text: async () => 'unauthorized' });

    await expect(call(new LlmClient())).rejects.toThrow();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('does NOT retry a permanent error_code', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ text: '', error_code: 'AI_AUTH_ERROR', error_message: 'bad key' }),
    });

    await expect(call(new LlmClient())).rejects.toThrow();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
