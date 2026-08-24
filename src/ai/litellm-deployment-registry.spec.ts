import { LitellmDeploymentRegistry } from './litellm-deployment-registry';

const SMART_ID = '8ebf6dfd489373f906f6596b761ffdd0cb8ea20514d971e4eff82003bb6cf451';
const SMART_MODEL = 'openrouter/google/gemma-4-31b-it:free';
const FALLBACK_ID = 'ec315304c9aa9a52';
const FALLBACK_MODEL = 'openrouter/nvidia/nemotron-3-super-120b-a12b:free';

/** Mirrors the real /model/info payload shape observed on LiteLLM 1.82.6. */
const modelInfoBody = {
  data: [
    { model_name: 'smart', litellm_params: { model: SMART_MODEL }, model_info: { id: SMART_ID } },
    { model_name: 'smart-fallback', litellm_params: { model: FALLBACK_MODEL }, model_info: { id: FALLBACK_ID } },
  ],
};

describe('LitellmDeploymentRegistry', () => {
  const ok = (body: unknown) => ({ ok: true, status: 200, json: async () => body }) as unknown as Response;

  it('resolves a deployment hash to the real upstream model', async () => {
    const fetchMock = jest.fn().mockResolvedValue(ok(modelInfoBody));
    const registry = new LitellmDeploymentRegistry('http://litellm:4000', 'k', fetchMock as never);

    expect(await registry.resolveModel(SMART_ID)).toBe(SMART_MODEL);
  });

  it('distinguishes a fallback deployment from the tier it stood in for', async () => {
    const fetchMock = jest.fn().mockResolvedValue(ok(modelInfoBody));
    const registry = new LitellmDeploymentRegistry('http://litellm:4000', 'k', fetchMock as never);

    expect(await registry.resolveModel(FALLBACK_ID)).toBe(FALLBACK_MODEL);
  });

  it('caches so a burst of completions does not stampede /model/info', async () => {
    const fetchMock = jest.fn().mockResolvedValue(ok(modelInfoBody));
    const registry = new LitellmDeploymentRegistry('http://litellm:4000', 'k', fetchMock as never);

    await Promise.all([SMART_ID, SMART_ID, FALLBACK_ID].map((id) => registry.resolveModel(id)));

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('returns undefined rather than a guess when /model/info fails', async () => {
    const fetchMock = jest.fn().mockResolvedValue({ ok: false, status: 503, json: async () => ({}) });
    const registry = new LitellmDeploymentRegistry('http://litellm:4000', 'k', fetchMock as never);

    // Undefined becomes model_resolved: false downstream. A wrong-but-plausible model
    // name would be worse than admitting the model is unknown.
    expect(await registry.resolveModel(SMART_ID)).toBeUndefined();
  });

  it('returns undefined when /model/info is unreachable', async () => {
    const fetchMock = jest.fn().mockRejectedValue(new Error('ECONNREFUSED'));
    const registry = new LitellmDeploymentRegistry('http://litellm:4000', 'k', fetchMock as never);

    expect(await registry.resolveModel(SMART_ID)).toBeUndefined();
  });

  it('returns undefined for a deployment id absent from the map', async () => {
    const fetchMock = jest.fn().mockResolvedValue(ok(modelInfoBody));
    const registry = new LitellmDeploymentRegistry('http://litellm:4000', 'k', fetchMock as never);

    expect(await registry.resolveModel('unknown-hash')).toBeUndefined();
  });

  it('returns undefined for an empty deployment id without calling out', async () => {
    const fetchMock = jest.fn();
    const registry = new LitellmDeploymentRegistry('http://litellm:4000', 'k', fetchMock as never);

    expect(await registry.resolveModel('')).toBeUndefined();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
