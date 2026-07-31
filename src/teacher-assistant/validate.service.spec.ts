import { ValidateService } from './validate.service';
import { ValidateDrillRequest } from './contracts';

const req: ValidateDrillRequest = {
  languageCode: 'de', materialLanguage: 'ru', level: 'A2',
  topics: [{ slug: 'prepositions', title: 'Предлоги' }],
  instructions: 'prepositions only',
  items: [
    { itemRef: 0, template: 'Ich warte [на]{auf} den Bus.', blanks: [], hint: null },
    { itemRef: 1, template: 'Ich sehe [die]{die} Schule.', blanks: [], hint: null },
  ],
  correlationId: 'c-1',
};

describe('ValidateService.validate', () => {
  it('maps a clean item to PASS with no issues', async () => {
    const llm = { completeJson: jest.fn().mockResolvedValue({
      data: { results: [{ itemRef: 0, verdicts: { topicAlignment: 'PASS', grammar: 'PASS', level: 'PASS', naturalness: 'PASS' }, issues: [], suggestedFix: null }] },
      meta: {} as any }) } as any;
    const svc = new ValidateService(llm);
    const res = await svc.validate(req);
    expect(res.results[0].state).toBe('PASS');
  });

  it('maps any FAIL verdict to state FAIL', async () => {
    const llm = { completeJson: jest.fn().mockResolvedValue({
      data: { results: [{ itemRef: 1,
        verdicts: { topicAlignment: 'FAIL', grammar: 'PASS', level: 'PASS', naturalness: 'PASS' },
        issues: [{ code: 'OFF_TOPIC', message: 'Blank is an article', span: 'die' }],
        suggestedFix: { template: 'Ich warte [на]{auf} die Schule.', blanks: [], hint: null } }] },
      meta: {} as any }) } as any;
    const svc = new ValidateService(llm);
    const res = await svc.validate(req);
    expect(res.results[0].state).toBe('FAIL');
    expect(res.results[0].issues[0].code).toBe('OFF_TOPIC');
  });

  it('maps WARN-only verdicts to state WARN', async () => {
    const llm = { completeJson: jest.fn().mockResolvedValue({
      data: { results: [{ itemRef: 0,
        verdicts: { topicAlignment: 'PASS', grammar: 'PASS', level: 'WARN', naturalness: 'WARN' },
        issues: [{ code: 'WRONG_LEVEL', message: 'B1 vocabulary' }], suggestedFix: null }] },
      meta: {} as any }) } as any;
    const svc = new ValidateService(llm);
    const res = await svc.validate(req);
    expect(res.results[0].state).toBe('WARN');
  });

  it('downgrades a FAIL with no suggestedFix to WARN and records it', async () => {
    const llm = { completeJson: jest.fn().mockResolvedValue({
      data: { results: [{ itemRef: 0,
        verdicts: { topicAlignment: 'FAIL', grammar: 'PASS', level: 'PASS', naturalness: 'PASS' },
        issues: [{ code: 'OFF_TOPIC', message: 'wrong' }], suggestedFix: null }] },
      meta: {} as any }) } as any;
    const svc = new ValidateService(llm);
    const res = await svc.validate(req);
    expect(res.results[0].state).toBe('WARN');
  });

  it('marks an item the model did not return as PENDING rather than dropping it', async () => {
    const llm = { completeJson: jest.fn().mockResolvedValue({
      data: { results: [{ itemRef: 0, verdicts: { topicAlignment: 'PASS', grammar: 'PASS', level: 'PASS', naturalness: 'PASS' }, issues: [], suggestedFix: null }] },
      meta: {} as any }) } as any;
    const svc = new ValidateService(llm);
    const res = await svc.validate(req);
    expect(res.results).toHaveLength(2);
    expect(res.results[1]).toMatchObject({ itemRef: 1, state: 'PENDING' });
  });

  it('never sends any hint about item provenance to the model', async () => {
    const llm = { completeJson: jest.fn().mockResolvedValue({ data: { results: [] }, meta: {} as any }) } as any;
    const svc = new ValidateService(llm);
    await svc.validate(req);
    const prompt = llm.completeJson.mock.calls[0][0].userPrompt.toLowerCase();
    for (const term of ['generated', 'generator', 'llm', 'model', 'bank', 'assistant', 'prompt']) {
      expect(prompt).not.toContain(term);
    }
    expect(prompt).not.toMatch(/\bai\b/);
  });
});
