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

  it('downgrades a FAIL with no issues and no suggestedFix to WARN with a synthesized issue', async () => {
    const llm = { completeJson: jest.fn().mockResolvedValue({
      data: { results: [{ itemRef: 0,
        verdicts: { topicAlignment: 'FAIL', grammar: 'PASS', level: 'PASS', naturalness: 'PASS' },
        issues: [], suggestedFix: null }] },
      meta: {} as any }) } as any;
    const svc = new ValidateService(llm);
    const res = await svc.validate(req);
    expect(res.results[0].state).toBe('WARN');
    expect(res.results[0].issues.length).toBeGreaterThan(0);
  });

  it('marks an item with malformed verdicts as PENDING rather than PASS', async () => {
    const llm = { completeJson: jest.fn().mockResolvedValue({
      data: { results: [{ itemRef: 0,
        verdicts: {},
        issues: [], suggestedFix: null }] },
      meta: {} as any }) } as any;
    const svc = new ValidateService(llm);
    const res = await svc.validate(req);
    expect(res.results[0].state).toBe('PENDING');
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

  // --- I2: provenance must not ride in on the teacher's own free text ------

  it('strips provenance and quantity phrasing out of the teacher instructions', async () => {
    // The fixed fixture above cannot catch this: the leak arrives through
    // whatever the teacher actually typed.
    const llm = { completeJson: jest.fn().mockResolvedValue({ data: { results: [] }, meta: {} as any }) } as any;
    const svc = new ValidateService(llm);
    await svc.validate({
      ...req,
      // Same shape as the real production order in run-eval.ts, plus the
      // "AI-generate" a teacher can plausibly type.
      instructions:
        'AI-generate 20 sentences drilling prepositions. Every blank must be a preposition, B1 level.',
    });

    const prompt: string = llm.completeJson.mock.calls[0][0].userPrompt;
    expect(prompt).not.toMatch(/\bAI\b/);
    expect(prompt).not.toMatch(/generate/i);
    expect(prompt).not.toMatch(/\b20\b/);
    // "20 sentences" is an order to a producer; a reviewer has no use for it.
    expect(prompt).not.toMatch(/sentences/i);
    // The editorial substance must survive the scrub.
    expect(prompt).toContain('Every blank must be a preposition');
    expect(prompt).toContain('B1 level');
  });

  it('frames the instructions as a standard rather than as an order to fulfil', async () => {
    const llm = { completeJson: jest.fn().mockResolvedValue({ data: { results: [] }, meta: {} as any }) } as any;
    const svc = new ValidateService(llm);
    await svc.validate(req);
    const prompt: string = llm.completeJson.mock.calls[0][0].userPrompt;
    expect(prompt).toContain('REQUIREMENTS THESE ITEMS MUST MEET');
    expect(prompt).not.toMatch(/instructions/i);
    expect(prompt).not.toMatch(/teacher/i);
  });

  it('omits the requirements block entirely when nothing survives sanitizing', async () => {
    const llm = { completeJson: jest.fn().mockResolvedValue({ data: { results: [] }, meta: {} as any }) } as any;
    const svc = new ValidateService(llm);
    await svc.validate({ ...req, instructions: 'AI-generate 10 sentences' });
    const prompt: string = llm.completeJson.mock.calls[0][0].userPrompt;
    expect(prompt).not.toContain('REQUIREMENTS THESE ITEMS MUST MEET');
  });

  // --- I1: suggestedFix must satisfy the DrillBlank contract Track D reads --

  it('normalizes suggestedFix blanks to the full DrillBlank shape', async () => {
    // The model is only asked for `prompt` and `answer`; `index` and
    // `alternatives` are required by DrillBlank and must be filled in here, or
    // Track D persists `index: undefined`.
    const llm = { completeJson: jest.fn().mockResolvedValue({
      data: { results: [{ itemRef: 0,
        verdicts: { topicAlignment: 'FAIL', grammar: 'PASS', level: 'PASS', naturalness: 'PASS' },
        issues: [{ code: 'OFF_TOPIC', message: 'wrong' }],
        suggestedFix: {
          template: 'Ich warte [на]{auf} den Bus.',
          blanks: [{ prompt: 'на', answer: 'auf' }, { prompt: 'в', answer: 'in' }],
          hint: null,
        } }] },
      meta: {} as any }) } as any;
    const svc = new ValidateService(llm);
    const res = await svc.validate(req);

    expect(res.results[0].state).toBe('FAIL');
    expect(res.results[0].suggestedFix!.blanks).toEqual([
      { index: 0, prompt: 'на', answer: 'auf', alternatives: [] },
      { index: 1, prompt: 'в', answer: 'in', alternatives: [] },
    ]);
  });

  it('coerces a missing blanks array and a non-string hint in suggestedFix', async () => {
    const llm = { completeJson: jest.fn().mockResolvedValue({
      data: { results: [{ itemRef: 0,
        verdicts: { topicAlignment: 'FAIL', grammar: 'PASS', level: 'PASS', naturalness: 'PASS' },
        issues: [{ code: 'OFF_TOPIC', message: 'wrong' }],
        suggestedFix: { template: 'Ich warte [на]{auf} den Bus.', hint: 42 } }] },
      meta: {} as any }) } as any;
    const svc = new ValidateService(llm);
    const res = await svc.validate(req);
    expect(res.results[0].suggestedFix).toEqual({
      template: 'Ich warte [на]{auf} den Bus.',
      blanks: [],
      hint: null,
    });
  });

  it('treats a suggestedFix without a usable template as absent and downgrades to WARN', async () => {
    // `{}` and `{ template: '' }` are truthy. Before this fix they held the
    // item at FAIL with `template: undefined` — unactionable AND blocking.
    for (const suggestedFix of [{}, { template: '' }, { template: '   ' }, 'nope', []]) {
      const llm = { completeJson: jest.fn().mockResolvedValue({
        data: { results: [{ itemRef: 0,
          verdicts: { topicAlignment: 'FAIL', grammar: 'PASS', level: 'PASS', naturalness: 'PASS' },
          issues: [{ code: 'OFF_TOPIC', message: 'wrong' }], suggestedFix }] },
        meta: {} as any }) } as any;
      const svc = new ValidateService(llm);
      const res = await svc.validate(req);
      expect(res.results[0].state).toBe('WARN');
      expect(res.results[0].suggestedFix).toBeNull();
    }
  });

  // --- M1: a non-array results field must not crash the endpoint -----------

  it('survives a non-array results field instead of throwing "not iterable"', async () => {
    const llm = { completeJson: jest.fn().mockResolvedValue({
      data: { results: { itemRef: 0 } }, meta: {} as any }) } as any;
    const svc = new ValidateService(llm);
    const res = await svc.validate(req);
    expect(res.results.every((r) => r.state === 'PENDING')).toBe(true);
    expect(res.results).toHaveLength(2);
  });
});
