import { GenerateService } from './generate.service';
import { GENERATE_SYSTEM_PROMPT } from './generate.prompt';
import { GenerateDrillRequest, VOCABULARY_MIN_KNOWN_RATIO } from './contracts';

const req: GenerateDrillRequest = {
  languageCode: 'de', materialLanguage: 'ru', level: 'A2',
  topics: [{ slug: 'prepositions', title: 'Предлоги', focus: 'an, bei, für' }],
  instructions: '50 sentences, present tense only', count: 2,
  knownVocabulary: ['bus', 'schule'], maxNewWordsPerSentence: 2,
  exampleItems: ['Ich gehe [in]{in} die Schule.'], avoidTexts: ['Ich gehe in die Schule.'],
  correlationId: 'c-1',
};

describe('GenerateService.generate', () => {
  it('normalizes blanks by adding index and defaulting alternatives', async () => {
    const llm = {
      completeJson: jest.fn().mockResolvedValue({
        data: { items: [{
          template: 'Ich warte [на]{auf} den Bus.',
          blanks: [{ prompt: 'на', answer: 'auf' }],
          hint: '(warten auf – ждать)', topicSlug: 'prepositions', newWords: ['warten'],
        }] },
        meta: { model: 'm', tier: 'smart', promptTokens: 1, completionTokens: 2 },
      }),
    } as any;
    const svc = new GenerateService(llm);
    const res = await svc.generate(req);
    expect(res.items[0].blanks[0]).toEqual({
      index: 0, prompt: 'на', answer: 'auf', alternatives: [],
    });
  });

  it('passes the teacher instructions through verbatim', async () => {
    const llm = { completeJson: jest.fn().mockResolvedValue({ data: { items: [] }, meta: {} as any }) } as any;
    const svc = new GenerateService(llm);
    await svc.generate(req);
    expect(llm.completeJson.mock.calls[0][0].userPrompt)
      .toContain('50 sentences, present tense only');
  });

  it('includes the avoid list so the model does not repeat known items', async () => {
    const llm = { completeJson: jest.fn().mockResolvedValue({ data: { items: [] }, meta: {} as any }) } as any;
    const svc = new GenerateService(llm);
    await svc.generate(req);
    expect(llm.completeJson.mock.calls[0][0].userPrompt).toContain('Ich gehe in die Schule.');
  });

  it('returns an empty item list rather than throwing when the model returns none', async () => {
    const llm = { completeJson: jest.fn().mockResolvedValue({ data: { items: [] }, meta: {} as any }) } as any;
    const svc = new GenerateService(llm);
    await expect(svc.generate(req)).resolves.toMatchObject({ items: [] });
  });

  it('drops an item whose blanks field is missing entirely', async () => {
    const llm = {
      completeJson: jest.fn().mockResolvedValue({
        data: { items: [{ template: 'x', topicSlug: 'prepositions', newWords: [] }] },
        meta: {} as any,
      }),
    } as any;
    const svc = new GenerateService(llm);
    const res = await svc.generate(req);
    expect(res.items).toEqual([]);
  });

  // M1 — nothing validates the parsed JSON, so a model that returns an object
  // where an array was asked for must not crash the endpoint.
  it('survives a non-array items field instead of throwing "not iterable"', async () => {
    const llm = {
      completeJson: jest.fn().mockResolvedValue({
        data: { items: { template: 'x' } },
        meta: {} as any,
      }),
    } as any;
    const svc = new GenerateService(llm);
    await expect(svc.generate(req)).resolves.toMatchObject({ items: [] });
  });

  it('survives a response with no data object at all', async () => {
    const llm = { completeJson: jest.fn().mockResolvedValue({ data: null, meta: {} as any }) } as any;
    const svc = new GenerateService(llm);
    await expect(svc.generate(req)).resolves.toMatchObject({ items: [] });
  });
});

// M5 — the prompt must quote the exported constants, not a hard-coded copy.
describe('GENERATE_SYSTEM_PROMPT', () => {
  it('interpolates the vocabulary constants rather than restating them', () => {
    expect(GENERATE_SYSTEM_PROMPT).toContain(
      `At least ${Math.round(VOCABULARY_MIN_KNOWN_RATIO * 100)}% of the content words`,
    );
    // The per-sentence new-word cap is deliberately NOT stated here: it is
    // request-driven (`Maximum new words per sentence: ${req.maxNewWordsPerSentence}`
    // in the user prompt) and a constant in the system prompt would contradict
    // any request asking for a different number.
    expect(GENERATE_SYSTEM_PROMPT).not.toMatch(/never\s+more than \d+/);
    // A hard-coded copy would still satisfy the assertion above while the
    // constants hold their current values, so assert there is no *other*
    // percentage in the prompt that could be a stale duplicate.
    const percentages = GENERATE_SYSTEM_PROMPT.match(/\d+%/g) ?? [];
    expect(percentages).toEqual([`${Math.round(VOCABULARY_MIN_KNOWN_RATIO * 100)}%`]);
  });
});
