import { AnalyzeService } from './analyze.service';
import type { AnalyzeErrorsRequest } from './contracts';

const request: AnalyzeErrorsRequest = {
  languageCode: 'en',
  materialLanguage: 'ru',
  level: 'A2',
  allowedTopicSlugs: ['en.prepositions-of-movement', 'en.other'],
  failures: [
    {
      answer: 'through',
      sentence: 'We will have to walk {{0}} this market.',
      prompt: 'через',
      wrongAttempts: ['across'],
      revealed: false,
      mistakeCount: 1,
    },
  ],
  correlationId: 'cid-1',
};

function llmStub(data: unknown) {
  return {
    completeJson: jest.fn(async () => ({ data, meta: { model: 'test', tokensIn: 1, tokensOut: 1 } })),
  };
}

describe('AnalyzeService.analyze', () => {
  it('returns the clusters the model produced', async () => {
    const llm = llmStub({
      clusters: [
        {
          topicSlug: 'en.prepositions-of-movement',
          title: 'Предлоги движения',
          explanation: 'through — сквозь что-то...',
          rules: ['through — внутри и наружу'],
          examples: [{ text: 'Walk through the park.', gloss: 'Пройди через парк.' }],
          answers: ['through'],
        },
      ],
    });
    const service = new AnalyzeService(llm as any);

    const result = await service.analyze(request);

    expect(result.clusters).toHaveLength(1);
    expect(result.clusters[0].topicSlug).toBe('en.prepositions-of-movement');
    expect(result.clusters[0].answers).toEqual(['through']);
  });

  it('returns an empty cluster list when the model returns an object instead of an array', async () => {
    const service = new AnalyzeService(llmStub({ clusters: { nope: true } }) as any);

    const result = await service.analyze(request);

    expect(result.clusters).toEqual([]);
  });

  it('drops a cluster with no topicSlug rather than emitting an unfileable one', async () => {
    const service = new AnalyzeService(
      llmStub({ clusters: [{ title: 'x', explanation: 'y', rules: [], examples: [], answers: [] }] }) as any,
    );

    expect((await service.analyze(request)).clusters).toEqual([]);
  });

  it('coerces missing rules, examples and answers to empty arrays', async () => {
    const service = new AnalyzeService(
      llmStub({
        clusters: [
          { topicSlug: 'en.other', title: 'x', explanation: 'y' },
        ],
      }) as any,
    );

    const cluster = (await service.analyze(request)).clusters[0];

    expect(cluster.rules).toEqual([]);
    expect(cluster.examples).toEqual([]);
    expect(cluster.answers).toEqual([]);
  });

  it('drops an example missing its text or gloss', async () => {
    const service = new AnalyzeService(
      llmStub({
        clusters: [
          {
            topicSlug: 'en.other',
            title: 'x',
            explanation: 'y',
            rules: [],
            examples: [{ text: 'ok', gloss: 'ок' }, { text: 'no gloss' }],
            answers: [],
          },
        ],
      }) as any,
    );

    expect((await service.analyze(request)).clusters[0].examples).toEqual([
      { text: 'ok', gloss: 'ок' },
    ]);
  });

  it('passes the correlation id through to the model client', async () => {
    const llm = llmStub({ clusters: [] });
    await new AnalyzeService(llm as any).analyze(request);

    expect(llm.completeJson).toHaveBeenCalledWith(
      expect.objectContaining({ correlationId: 'cid-1' }),
    );
  });
});
