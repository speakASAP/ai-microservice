import { ANALYZE_SYSTEM_PROMPT, buildAnalyzeUserPrompt } from './analyze.prompt';
import type { AnalyzeErrorsRequest } from './contracts';

const request: AnalyzeErrorsRequest = {
  languageCode: 'en',
  materialLanguage: 'ru',
  level: 'A2',
  allowedTopicSlugs: ['en.prepositions-of-place', 'en.prepositions-of-movement', 'en.other'],
  failures: [
    {
      answer: 'through',
      sentence: 'We will have to walk {{0}} this market.',
      prompt: 'через',
      wrongAttempts: ['acros', 'across', 'to across'],
      revealed: true,
      mistakeCount: 3,
    },
  ],
  correlationId: 'cid-1',
};

describe('buildAnalyzeUserPrompt', () => {
  it('lists every allowed slug and forbids inventing others', () => {
    const prompt = buildAnalyzeUserPrompt(request);

    expect(prompt).toContain('en.prepositions-of-movement');
    expect(prompt).toContain('en.other');
    expect(ANALYZE_SYSTEM_PROMPT + prompt).toMatch(/only|exclusively/i);
  });

  it('includes what the student actually typed', () => {
    const prompt = buildAnalyzeUserPrompt(request);

    expect(prompt).toContain('across');
    expect(prompt).toContain('acros');
  });

  it('includes the sentence and the correct answer', () => {
    const prompt = buildAnalyzeUserPrompt(request);

    expect(prompt).toContain('We will have to walk');
    expect(prompt).toContain('through');
  });

  it('names the explanation language explicitly', () => {
    expect(buildAnalyzeUserPrompt(request)).toMatch(/Russian|русск/i);
    expect(buildAnalyzeUserPrompt({ ...request, materialLanguage: 'en' })).toMatch(/English/i);
  });

  it('says a revealed blank means the student did not know the answer', () => {
    expect(buildAnalyzeUserPrompt(request)).toMatch(/reveal/i);
  });

  it('requires every submitted answer to appear in exactly one cluster', () => {
    expect(ANALYZE_SYSTEM_PROMPT).toMatch(/every|each/i);
    expect(ANALYZE_SYSTEM_PROMPT).toMatch(/exactly one cluster/i);
  });
});
