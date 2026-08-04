import { GENERATE_SYSTEM_PROMPT, buildGenerateUserPrompt } from './generate.prompt';

/**
 * A German course produced English prompts: `Wir [have come]{sind gekommen}.` The
 * student, who is Russian-speaking and learning German, saw English placeholders on
 * every blank.
 *
 * Two causes. The system prompt stated the rule once, in prose, with no example showing
 * a filled prompt — and the user prompt passed bare ISO codes (`ru`, `de`) rather than
 * language names, so "MATERIAL language: ru" was a weaker signal than the English
 * examples the model had in front of it from the bank.
 */
describe('generate prompt', () => {
  const req = {
    languageCode: 'de',
    materialLanguage: 'ru',
    level: 'B1',
    count: 3,
    topics: ['present-perfect'],
    instructions: 'train present perfect',
    knownVocabulary: ['machen'],
    exampleItems: [],
    avoidTexts: [],
    maxNewWordsPerSentence: 2,
  } as any;

  describe('system prompt', () => {
    it('shows a worked example of a prompt in the material language', () => {
      // Prose alone did not hold. An example the model can pattern-match does.
      expect(GENERATE_SYSTEM_PROMPT).toMatch(/\[[^\]]*[а-яА-Я][^\]]*\]\{/);
    });

    it('states that the prompt must never be in English unless English is the material', () => {
      expect(GENERATE_SYSTEM_PROMPT.toLowerCase()).toContain('never');
    });
  });

  describe('user prompt', () => {
    it('names the material language rather than passing a bare code', () => {
      const prompt = buildGenerateUserPrompt(req);

      expect(prompt).toContain('Russian');
      expect(prompt).not.toMatch(/MATERIAL language \(prompts and hints\): ru$/m);
    });

    it('names the target language too', () => {
      expect(buildGenerateUserPrompt(req)).toContain('German');
    });

    it('falls back to the raw code for a language it does not know', () => {
      const prompt = buildGenerateUserPrompt({ ...req, materialLanguage: 'xx' });

      expect(prompt).toContain('xx');
    });

    it('repeats the material language next to the markup rule, where it is used', () => {
      // The rule and the language were paragraphs apart; the model followed the nearer
      // signal, which was the English examples.
      const prompt = buildGenerateUserPrompt(req);

      expect(prompt).toMatch(/prompt.*Russian|Russian.*prompt/i);
    });
  });
});
