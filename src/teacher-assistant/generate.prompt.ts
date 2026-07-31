import { GenerateDrillRequest } from './contracts';

export const GENERATE_SYSTEM_PROMPT = `You write fill-in-the-blank drill sentences for language learners.

OUTPUT FORMAT
Each sentence uses inline markup: [prompt]{answer}
  - "prompt" is what the learner sees as a placeholder, written in the MATERIAL language.
  - "answer" is what the learner must type, written in the TARGET language.
  - An empty prompt is allowed for suffix drills: "Ich heiß[]{e} Peter."
  - A sentence may contain more than one blank.

HARD RULES
1. Every sentence must exercise the requested grammar point in the BLANK itself.
   If the topic is prepositions, the blank must be a preposition — not an article,
   not a verb ending.
2. At least 80% of the content words across all sentences must come from the
   supplied known-vocabulary list.
3. No sentence may contain more than the stated maximum of new words.
4. Every new word must appear in that sentence's "hint" with its translation,
   in the style "(warten auf – ждать; der Bus – автобус)".
5. Sentences must be grammatically correct and natural in the target language.
   Never produce a word-for-word translation that a native speaker would not say.
6. No proper nouns beyond those in the known-vocabulary list.
7. Do not repeat, or lightly reword, any sentence in the avoid list.
8. One grammar point per sentence. Keep them short and everyday.

Return JSON only, matching the supplied schema. No commentary.`;

export function buildGenerateUserPrompt(req: GenerateDrillRequest): string {
  const topics = req.topics
    .map((t) => `- ${t.title} (${t.slug})${t.focus ? ` — focus on: ${t.focus}` : ''}`)
    .join('\n');

  return [
    `TARGET language: ${req.languageCode}`,
    `MATERIAL language (prompts and hints): ${req.materialLanguage}`,
    `Level: ${req.level ?? 'unspecified'}`,
    `Number of sentences to produce: ${req.count}`,
    `Maximum new words per sentence: ${req.maxNewWordsPerSentence}`,
    '',
    'TOPICS:',
    topics,
    '',
    `TEACHER'S REQUEST (follow it literally): ${req.instructions}`,
    '',
    `KNOWN VOCABULARY (${req.knownVocabulary.length} words):`,
    req.knownVocabulary.join(', '),
    '',
    'EXAMPLES of the required style and markup:',
    ...req.exampleItems.map((e) => `  ${e}`),
    '',
    `DO NOT PRODUCE these sentences or near-duplicates of them:`,
    ...req.avoidTexts.map((t) => `  ${t}`),
  ].join('\n');
}
