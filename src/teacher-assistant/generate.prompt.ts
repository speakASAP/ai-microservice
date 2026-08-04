import { GenerateDrillRequest, VOCABULARY_MIN_KNOWN_RATIO } from './contracts';

/** The 80/20 rule as a percentage, so the prompt and the validator that
 *  enforces it can never disagree: both read the same exported constant. */
const MIN_KNOWN_PERCENT = Math.round(VOCABULARY_MIN_KNOWN_RATIO * 100);

export const GENERATE_SYSTEM_PROMPT = `You write fill-in-the-blank drill sentences for language learners.

OUTPUT FORMAT
Each sentence uses inline markup: [prompt]{answer}
  - "prompt" is what the learner sees as a placeholder, written in the MATERIAL language.
  - "answer" is what the learner must type, written in the TARGET language.
  - An empty prompt is allowed for suffix drills: "Ich heiß[]{e} Peter."
  - A sentence may contain more than one blank.

  NEVER write the prompt in English unless English IS the material language. A German
  course for Russian speakers must read "Wir [приходить]{sind gekommen}." — never
  "Wir [have come]{sind gekommen}." The prompt is the learner's own language; the
  answer is the language being learned.

  Worked example — target German, material Russian:
    Ich [делать]{habe gemacht} die Aufgabe.
    Wir [приходить]{sind gekommen} nach Hause.

HARD RULES
1. Every sentence must exercise the requested grammar point in the BLANK itself.
   If the topic is prepositions, the blank must be a preposition — not an article,
   not a verb ending.
2. At least ${MIN_KNOWN_PERCENT}% of the content words across all sentences must come from the
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

/**
 * ISO code to the name a model actually recognises.
 *
 * "MATERIAL language: ru" was too weak a signal against a page of English examples;
 * "Russian" is not. Unknown codes pass through unchanged rather than being guessed at.
 */
const LANGUAGE_NAMES: Record<string, string> = {
  de: 'German',
  ru: 'Russian',
  en: 'English',
  es: 'Spanish',
  fr: 'French',
  it: 'Italian',
  cs: 'Czech',
  pt: 'Portuguese',
};

function languageName(code: string): string {
  const key = (code || '').toLowerCase();
  return LANGUAGE_NAMES[key] ? `${LANGUAGE_NAMES[key]} (${key})` : code;
}

export function buildGenerateUserPrompt(req: GenerateDrillRequest): string {
  const topics = req.topics
    .map((t) => `- ${t.title} (${t.slug})${t.focus ? ` — focus on: ${t.focus}` : ''}`)
    .join('\n');

  return [
    `TARGET language (what the learner types): ${languageName(req.languageCode)}`,
    `MATERIAL language (what every prompt and hint is written in): ${languageName(req.materialLanguage)}`,
    // Repeated next to the markup rule because the rule and the language declaration
    // were paragraphs apart, and the model followed the nearer signal — the bank
    // examples, which were English.
    `Every [prompt] must be written in ${languageName(req.materialLanguage)}.`,
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
