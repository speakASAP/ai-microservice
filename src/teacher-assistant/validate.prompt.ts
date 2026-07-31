import { ValidateDrillRequest } from './contracts';

export const VALIDATE_SYSTEM_PROMPT = `You are a strict language-teaching editor. You review fill-in-the-blank drill sentences written by someone else.

You do NOT know who wrote them or why. Judge only what is in front of you.

Markup: [prompt]{answer} — "prompt" is the placeholder shown to the learner,
"answer" is what they must type.

For EVERY item, judge four things independently:

1. topicAlignment — does the BLANK actually exercise the requested grammar point?
   A preposition drill whose blank is an article is OFF_TOPIC, however good the
   sentence is. This is the check that matters most; be strict.
2. grammar — is the sentence correct in the target language once the answer is
   substituted? Judge the target language only; the prompt is in another language.
3. level — is the vocabulary and structure appropriate for the stated level?
4. naturalness — would a native speaker say this? Word-for-word translations
   from the material language are UNNATURAL even when grammatical.

Verdicts: PASS, WARN, or FAIL.
  - grammar may only be PASS or FAIL.
  - Use FAIL for topicAlignment when the blank tests the wrong thing.
  - Use WARN, not FAIL, for style you merely dislike.

When ANY verdict is FAIL you MUST supply suggestedFix: a corrected version of
the whole item in the same markup, preserving the intent and the topic. Never
return a FAIL with a null suggestedFix. A complaint without a correction is
useless to the teacher.

Issue codes: OFF_TOPIC, UNGRAMMATICAL, WRONG_LEVEL, UNNATURAL.

Return JSON only.`;

/**
 * Terms that would tell the reviewer where the items came from — that a machine
 * wrote them, or that they were written to fill an order at all. A teacher who
 * types "AI-generate 20 preposition sentences" would otherwise put "AI" and
 * "generate" straight into the reviewer's context, and the reviewer would be
 * grading a fulfilment rather than reviewing found material.
 *
 * Matched case-insensitively against hyphen/underscore-separated word parts, so
 * "AI-generate" and "auto_generated" are both caught.
 */
const PROVENANCE_TERMS = new Set([
  'ai', 'a.i', 'llm', 'llms', 'gpt', 'chatgpt', 'claude', 'gemini', 'openai',
  'model', 'models', 'bot', 'bots', 'robot', 'machine', 'automatically', 'auto',
  'generate', 'generates', 'generated', 'generating', 'generation', 'generator',
  'create', 'creates', 'created', 'creating',
  'write', 'writes', 'wrote', 'writing', 'written',
  'produce', 'produces', 'produced', 'producing',
  'compose', 'composes', 'composed', 'composing',
  'make', 'makes', 'made', 'making',
  'invent', 'invents', 'invented',
  'prompt', 'prompts', 'assistant', 'bank',
]);

/** "10 sentences", "a few items", "several examples" — a count is an instruction
 *  to a producer, and a reviewer has no use for it. */
const QUANTITY_PHRASE =
  /\b(?:\d+|a\s+few|a\s+couple\s+of|several|some|lots\s+of)\s+(?:more\s+)?(?:sentences?|items?|examples?|questions?|drills?|exercises?|phrases?|tasks?)\b/gi;

function isProvenanceToken(token: string): boolean {
  const cleaned = token.toLowerCase().replace(/[^a-z0-9._-]/g, '');
  if (cleaned === '') return false;
  if (PROVENANCE_TERMS.has(cleaned)) return true;
  // Split compounds ("AI-generate", "auto_generated") and reject if any part is
  // a provenance term.
  const parts = cleaned.split(/[-_.]+/).filter(Boolean);
  return parts.length > 1 && parts.some((p) => PROVENANCE_TERMS.has(p));
}

/**
 * Strips quantity phrasing, bare counts and provenance terms from the teacher's
 * free text so what reaches the reviewer reads as an editorial standard rather
 * than as the order the items were produced against.
 */
export function sanitizeInstructionsForReview(raw: string): string {
  const withoutCounts = raw.replace(QUANTITY_PHRASE, ' ').replace(/\b\d+\b/g, ' ');

  const kept = withoutCounts
    .split(/\s+/)
    .filter((token) => token !== '' && !isProvenanceToken(token))
    .join(' ');

  return kept
    // Punctuation debris left behind by removed tokens.
    .replace(/\s+([.,;:!?])/g, '$1')
    .replace(/([.,;:!?])\1+/g, '$1')
    .replace(/^[\s.,;:!?—-]+/, '')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

export function buildValidateUserPrompt(req: ValidateDrillRequest): string {
  const topics = req.topics
    .map((t) => `- ${t.title} (${t.slug})${t.focus ? ` — focus on: ${t.focus}` : ''}`)
    .join('\n');

  const items = req.items
    .map((item) => `#${item.itemRef}: ${item.template}${item.hint ? ` (hint: ${item.hint})` : ''}`)
    .join('\n');

  const requirements = sanitizeInstructionsForReview(req.instructions);

  return [
    `TARGET language: ${req.languageCode}`,
    `MATERIAL language (used in placeholders and hints): ${req.materialLanguage}`,
    `Level: ${req.level ?? 'unspecified'}`,
    '',
    'TOPICS:',
    topics,
    // Framed as a standard the material must meet, not as an order it was
    // written to fill. "TEACHER'S INSTRUCTIONS" told the reviewer it was
    // grading a fulfilment — exactly the anchoring this agent exists to avoid.
    ...(requirements
      ? ['', 'REQUIREMENTS THESE ITEMS MUST MEET:', requirements]
      : []),
    '',
    'ITEMS TO REVIEW:',
    items,
  ].join('\n');
}
