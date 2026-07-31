/**
 * MANUAL EVAL SCRIPT — DO NOT RUN IN CI. DO NOT IMPORT FROM ANY OTHER FILE.
 *
 * This calls the real LLM through ai-microservice's own `/ai/complete` route
 * (via LlmClient -> AI_ORCHESTRATOR_URL) three times over, once per language
 * pair below, at 10 items each. That costs real tokens/money and needs
 * network access to a running orchestrator. It is not a jest test:
 *   - it lives under `__evals__/`, which the jest config's `testRegex`
 *     (`.*\.spec\.ts$` in jest.config.js) does not match, so `npm test`
 *     never collects it.
 *   - the filename itself does not end in `.spec.ts`.
 *   - as a second belt-and-braces check, it refuses to run at all when
 *     `CI` is set (see the guard immediately below `main()`).
 *
 * This is the baseline harness for judging future prompt changes to
 * generate.prompt.ts / validate.prompt.ts: run it before and after a prompt
 * edit and diff the tables.
 *
 * Usage — run manually, from the ai-microservice repo root, never in CI:
 *
 *   rtk npx ts-node src/teacher-assistant/__evals__/run-eval.ts
 *
 * Requires AI_ORCHESTRATOR_URL (and optionally DRILL_GENERATION_MODEL_TIER)
 * to point at a reachable orchestrator — see LlmClient. ts-node is not a
 * declared devDependency of this package; `npx` will fetch it on demand, or
 * install it locally first (`npm install -D ts-node`) if npx has no network
 * access to the registry.
 */

import { LlmClient } from '../llm.client';
import { GenerateService } from '../generate.service';
import { ValidateService } from '../validate.service';
import {
  GenerateDrillRequest,
  ItemValidationResult,
  ValidateDrillRequest,
  ValidationState,
} from '../contracts';

interface LanguagePair {
  label: string;
  languageCode: string;
  materialLanguage: string;
  knownVocabulary: string[];
  exampleItems: string[];
}

const TOPIC = { slug: 'prepositions', title: 'Prepositions' };
const ITEMS_PER_PAIR = 10;

const LANGUAGE_PAIRS: LanguagePair[] = [
  {
    label: 'de/ru',
    languageCode: 'de',
    materialLanguage: 'ru',
    knownVocabulary: [
      'Bus', 'Schule', 'Arbeit', 'Freund', 'Stadt', 'Bahnhof', 'Park', 'Haus',
      'Straße', 'Tag', 'warten', 'gehen', 'fahren', 'kommen', 'sein', 'arbeiten',
      'wohnen', 'sprechen', 'sehen', 'Woche',
    ],
    exampleItems: ['Ich warte [на]{auf} den Bus.'],
  },
  {
    label: 'en/ru',
    languageCode: 'en',
    materialLanguage: 'ru',
    knownVocabulary: [
      'bus', 'school', 'work', 'friend', 'city', 'station', 'park', 'house',
      'street', 'day', 'wait', 'go', 'drive', 'come', 'be', 'live', 'speak',
      'see', 'week', 'morning',
    ],
    exampleItems: ["I'm waiting [-]{for} the bus."],
  },
  {
    label: 'fr/ru',
    languageCode: 'fr',
    materialLanguage: 'ru',
    knownVocabulary: [
      'bus', 'école', 'travail', 'ami', 'ville', 'gare', 'parc', 'maison',
      'rue', 'jour', 'attendre', 'aller', 'venir', 'être', 'habiter', 'parler',
      'voir', 'semaine', 'matin',
    ],
    exampleItems: ["Je vais [в]{à} l'école."],
  },
];

interface PairSummary {
  pair: string;
  itemsGenerated: number;
  pass: number;
  warn: number;
  fail: number;
  pending: number;
}

interface OffTopicIssue {
  pair: string;
  itemRef: number;
  template: string;
  message: string;
}

function buildGenerateRequest(pair: LanguagePair, correlationId: string): GenerateDrillRequest {
  return {
    languageCode: pair.languageCode,
    materialLanguage: pair.materialLanguage,
    level: 'A2',
    topics: [{ slug: TOPIC.slug, title: TOPIC.title }],
    instructions: `${ITEMS_PER_PAIR} sentences drilling prepositions. Every blank must be a preposition.`,
    count: ITEMS_PER_PAIR,
    knownVocabulary: pair.knownVocabulary,
    maxNewWordsPerSentence: 2,
    exampleItems: pair.exampleItems,
    avoidTexts: [],
    correlationId,
  };
}

function stateCounts(results: ItemValidationResult[]): Record<ValidationState, number> {
  const counts: Record<ValidationState, number> = { PENDING: 0, PASS: 0, WARN: 0, FAIL: 0, OVERRIDDEN: 0 };
  for (const r of results) counts[r.state] += 1;
  return counts;
}

async function runPair(
  pair: LanguagePair,
  generateService: GenerateService,
  validateService: ValidateService,
): Promise<{ summary: PairSummary; offTopicIssues: OffTopicIssue[] }> {
  const correlationId = `eval-${pair.label}-${Date.now()}`;

  const generateRes = await generateService.generate(buildGenerateRequest(pair, correlationId));

  // itemRef is assigned here, by construction, as the position in the array
  // we are building for the validator — that is the one place it is
  // legitimate to use array position. Once results come back, everything
  // downstream must key off itemRef, never off array index again (the
  // validator does not preserve request order; PENDING entries for
  // un-judged items are appended at the end).
  const validateReq: ValidateDrillRequest = {
    languageCode: pair.languageCode,
    materialLanguage: pair.materialLanguage,
    level: 'A2',
    topics: [{ slug: TOPIC.slug, title: TOPIC.title }],
    instructions: `${ITEMS_PER_PAIR} sentences drilling prepositions. Every blank must be a preposition.`,
    items: generateRes.items.map((item, itemRef) => ({
      itemRef,
      template: item.template,
      blanks: item.blanks,
      hint: item.hint,
    })),
    correlationId,
  };

  const validateRes = await validateService.validate(validateReq);

  const templateByRef = new Map(validateReq.items.map((item) => [item.itemRef, item.template]));
  const offTopicIssues: OffTopicIssue[] = [];
  for (const result of validateRes.results) {
    for (const issue of result.issues) {
      if (issue.code !== 'OFF_TOPIC') continue;
      offTopicIssues.push({
        pair: pair.label,
        itemRef: result.itemRef,
        template: templateByRef.get(result.itemRef) ?? '(unknown item)',
        message: issue.message,
      });
    }
  }

  const counts = stateCounts(validateRes.results);
  return {
    summary: {
      pair: pair.label,
      itemsGenerated: generateRes.items.length,
      pass: counts.PASS,
      warn: counts.WARN,
      fail: counts.FAIL,
      pending: counts.PENDING,
    },
    offTopicIssues,
  };
}

async function main(): Promise<void> {
  const llm = new LlmClient();
  const generateService = new GenerateService(llm);
  const validateService = new ValidateService(llm);

  const summaries: PairSummary[] = [];
  const allOffTopicIssues: OffTopicIssue[] = [];

  for (const pair of LANGUAGE_PAIRS) {
    console.log(`Running ${pair.label} (${ITEMS_PER_PAIR} items, topic: ${TOPIC.slug})...`);
    const { summary, offTopicIssues } = await runPair(pair, generateService, validateService);
    summaries.push(summary);
    allOffTopicIssues.push(...offTopicIssues);
  }

  console.log('\n=== Generation + validation summary ===');
  console.table(summaries);

  console.log('\n=== OFF_TOPIC issues (in full) ===');
  if (allOffTopicIssues.length === 0) {
    console.log('None.');
  } else {
    for (const issue of allOffTopicIssues) {
      console.log(`\n[${issue.pair}] itemRef ${issue.itemRef}`);
      console.log(`  template: ${issue.template}`);
      console.log(`  issue:    ${issue.message}`);
    }
  }
}

// Second guard, independent of jest's testRegex not matching this path:
// refuse outright if invoked in a CI environment.
if (process.env.CI) {
  console.error('run-eval.ts is a manual, token-spending eval script and must never run in CI. Aborting.');
  process.exit(1);
}

main().catch((err) => {
  console.error('Eval run failed:', err);
  process.exit(1);
});
