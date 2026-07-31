import { Injectable } from '@nestjs/common';
import { LlmClient } from './llm.client';
import { VALIDATE_SYSTEM_PROMPT, buildValidateUserPrompt } from './validate.prompt';
import { VALIDATE_OUTPUT_SCHEMA } from './validate.schema';
import {
  DrillBlank,
  ItemValidationResult,
  ValidateDrillRequest,
  ValidateDrillResponse,
  ValidationIssue,
  ValidationIssueCode,
  ValidationState,
} from './contracts';

interface RawVerdicts {
  topicAlignment: 'PASS' | 'WARN' | 'FAIL';
  grammar: 'PASS' | 'FAIL';
  level: 'PASS' | 'WARN' | 'FAIL';
  naturalness: 'PASS' | 'WARN' | 'FAIL';
}

interface RawResult {
  itemRef: number;
  verdicts: RawVerdicts;
  issues: ValidationIssue[];
  /** Deliberately `unknown`: this comes off the wire from a language model and
   *  is normalized by `normalizeSuggestedFix` before it is allowed to be typed
   *  as the contract shape. */
  suggestedFix: unknown;
}

const TOPIC_LEVEL_NATURAL = ['PASS', 'WARN', 'FAIL'];
const GRAMMAR_ONLY = ['PASS', 'FAIL'];

/** Which issue code best describes a FAIL in each verdict category, used only
 *  to synthesize an explanation when the model gave none. */
const CATEGORY_ISSUE_CODE: Record<keyof RawVerdicts, ValidationIssueCode> = {
  topicAlignment: 'OFF_TOPIC',
  grammar: 'UNGRAMMATICAL',
  level: 'WRONG_LEVEL',
  naturalness: 'UNNATURAL',
};

/** Guards against a missing or malformed `verdicts` object so it can never
 *  silently fail-open to PASS (all four lookups would be `undefined`, which
 *  matches neither 'FAIL' nor 'WARN'). */
function hasValidVerdicts(v: unknown): v is RawVerdicts {
  if (typeof v !== 'object' || v === null) return false;
  const o = v as Record<string, unknown>;
  return (
    TOPIC_LEVEL_NATURAL.includes(o.topicAlignment as string) &&
    GRAMMAR_ONLY.includes(o.grammar as string) &&
    TOPIC_LEVEL_NATURAL.includes(o.level as string) &&
    TOPIC_LEVEL_NATURAL.includes(o.naturalness as string)
  );
}

/**
 * Coerces a model-supplied fix into the `ItemValidationResult['suggestedFix']`
 * contract, or rejects it as absent.
 *
 * Two problems this closes:
 *  1. `validate.schema.ts` only ever asked the model for `prompt` and `answer`
 *     inside `suggestedFix.blanks[]`, but `DrillBlank` requires `index` and
 *     `alternatives` too. Passing the raw value through typed as `DrillBlank[]`
 *     is a lie TypeScript cannot catch across an HTTP boundary, and Track D
 *     applying such a fix would persist blanks with `index: undefined`.
 *     Normalize exactly as `generate.service.ts` normalizes its own items.
 *  2. A fix with no usable `template` (`{}`, `{ template: '' }`) is not a fix.
 *     Treating it as present kept the item at FAIL with `template: undefined`
 *     and suppressed the FAIL -> WARN downgrade, which exists precisely so a
 *     teacher is never blocked by a complaint they cannot act on.
 */
function normalizeSuggestedFix(raw: unknown): ItemValidationResult['suggestedFix'] {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return null;

  const o = raw as Record<string, unknown>;
  if (typeof o.template !== 'string' || o.template.trim() === '') return null;

  const rawBlanks = Array.isArray(o.blanks) ? o.blanks : [];
  const blanks: DrillBlank[] = rawBlanks.map((b: unknown, index: number) => {
    const blank = (typeof b === 'object' && b !== null ? b : {}) as Record<string, unknown>;
    return {
      index,
      prompt: String(blank.prompt ?? ''),
      answer: String(blank.answer ?? ''),
      alternatives: Array.isArray(blank.alternatives) ? blank.alternatives.map(String) : [],
    };
  });

  return {
    template: o.template,
    blanks,
    hint: typeof o.hint === 'string' ? o.hint : null,
  };
}

@Injectable()
export class ValidateService {
  constructor(private readonly llm: LlmClient) {}

  async validate(req: ValidateDrillRequest): Promise<ValidateDrillResponse> {
    const { data, meta } = await this.llm.completeJson<{ results: unknown[] }>({
      systemPrompt: VALIDATE_SYSTEM_PROMPT,
      userPrompt: buildValidateUserPrompt(req),
      outputSchema: VALIDATE_OUTPUT_SCHEMA,
      correlationId: req.correlationId,
    });

    // Items the request actually contains — anything else in the model's
    // response refers to nothing we submitted and is ignored.
    const validRefs = new Set(req.items.map((item) => item.itemRef));
    const byRef = new Map<number, ItemValidationResult>();

    // A model that returns an object instead of an array would otherwise throw
    // `TypeError: results is not iterable` — nothing validates the parsed JSON.
    const rawResults = Array.isArray(data?.results) ? data.results : [];

    for (const raw of rawResults) {
      const r = raw as RawResult;
      if (typeof r?.itemRef !== 'number' || !validRefs.has(r.itemRef)) continue;

      if (!hasValidVerdicts(r.verdicts)) {
        // A missing/malformed verdicts object means the item was never
        // actually judged. Reporting that as PASS would be silent
        // information loss of the worst kind: a teacher approving an item
        // believing it was checked. PENDING already means "not judged".
        byRef.set(r.itemRef, { itemRef: r.itemRef, state: 'PENDING', issues: [], suggestedFix: null });
        continue;
      }

      const categories = Object.keys(CATEGORY_ISSUE_CODE) as (keyof RawVerdicts)[];
      const verdicts = categories.map((c) => r.verdicts[c]);
      const issues: ValidationIssue[] = Array.isArray(r.issues) ? [...r.issues] : [];
      const hasFail = verdicts.includes('FAIL');
      const hasWarn = verdicts.includes('WARN');
      const suggestedFix = normalizeSuggestedFix(r.suggestedFix);

      let state: ValidationState;
      if (hasFail && suggestedFix) {
        state = 'FAIL';
      } else if (hasFail && !suggestedFix) {
        // Downgrade: a FAIL the teacher can't act on (no correction) must not
        // block approval, but the model's own explanation must survive the
        // downgrade. The model's issue(s) are already in `issues` verbatim;
        // preserve them rather than dropping or rewriting the message.
        state = 'WARN';
        if (issues.length === 0) {
          // The model gave no issue at all — a downgrade must never destroy
          // the reason for it. Synthesize one per failing category so the
          // teacher learns at least what kind of problem was found.
          for (const category of categories) {
            if (r.verdicts[category] === 'FAIL') {
              issues.push({
                code: CATEGORY_ISSUE_CODE[category],
                message: `Model flagged ${category} as FAIL but supplied no explanation.`,
              });
            }
          }
        }
      } else if (hasWarn) {
        state = 'WARN';
      } else {
        state = 'PASS';
      }

      // A duplicate itemRef overwrites the earlier entry but keeps its
      // original position (Map.set preserves insertion order on update).
      byRef.set(r.itemRef, {
        itemRef: r.itemRef,
        state,
        issues,
        suggestedFix,
      });
    }

    const results: ItemValidationResult[] = [...byRef.values()];
    for (const item of req.items) {
      if (!byRef.has(item.itemRef)) {
        results.push({ itemRef: item.itemRef, state: 'PENDING', issues: [], suggestedFix: null });
      }
    }

    return { results, meta };
  }
}
