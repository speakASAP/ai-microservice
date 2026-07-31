import { Injectable } from '@nestjs/common';
import { LlmClient } from './llm.client';
import { VALIDATE_SYSTEM_PROMPT, buildValidateUserPrompt } from './validate.prompt';
import { VALIDATE_OUTPUT_SCHEMA } from './validate.schema';
import {
  ItemValidationResult,
  ValidateDrillRequest,
  ValidateDrillResponse,
  ValidationIssue,
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
  suggestedFix: ItemValidationResult['suggestedFix'];
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

    for (const raw of data.results ?? []) {
      const r = raw as RawResult;
      if (typeof r?.itemRef !== 'number' || !validRefs.has(r.itemRef)) continue;

      const verdicts = [r.verdicts?.topicAlignment, r.verdicts?.grammar, r.verdicts?.level, r.verdicts?.naturalness];
      const issues: ValidationIssue[] = Array.isArray(r.issues) ? [...r.issues] : [];
      const hasFail = verdicts.includes('FAIL');
      const hasWarn = verdicts.includes('WARN');

      let state: ValidationState;
      if (hasFail && r.suggestedFix) {
        state = 'FAIL';
      } else if (hasFail && !r.suggestedFix) {
        // Downgrade: a FAIL the teacher can't act on (no correction) must not
        // block approval, but the model's own explanation must survive the
        // downgrade. The model's issue(s) are already in `issues` verbatim;
        // preserve them rather than dropping or rewriting the message.
        state = 'WARN';
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
        suggestedFix: r.suggestedFix ?? null,
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
