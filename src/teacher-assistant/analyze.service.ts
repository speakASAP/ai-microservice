import { Injectable } from '@nestjs/common';
import { LlmClient } from './llm.client';
import { ANALYZE_SYSTEM_PROMPT, buildAnalyzeUserPrompt } from './analyze.prompt';
import { ANALYZE_OUTPUT_SCHEMA } from './analyze.schema';
import { AnalyzeErrorsRequest, AnalyzeErrorsResponse, AnalyzedGapCluster } from './contracts';

/**
 * Clusters a student's drill mistakes into grammar gaps with the theory that closes them.
 *
 * Mirrors `GenerateService`: nothing validates the model's parsed JSON for us, so every
 * field is checked here. A cluster without a `topicSlug` cannot be filed under a topic and
 * is dropped — education-service's coercion needs something to coerce.
 */
@Injectable()
export class AnalyzeService {
  constructor(private readonly llm: LlmClient) {}

  async analyze(req: AnalyzeErrorsRequest): Promise<AnalyzeErrorsResponse> {
    const { data, meta } = await this.llm.completeJson<{ clusters: unknown[] }>({
      systemPrompt: ANALYZE_SYSTEM_PROMPT,
      userPrompt: buildAnalyzeUserPrompt(req),
      outputSchema: ANALYZE_OUTPUT_SCHEMA,
      correlationId: req.correlationId,
    });

    // A model that returns an object instead of an array would otherwise throw
    // `TypeError: data.clusters is not iterable` — nothing validates the parsed JSON.
    const raw = Array.isArray(data?.clusters) ? data.clusters : [];
    const clusters: AnalyzedGapCluster[] = [];

    for (const candidate of raw) {
      const c = candidate as Record<string, any>;
      const topicSlug = typeof c?.topicSlug === 'string' ? c.topicSlug.trim() : '';
      if (!topicSlug) {
        // Cannot be filed under a topic; the calling service needs something
        // real to coerce, not a default that hides the gap.
        continue;
      }

      clusters.push({
        topicSlug,
        title: String(c.title ?? ''),
        explanation: String(c.explanation ?? ''),
        rules: Array.isArray(c.rules) ? c.rules.map(String) : [],
        examples: Array.isArray(c.examples)
          ? c.examples
              .filter(
                (e: any) => typeof e?.text === 'string' && typeof e?.gloss === 'string',
              )
              .map((e: any) => ({ text: String(e.text), gloss: String(e.gloss) }))
          : [],
        answers: Array.isArray(c.answers) ? c.answers.map(String) : [],
      });
    }

    return { clusters, meta };
  }
}
