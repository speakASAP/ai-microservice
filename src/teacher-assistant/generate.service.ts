import { Injectable } from '@nestjs/common';
import { LlmClient } from './llm.client';
import { GENERATE_SYSTEM_PROMPT, buildGenerateUserPrompt } from './generate.prompt';
import { GENERATE_OUTPUT_SCHEMA } from './generate.schema';
import { GenerateDrillRequest, GenerateDrillResponse, GeneratedDrillItem } from './contracts';

@Injectable()
export class GenerateService {
  constructor(private readonly llm: LlmClient) {}

  async generate(req: GenerateDrillRequest): Promise<GenerateDrillResponse> {
    const { data, meta } = await this.llm.completeJson<{ items: unknown[] }>({
      systemPrompt: GENERATE_SYSTEM_PROMPT,
      userPrompt: buildGenerateUserPrompt(req),
      outputSchema: GENERATE_OUTPUT_SCHEMA,
      correlationId: req.correlationId,
    });

    const items: GeneratedDrillItem[] = [];
    // A model that returns an object instead of an array would otherwise throw
    // `TypeError: items is not iterable` — nothing validates the parsed JSON.
    const rawItems = Array.isArray(data?.items) ? data.items : [];

    for (const raw of rawItems) {
      const r = raw as Record<string, any>;
      if (typeof r.template !== 'string' || !Array.isArray(r.blanks)) continue;
      items.push({
        template: r.template,
        blanks: r.blanks.map((b: any, index: number) => ({
          index,
          prompt: String(b?.prompt ?? ''),
          answer: String(b?.answer ?? ''),
          alternatives: Array.isArray(b?.alternatives) ? b.alternatives.map(String) : [],
        })),
        hint: typeof r.hint === 'string' ? r.hint : null,
        topicSlug: String(r.topicSlug ?? ''),
        newWords: Array.isArray(r.newWords) ? r.newWords.map(String) : [],
      });
    }

    return { items, meta };
  }
}
