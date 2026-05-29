import { Injectable, BadRequestException } from '@nestjs/common';
import { AiService } from '../ai/ai.service';
import type { TaskDraftRequestInput, TaskDraftResponseInput } from '../contracts';

type TaskPriority = 'low' | 'normal' | 'high';

const SYSTEM_PROMPT = `You are a school task assistant. Given a voice transcript and/or text note from a teacher, extract and reformat into a structured volunteer task for parents.
Output ONLY valid JSON (no markdown, no explanation): { "title": string, "description": string, "priority": "low"|"normal"|"high", "deadline"?: string }
- title: concise, max 80 chars
- description: clear, actionable, 2-5 sentences
- priority: "low" | "normal" | "high"
- deadline: ISO date string (YYYY-MM-DD) if mentioned, omit otherwise
- Language: match input language (default Czech)`;

function toTaskPriority(value: unknown): TaskPriority {
  return (['low', 'normal', 'high'] as const).includes(value as TaskPriority) ? (value as TaskPriority) : 'normal';
}

@Injectable()
export class TaskService {
  constructor(private readonly aiService: AiService) {}

  async draftTask(dto: TaskDraftRequestInput): Promise<TaskDraftResponseInput> {
    if (!dto.transcript && !dto.textNote) {
      throw new BadRequestException('At least one of transcript or textNote is required');
    }

    const userContent = [
      dto.transcript ? `Transcript: ${dto.transcript}` : '',
      dto.textNote ? `Note: ${dto.textNote}` : '',
    ].filter(Boolean).join('\n');

    const result = await this.aiService.complete({
      model_tier: 'smart',
      system_prompt: SYSTEM_PROMPT,
      user_prompt: userContent,
      max_tokens: 512,
      output_schema: { type: 'object' },
    });

    const raw = result.text ?? '';

    try {
      const parsed = JSON.parse(raw) as { title?: string; description?: string; priority?: string; deadline?: string };
      return {
        title: parsed.title ?? dto.transcript?.slice(0, 80) ?? 'Nový úkol',
        description: parsed.description ?? dto.transcript ?? dto.textNote ?? '',
        priority: toTaskPriority(parsed.priority),
        deadline: parsed.deadline,
        modelTier: 'smart',
      };
    } catch {
      return {
        title: (dto.transcript ?? dto.textNote ?? 'Nový úkol').slice(0, 80),
        description: raw || dto.transcript || dto.textNote || '',
        priority: 'normal',
        modelTier: 'smart',
      };
    }
  }
}
