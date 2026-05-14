import { Injectable, BadRequestException, InternalServerErrorException } from '@nestjs/common';
import type { TaskDraftDto } from './dto/task-draft.dto';
import type { TaskDraftResponse, TaskPriority } from './dto/task-draft-response.dto';

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

  async draftTask(dto: TaskDraftDto): Promise<TaskDraftResponse> {
    if (!dto.transcript && !dto.textNote) {
      throw new BadRequestException('At least one of transcript or textNote is required');
    }

    // Read at call time so tests can control the env var per-test
    const litellmBaseUrl = process.env.LITELLM_BASE_URL;
    const litellmApiKey = process.env.LITELLM_MASTER_KEY ?? '';

    if (!litellmBaseUrl) {
      throw new InternalServerErrorException('LITELLM_BASE_URL is not configured');
    }

    const userContent = [
      dto.transcript ? `Transcript: ${dto.transcript}` : '',
      dto.textNote ? `Note: ${dto.textNote}` : '',
    ].filter(Boolean).join('\n');

    const res = await fetch(`${litellmBaseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${litellmApiKey}`,
      },
      body: JSON.stringify({
        model: 'smart',
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          { role: 'user', content: userContent },
        ],
        temperature: 0.3,
      }),
      signal: AbortSignal.timeout(30_000),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new InternalServerErrorException(`LiteLLM error: ${errText}`);
    }

    const body = await res.json() as { choices: { message: { content: string } }[] };
    const raw = body.choices?.[0]?.message?.content ?? '';

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
      // AI returned non-JSON — fall back to raw text as description
      return {
        title: (dto.transcript ?? dto.textNote ?? 'Nový úkol').slice(0, 80),
        description: raw || dto.transcript || dto.textNote || '',
        priority: 'normal',
        modelTier: 'smart',
      };
    }
  }
}
