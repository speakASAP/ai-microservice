import {
  Controller,
  Post,
  Get,
  Body,
  HttpCode,
  HttpStatus,
  HttpException,
  Logger,
} from '@nestjs/common';
import { Public } from '../service-identity/public.decorator';
import { EmailTriageService } from './email-triage.service';
import { AiService } from '../ai/ai.service';
import {
  parseOrThrow,
  EmailIngestRequestSchema,
  EmailIngestResponseSchema,
  EmailClassifyRequestSchema,
  EmailClassifyResponseSchema,
  EmailExtractRequestSchema,
  EmailExtractResponseSchema,
  EmailDecideRequestSchema,
  EmailDecideResponseSchema,
} from '../contracts';

function coerceUseLlm(value: unknown): boolean {
  if (value == null) return false;
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') return ['true', '1', 'yes'].includes(value.trim().toLowerCase());
  if (typeof value === 'number') return Boolean(value);
  return false;
}

@Controller('api/email-triage')
@Public()
export class EmailTriageController {
  private readonly logger = new Logger(EmailTriageController.name);
  private readonly llmClassifierDefault: boolean;
  private readonly llmDeciderDefault: boolean;

  constructor(
    private readonly emailTriageService: EmailTriageService,
    private readonly aiService: AiService,
  ) {
    const toLlmFlag = (env: string | undefined) =>
      ['true', '1', 'yes'].includes((env ?? '').trim().toLowerCase());
    this.llmClassifierDefault = toLlmFlag(process.env.EMAIL_TRIAGE_LLM_CLASSIFIER);
    this.llmDeciderDefault = toLlmFlag(process.env.EMAIL_TRIAGE_LLM_DECIDER);
  }

  @Get('ready')
  ready() {
    return { ready: true, service: 'email-triage' };
  }

  @Post('ingest')
  @HttpCode(HttpStatus.OK)
  async ingest(@Body() body: unknown) {
    if (!body || typeof body !== 'object' || Array.isArray(body)) {
      throw new HttpException(
        { error: 'Payload must be an object', escalation_reason: 'incomplete_data' },
        HttpStatus.BAD_REQUEST,
      );
    }
    const validated = parseOrThrow(EmailIngestRequestSchema, body, 'email-triage.ingest.request');
    const t0 = Date.now();
    const { payload, error, escalation_reason } = this.emailTriageService.validateAndNormalize(validated);
    const duration_ms = Date.now() - t0;
    if (error) {
      throw new HttpException({ error, escalation_reason: escalation_reason ?? 'incomplete_data' }, HttpStatus.BAD_REQUEST);
    }
    return parseOrThrow(
      EmailIngestResponseSchema,
      { success: true as const, payload, duration_ms, model_used: 'schema-validator' },
      'email-triage.ingest.response',
    );
  }

  @Post('classify')
  @HttpCode(HttpStatus.OK)
  async classify(@Body() body: unknown) {
    const validated = parseOrThrow(EmailClassifyRequestSchema, body, 'email-triage.classify.request');
    const t0 = Date.now();
    const payload = (validated.payload && typeof validated.payload === 'object' && !Array.isArray(validated.payload)
      ? validated.payload
      : validated) as Record<string, unknown>;
    const msgId = payload.message_id != null ? String(payload.message_id) : undefined;

    const useLlm = validated.use_llm !== undefined ? coerceUseLlm(validated.use_llm) : this.llmClassifierDefault;

    let result: Record<string, unknown> | null = null;
    let llmFallbackReason: string | null = null;

    if (useLlm) {
      try {
        const text = this.emailTriageService.getEmailTextForLlm(payload);
        if (!text.trim()) throw new Error('Empty email text for LLM classify');
        const llmResult = await this.aiService.complete({
          model_tier: 'free',
          system_prompt:
            'You are an email classifier. Classify the email intent. Reply with JSON only: { "intent": string, "confidence": number (0-1), "raw_scores": { [intent]: number } }. Intents: support, sales, contract, technical, billing, spam, unknown, multi_intent.',
          user_prompt: text,
          max_tokens: 200,
        });
        const intent = llmResult.intent as string | undefined;
        const confidence = llmResult.confidence as number | undefined;
        if (intent && confidence != null && confidence >= 0 && confidence <= 1) {
          result = { intent, confidence, raw_scores: llmResult.raw_scores, model_used: llmResult.model_used, llm_output: { intent, confidence, raw_scores: llmResult.raw_scores } };
        } else {
          llmFallbackReason = 'LLM returned invalid structure';
        }
      } catch (e: unknown) {
        llmFallbackReason = e instanceof Error ? e.message : String(e);
        this.logger.warn('Email-triage LLM classify failed, falling back to rule-based', { msgId, error: llmFallbackReason });
      }
    }

    const classifyResult: Record<string, unknown> = result ?? (this.emailTriageService.classifyPayload(payload) as unknown as Record<string, unknown>);
    const duration_ms = Date.now() - t0;

    const out = {
      success: true as const,
      intent: classifyResult.intent,
      confidence: classifyResult.confidence,
      raw_scores: (classifyResult.raw_scores as Record<string, number> | null | undefined) ?? null,
      model_used: (classifyResult.model_used as string | undefined) ?? 'rule-based',
      duration_ms,
      ...(classifyResult.llm_output != null ? { llm_output: classifyResult.llm_output } : {}),
      ...(llmFallbackReason && useLlm ? { llm_fallback_reason: llmFallbackReason.slice(0, 500) } : {}),
    };
    return parseOrThrow(EmailClassifyResponseSchema, out, 'email-triage.classify.response');
  }

  @Post('extract')
  @HttpCode(HttpStatus.OK)
  extract(@Body() body: unknown) {
    const validated = parseOrThrow(EmailExtractRequestSchema, body, 'email-triage.extract.request');
    const payload = (validated.payload && typeof validated.payload === 'object' && !Array.isArray(validated.payload)
      ? validated.payload
      : validated) as Record<string, unknown>;
    const intent = typeof validated.intent === 'string' ? validated.intent : undefined;
    const result = this.emailTriageService.extractPayload(payload, intent);
    return parseOrThrow(
      EmailExtractResponseSchema,
      { success: true as const, model_used: 'pattern-extractor', ...result },
      'email-triage.extract.response',
    );
  }

  @Post('decide')
  @HttpCode(HttpStatus.OK)
  async decide(@Body() body: unknown) {
    const validated = parseOrThrow(EmailDecideRequestSchema, body, 'email-triage.decide.request');
    const confFloat = parseFloat(String(validated.confidence));
    const entities = validated.entities;

    const useLlm = validated.use_llm !== undefined ? coerceUseLlm(validated.use_llm) : this.llmDeciderDefault;

    let result: Record<string, unknown> | null = null;
    let llmFallbackReason: string | null = null;

    if (useLlm) {
      try {
        const text = JSON.stringify({ intent: validated.intent, confidence: confFloat, entities: entities ?? {} });
        const llmResult = await this.aiService.complete({
          model_tier: 'free',
          system_prompt:
            'You are an email action decider. Given intent, confidence, and entities, choose the action. Reply with JSON only: { "action": "auto_respond"|"route_to_queue"|"escalate", "escalation_reason": string|null, "queue": string|null }.',
          user_prompt: text,
          max_tokens: 200,
        });
        const action = llmResult.action as string | undefined;
        if (action && ['auto_respond', 'route_to_queue', 'escalate'].includes(action)) {
          result = { action, escalation_reason: llmResult.escalation_reason ?? null, queue: llmResult.queue ?? null, model_used: llmResult.model_used, llm_output: { action, escalation_reason: llmResult.escalation_reason, queue: llmResult.queue } };
        } else {
          llmFallbackReason = 'LLM returned invalid action';
        }
      } catch (e: unknown) {
        llmFallbackReason = e instanceof Error ? e.message : String(e);
        this.logger.warn('Email-triage LLM decide failed, falling back to rule-based', { intent: validated.intent, error: llmFallbackReason });
      }
    }

    const decideResult: Record<string, unknown> = result ?? (this.emailTriageService.decideAction(validated.intent, confFloat, undefined, entities) as unknown as Record<string, unknown>);

    const out = {
      success: true as const,
      model_used: (decideResult.model_used as string | undefined) ?? 'rule-based',
      action: decideResult.action,
      escalation_reason: (decideResult.escalation_reason as string | null | undefined) ?? null,
      queue: (decideResult.queue as string | null | undefined) ?? null,
      ...(decideResult.llm_output != null ? { llm_output: decideResult.llm_output } : {}),
      ...(llmFallbackReason && useLlm ? { llm_fallback_reason: llmFallbackReason.slice(0, 500) } : {}),
    };
    return parseOrThrow(EmailDecideResponseSchema, out, 'email-triage.decide.response');
  }
}
