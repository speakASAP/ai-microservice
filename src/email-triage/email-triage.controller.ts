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
    const t0 = Date.now();
    const { payload, error, escalation_reason } = this.emailTriageService.validateAndNormalize(body);
    const duration_ms = Date.now() - t0;
    if (error) {
      throw new HttpException({ error, escalation_reason: escalation_reason ?? 'incomplete_data' }, HttpStatus.BAD_REQUEST);
    }
    return { success: true, payload, duration_ms, model_used: 'schema-validator' };
  }

  @Post('classify')
  @HttpCode(HttpStatus.OK)
  async classify(@Body() body: Record<string, unknown>) {
    const t0 = Date.now();
    const payload = (body.payload && typeof body.payload === 'object' && !Array.isArray(body.payload)
      ? body.payload
      : body) as Record<string, unknown>;
    const msgId = payload.message_id != null ? String(payload.message_id) : undefined;

    const useLlmRaw = body.use_llm;
    const useLlm = useLlmRaw !== undefined ? coerceUseLlm(useLlmRaw) : this.llmClassifierDefault;

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
    const out: Record<string, unknown> = {
      success: true,
      intent: classifyResult.intent,
      confidence: classifyResult.confidence,
      raw_scores: classifyResult.raw_scores ?? null,
      model_used: classifyResult.model_used ?? 'rule-based',
      duration_ms,
    };
    if (classifyResult.llm_output != null) out.llm_output = classifyResult.llm_output;
    if (llmFallbackReason && useLlm) out.llm_fallback_reason = llmFallbackReason.slice(0, 500);
    return out;
  }

  @Post('extract')
  @HttpCode(HttpStatus.OK)
  extract(@Body() body: Record<string, unknown>) {
    const payload = (body.payload && typeof body.payload === 'object' && !Array.isArray(body.payload)
      ? body.payload
      : body) as Record<string, unknown>;
    const intent = typeof body.intent === 'string' ? body.intent : undefined;
    const result = this.emailTriageService.extractPayload(payload, intent);
    return { success: true, model_used: 'pattern-extractor', ...result };
  }

  @Post('decide')
  @HttpCode(HttpStatus.OK)
  async decide(@Body() body: Record<string, unknown>) {
    const intent = body.intent;
    const confidence = body.confidence;
    if (!intent || typeof intent !== 'string') {
      throw new HttpException({ error: 'intent is required', escalation_reason: 'incomplete_data' }, HttpStatus.BAD_REQUEST);
    }
    if (confidence == null || (typeof confidence !== 'number' && typeof confidence !== 'string')) {
      throw new HttpException({ error: 'confidence is required', escalation_reason: 'incomplete_data' }, HttpStatus.BAD_REQUEST);
    }
    const confFloat = parseFloat(String(confidence));
    const entities = body.entities && typeof body.entities === 'object' && !Array.isArray(body.entities)
      ? (body.entities as Record<string, unknown>)
      : undefined;

    const useLlmRaw = body.use_llm;
    const useLlm = useLlmRaw !== undefined ? coerceUseLlm(useLlmRaw) : this.llmDeciderDefault;

    let result: Record<string, unknown> | null = null;
    let llmFallbackReason: string | null = null;

    if (useLlm) {
      try {
        const text = JSON.stringify({ intent, confidence: confFloat, entities: entities ?? {} });
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
        this.logger.warn('Email-triage LLM decide failed, falling back to rule-based', { intent, error: llmFallbackReason });
      }
    }

    const decideResult: Record<string, unknown> = result ?? (this.emailTriageService.decideAction(intent, confFloat, undefined, entities) as unknown as Record<string, unknown>);

    const out: Record<string, unknown> = {
      success: true,
      model_used: (decideResult.model_used as string | undefined) ?? 'rule-based',
      ...decideResult,
    };
    if (decideResult.llm_output != null) out.llm_output = decideResult.llm_output;
    if (llmFallbackReason && useLlm) out.llm_fallback_reason = llmFallbackReason.slice(0, 500);
    return out;
  }
}
