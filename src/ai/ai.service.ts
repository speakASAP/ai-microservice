import { Injectable, Logger, Optional } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { spawn } from 'child_process';
import { writeFileSync, unlinkSync, existsSync, openSync, closeSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { randomUUID } from 'crypto';
import type { AiCompleteRequestInput } from '../contracts';
import { LoggingClient } from '../claude-code/logging.client';
import { AiAgent } from '../database/entities/ai-agent.entity';
import { Repository } from 'typeorm';

const LITELLM_TIMEOUT_MS = Number(process.env.LITELLM_TIMEOUT_MS || 120_000);

type AiCompleteRouter = 'auto' | 'litellm' | 'claude_cli' | 'claude_cli_with_litellm_fallback';

function litellmBaseUrl(): string {
  return (process.env.LITELLM_BASE_URL ?? '').replace(/\/$/, '');
}

function isLitellmConfigured(): boolean {
  return litellmBaseUrl().length > 0 && Boolean(process.env.LITELLM_MASTER_KEY);
}

function litellmChatCompletionsUrl(): string {
  const base = litellmBaseUrl();
  return base.endsWith('/v1') ? `${base}/chat/completions` : `${base}/v1/chat/completions`;
}

function resolveRouterMode(): AiCompleteRouter {
  const raw = (process.env.AI_COMPLETE_ROUTER ?? 'auto').trim();
  if (raw === 'litellm' || raw === 'claude_cli' || raw === 'claude_cli_with_litellm_fallback') {
    return raw;
  }
  return 'auto';
}

const CC_FALLBACK_ERROR_CODES = new Set(['CLI_FAILED', 'RATE_LIMIT', 'MODEL_ERROR']);

function shouldFallbackFromCc(errorCode: string | undefined): boolean {
  return errorCode !== undefined && CC_FALLBACK_ERROR_CODES.has(errorCode);
}

const DEFAULT_CC_MODEL = 'sonnet';
const CC_CLI = process.env.CC_CLI_PATH || '/home/ssf/.local/bin/claude';
const CC_TIMEOUT_MS = Number(process.env.CC_CLI_TIMEOUT_MS || 120_000);
const CC_MAX_CONCURRENT = Number(process.env.CC_MAX_CONCURRENT || 2);

interface LiteLLMResponse {
  choices: Array<{ message: { content: string } }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
  model?: string;
}

interface CcJsonResult {
  result?: string;
  is_error?: boolean;
  api_error_status?: number;
  usage?: { input_tokens?: number; output_tokens?: number };
  modelUsage?: Record<string, { inputTokens?: number; outputTokens?: number }>;
}

function parseCcStdout(stdout: string): CcJsonResult | null {
  const trimmed = stdout.trim();
  if (!trimmed) return null;
  try {
    return JSON.parse(trimmed) as CcJsonResult;
  } catch {
    return null;
  }
}

function isCcEnvelope(cc: CcJsonResult): boolean {
  return typeof cc.result === 'string'
    || cc.is_error === true
    || typeof cc.api_error_status === 'number'
    || cc.usage !== undefined
    || cc.modelUsage !== undefined;
}

function ccApiErrorResult(model: string, cc: CcJsonResult): AiCompleteResult {
  const message = (cc.result ?? '').trim()
    || `Claude API error (HTTP ${cc.api_error_status ?? 'unknown'})`;
  let error_code = 'CLI_FAILED';
  if (cc.api_error_status === 429) {
    error_code = 'RATE_LIMIT';
  } else if (cc.is_error) {
    error_code = 'MODEL_ERROR';
  }
  return {
    text: '',
    model_used: `claude-${model}`,
    inputTokens: 0,
    outputTokens: 0,
    token_usage_estimate: 0,
    error_code,
    error_message: message.slice(0, 500),
  };
}

export type AiCompleteResult = Record<string, unknown> & {
  text: string;
  model_used: string;
  inputTokens: number;
  outputTokens: number;
  token_usage_estimate: number;
  error_code?: string;
  error_message?: string;
};

interface AgentRouteAudit {
  agent_id: string;
  agent_slug: string;
  agent_name: string;
  agent_service_scope: string;
}

interface ResolvedAgentRoute {
  dto: AiCompleteRequestInput;
  audit?: AgentRouteAudit;
  error?: AiCompleteResult;
}

function cliFailureResult(model: string, detail: string, errorCode = 'CLI_FAILED'): AiCompleteResult {
  return {
    text: '',
    model_used: `claude-${model}`,
    inputTokens: 0,
    outputTokens: 0,
    token_usage_estimate: 0,
    error_code: errorCode,
    error_message: detail.slice(0, 500),
  };
}

function litellmErrorResult(model: string, status: number, detail: string): AiCompleteResult {
  let error_code = 'AI_SERVICE_ERROR';
  if (status === 429) error_code = 'RATE_LIMIT';
  else if (status === 401 || status === 403) error_code = 'AI_AUTH_ERROR';
  return {
    text: '',
    model_used: model,
    inputTokens: 0,
    outputTokens: 0,
    token_usage_estimate: 0,
    error_code,
    error_message: detail.slice(0, 500),
  };
}

function resolveBusinessId(dto: AiCompleteRequestInput): string | undefined {
  return dto.business_id ?? dto.businessId;
}

function spreadParsedJson(rawText: string, outputSchema?: unknown): { parsedData: Record<string, unknown>; text: string } {
  let parsedData: Record<string, unknown> = {};
  const trimmed = rawText.trim();
  if (trimmed.startsWith('{') || trimmed.startsWith('[') || outputSchema) {
    try {
      const cleaned = trimmed.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim();
      const parsed = JSON.parse(cleaned) as unknown;
      if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
        parsedData = parsed as Record<string, unknown>;
      } else if (Array.isArray(parsed)) {
        parsedData = { data: parsed };
      }
    } catch {
      // Not JSON — callers read .text
    }
  }
  return { parsedData, text: rawText };
}

@Injectable()
export class AiService {
  private readonly logger = new Logger(AiService.name);
  private activeProcesses = 0;

  constructor(
    private readonly loggingClient: LoggingClient,
    @Optional()
    @InjectRepository(AiAgent)
    private readonly agents?: Repository<AiAgent>,
  ) {}

  async complete(dto: AiCompleteRequestInput): Promise<AiCompleteResult> {
    const resolved = await this.resolveAgentRoute(dto);
    if (resolved.error) {
      return resolved.error;
    }
    const effectiveDto = resolved.dto;

    if (effectiveDto.model_tier === 'premium') {
      return this.withAgentAudit(
        litellmErrorResult('premium', 403, 'Premium tier requires explicit human approval per call'),
        resolved.audit,
      );
    }

    const router = resolveRouterMode();
    const litellmReady = isLitellmConfigured();

    if (router === 'litellm') {
      if (!litellmReady) {
        return this.withAgentAudit(
          litellmErrorResult(effectiveDto.model_tier ?? 'free', 503, 'LiteLLM not configured (LITELLM_BASE_URL + LITELLM_MASTER_KEY required)'),
          resolved.audit,
        );
      }
      return this.withAgentAudit(await this.completeViaLiteLLM(effectiveDto, resolved.audit), resolved.audit);
    }

    if (router === 'claude_cli') {
      return this.withAgentAudit(await this.completeViaCcCli(effectiveDto, resolved.audit), resolved.audit);
    }

    if (router === 'claude_cli_with_litellm_fallback') {
      const ccResult = await this.completeViaCcCli(effectiveDto, resolved.audit);
      if (!ccResult.error_code) {
        return this.withAgentAudit(ccResult, resolved.audit);
      }
      if (litellmReady && shouldFallbackFromCc(ccResult.error_code)) {
        this.logger.warn(
          `CC CLI ${ccResult.error_code} — falling back to LiteLLM tier=${effectiveDto.model_tier ?? 'free'}`,
        );
        return this.withAgentAudit(await this.completeViaLiteLLM(effectiveDto, resolved.audit), resolved.audit);
      }
      return this.withAgentAudit(ccResult, resolved.audit);
    }

    // auto: LiteLLM when configured (K8s default); CC only when LiteLLM unset; CC→LiteLLM on CLI failure
    if (litellmReady) {
      return this.withAgentAudit(await this.completeViaLiteLLM(effectiveDto, resolved.audit), resolved.audit);
    }

    const ccResult = await this.completeViaCcCli(effectiveDto, resolved.audit);
    if (!ccResult.error_code) {
      return this.withAgentAudit(ccResult, resolved.audit);
    }
    this.logger.warn('LITELLM_BASE_URL unset — CC CLI is the only backend');
    return this.withAgentAudit(ccResult, resolved.audit);
  }

  private async completeViaLiteLLM(dto: AiCompleteRequestInput, audit?: AgentRouteAudit): Promise<AiCompleteResult> {
    const model = dto.model_tier ?? 'free';
    const messages: Array<{ role: string; content: string }> = [];
    let userContent = dto.user_prompt;
    if (dto.system_prompt) {
      userContent = `${dto.system_prompt.trim()}\n\n${dto.user_prompt}`;
    }
    if (dto.output_schema) {
      userContent = 'Respond with valid JSON only. No markdown fences.\n\n' + userContent;
    }
    messages.push({ role: 'user', content: userContent });

    const requestBody: Record<string, unknown> = {
      model,
      messages,
      temperature: 0.2,
    };
    if (dto.max_tokens) {
      requestBody.max_tokens = dto.max_tokens;
    }
    if (dto.output_schema) {
      requestBody.response_format = { type: 'json_object' };
    }

    let res: Response;
    try {
      res = await fetch(litellmChatCompletionsUrl(), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${process.env.LITELLM_MASTER_KEY ?? ''}`,
        },
        body: JSON.stringify(requestBody),
        signal: AbortSignal.timeout(LITELLM_TIMEOUT_MS),
      });
    } catch (err: unknown) {
      const detail = err instanceof Error ? err.message : String(err);
      if (detail.includes('timeout') || detail.includes('aborted')) {
        throw new Error(`AI_HTTP_TIMEOUT: LiteLLM did not respond within ${LITELLM_TIMEOUT_MS}ms`);
      }
      this.emitTelemetry(dto.correlation_id, resolveBusinessId(dto), model, 0, 0, audit);
      return litellmErrorResult(model, 0, `LiteLLM unreachable: ${detail}`);
    }

    if (!res.ok) {
      const errText = await res.text();
      this.logger.error(`LiteLLM error ${res.status}: ${errText.slice(0, 300)}`);
      this.emitTelemetry(dto.correlation_id, resolveBusinessId(dto), model, 0, 0, audit);
      return litellmErrorResult(model, res.status, errText || `HTTP ${res.status}`);
    }

    const body = (await res.json()) as LiteLLMResponse;
    const rawText = body.choices?.[0]?.message?.content ?? '';
    const inputTokens = body.usage?.prompt_tokens ?? 0;
    const outputTokens = body.usage?.completion_tokens ?? 0;
    const { parsedData } = spreadParsedJson(rawText, dto.output_schema);

    this.emitTelemetry(dto.correlation_id, resolveBusinessId(dto), body.model ?? model, inputTokens, outputTokens, audit);
    return {
      ...parsedData,
      text: rawText,
      model_used: body.model ?? model,
      inputTokens,
      outputTokens,
      token_usage_estimate: inputTokens + outputTokens,
    };
  }

  private async completeViaCcCli(dto: AiCompleteRequestInput, audit?: AgentRouteAudit): Promise<AiCompleteResult> {
    const model = DEFAULT_CC_MODEL;

    let fullPrompt = '';
    if (dto.system_prompt) {
      fullPrompt += dto.system_prompt.trim() + '\n\n';
    }
    if (dto.output_schema) {
      fullPrompt += 'Respond with valid JSON only. No markdown fences.\n\n';
    }
    fullPrompt += dto.user_prompt;

    if (this.activeProcesses >= CC_MAX_CONCURRENT) {
      this.logger.warn(`claude CLI concurrency limit reached (${CC_MAX_CONCURRENT} active); rejecting request`);
      this.emitTelemetry(dto.correlation_id, resolveBusinessId(dto), `claude-${model}`, 0, 0, audit);
      return cliFailureResult(model, `claude CLI concurrency limit reached (${CC_MAX_CONCURRENT} active)`);
    }

    const tmpFile = join(tmpdir(), `ai-complete-${randomUUID()}.txt`);
    writeFileSync(tmpFile, fullPrompt, 'utf-8');

    let stdinFd: number | undefined;
    try {
      stdinFd = openSync(tmpFile, 'r');
    } catch {
      stdinFd = undefined;
    }

    this.activeProcesses++;
    let stdout = '';
    try {
      stdout = await this.spawnCcCli(stdinFd);
    } catch (err: unknown) {
      const detail = err instanceof Error ? err.message : String(err);
      if (detail.startsWith('AI_HTTP_TIMEOUT')) {
        throw new Error(detail);
      }
      this.logger.error(`claude CLI failed: ${detail.slice(0, 300)}`);
      this.emitTelemetry(dto.correlation_id, resolveBusinessId(dto), `claude-${model}`, 0, 0, audit);
      return cliFailureResult(model, `claude CLI failed: ${detail}`);
    } finally {
      this.activeProcesses--;
      if (stdinFd !== undefined) { try { closeSync(stdinFd); } catch { /* ignore */ } }
      try { if (existsSync(tmpFile)) unlinkSync(tmpFile); } catch { /* ignore */ }
    }

    let rawText = '';
    let inputTokens = 0;
    let outputTokens = 0;
    try {
      const ccResult = parseCcStdout(stdout) as CcJsonResult;
      if (!ccResult) {
        throw new Error('not json');
      }
      if (ccResult.is_error || ccResult.api_error_status) {
        this.emitTelemetry(dto.correlation_id, resolveBusinessId(dto), `claude-${model}`, 0, 0, audit);
        return ccApiErrorResult(model, ccResult);
      }
      rawText = isCcEnvelope(ccResult) ? (ccResult.result ?? '') : stdout.trim();
      if (ccResult.usage) {
        inputTokens = ccResult.usage.input_tokens ?? 0;
        outputTokens = ccResult.usage.output_tokens ?? 0;
      } else if (ccResult.modelUsage) {
        const first = Object.values(ccResult.modelUsage)[0];
        if (first) {
          inputTokens = first.inputTokens ?? 0;
          outputTokens = first.outputTokens ?? 0;
        }
      }
    } catch {
      rawText = stdout.trim();
    }

    const { parsedData } = spreadParsedJson(rawText, dto.output_schema);
    this.emitTelemetry(dto.correlation_id, resolveBusinessId(dto), `claude-${model}`, inputTokens, outputTokens, audit);
    return {
      ...parsedData,
      text: rawText,
      model_used: `claude-${model}`,
      inputTokens,
      outputTokens,
      token_usage_estimate: inputTokens + outputTokens,
    };
  }

  private spawnCcCli(stdinFd: number | undefined): Promise<string> {
    return new Promise<string>((resolve, reject) => {
      const child = spawn(
        CC_CLI,
        ['--print', '--output-format', 'json', '--model', DEFAULT_CC_MODEL],
        {
          stdio: [stdinFd !== undefined ? stdinFd : 'pipe', 'pipe', 'pipe'],
          env: { ...process.env, CLAUDE_CONFIG_DIR: process.env.CLAUDE_CONFIG_DIR || '/home/ssf/.claude' },
        },
      );

      const stdoutChunks: Buffer[] = [];
      const stderrChunks: Buffer[] = [];
      child.stdout?.on('data', (chunk: Buffer) => stdoutChunks.push(chunk));
      child.stderr?.on('data', (chunk: Buffer) => stderrChunks.push(chunk));

      const timer = setTimeout(() => {
        child.kill('SIGTERM');
        const killTimer = setTimeout(() => { try { child.kill('SIGKILL'); } catch { /* already dead */ } }, 5000);
        child.once('close', () => clearTimeout(killTimer));
        reject(new Error(`AI_HTTP_TIMEOUT: claude CLI did not respond within ${CC_TIMEOUT_MS}ms`));
      }, CC_TIMEOUT_MS);

      child.on('close', (code, signal) => {
        clearTimeout(timer);
        const out = Buffer.concat(stdoutChunks).toString('utf-8');
        const err = Buffer.concat(stderrChunks).toString('utf-8');
        if (signal === 'SIGKILL' || signal === 'SIGTERM') {
          reject(new Error(`claude CLI killed by signal ${signal}: ${err.slice(0, 200)}`));
        } else if (code !== 0) {
          const ccFromStdout = parseCcStdout(out);
          if (ccFromStdout && (ccFromStdout.is_error || ccFromStdout.api_error_status)) {
            resolve(out);
          } else {
            reject(new Error(err.trim() || out.trim() || `claude CLI exited with code ${code}`));
          }
        } else {
          resolve(out);
        }
      });

      child.on('error', (err) => {
        clearTimeout(timer);
        reject(err);
      });
    });
  }

  private emitTelemetry(
    correlationId: string | undefined,
    businessId: string | undefined,
    modelUsed: string,
    inputTokens: number,
    outputTokens: number,
    audit?: AgentRouteAudit,
  ): void {
    this.loggingClient
      .log('info', 'ai_complete', {
        correlation_id: correlationId,
        business_id: businessId,
        model_used: modelUsed,
        inputTokens,
        outputTokens,
        token_usage_estimate: inputTokens + outputTokens,
        agent_id: audit?.agent_id,
        agent_slug: audit?.agent_slug,
        agent_service_scope: audit?.agent_service_scope,
        compression: { rtk: true, caveman: 'lite' },
      })
      .catch(() => { /* logging must never crash the service */ });
  }

  private async resolveAgentRoute(dto: AiCompleteRequestInput): Promise<ResolvedAgentRoute> {
    const slug = cleanOptionalString(dto.agent_slug);
    if (!slug) {
      return { dto };
    }
    if (!this.agents) {
      return {
        dto,
        error: agentRoutingError(slug, 'AGENT_ROUTING_UNAVAILABLE', 'Agent registry routing is not configured'),
      };
    }

    const agent = await this.agents.findOne({ where: { slug } });
    if (!agent || agent.status !== 'active') {
      return {
        dto,
        error: agentRoutingError(slug, 'AGENT_NOT_AVAILABLE', 'Active agent definition not found'),
      };
    }

    const requestedScope = cleanOptionalString(dto.agent_service_scope);
    if (requestedScope && agent.serviceScope !== requestedScope) {
      return {
        dto,
        error: agentRoutingError(slug, 'AGENT_SCOPE_MISMATCH', 'Agent is not registered for the requested service scope', {
          agent_id: agent.id,
          agent_slug: agent.slug,
          agent_name: agent.name,
          agent_service_scope: agent.serviceScope,
        }),
      };
    }

    if (agent.routePath && agent.routePath !== '/ai/complete') {
      return {
        dto,
        error: agentRoutingError(slug, 'AGENT_ROUTE_MISMATCH', 'Agent is not registered for /ai/complete', {
          agent_id: agent.id,
          agent_slug: agent.slug,
          agent_name: agent.name,
          agent_service_scope: agent.serviceScope,
        }),
      };
    }

    const audit: AgentRouteAudit = {
      agent_id: agent.id,
      agent_slug: agent.slug,
      agent_name: agent.name,
      agent_service_scope: agent.serviceScope,
    };

    return {
      audit,
      dto: {
        ...dto,
        model_tier: agent.modelTier,
        system_prompt: agent.systemPrompt || dto.system_prompt,
        user_prompt: renderAgentPrompt(agent.userPromptTemplate, dto.user_prompt),
        output_schema: agent.outputSchema || dto.output_schema,
        max_tokens: agent.maxTokens || dto.max_tokens,
      },
    };
  }

  private withAgentAudit(result: AiCompleteResult, audit?: AgentRouteAudit): AiCompleteResult {
    if (!audit) return result;
    return { ...result, ...audit };
  }
}

function cleanOptionalString(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined;
  const cleaned = value.trim();
  return cleaned || undefined;
}

function renderAgentPrompt(template: string | undefined, userPrompt: string): string {
  const cleaned = (template ?? '').trim();
  if (!cleaned) return userPrompt;
  return cleaned
    .split('{{user_prompt}}').join(userPrompt)
    .split('{{input}}').join(userPrompt)
    .split('{{prompt}}').join(userPrompt);
}

function agentRoutingError(
  slug: string,
  error_code: string,
  error_message: string,
  audit?: AgentRouteAudit,
): AiCompleteResult {
  return {
    ...(audit || { agent_slug: slug }),
    text: '',
    model_used: 'agent-registry',
    inputTokens: 0,
    outputTokens: 0,
    token_usage_estimate: 0,
    error_code,
    error_message,
  };
}
