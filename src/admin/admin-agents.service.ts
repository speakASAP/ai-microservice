import { BadRequestException, ConflictException, Injectable, NotFoundException, OnModuleInit } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { FindOptionsWhere, ILike, Repository } from 'typeorm';
import { AiAgent, AiAgentModelTier, AiAgentStatus } from '../database/entities/ai-agent.entity';

const STATUSES: AiAgentStatus[] = ['draft', 'active', 'disabled'];
const MODEL_TIERS: AiAgentModelTier[] = ['free', 'cheap', 'smart', 'premium'];

export interface AiAgentPayload {
  name?: unknown;
  slug?: unknown;
  description?: unknown;
  status?: unknown;
  serviceScope?: unknown;
  routePath?: unknown;
  modelTier?: unknown;
  providerModel?: unknown;
  temperature?: unknown;
  maxTokens?: unknown;
  systemPrompt?: unknown;
  userPromptTemplate?: unknown;
  outputSchema?: unknown;
  metadata?: unknown;
  tags?: unknown;
}

export interface AiAgentListQuery {
  q?: string;
  status?: AiAgentStatus;
  modelTier?: AiAgentModelTier;
  serviceScope?: string;
}

@Injectable()
export class AdminAgentsService implements OnModuleInit {
  constructor(
    @InjectRepository(AiAgent)
    private readonly agents: Repository<AiAgent>,
  ) {}

  async onModuleInit(): Promise<void> {
    await this.seedMissingDefaults();
  }

  async list(query: AiAgentListQuery): Promise<AiAgent[]> {
    const where: FindOptionsWhere<AiAgent>[] = [];
    const base: FindOptionsWhere<AiAgent> = {};

    if (query.status) base.status = query.status;
    if (query.modelTier) base.modelTier = query.modelTier;
    if (query.serviceScope) base.serviceScope = query.serviceScope;

    if (query.q) {
      const like = ILike(`%${query.q}%`);
      where.push({ ...base, name: like }, { ...base, slug: like }, { ...base, serviceScope: like });
    }

    return this.agents.find({
      where: where.length ? where : base,
      order: { updatedAt: 'DESC' },
      take: 250,
    });
  }

  async get(id: string): Promise<AiAgent> {
    const agent = await this.agents.findOne({ where: { id } });
    if (!agent) throw new NotFoundException('AI agent not found');
    return agent;
  }

  async create(payload: AiAgentPayload): Promise<AiAgent> {
    const data = this.normalizePayload(payload, true);
    if (!data.slug) throw new BadRequestException('Agent slug is required');
    await this.ensureSlugAvailable(data.slug);
    return this.agents.save(this.agents.create(data));
  }

  async update(id: string, payload: AiAgentPayload): Promise<AiAgent> {
    const agent = await this.get(id);
    const data = this.normalizePayload(payload, false);
    if (data.slug && data.slug !== agent.slug) {
      await this.ensureSlugAvailable(data.slug, id);
    }
    Object.assign(agent, data);
    return this.agents.save(agent);
  }

  async remove(id: string): Promise<void> {
    const agent = await this.get(id);
    await this.agents.remove(agent);
  }

  private async seedMissingDefaults(): Promise<void> {
    for (const defaultAgent of DEFAULT_AGENTS) {
      if (!defaultAgent.slug) continue;
      const existing = await this.agents.findOne({ where: { slug: defaultAgent.slug } });
      if (existing) continue;
      await this.agents.save(this.agents.create(defaultAgent));
    }
  }

  private async ensureSlugAvailable(slug: string, exceptId?: string): Promise<void> {
    const existing = await this.agents.findOne({ where: { slug } });
    if (existing && existing.id !== exceptId) {
      throw new ConflictException('Agent slug already exists');
    }
  }

  private normalizePayload(payload: AiAgentPayload, creating: boolean): Partial<AiAgent> {
    const name = cleanString(payload.name);
    const slug = slugify(cleanString(payload.slug) || name);
    const serviceScope = cleanString(payload.serviceScope);

    if (creating && !name) throw new BadRequestException('Agent name is required');
    if (creating && !serviceScope) throw new BadRequestException('Service scope is required');
    if ((payload.slug !== undefined || creating) && !slug) throw new BadRequestException('Agent slug is required');

    const status = payload.status === undefined ? undefined : enumValue(payload.status, STATUSES, 'status');
    const modelTier = payload.modelTier === undefined ? undefined : enumValue(payload.modelTier, MODEL_TIERS, 'modelTier');
    const temperature = payload.temperature === undefined ? undefined : numberInRange(payload.temperature, 0, 2, 'temperature');
    const maxTokens = payload.maxTokens === undefined ? undefined : integerInRange(payload.maxTokens, 1, 200000, 'maxTokens');

    const normalized: Partial<AiAgent> = {};

    if (payload.name !== undefined || creating) normalized.name = name;
    if (payload.slug !== undefined || creating) normalized.slug = slug;
    if (payload.serviceScope !== undefined || creating) normalized.serviceScope = serviceScope;
    if (!creating || payload.description !== undefined) normalized.description = nullableString(payload.description);
    if (status) normalized.status = status;
    if (modelTier) normalized.modelTier = modelTier;
    if (payload.routePath !== undefined) normalized.routePath = nullableString(payload.routePath);
    if (payload.providerModel !== undefined) normalized.providerModel = nullableString(payload.providerModel);
    if (temperature !== undefined) normalized.temperature = temperature.toFixed(2);
    if (maxTokens !== undefined) normalized.maxTokens = maxTokens;
    if (payload.systemPrompt !== undefined || creating) normalized.systemPrompt = cleanString(payload.systemPrompt);
    if (payload.userPromptTemplate !== undefined || creating) normalized.userPromptTemplate = cleanString(payload.userPromptTemplate);
    if (payload.outputSchema !== undefined) normalized.outputSchema = jsonObjectOrNull(payload.outputSchema, 'outputSchema');
    if (payload.metadata !== undefined) normalized.metadata = jsonObjectOrNull(payload.metadata, 'metadata');
    if (payload.tags !== undefined || creating) normalized.tags = normalizeTags(payload.tags);

    if (creating) {
      normalized.status ??= 'draft';
      normalized.modelTier ??= 'free';
      normalized.temperature ??= '0.20';
      normalized.maxTokens ??= 1000;
    }

    return normalized;
  }
}

function cleanString(value: unknown): string {
  return typeof value === 'string' ? value.trim() : '';
}

function nullableString(value: unknown): string | null {
  const cleaned = cleanString(value);
  return cleaned ? cleaned : null;
}

function slugify(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 160);
}

function enumValue<T extends string>(value: unknown, allowed: T[], field: string): T {
  if (typeof value === 'string' && allowed.includes(value as T)) return value as T;
  throw new BadRequestException(`${field} must be one of: ${allowed.join(', ')}`);
}

function numberInRange(value: unknown, min: number, max: number, field: string): number {
  const parsed = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(parsed) || parsed < min || parsed > max) {
    throw new BadRequestException(`${field} must be between ${min} and ${max}`);
  }
  return parsed;
}

function integerInRange(value: unknown, min: number, max: number, field: string): number {
  const parsed = numberInRange(value, min, max, field);
  if (!Number.isInteger(parsed)) throw new BadRequestException(`${field} must be an integer`);
  return parsed;
}

function jsonObjectOrNull(value: unknown, field: string): Record<string, unknown> | null {
  if (value === null || value === undefined || value === '') return null;
  if (typeof value === 'object' && !Array.isArray(value)) return value as Record<string, unknown>;
  throw new BadRequestException(`${field} must be a JSON object`);
}

function normalizeTags(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((tag) => cleanString(tag)).filter(Boolean).slice(0, 30);
  }
  if (typeof value === 'string') {
    return value.split(',').map((tag) => tag.trim()).filter(Boolean).slice(0, 30);
  }
  return [];
}

const DEFAULT_AGENTS: Array<Partial<AiAgent>> = [
  {
    name: 'AI Completion Gateway',
    slug: 'ai-completion-gateway',
    description: 'Central model-tier gateway used by services through POST /ai/complete.',
    status: 'active',
    serviceScope: 'ai-microservice',
    routePath: '/ai/complete',
    modelTier: 'free',
    temperature: '0.20',
    maxTokens: 1000,
    systemPrompt: '',
    userPromptTemplate: '{{user_prompt}}',
    tags: ['gateway', 'llm', 'shared'],
    metadata: { source: 'src/ai/ai.controller.ts', runtime: 'LiteLLM or Claude CLI fallback' },
  },
  {
    name: 'Shop Query Refiner',
    slug: 'shop-query-refiner',
    description: 'Turns conversational shopping input into one concise product search query.',
    status: 'active',
    serviceScope: 'shop-assistant',
    routePath: '/api/shop-assistant/refine-query',
    modelTier: 'free',
    temperature: '0.20',
    maxTokens: 100,
    systemPrompt: 'You are a search query refiner. Output only the refined query, no explanation.',
    userPromptTemplate: 'Extract a single web product search query from: "{{user_input}}". Reply with ONE short search query only (max 200 chars), no other text.',
    tags: ['shop-assistant', 'communication', 'search'],
    metadata: { variables: ['user_input', 'previous_params'], source: 'src/shop-assistant/shop-assistant.service.ts' },
  },
  {
    name: 'Shop Result Presenter',
    slug: 'shop-result-presenter',
    description: 'Formats product search results into clear user-facing recommendations.',
    status: 'active',
    serviceScope: 'shop-assistant',
    routePath: '/api/shop-assistant/format-presentation',
    modelTier: 'free',
    temperature: '0.20',
    maxTokens: 500,
    systemPrompt: 'Format search results clearly for the user.',
    userPromptTemplate: 'Format these product search results for the user in a clear, readable way (dialog or list). Query: {{queryText}}. Results: {{searchResults}}. Return only the formatted text, no extra commentary.',
    tags: ['shop-assistant', 'presentation'],
    metadata: { variables: ['queryText', 'searchResults'], source: 'src/shop-assistant/shop-assistant.service.ts' },
  },
  {
    name: 'Shop Price Comparator',
    slug: 'shop-price-comparator',
    description: 'Compares product options according to user priorities.',
    status: 'active',
    serviceScope: 'shop-assistant',
    routePath: '/api/shop-assistant/compare-prices',
    modelTier: 'free',
    temperature: '0.20',
    maxTokens: 500,
    systemPrompt: 'You are a shopping comparison assistant.',
    userPromptTemplate: 'You are a shopping assistant comparing products for a user.\n\nUser query: {{queryText}}\nPriorities (price, quality, location, etc.): {{priorityOrder}}\n\nSearch results:\n{{searchResults}}\n\nWrite a short comparison focusing on the user priorities. Recommend several best options and explain briefly. Keep it concise.',
    tags: ['shop-assistant', 'comparison'],
    metadata: { variables: ['queryText', 'priorityOrder', 'searchResults'], source: 'src/shop-assistant/shop-assistant.service.ts' },
  },
  {
    name: 'Shop Location Extractor',
    slug: 'shop-location-extractor',
    description: 'Extracts delivery or shipping region from a shopping conversation.',
    status: 'active',
    serviceScope: 'shop-assistant',
    routePath: '/api/shop-assistant/extract-location',
    modelTier: 'free',
    temperature: '0.20',
    maxTokens: 60,
    systemPrompt: 'Extract delivery region from user shopping request. Return short phrase or empty string.',
    userPromptTemplate: 'User is shopping online.\n\nUser input: {{userInput}}\nCurrent query: {{queryText}}\nPriorities (price, quality, location, etc.): {{priorityOrder}}\n\nExtract a short phrase describing the delivery region or shipping location that matters for this request, for example "delivery Czech Republic" or "ships to EU". If region is not specified, respond with an empty string.\nAnswer with this short phrase only, no other text.',
    tags: ['shop-assistant', 'location'],
    metadata: { variables: ['userInput', 'queryText', 'priorityOrder'], source: 'src/shop-assistant/shop-assistant.service.ts' },
  },
  {
    name: 'Email Triage Classifier',
    slug: 'email-triage-classifier',
    description: 'Classifies inbound email intent for the agentic email processing system.',
    status: 'active',
    serviceScope: 'agentic-email-processing-system',
    routePath: '/api/email-triage/classify',
    modelTier: 'free',
    temperature: '0.20',
    maxTokens: 500,
    systemPrompt: 'Classify inbound email intent and confidence. Return valid JSON matching the expected triage schema.',
    userPromptTemplate: '{{email_text}}',
    outputSchema: { intent: 'string', confidence: 'number', raw_scores: 'object' },
    tags: ['email-triage', 'classifier'],
    metadata: { source: 'src/email-triage/email-triage.controller.ts' },
  },
  {
    name: 'Email Triage Decider',
    slug: 'email-triage-decider',
    description: 'Decides action, queue, and escalation reason for triaged email.',
    status: 'active',
    serviceScope: 'agentic-email-processing-system',
    routePath: '/api/email-triage/decide',
    modelTier: 'free',
    temperature: '0.20',
    maxTokens: 500,
    systemPrompt: 'Decide the next email-handling action. Return valid JSON only.',
    userPromptTemplate: '{{email_text}}',
    outputSchema: { action: 'string', escalation_reason: 'string|null', queue: 'string|null' },
    tags: ['email-triage', 'decision'],
    metadata: { source: 'src/email-triage/email-triage.controller.ts' },
  },
  {
    name: 'Task Draft Agent',
    slug: 'task-draft-agent',
    description: 'Drafts structured task output from user instructions.',
    status: 'active',
    serviceScope: 'task',
    routePath: '/task/draft',
    modelTier: 'smart',
    temperature: '0.20',
    maxTokens: 1200,
    systemPrompt: 'Draft a structured implementation task from user intent.',
    userPromptTemplate: '{{task_input}}',
    tags: ['task', 'drafting'],
    metadata: { source: 'src/task/task.service.ts' },
  },
  {
    name: 'Claude Code Executor',
    slug: 'claude-code-executor',
    description: 'Queues and executes Claude Code jobs against repository worktrees.',
    status: 'active',
    serviceScope: 'runlayer',
    routePath: '/ai/claude-code-execute',
    modelTier: 'smart',
    providerModel: 'claude-sonnet',
    temperature: '0.20',
    maxTokens: 8000,
    systemPrompt: 'Execute the requested repository task and report validation output.',
    userPromptTemplate: '{{instructions}}',
    tags: ['claude-code', 'automation', 'runlayer'],
    metadata: { source: 'src/claude-code/claude-code.controller.ts' },
  },
  {
    name: 'Voice Transcriber',
    slug: 'voice-transcriber',
    description: 'Transcribes voice input through the ASR service path.',
    status: 'active',
    serviceScope: 'voice',
    routePath: '/voice/transcribe',
    modelTier: 'free',
    providerModel: 'openai/whisper-1',
    temperature: '0.00',
    maxTokens: 1000,
    systemPrompt: 'Transcribe audio input faithfully.',
    userPromptTemplate: '{{voice_file_url}}',
    tags: ['voice', 'asr'],
    metadata: { source: 'src/voice/voice.service.ts' },
  },
];
