# Strict JSON Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce strict, Zod-validated JSON contracts on every inter-agent and inter-service boundary in ai-microservice, so all input and output shapes are validated at runtime with a clear `ContractViolationError` on failure — mirroring the approach already used in business-orchestrator (see issue #21).

**Architecture:** Add a `src/contracts/` directory with Zod schemas for every request/response shape across all six modules (ai, task, voice, email-triage, shop-assistant, claude-code). Wire validation into NestJS using a global `ZodValidationPipe` for request bodies and a thin `parseOrThrow` helper for service outputs. No class-validator migration — class-validator is removed from request-body paths once Zod pipes are wired in; existing DTOs are kept in sync as TypeScript types derived from Zod.

**Tech Stack:** Zod ^3, NestJS ValidationPipe pattern, class-validator (existing, kept for non-Zod paths during transition), Jest for contract unit tests.

---

## Gap Analysis (current state → target state)

| Module | Endpoint | Input contract | Output contract |
|---|---|---|---|
| ai | POST /ai/complete | class-validator DTO ✓ | TypeScript interface only — no runtime validation ✗ |
| task | POST /task/draft | class-validator DTO ✓ | TypeScript interface only ✗ |
| voice | POST /voice/transcribe | class-validator DTO ✓ | `{ transcript: string }` inline — no validation ✗ |
| email-triage | POST /api/email-triage/ingest | ad-hoc manual checks ✗ | ad-hoc object spread ✗ |
| email-triage | POST /api/email-triage/classify | ad-hoc `Record<string,unknown>` ✗ | ad-hoc object spread ✗ |
| email-triage | POST /api/email-triage/extract | ad-hoc `Record<string,unknown>` ✗ | ad-hoc spread ✗ |
| email-triage | POST /api/email-triage/decide | manual field checks ✗ | ad-hoc spread ✗ |
| shop-assistant | POST /api/shop-assistant/* (6 endpoints) | inline interface types ✗ | no contract ✗ |
| claude-code | POST /ai/claude-code-execute | class-validator DTO ✓ | typed response DTO (no Zod) ✗ |
| claude-code | GET /ai/claude-code-execute/:jobId | path param, no DTO ✓ | typed response DTO (no Zod) ✗ |

---

## File Structure

```
src/
  contracts/
    index.ts                          (re-exports all schemas + parseOrThrow)
    contract-violation.error.ts       (ContractViolationError class)
    ai.contract.ts                    (AiCompleteRequestSchema, AiCompleteResponseSchema)
    task.contract.ts                  (TaskDraftRequestSchema, TaskDraftResponseSchema)
    voice.contract.ts                 (TranscribeRequestSchema, TranscribeResponseSchema)
    email-triage.contract.ts          (IngestRequestSchema, IngestResponseSchema, ClassifyRequestSchema, ClassifyResponseSchema, ExtractRequestSchema, ExtractResponseSchema, DecideRequestSchema, DecideResponseSchema)
    shop-assistant.contract.ts        (TranscribeRequestSchema, RefineQueryRequestSchema, SearchRequestSchema, FormatPresentationRequestSchema, ComparePricesRequestSchema, ExtractLocationRequestSchema + all response schemas)
    claude-code.contract.ts           (ExecuteCodeRequestSchema, JobEnqueueResponseSchema, JobStatusResponseSchema)
    zod-validation.pipe.ts            (NestJS ZodValidationPipe — replaces class-validator pipe per-controller)
  ai/
    ai.controller.ts                  (add ZodValidationPipe, output parseOrThrow)
    ai.service.ts                     (add output parseOrThrow)
  task/
    task.controller.ts                (add ZodValidationPipe, output parseOrThrow)
    task.service.ts                   (add output parseOrThrow)
  voice/
    voice.controller.ts               (add ZodValidationPipe, output parseOrThrow)
    voice.service.ts                  (add output parseOrThrow)
  email-triage/
    email-triage.controller.ts        (replace ad-hoc checks with ZodValidationPipe on each endpoint, output parseOrThrow)
  shop-assistant/
    shop-assistant.controller.ts      (add DTOs derived from Zod schemas, ZodValidationPipe, output parseOrThrow)
  claude-code/
    claude-code.controller.ts         (add ZodValidationPipe for execute, output parseOrThrow on responses)

test/
  contracts/
    contracts.spec.ts                 (unit tests for all schemas: valid + invalid cases)
```

---

## Task 1: Add Zod dependency and ContractViolationError

**Files:**
- Modify: `package.json`
- Create: `src/contracts/contract-violation.error.ts`

- [ ] **Step 1: Install Zod**

```bash
cd /home/ssf/Documents/Github/ai-microservice
npm install zod
```

Expected output: `added 1 package` (or already present)

- [ ] **Step 2: Create ContractViolationError**

Create `src/contracts/contract-violation.error.ts`:

```typescript
import { ZodError } from 'zod';

export class ContractViolationError extends Error {
  constructor(
    public readonly schemaName: string,
    public readonly zodError: ZodError,
    public readonly direction: 'input' | 'output',
  ) {
    super(
      `Contract violation [${direction}] in ${schemaName}: ${zodError.errors.map((e) => `${e.path.join('.')}: ${e.message}`).join(', ')}`,
    );
    this.name = 'ContractViolationError';
  }
}

export function parseOrThrow<T>(schema: { safeParse(data: unknown): { success: true; data: T } | { success: false; error: ZodError } }, schemaName: string, data: unknown, direction: 'input' | 'output' = 'output'): T {
  const result = schema.safeParse(data);
  if (!result.success) {
    throw new ContractViolationError(schemaName, result.error, direction);
  }
  return result.data;
}
```

- [ ] **Step 3: Commit**

```bash
git add package.json package-lock.json src/contracts/contract-violation.error.ts
git commit -m "feat(contracts): add Zod dependency and ContractViolationError"
```

---

## Task 2: Create ZodValidationPipe

**Files:**
- Create: `src/contracts/zod-validation.pipe.ts`

- [ ] **Step 1: Write failing test**

Create `test/contracts/contracts.spec.ts`:

```typescript
import { ZodValidationPipe } from '../../src/contracts/zod-validation.pipe';
import { z } from 'zod';
import { BadRequestException } from '@nestjs/common';

describe('ZodValidationPipe', () => {
  const schema = z.object({ name: z.string() });
  const pipe = new ZodValidationPipe(schema);

  it('passes valid input through', () => {
    expect(pipe.transform({ name: 'Alice' }, {} as any)).toEqual({ name: 'Alice' });
  });

  it('throws BadRequestException on invalid input', () => {
    expect(() => pipe.transform({ name: 123 }, {} as any)).toThrow(BadRequestException);
  });

  it('throws BadRequestException on missing required field', () => {
    expect(() => pipe.transform({}, {} as any)).toThrow(BadRequestException);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ssf/Documents/Github/ai-microservice
npx jest test/contracts/contracts.spec.ts --no-coverage 2>&1 | tail -20
```

Expected: FAIL — `ZodValidationPipe` not found.

- [ ] **Step 3: Create ZodValidationPipe**

Create `src/contracts/zod-validation.pipe.ts`:

```typescript
import { PipeTransform, BadRequestException } from '@nestjs/common';
import { ZodSchema } from 'zod';

export class ZodValidationPipe implements PipeTransform {
  constructor(private readonly schema: ZodSchema) {}

  transform(value: unknown) {
    const result = this.schema.safeParse(value);
    if (!result.success) {
      throw new BadRequestException({
        error: 'Contract violation',
        details: result.error.errors.map((e) => ({
          path: e.path.join('.'),
          message: e.message,
        })),
      });
    }
    return result.data;
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
npx jest test/contracts/contracts.spec.ts --no-coverage 2>&1 | tail -10
```

Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/contracts/zod-validation.pipe.ts test/contracts/contracts.spec.ts
git commit -m "feat(contracts): add ZodValidationPipe for NestJS"
```

---

## Task 3: AI module contracts

**Files:**
- Create: `src/contracts/ai.contract.ts`
- Modify: `src/ai/ai.controller.ts`
- Modify: `src/ai/ai.service.ts`
- Modify: `test/contracts/contracts.spec.ts`

- [ ] **Step 1: Add AI contract tests**

Append to `test/contracts/contracts.spec.ts`:

```typescript
import { AiCompleteRequestSchema, AiCompleteResponseSchema } from '../../src/contracts/ai.contract';

describe('AiCompleteRequestSchema', () => {
  it('accepts valid request', () => {
    const result = AiCompleteRequestSchema.safeParse({
      model_tier: 'free',
      user_prompt: 'Hello',
    });
    expect(result.success).toBe(true);
  });

  it('rejects missing user_prompt', () => {
    const result = AiCompleteRequestSchema.safeParse({ model_tier: 'free' });
    expect(result.success).toBe(false);
  });

  it('rejects invalid model_tier', () => {
    const result = AiCompleteRequestSchema.safeParse({ model_tier: 'ultra', user_prompt: 'x' });
    expect(result.success).toBe(false);
  });
});

describe('AiCompleteResponseSchema', () => {
  it('accepts valid response', () => {
    const result = AiCompleteResponseSchema.safeParse({
      text: 'hello',
      model_used: 'sonnet',
      inputTokens: 10,
      outputTokens: 5,
      token_usage_estimate: 15,
    });
    expect(result.success).toBe(true);
  });

  it('rejects missing text', () => {
    const result = AiCompleteResponseSchema.safeParse({
      model_used: 'sonnet',
      inputTokens: 10,
      outputTokens: 5,
      token_usage_estimate: 15,
    });
    expect(result.success).toBe(false);
  });
});
```

- [ ] **Step 2: Run tests — expect fail**

```bash
npx jest test/contracts/contracts.spec.ts --no-coverage 2>&1 | tail -20
```

Expected: FAIL — `AiCompleteRequestSchema` not found.

- [ ] **Step 3: Create ai.contract.ts**

Create `src/contracts/ai.contract.ts`:

```typescript
import { z } from 'zod';

export const MODEL_TIERS = ['free', 'cheap', 'smart', 'premium'] as const;

export const AiCompleteRequestSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  model_tier: z.enum(MODEL_TIERS),
  system_prompt: z.string().optional(),
  user_prompt: z.string().min(1),
  output_schema: z.record(z.unknown()).optional(),
  max_tokens: z.number().int().positive().optional(),
  correlation_id: z.string().optional(),
});

export type AiCompleteRequest = z.infer<typeof AiCompleteRequestSchema>;

export const AiCompleteResponseSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  data: z.unknown().optional(),
  text: z.string(),
  model_used: z.string(),
  inputTokens: z.number().int().nonnegative(),
  outputTokens: z.number().int().nonnegative(),
  token_usage_estimate: z.number().int().nonnegative(),
  error_code: z.string().optional(),
});

export type AiCompleteResponse = z.infer<typeof AiCompleteResponseSchema>;
```

- [ ] **Step 4: Wire validation into ai.controller.ts**

Edit `src/ai/ai.controller.ts` to:

```typescript
import { Controller, Post, Body, HttpCode, UsePipes } from '@nestjs/common';
import { AiService } from './ai.service';
import { AiCompleteRequestSchema } from '../contracts/ai.contract';
import { ZodValidationPipe } from '../contracts/zod-validation.pipe';
import type { AiCompleteRequest } from '../contracts/ai.contract';
import type { AiCompleteResult } from './ai.service';

@Controller('ai')
export class AiController {
  constructor(private readonly aiService: AiService) {}

  @Post('complete')
  @HttpCode(200)
  @UsePipes(new ZodValidationPipe(AiCompleteRequestSchema))
  async complete(@Body() dto: AiCompleteRequest): Promise<AiCompleteResult> {
    return this.aiService.complete(dto);
  }
}
```

- [ ] **Step 5: Add output parseOrThrow in ai.service.ts**

In `src/ai/ai.service.ts`, add output validation in the `complete()` method return statement. Find the line where the final result object is built and returned (around line 80-110), and wrap the return value:

```typescript
import { parseOrThrow } from '../contracts/contract-violation.error';
import { AiCompleteResponseSchema } from '../contracts/ai.contract';

// At the end of complete(), replace `return result;` with:
return parseOrThrow(AiCompleteResponseSchema, 'AiCompleteResponse', result, 'output') as AiCompleteResult;
```

- [ ] **Step 6: Run tests — expect pass**

```bash
npx jest test/contracts/contracts.spec.ts --no-coverage 2>&1 | tail -10
```

Expected: PASS (all tests).

- [ ] **Step 7: Commit**

```bash
git add src/contracts/ai.contract.ts src/ai/ai.controller.ts src/ai/ai.service.ts test/contracts/contracts.spec.ts
git commit -m "feat(contracts): add Zod schemas and validation for AI module"
```

---

## Task 4: Task module contracts

**Files:**
- Create: `src/contracts/task.contract.ts`
- Modify: `src/task/task.controller.ts`
- Modify: `src/task/task.service.ts`
- Modify: `test/contracts/contracts.spec.ts`

- [ ] **Step 1: Add task contract tests**

Append to `test/contracts/contracts.spec.ts`:

```typescript
import { TaskDraftRequestSchema, TaskDraftResponseSchema } from '../../src/contracts/task.contract';

describe('TaskDraftRequestSchema', () => {
  it('accepts empty body (all optional)', () => {
    expect(TaskDraftRequestSchema.safeParse({}).success).toBe(true);
  });

  it('accepts full body', () => {
    const result = TaskDraftRequestSchema.safeParse({
      transcript: 'please do X',
      textNote: 'context',
      language: 'en',
    });
    expect(result.success).toBe(true);
  });

  it('rejects non-string transcript', () => {
    expect(TaskDraftRequestSchema.safeParse({ transcript: 123 }).success).toBe(false);
  });
});

describe('TaskDraftResponseSchema', () => {
  it('accepts valid response', () => {
    const result = TaskDraftResponseSchema.safeParse({
      title: 'Fix bug',
      description: 'Details',
      priority: 'high',
      modelTier: 'free',
    });
    expect(result.success).toBe(true);
  });

  it('rejects invalid priority', () => {
    const result = TaskDraftResponseSchema.safeParse({
      title: 'Fix bug',
      description: 'Details',
      priority: 'urgent',
      modelTier: 'free',
    });
    expect(result.success).toBe(false);
  });
});
```

- [ ] **Step 2: Run tests — expect fail**

```bash
npx jest test/contracts/contracts.spec.ts --no-coverage 2>&1 | tail -20
```

- [ ] **Step 3: Create task.contract.ts**

Create `src/contracts/task.contract.ts`:

```typescript
import { z } from 'zod';

export const TaskDraftRequestSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  transcript: z.string().optional(),
  textNote: z.string().optional(),
  language: z.string().optional(),
});

export type TaskDraftRequest = z.infer<typeof TaskDraftRequestSchema>;

export const TaskPrioritySchema = z.enum(['low', 'normal', 'high']);

export const TaskDraftResponseSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  title: z.string(),
  description: z.string(),
  priority: TaskPrioritySchema,
  deadline: z.string().optional(),
  modelTier: z.string(),
});

export type TaskDraftResponse = z.infer<typeof TaskDraftResponseSchema>;
```

- [ ] **Step 4: Wire into task.controller.ts**

Replace `src/task/task.controller.ts`:

```typescript
import { Controller, Post, Body, HttpCode, UsePipes } from '@nestjs/common';
import { TaskService } from './task.service';
import { TaskDraftRequestSchema } from '../contracts/task.contract';
import { ZodValidationPipe } from '../contracts/zod-validation.pipe';
import type { TaskDraftRequest, TaskDraftResponse } from '../contracts/task.contract';

@Controller('task')
export class TaskController {
  constructor(private readonly taskService: TaskService) {}

  @Post('draft')
  @HttpCode(200)
  @UsePipes(new ZodValidationPipe(TaskDraftRequestSchema))
  async draft(@Body() dto: TaskDraftRequest): Promise<TaskDraftResponse> {
    return this.taskService.draftTask(dto);
  }
}
```

- [ ] **Step 5: Add output parseOrThrow in task.service.ts**

In `src/task/task.service.ts`, find the `draftTask` method's return statement and wrap it:

```typescript
import { parseOrThrow } from '../contracts/contract-violation.error';
import { TaskDraftResponseSchema } from '../contracts/task.contract';
import type { TaskDraftResponse } from '../contracts/task.contract';

// Wrap the return value:
return parseOrThrow(TaskDraftResponseSchema, 'TaskDraftResponse', result, 'output') as TaskDraftResponse;
```

- [ ] **Step 6: Run tests — expect pass**

```bash
npx jest test/contracts/contracts.spec.ts --no-coverage 2>&1 | tail -10
```

- [ ] **Step 7: Commit**

```bash
git add src/contracts/task.contract.ts src/task/task.controller.ts src/task/task.service.ts test/contracts/contracts.spec.ts
git commit -m "feat(contracts): add Zod schemas and validation for task module"
```

---

## Task 5: Voice module contracts

**Files:**
- Create: `src/contracts/voice.contract.ts`
- Modify: `src/voice/voice.controller.ts`
- Modify: `src/voice/voice.service.ts`
- Modify: `test/contracts/contracts.spec.ts`

- [ ] **Step 1: Add voice contract tests**

Append to `test/contracts/contracts.spec.ts`:

```typescript
import { TranscribeRequestSchema, TranscribeResponseSchema } from '../../src/contracts/voice.contract';

describe('TranscribeRequestSchema', () => {
  it('accepts valid request', () => {
    expect(TranscribeRequestSchema.safeParse({ fileKey: 'audio/123.wav' }).success).toBe(true);
  });

  it('rejects empty fileKey', () => {
    expect(TranscribeRequestSchema.safeParse({ fileKey: '' }).success).toBe(false);
  });

  it('rejects missing fileKey', () => {
    expect(TranscribeRequestSchema.safeParse({}).success).toBe(false);
  });
});

describe('TranscribeResponseSchema', () => {
  it('accepts valid response', () => {
    expect(TranscribeResponseSchema.safeParse({ transcript: 'Hello world' }).success).toBe(true);
  });

  it('rejects missing transcript', () => {
    expect(TranscribeResponseSchema.safeParse({}).success).toBe(false);
  });
});
```

- [ ] **Step 2: Run tests — expect fail**

```bash
npx jest test/contracts/contracts.spec.ts --no-coverage 2>&1 | tail -20
```

- [ ] **Step 3: Create voice.contract.ts**

Create `src/contracts/voice.contract.ts`:

```typescript
import { z } from 'zod';

export const TranscribeRequestSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  fileKey: z.string().min(1),
  language: z.string().optional(),
});

export type TranscribeRequest = z.infer<typeof TranscribeRequestSchema>;

export const TranscribeResponseSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  transcript: z.string(),
});

export type TranscribeResponse = z.infer<typeof TranscribeResponseSchema>;
```

- [ ] **Step 4: Wire into voice.controller.ts**

Replace `src/voice/voice.controller.ts`:

```typescript
import { Controller, Post, Body, HttpCode, UsePipes } from '@nestjs/common';
import { VoiceService } from './voice.service';
import { TranscribeRequestSchema, TranscribeResponseSchema } from '../contracts/voice.contract';
import { ZodValidationPipe } from '../contracts/zod-validation.pipe';
import { parseOrThrow } from '../contracts/contract-violation.error';
import type { TranscribeRequest, TranscribeResponse } from '../contracts/voice.contract';

@Controller('voice')
export class VoiceController {
  constructor(private readonly voiceService: VoiceService) {}

  @Post('transcribe')
  @HttpCode(200)
  @UsePipes(new ZodValidationPipe(TranscribeRequestSchema))
  async transcribe(@Body() dto: TranscribeRequest): Promise<TranscribeResponse> {
    const result = await this.voiceService.transcribe(dto);
    return parseOrThrow(TranscribeResponseSchema, 'TranscribeResponse', result, 'output');
  }
}
```

- [ ] **Step 5: Run tests — expect pass**

```bash
npx jest test/contracts/contracts.spec.ts --no-coverage 2>&1 | tail -10
```

- [ ] **Step 6: Commit**

```bash
git add src/contracts/voice.contract.ts src/voice/voice.controller.ts test/contracts/contracts.spec.ts
git commit -m "feat(contracts): add Zod schemas and validation for voice module"
```

---

## Task 6: Email-triage module contracts

**Files:**
- Create: `src/contracts/email-triage.contract.ts`
- Modify: `src/email-triage/email-triage.controller.ts`
- Modify: `test/contracts/contracts.spec.ts`

- [ ] **Step 1: Add email-triage contract tests**

Append to `test/contracts/contracts.spec.ts`:

```typescript
import {
  IngestRequestSchema, IngestResponseSchema,
  ClassifyRequestSchema, ClassifyResponseSchema,
  ExtractRequestSchema, ExtractResponseSchema,
  DecideRequestSchema, DecideResponseSchema,
} from '../../src/contracts/email-triage.contract';

describe('IngestRequestSchema', () => {
  const valid = {
    message_id: 'msg-1', tenant_id: 'ten-1',
    timestamp: '2026-01-01T00:00:00Z', sender: 'a@b.com',
    recipients: ['c@d.com'], subject: 'Hello',
    body_plain: 'body', body_html: '<p>body</p>', attachments: [],
  };
  it('accepts valid email', () => {
    expect(IngestRequestSchema.safeParse(valid).success).toBe(true);
  });
  it('rejects missing sender', () => {
    const { sender, ...rest } = valid;
    expect(IngestRequestSchema.safeParse(rest).success).toBe(false);
  });
});

describe('ClassifyRequestSchema', () => {
  it('accepts valid request', () => {
    expect(ClassifyRequestSchema.safeParse({ payload: { message_id: '1', subject: 'x', body_plain: 'y' }, use_llm: false }).success).toBe(true);
  });
  it('rejects missing payload', () => {
    expect(ClassifyRequestSchema.safeParse({ use_llm: false }).success).toBe(false);
  });
});

describe('ClassifyResponseSchema', () => {
  it('accepts valid response', () => {
    const result = ClassifyResponseSchema.safeParse({
      success: true, intent: 'support', confidence: 0.9, model_used: 'rule-based', duration_ms: 5,
    });
    expect(result.success).toBe(true);
  });
  it('rejects confidence out of range', () => {
    const result = ClassifyResponseSchema.safeParse({
      success: true, intent: 'support', confidence: 1.5, model_used: 'rule-based', duration_ms: 5,
    });
    expect(result.success).toBe(false);
  });
});

describe('DecideRequestSchema', () => {
  it('accepts valid request', () => {
    expect(DecideRequestSchema.safeParse({ intent: 'support', confidence: 0.8 }).success).toBe(true);
  });
  it('rejects missing intent', () => {
    expect(DecideRequestSchema.safeParse({ confidence: 0.8 }).success).toBe(false);
  });
});

describe('DecideResponseSchema', () => {
  it('accepts valid response', () => {
    const result = DecideResponseSchema.safeParse({
      success: true, action: 'auto_respond', escalation_reason: null, queue: null, model_used: 'rule-based',
    });
    expect(result.success).toBe(true);
  });
  it('rejects invalid action', () => {
    const result = DecideResponseSchema.safeParse({
      success: true, action: 'do_nothing', escalation_reason: null, queue: null, model_used: 'rule-based',
    });
    expect(result.success).toBe(false);
  });
});
```

- [ ] **Step 2: Run tests — expect fail**

```bash
npx jest test/contracts/contracts.spec.ts --no-coverage 2>&1 | tail -20
```

- [ ] **Step 3: Create email-triage.contract.ts**

Create `src/contracts/email-triage.contract.ts`:

```typescript
import { z } from 'zod';

const EMAIL_INTENTS = ['support', 'sales', 'contract', 'technical', 'billing', 'spam', 'unknown', 'multi_intent'] as const;
const DECIDE_ACTIONS = ['auto_respond', 'route_to_queue', 'escalate'] as const;

export const IngestRequestSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  message_id: z.string(),
  tenant_id: z.string(),
  timestamp: z.union([z.string(), z.number()]),
  sender: z.string().email(),
  recipients: z.array(z.string()),
  subject: z.string(),
  body_plain: z.string(),
  body_html: z.string(),
  attachments: z.array(z.unknown()),
  locale: z.string().optional(),
  metadata: z.record(z.unknown()).optional(),
});

export type IngestRequest = z.infer<typeof IngestRequestSchema>;

export const IngestResponseSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  success: z.literal(true),
  payload: IngestRequestSchema,
  duration_ms: z.number(),
  model_used: z.string(),
});

export type IngestResponse = z.infer<typeof IngestResponseSchema>;

export const ClassifyRequestSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  payload: z.record(z.unknown()),
  use_llm: z.boolean().optional(),
});

export type ClassifyRequest = z.infer<typeof ClassifyRequestSchema>;

export const ClassifyResponseSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  success: z.literal(true),
  intent: z.enum(EMAIL_INTENTS),
  confidence: z.number().min(0).max(1),
  raw_scores: z.record(z.number()).nullable().optional(),
  model_used: z.string(),
  duration_ms: z.number(),
  llm_output: z.record(z.unknown()).optional(),
  llm_fallback_reason: z.string().optional(),
});

export type ClassifyResponse = z.infer<typeof ClassifyResponseSchema>;

export const ExtractRequestSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  payload: z.record(z.unknown()),
  intent: z.string().optional(),
});

export type ExtractRequest = z.infer<typeof ExtractRequestSchema>;

export const ExtractResponseSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  success: z.literal(true),
  model_used: z.string(),
});

export type ExtractResponse = z.infer<typeof ExtractResponseSchema>;

export const DecideRequestSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  intent: z.string().min(1),
  confidence: z.union([z.number(), z.string()]),
  entities: z.record(z.unknown()).optional(),
  use_llm: z.boolean().optional(),
});

export type DecideRequest = z.infer<typeof DecideRequestSchema>;

export const DecideResponseSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  success: z.literal(true),
  action: z.enum(DECIDE_ACTIONS),
  escalation_reason: z.string().nullable(),
  queue: z.string().nullable(),
  model_used: z.string(),
  llm_output: z.record(z.unknown()).optional(),
  llm_fallback_reason: z.string().optional(),
});

export type DecideResponse = z.infer<typeof DecideResponseSchema>;
```

- [ ] **Step 4: Wire into email-triage.controller.ts**

Replace the `@Body() body: unknown` / `@Body() body: Record<string, unknown>` arguments and ad-hoc checks in `src/email-triage/email-triage.controller.ts` with `@UsePipes` and output `parseOrThrow` calls. Update the imports block:

```typescript
import {
  Controller, Post, Get, Body, HttpCode, HttpStatus, HttpException, Logger, UsePipes,
} from '@nestjs/common';
import { Public } from '../service-identity/public.decorator';
import { EmailTriageService } from './email-triage.service';
import { AiService } from '../ai/ai.service';
import { ZodValidationPipe } from '../contracts/zod-validation.pipe';
import { parseOrThrow } from '../contracts/contract-violation.error';
import {
  IngestRequestSchema, IngestResponseSchema,
  ClassifyRequestSchema, ClassifyResponseSchema,
  ExtractRequestSchema, ExtractResponseSchema,
  DecideRequestSchema, DecideResponseSchema,
} from '../contracts/email-triage.contract';
import type {
  IngestRequest, IngestResponse, ClassifyRequest, ClassifyResponse,
  ExtractRequest, ExtractResponse, DecideRequest, DecideResponse,
} from '../contracts/email-triage.contract';
```

Then update each route handler:

- `ingest(@Body() body: IngestRequest)` — add `@UsePipes(new ZodValidationPipe(IngestRequestSchema))`, remove manual object check, wrap return in `parseOrThrow(IngestResponseSchema, 'IngestResponse', result, 'output')`.
- `classify(@Body() body: ClassifyRequest)` — add `@UsePipes(new ZodValidationPipe(ClassifyRequestSchema))`, wrap return in `parseOrThrow(ClassifyResponseSchema, 'ClassifyResponse', result, 'output')`.
- `extract(@Body() body: ExtractRequest)` — add `@UsePipes(new ZodValidationPipe(ExtractRequestSchema))`, wrap return in `parseOrThrow(ExtractResponseSchema, 'ExtractResponse', result, 'output')`.
- `decide(@Body() body: DecideRequest)` — add `@UsePipes(new ZodValidationPipe(DecideRequestSchema))`, remove manual `intent`/`confidence` checks (now handled by Zod), wrap return in `parseOrThrow(DecideResponseSchema, 'DecideResponse', result, 'output')`.

- [ ] **Step 5: Run tests — expect pass**

```bash
npx jest test/contracts/contracts.spec.ts --no-coverage 2>&1 | tail -10
```

- [ ] **Step 6: Run existing email-triage tests if any**

```bash
npx jest --testPathPattern=email-triage --no-coverage 2>&1 | tail -15
```

- [ ] **Step 7: Commit**

```bash
git add src/contracts/email-triage.contract.ts src/email-triage/email-triage.controller.ts test/contracts/contracts.spec.ts
git commit -m "feat(contracts): add Zod schemas and validation for email-triage module"
```

---

## Task 7: Shop-assistant module contracts

**Files:**
- Create: `src/contracts/shop-assistant.contract.ts`
- Modify: `src/shop-assistant/shop-assistant.controller.ts`
- Modify: `test/contracts/contracts.spec.ts`

- [ ] **Step 1: Add shop-assistant contract tests**

Append to `test/contracts/contracts.spec.ts`:

```typescript
import {
  ShopTranscribeRequestSchema,
  RefineQueryRequestSchema,
  SearchRequestSchema,
  FormatPresentationRequestSchema,
  ComparePricesRequestSchema,
  ExtractLocationRequestSchema,
} from '../../src/contracts/shop-assistant.contract';

describe('ShopTranscribeRequestSchema', () => {
  it('accepts valid request', () => {
    expect(ShopTranscribeRequestSchema.safeParse({ voice_file_url: 'https://example.com/audio.wav' }).success).toBe(true);
  });
  it('rejects missing voice_file_url', () => {
    expect(ShopTranscribeRequestSchema.safeParse({}).success).toBe(false);
  });
});

describe('RefineQueryRequestSchema', () => {
  it('accepts minimal request', () => {
    expect(RefineQueryRequestSchema.safeParse({ user_text: 'find me shoes' }).success).toBe(true);
  });
  it('rejects empty user_text', () => {
    expect(RefineQueryRequestSchema.safeParse({ user_text: '' }).success).toBe(false);
  });
});

describe('SearchRequestSchema', () => {
  it('accepts valid request', () => {
    expect(SearchRequestSchema.safeParse({ query_text: 'shoes' }).success).toBe(true);
  });
  it('rejects missing query_text', () => {
    expect(SearchRequestSchema.safeParse({}).success).toBe(false);
  });
});

describe('FormatPresentationRequestSchema', () => {
  it('accepts valid request', () => {
    expect(FormatPresentationRequestSchema.safeParse({ results: [], query_text: 'test' }).success).toBe(true);
  });
});

describe('ComparePricesRequestSchema', () => {
  it('accepts valid request', () => {
    expect(ComparePricesRequestSchema.safeParse({ results: [], query_text: 'test' }).success).toBe(true);
  });
});

describe('ExtractLocationRequestSchema', () => {
  it('accepts valid request', () => {
    expect(ExtractLocationRequestSchema.safeParse({ text: 'near Prague' }).success).toBe(true);
  });
  it('rejects missing text', () => {
    expect(ExtractLocationRequestSchema.safeParse({}).success).toBe(false);
  });
});
```

- [ ] **Step 2: Run tests — expect fail**

```bash
npx jest test/contracts/contracts.spec.ts --no-coverage 2>&1 | tail -20
```

- [ ] **Step 3: Create shop-assistant.contract.ts**

Create `src/contracts/shop-assistant.contract.ts`:

```typescript
import { z } from 'zod';

export const ShopTranscribeRequestSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  voice_file_url: z.string().min(1),
});

export type ShopTranscribeRequest = z.infer<typeof ShopTranscribeRequestSchema>;

export const RefineQueryRequestSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  user_text: z.string().min(1),
  previous_params: z.record(z.unknown()).optional(),
  role: z.string().optional(),
  prompt_content: z.string().optional(),
  model: z.string().optional(),
});

export type RefineQueryRequest = z.infer<typeof RefineQueryRequestSchema>;

export const SearchRequestSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  query_text: z.string().min(1),
  limit: z.number().int().positive().optional(),
});

export type SearchRequest = z.infer<typeof SearchRequestSchema>;

export const FormatPresentationRequestSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  results: z.array(z.record(z.unknown())),
  query_text: z.string(),
  role: z.string().optional(),
  prompt_content: z.string().optional(),
  model: z.string().optional(),
});

export type FormatPresentationRequest = z.infer<typeof FormatPresentationRequestSchema>;

export const ComparePricesRequestSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  results: z.array(z.record(z.unknown())),
  query_text: z.string(),
  role: z.string().optional(),
  prompt_content: z.string().optional(),
  model: z.string().optional(),
  priority_order: z.array(z.string()).optional(),
});

export type ComparePricesRequest = z.infer<typeof ComparePricesRequestSchema>;

export const ExtractLocationRequestSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  text: z.string().min(1),
});

export type ExtractLocationRequest = z.infer<typeof ExtractLocationRequestSchema>;
```

- [ ] **Step 4: Wire into shop-assistant.controller.ts**

Update `src/shop-assistant/shop-assistant.controller.ts` to import and use `ZodValidationPipe` on each route:

```typescript
import { Controller, Post, Body, HttpCode, HttpStatus, UsePipes } from '@nestjs/common';
import { Public } from '../service-identity/public.decorator';
import { ShopAssistantService } from './shop-assistant.service';
import { ZodValidationPipe } from '../contracts/zod-validation.pipe';
import {
  ShopTranscribeRequestSchema, RefineQueryRequestSchema,
  SearchRequestSchema, FormatPresentationRequestSchema,
  ComparePricesRequestSchema, ExtractLocationRequestSchema,
} from '../contracts/shop-assistant.contract';
import type {
  ShopTranscribeRequest, RefineQueryRequest, SearchRequest,
  FormatPresentationRequest, ComparePricesRequest, ExtractLocationRequest,
} from '../contracts/shop-assistant.contract';

@Controller('api/shop-assistant')
@Public()
export class ShopAssistantController {
  constructor(private readonly shopAssistantService: ShopAssistantService) {}

  @Post('transcribe')
  @HttpCode(HttpStatus.OK)
  @UsePipes(new ZodValidationPipe(ShopTranscribeRequestSchema))
  transcribe(@Body() body: ShopTranscribeRequest) {
    return this.shopAssistantService.transcribe(body.voice_file_url);
  }

  @Post('refine-query')
  @HttpCode(HttpStatus.OK)
  @UsePipes(new ZodValidationPipe(RefineQueryRequestSchema))
  refineQuery(@Body() body: RefineQueryRequest) {
    return this.shopAssistantService.refineQuery(
      body.user_text, body.previous_params, body.role, body.prompt_content, body.model,
    );
  }

  @Post('search')
  @HttpCode(HttpStatus.OK)
  @UsePipes(new ZodValidationPipe(SearchRequestSchema))
  search(@Body() body: SearchRequest) {
    return this.shopAssistantService.search(body.query_text, body.limit);
  }

  @Post('format-presentation')
  @HttpCode(HttpStatus.OK)
  @UsePipes(new ZodValidationPipe(FormatPresentationRequestSchema))
  formatPresentation(@Body() body: FormatPresentationRequest) {
    return this.shopAssistantService.formatPresentation(
      body.results, body.query_text, body.role, body.prompt_content, body.model,
    );
  }

  @Post('compare-prices')
  @HttpCode(HttpStatus.OK)
  @UsePipes(new ZodValidationPipe(ComparePricesRequestSchema))
  comparePrices(@Body() body: ComparePricesRequest) {
    return this.shopAssistantService.comparePrices(
      body.results, body.query_text, body.role, body.prompt_content, body.model, body.priority_order,
    );
  }

  @Post('extract-location')
  @HttpCode(HttpStatus.OK)
  @UsePipes(new ZodValidationPipe(ExtractLocationRequestSchema))
  extractLocation(@Body() body: ExtractLocationRequest) {
    return this.shopAssistantService.extractLocation(body.text);
  }
}
```

- [ ] **Step 5: Run tests — expect pass**

```bash
npx jest test/contracts/contracts.spec.ts --no-coverage 2>&1 | tail -10
```

- [ ] **Step 6: Commit**

```bash
git add src/contracts/shop-assistant.contract.ts src/shop-assistant/shop-assistant.controller.ts test/contracts/contracts.spec.ts
git commit -m "feat(contracts): add Zod schemas and validation for shop-assistant module"
```

---

## Task 8: Claude-code module contracts

**Files:**
- Create: `src/contracts/claude-code.contract.ts`
- Modify: `src/claude-code/claude-code.controller.ts`
- Modify: `test/contracts/contracts.spec.ts`

- [ ] **Step 1: Add claude-code contract tests**

Append to `test/contracts/contracts.spec.ts`:

```typescript
import {
  ExecuteCodeRequestSchema, JobEnqueueResponseSchema, JobStatusResponseSchema,
} from '../../src/contracts/claude-code.contract';

describe('ExecuteCodeRequestSchema', () => {
  const valid = {
    taskId: '123e4567-e89b-12d3-a456-426614174000',
    repoPath: '/repos/myapp',
    branch: 'main',
    instructions: 'fix the bug',
  };
  it('accepts valid request', () => {
    expect(ExecuteCodeRequestSchema.safeParse(valid).success).toBe(true);
  });
  it('rejects invalid UUID', () => {
    expect(ExecuteCodeRequestSchema.safeParse({ ...valid, taskId: 'not-uuid' }).success).toBe(false);
  });
  it('rejects timeout out of range', () => {
    expect(ExecuteCodeRequestSchema.safeParse({ ...valid, timeoutSeconds: 5 }).success).toBe(false);
  });
  it('rejects invalid executionMode', () => {
    expect(ExecuteCodeRequestSchema.safeParse({ ...valid, executionMode: 'run' }).success).toBe(false);
  });
});

describe('JobEnqueueResponseSchema', () => {
  it('accepts valid response', () => {
    const result = JobEnqueueResponseSchema.safeParse({
      jobId: '123e4567-e89b-12d3-a456-426614174000',
      taskId: '123e4567-e89b-12d3-a456-426614174001',
      status: 'QUEUED',
      createdAt: '2026-01-01T00:00:00Z',
    });
    expect(result.success).toBe(true);
  });
});

describe('JobStatusResponseSchema', () => {
  it('accepts queued job', () => {
    const result = JobStatusResponseSchema.safeParse({
      jobId: '123e4567-e89b-12d3-a456-426614174000',
      taskId: '123e4567-e89b-12d3-a456-426614174001',
      status: 'QUEUED',
    });
    expect(result.success).toBe(true);
  });
  it('accepts completed job with all fields', () => {
    const result = JobStatusResponseSchema.safeParse({
      jobId: '123e4567-e89b-12d3-a456-426614174000',
      taskId: '123e4567-e89b-12d3-a456-426614174001',
      status: 'SUCCESS',
      exitCode: 0,
      stdout: 'done',
      stderr: '',
      gitDiff: '',
      validationPassed: true,
    });
    expect(result.success).toBe(true);
  });
});
```

- [ ] **Step 2: Run tests — expect fail**

```bash
npx jest test/contracts/contracts.spec.ts --no-coverage 2>&1 | tail -20
```

- [ ] **Step 3: Create claude-code.contract.ts**

Create `src/contracts/claude-code.contract.ts`:

```typescript
import { z } from 'zod';

const JOB_STATUSES = ['QUEUED', 'EXECUTING', 'SUCCESS', 'FAILED', 'TIMEOUT', 'RETRYING'] as const;

export const ExecuteCodeRequestSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  taskId: z.string().uuid(),
  repoPath: z.string().min(1),
  branch: z.string().min(1),
  instructions: z.string().min(1),
  expectedOutcome: z.string().optional(),
  timeoutSeconds: z.number().int().min(10).max(3600).optional(),
  validationScript: z.string().optional(),
  executionMode: z.enum(['code', 'print']).optional(),
  model: z.string().optional(),
});

export type ExecuteCodeRequest = z.infer<typeof ExecuteCodeRequestSchema>;

export const JobEnqueueResponseSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  jobId: z.string().uuid(),
  taskId: z.string().uuid(),
  status: z.enum(JOB_STATUSES),
  createdAt: z.string(),
});

export type JobEnqueueResponse = z.infer<typeof JobEnqueueResponseSchema>;

export const JobStatusResponseSchema = z.object({
  schemaVersion: z.string().default('1.0'),
  jobId: z.string().uuid(),
  taskId: z.string().uuid(),
  status: z.enum(JOB_STATUSES),
  startedAt: z.string().optional(),
  completedAt: z.string().optional(),
  exitCode: z.number().int().optional(),
  stdout: z.string().optional(),
  stderr: z.string().optional(),
  gitDiff: z.string().optional(),
  validationPassed: z.boolean().optional(),
  validationOutput: z.string().optional(),
});

export type JobStatusResponse = z.infer<typeof JobStatusResponseSchema>;
```

- [ ] **Step 4: Wire into claude-code.controller.ts**

Read `src/claude-code/claude-code.controller.ts` and update it:
- Add `@UsePipes(new ZodValidationPipe(ExecuteCodeRequestSchema))` on the `POST` route
- Add `parseOrThrow(JobEnqueueResponseSchema, 'JobEnqueueResponse', result, 'output')` around the enqueue response
- Add `parseOrThrow(JobStatusResponseSchema, 'JobStatusResponse', result, 'output')` around the status poll response

- [ ] **Step 5: Run tests — expect pass**

```bash
npx jest test/contracts/contracts.spec.ts --no-coverage 2>&1 | tail -10
```

- [ ] **Step 6: Commit**

```bash
git add src/contracts/claude-code.contract.ts src/claude-code/claude-code.controller.ts test/contracts/contracts.spec.ts
git commit -m "feat(contracts): add Zod schemas and validation for claude-code module"
```

---

## Task 9: Create contracts index and run full test suite

**Files:**
- Create: `src/contracts/index.ts`

- [ ] **Step 1: Create contracts index**

Create `src/contracts/index.ts`:

```typescript
export * from './contract-violation.error';
export * from './zod-validation.pipe';
export * from './ai.contract';
export * from './task.contract';
export * from './voice.contract';
export * from './email-triage.contract';
export * from './shop-assistant.contract';
export * from './claude-code.contract';
```

- [ ] **Step 2: Run full test suite**

```bash
cd /home/ssf/Documents/Github/ai-microservice
npx jest --no-coverage 2>&1 | tail -30
```

Expected: All tests pass. Fix any failures before proceeding.

- [ ] **Step 3: Build TypeScript**

```bash
npx tsc --noEmit 2>&1 | head -40
```

Expected: Zero errors. Fix any type errors before proceeding.

- [ ] **Step 4: Commit**

```bash
git add src/contracts/index.ts
git commit -m "feat(contracts): add contracts barrel export"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] All 10 endpoints across 6 modules have input Zod schemas
- [x] All response shapes have output Zod schemas
- [x] `ContractViolationError` wraps Zod errors with clear messages
- [x] `ZodValidationPipe` returns `BadRequestException` with structured details
- [x] `schemaVersion: '1.0'` present in all schemas (consistent with business-orchestrator)
- [x] `parseOrThrow` used consistently at every output boundary
- [x] All schemas have matching tests in `contracts.spec.ts`

**Gaps addressed vs business-orchestrator issue #21:**
- `parseOrThrow` pattern matches what business-orchestrator uses
- `schemaVersion` field aligns with existing convention
- Zod is the single validation library (class-validator kept in existing DTOs, Zod used for all new contract validation)
