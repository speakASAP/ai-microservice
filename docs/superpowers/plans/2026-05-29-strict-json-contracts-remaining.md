# Strict JSON Contracts — Remaining Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining contract enforcement gaps in ai-microservice so every request body is validated by Zod (not class-validator) and every response is validated by `parseOrThrow` — making enforcement 100% consistent with business-orchestrator.

**Architecture:** The `src/contracts/` directory and all Zod schemas already exist. The gap is: 4 controllers still use class-validator DTOs for request body validation instead of `ZodValidationPipe`, shop-assistant controller has no `parseOrThrow` on responses, and the claude-code 404 branch returns an uncontracted plain object. This plan replaces each class-validator DTO with the corresponding Zod schema already in `src/contracts/`, adds response schemas for shop-assistant, and wires `parseOrThrow` at every output boundary.

**Tech Stack:** NestJS, Zod ^4, `ZodValidationPipe` (already in `src/contracts/zod-validation.pipe.ts`), Jest.

---

## Current State vs Target

| Module | Endpoint | Input NOW | Input TARGET | Output NOW | Output TARGET |
|--------|----------|-----------|--------------|------------|---------------|
| ai | POST /ai/complete | class-validator `CompleteRequestDto` | `ZodValidationPipe(AiCompleteRequestSchema)` | `parseOrThrow` ✓ | no change |
| task | POST /task/draft | class-validator `TaskDraftDto` | `ZodValidationPipe(TaskDraftRequestSchema)` | `parseOrThrow` ✓ | no change |
| voice | POST /voice/transcribe | class-validator `TranscribeDto` | `ZodValidationPipe(TranscribeRequestSchema)` | `parseOrThrow` ✓ | no change |
| claude-code | POST /ai/claude-code-execute | class-validator `ExecuteCodeDto` | `ZodValidationPipe(ExecuteCodeRequestSchema)` | `parseOrThrow` ✓ | no change |
| claude-code | GET /ai/claude-code-execute/:jobId | path param | no change | raw `{ error: 'Job not found' }` | `parseOrThrow(NotFoundResponseSchema)` |
| shop-assistant | all 6 POST endpoints | `ZodValidationPipe` ✓ | no change | no `parseOrThrow` | add `parseOrThrow` on each |

---

## File Map

| File | Action | What changes |
|------|--------|--------------|
| `src/ai/ai.controller.ts` | Modify | Replace `CompleteRequestDto` import + `@Body()` type with `ZodValidationPipe` + `AiCompleteRequestSchema` |
| `src/ai/dto/complete-request.dto.ts` | Delete | Replaced by `AiCompleteRequestSchema` from contracts |
| `src/task/task.controller.ts` | Modify | Replace `TaskDraftDto` with `ZodValidationPipe(TaskDraftRequestSchema)` |
| `src/task/dto/task-draft.dto.ts` | Delete | Replaced by `TaskDraftRequestSchema` from contracts |
| `src/task/task.service.ts` | Modify | Change param type from `TaskDraftDto` to `TaskDraftRequest` (from contracts) |
| `src/voice/voice.controller.ts` | Modify | Replace `TranscribeDto` with `ZodValidationPipe(TranscribeRequestSchema)` |
| `src/voice/dto/transcribe.dto.ts` | Delete | Replaced by `TranscribeRequestSchema` from contracts |
| `src/voice/voice.service.ts` | Modify | Change param type from `TranscribeDto` to `TranscribeRequest` (from contracts) |
| `src/claude-code/claude-code.controller.ts` | Modify | Replace `ExecuteCodeDto` with `ZodValidationPipe(ExecuteCodeRequestSchema)`; add `parseOrThrow` on 404 branch |
| `src/claude-code/dto/execute-code.dto.ts` | Delete | Replaced by `ExecuteCodeRequestSchema` from contracts |
| `src/claude-code/claude-code.service.ts` | Modify | Change `enqueueJob` param type from `ExecuteCodeDto` to `ExecuteCodeRequest` |
| `src/contracts/http-responses.contract.ts` | Modify | Add `NotFoundResponseSchema` |
| `src/contracts/shop-assistant.contract.ts` | Modify | Add 6 response schemas: `ShopTranscribeResponseSchema`, `ShopRefineQueryResponseSchema`, `ShopSearchResponseSchema`, `ShopPresentationResponseSchema`, `ShopComparePricesResponseSchema`, `ShopExtractLocationResponseSchema` |
| `src/contracts/index.ts` | Modify | Export new response schemas |
| `src/shop-assistant/shop-assistant.controller.ts` | Modify | Add `parseOrThrow` on each of the 6 method returns |
| `src/contracts/contracts.spec.ts` | Modify | Add tests for new response schemas and `NotFoundResponseSchema` |

---

## Task 1: Add shop-assistant response schemas and wire parseOrThrow

The shop-assistant controller already has `ZodValidationPipe` on all inputs but no output validation. Service return shapes are well-defined; we add Zod schemas and wire `parseOrThrow`.

**Files:**
- Modify: `src/contracts/shop-assistant.contract.ts`
- Modify: `src/contracts/index.ts`
- Modify: `src/shop-assistant/shop-assistant.controller.ts`
- Modify: `src/contracts/contracts.spec.ts`

- [ ] **Step 1: Add response schemas to shop-assistant.contract.ts**

The service returns these shapes (from `shop-assistant.service.ts`):
- `transcribe` → `{ transcript: string }`
- `refineQuery` → `{ query_text: string; refined_params: Record<string, unknown> }`
- `search` → `{ items: SearchItem[] }` where `SearchItem = { title, url, price?, source?, position, snippet? }`
- `formatPresentation` → `{ formatted_content: string }`
- `comparePrices` → `{ summary: string }`
- `extractLocation` → `{ region: string | null; augmented_query: string | null }`

Append to `src/contracts/shop-assistant.contract.ts`:

```typescript
export const ShopTranscribeResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  transcript: z.string(),
});

export const ShopRefineQueryResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  query_text: z.string(),
  refined_params: z.record(z.string(), z.unknown()),
});

const ShopSearchItemSchema = z.object({
  title: z.string(),
  url: z.string(),
  price: z.unknown().optional(),
  source: z.string().optional(),
  position: z.number().int().nonnegative(),
  snippet: z.string().optional(),
});

export const ShopSearchResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  items: z.array(ShopSearchItemSchema),
});

export const ShopPresentationResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  formatted_content: z.string(),
});

export const ShopComparePricesResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  summary: z.string(),
});

export const ShopExtractLocationResponseSchema = z.object({
  schemaVersion: z.literal('1.0').default('1.0'),
  region: z.string().nullable(),
  augmented_query: z.string().nullable(),
});

export type ShopTranscribeResponse = z.infer<typeof ShopTranscribeResponseSchema>;
export type ShopRefineQueryResponse = z.infer<typeof ShopRefineQueryResponseSchema>;
export type ShopSearchResponse = z.infer<typeof ShopSearchResponseSchema>;
export type ShopPresentationResponse = z.infer<typeof ShopPresentationResponseSchema>;
export type ShopComparePricesResponse = z.infer<typeof ShopComparePricesResponseSchema>;
export type ShopExtractLocationResponse = z.infer<typeof ShopExtractLocationResponseSchema>;
```

- [ ] **Step 2: Export new response schemas from contracts index**

In `src/contracts/index.ts`, the existing line is `export * from './shop-assistant.contract';` — the new types auto-export, no change needed. Verify with:

```bash
grep "shop-assistant" /home/ssf/Documents/Github/ai-microservice/src/contracts/index.ts
```

Expected: `export * from './shop-assistant.contract';`

- [ ] **Step 3: Write failing tests for response schemas**

Append to `src/contracts/contracts.spec.ts`:

```typescript
import {
  ShopTranscribeResponseSchema,
  ShopRefineQueryResponseSchema,
  ShopSearchResponseSchema,
  ShopPresentationResponseSchema,
  ShopComparePricesResponseSchema,
  ShopExtractLocationResponseSchema,
} from './shop-assistant.contract';

describe('ShopTranscribeResponseSchema', () => {
  it('accepts valid response', () => {
    expect(ShopTranscribeResponseSchema.safeParse({ transcript: 'hello' }).success).toBe(true);
  });
  it('rejects missing transcript', () => {
    expect(ShopTranscribeResponseSchema.safeParse({}).success).toBe(false);
  });
  it('adds schemaVersion default', () => {
    const r = ShopTranscribeResponseSchema.safeParse({ transcript: 'x' });
    expect(r.success && r.data.schemaVersion).toBe('1.0');
  });
});

describe('ShopRefineQueryResponseSchema', () => {
  it('accepts valid response', () => {
    expect(ShopRefineQueryResponseSchema.safeParse({ query_text: 'shoes', refined_params: {} }).success).toBe(true);
  });
  it('rejects missing query_text', () => {
    expect(ShopRefineQueryResponseSchema.safeParse({ refined_params: {} }).success).toBe(false);
  });
});

describe('ShopSearchResponseSchema', () => {
  it('accepts valid items array', () => {
    expect(ShopSearchResponseSchema.safeParse({
      items: [{ title: 'Shoe', url: 'http://x.com', position: 1 }],
    }).success).toBe(true);
  });
  it('accepts empty items array', () => {
    expect(ShopSearchResponseSchema.safeParse({ items: [] }).success).toBe(true);
  });
  it('rejects item missing required position', () => {
    expect(ShopSearchResponseSchema.safeParse({
      items: [{ title: 'x', url: 'http://x.com' }],
    }).success).toBe(false);
  });
});

describe('ShopPresentationResponseSchema', () => {
  it('accepts valid response', () => {
    expect(ShopPresentationResponseSchema.safeParse({ formatted_content: 'Here are results...' }).success).toBe(true);
  });
  it('rejects missing formatted_content', () => {
    expect(ShopPresentationResponseSchema.safeParse({}).success).toBe(false);
  });
});

describe('ShopComparePricesResponseSchema', () => {
  it('accepts valid response', () => {
    expect(ShopComparePricesResponseSchema.safeParse({ summary: 'Best pick is X' }).success).toBe(true);
  });
  it('rejects missing summary', () => {
    expect(ShopComparePricesResponseSchema.safeParse({}).success).toBe(false);
  });
});

describe('ShopExtractLocationResponseSchema', () => {
  it('accepts valid response with region', () => {
    expect(ShopExtractLocationResponseSchema.safeParse({ region: 'Czech Republic', augmented_query: 'shoes Czech Republic' }).success).toBe(true);
  });
  it('accepts null values', () => {
    expect(ShopExtractLocationResponseSchema.safeParse({ region: null, augmented_query: null }).success).toBe(true);
  });
  it('rejects missing region field', () => {
    expect(ShopExtractLocationResponseSchema.safeParse({ augmented_query: null }).success).toBe(false);
  });
});
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
cd /home/ssf/Documents/Github/ai-microservice
npx jest src/contracts/contracts.spec.ts --no-coverage 2>&1 | tail -20
```

Expected: FAIL — `ShopTranscribeResponseSchema` not found (because schemas not appended yet, or test file path mismatch). If contracts.spec.ts is in `src/contracts/`, adjust the import paths — they should be relative within the same directory:

```typescript
import { ShopTranscribeResponseSchema, ... } from './shop-assistant.contract';
```

- [ ] **Step 5: Wire parseOrThrow in shop-assistant controller**

Replace `src/shop-assistant/shop-assistant.controller.ts` with:

```typescript
import { Controller, Post, Body, HttpCode, HttpStatus, UsePipes } from '@nestjs/common';
import { Public } from '../service-identity/public.decorator';
import { ShopAssistantService } from './shop-assistant.service';
import { ZodValidationPipe } from '../contracts/zod-validation.pipe';
import { parseOrThrow } from '../contracts/parse-or-throw';
import {
  ShopTranscribeRequestSchema,
  ShopRefineQueryRequestSchema,
  ShopSearchRequestSchema,
  ShopPresentationRequestSchema,
  ShopComparePricesRequestSchema,
  ShopExtractLocationRequestSchema,
  ShopTranscribeResponseSchema,
  ShopRefineQueryResponseSchema,
  ShopSearchResponseSchema,
  ShopPresentationResponseSchema,
  ShopComparePricesResponseSchema,
  ShopExtractLocationResponseSchema,
} from '../contracts';
import type {
  ShopTranscribeRequest,
  ShopRefineQueryRequest,
  ShopSearchRequest,
  ShopPresentationRequest,
  ShopComparePricesRequest,
  ShopExtractLocationRequest,
} from '../contracts';

@Controller('api/shop-assistant')
@Public()
export class ShopAssistantController {
  constructor(private readonly shopAssistantService: ShopAssistantService) {}

  @Post('transcribe')
  @HttpCode(HttpStatus.OK)
  @UsePipes(new ZodValidationPipe(ShopTranscribeRequestSchema))
  async transcribe(@Body() body: ShopTranscribeRequest) {
    const result = await this.shopAssistantService.transcribe(body.voice_file_url);
    return parseOrThrow(ShopTranscribeResponseSchema, result, 'shop-assistant.transcribe.response');
  }

  @Post('refine-query')
  @HttpCode(HttpStatus.OK)
  @UsePipes(new ZodValidationPipe(ShopRefineQueryRequestSchema))
  async refineQuery(@Body() body: ShopRefineQueryRequest) {
    const result = await this.shopAssistantService.refineQuery(
      body.user_text,
      body.previous_params,
      body.role,
      body.prompt_content,
      body.model,
    );
    return parseOrThrow(ShopRefineQueryResponseSchema, result, 'shop-assistant.refine-query.response');
  }

  @Post('search')
  @HttpCode(HttpStatus.OK)
  @UsePipes(new ZodValidationPipe(ShopSearchRequestSchema))
  async search(@Body() body: ShopSearchRequest) {
    const result = await this.shopAssistantService.search(body.query_text, body.limit);
    return parseOrThrow(ShopSearchResponseSchema, result, 'shop-assistant.search.response');
  }

  @Post('format-presentation')
  @HttpCode(HttpStatus.OK)
  @UsePipes(new ZodValidationPipe(ShopPresentationRequestSchema))
  async formatPresentation(@Body() body: ShopPresentationRequest) {
    const result = await this.shopAssistantService.formatPresentation(
      body.results,
      body.query_text,
      body.role,
      body.prompt_content,
      body.model,
    );
    return parseOrThrow(ShopPresentationResponseSchema, result, 'shop-assistant.format-presentation.response');
  }

  @Post('compare-prices')
  @HttpCode(HttpStatus.OK)
  @UsePipes(new ZodValidationPipe(ShopComparePricesRequestSchema))
  async comparePrices(@Body() body: ShopComparePricesRequest) {
    const result = await this.shopAssistantService.comparePrices(
      body.results,
      body.query_text,
      body.role,
      body.prompt_content,
      body.model,
      body.priority_order,
    );
    return parseOrThrow(ShopComparePricesResponseSchema, result, 'shop-assistant.compare-prices.response');
  }

  @Post('extract-location')
  @HttpCode(HttpStatus.OK)
  @UsePipes(new ZodValidationPipe(ShopExtractLocationRequestSchema))
  async extractLocation(@Body() body: ShopExtractLocationRequest) {
    const result = await this.shopAssistantService.extractLocation(
      body.user_text,
      body.query_text,
      body.role,
      body.prompt_content,
      body.model,
      body.priority_order,
    );
    return parseOrThrow(ShopExtractLocationResponseSchema, result, 'shop-assistant.extract-location.response');
  }
}
```

- [ ] **Step 6: Run contract tests**

```bash
cd /home/ssf/Documents/Github/ai-microservice
npx jest src/contracts/contracts.spec.ts --no-coverage 2>&1 | tail -20
```

Expected: All tests PASS.

- [ ] **Step 7: TypeScript build check**

```bash
cd /home/ssf/Documents/Github/ai-microservice
npx tsc --noEmit 2>&1 | head -30
```

Expected: No errors.

---

## Task 2: Migrate ai controller from class-validator to ZodValidationPipe

The `AiCompleteRequestSchema` already exists in `src/contracts/ai-complete.contract.ts`. We remove the class-validator DTO and wire the Zod pipe.

**Files:**
- Modify: `src/ai/ai.controller.ts`
- Delete: `src/ai/dto/complete-request.dto.ts`
- Modify: `src/ai/ai.service.ts` (change param type)

- [ ] **Step 1: Replace controller body validation**

Replace `src/ai/ai.controller.ts` with:

```typescript
import { Controller, Post, Body, HttpCode, UsePipes } from '@nestjs/common';
import { AiService } from './ai.service';
import { ZodValidationPipe } from '../contracts/zod-validation.pipe';
import { parseOrThrow, AiCompleteRequestSchema, AiCompleteResponseSchema } from '../contracts';
import type { AiCompleteRequest, AiCompleteResponse } from '../contracts';

@Controller('ai')
export class AiController {
  constructor(private readonly aiService: AiService) {}

  @Post('complete')
  @HttpCode(200)
  @UsePipes(new ZodValidationPipe(AiCompleteRequestSchema))
  async complete(@Body() dto: AiCompleteRequest): Promise<AiCompleteResponse> {
    const result = await this.aiService.complete(dto);
    return parseOrThrow(AiCompleteResponseSchema, result, 'ai.complete.response');
  }
}
```

- [ ] **Step 2: Update ai.service.ts param type**

In `src/ai/ai.service.ts`, change the import and method signature:

Remove:
```typescript
import type { CompleteRequestDto } from './dto/complete-request.dto';
```

Add at top:
```typescript
import type { AiCompleteRequest } from '../contracts';
```

Change method signature from:
```typescript
async complete(dto: CompleteRequestDto): Promise<AiCompleteResult> {
```
To:
```typescript
async complete(dto: AiCompleteRequest): Promise<AiCompleteResult> {
```

- [ ] **Step 3: Delete the class-validator DTO file**

```bash
rm /home/ssf/Documents/Github/ai-microservice/src/ai/dto/complete-request.dto.ts
```

Check if the `dto/` directory is now empty and should be removed:

```bash
ls /home/ssf/Documents/Github/ai-microservice/src/ai/dto/
```

If only `complete-request.dto.ts` and `complete-response.dto.ts` remain, also delete the response DTO (it's now replaced by the Zod type):

```bash
rm /home/ssf/Documents/Github/ai-microservice/src/ai/dto/complete-response.dto.ts
rmdir /home/ssf/Documents/Github/ai-microservice/src/ai/dto/ 2>/dev/null || true
```

- [ ] **Step 4: TypeScript build check**

```bash
cd /home/ssf/Documents/Github/ai-microservice
npx tsc --noEmit 2>&1 | head -30
```

Expected: No errors. If errors mention `CompleteRequestDto`, there are other importers — fix them.

- [ ] **Step 5: Run full test suite**

```bash
cd /home/ssf/Documents/Github/ai-microservice
npx jest --no-coverage 2>&1 | tail -20
```

Expected: All tests PASS (or same as before this task — we haven't changed behavior).

---

## Task 3: Migrate task controller from class-validator to ZodValidationPipe

`TaskDraftRequestSchema` already exists in `src/contracts/task.contract.ts`.

**Files:**
- Modify: `src/task/task.controller.ts`
- Modify: `src/task/task.service.ts`
- Delete: `src/task/dto/task-draft.dto.ts`

- [ ] **Step 1: Replace controller body validation**

Replace `src/task/task.controller.ts` with:

```typescript
import { Controller, Post, Body, HttpCode, UsePipes } from '@nestjs/common';
import { TaskService } from './task.service';
import { ZodValidationPipe } from '../contracts/zod-validation.pipe';
import { parseOrThrow, TaskDraftRequestSchema, TaskDraftResponseSchema } from '../contracts';
import type { TaskDraftRequest, TaskDraftResponse } from '../contracts';

@Controller('task')
export class TaskController {
  constructor(private readonly taskService: TaskService) {}

  @Post('draft')
  @HttpCode(200)
  @UsePipes(new ZodValidationPipe(TaskDraftRequestSchema))
  async draft(@Body() dto: TaskDraftRequest): Promise<TaskDraftResponse> {
    const result = await this.taskService.draftTask(dto);
    return parseOrThrow(TaskDraftResponseSchema, result, 'task.draft.response');
  }
}
```

- [ ] **Step 2: Update task.service.ts param type**

In `src/task/task.service.ts`:

Remove:
```typescript
import type { TaskDraftDto } from './dto/task-draft.dto';
```

Add at top (with other imports):
```typescript
import type { TaskDraftRequest } from '../contracts';
```

Change method signature from:
```typescript
async draftTask(dto: TaskDraftDto): Promise<TaskDraftResponse> {
```
To:
```typescript
async draftTask(dto: TaskDraftRequest): Promise<TaskDraftResponse> {
```

Also remove the `TaskDraftResponse` and `TaskPriority` import from the local dto file (it now comes from contracts). The `TaskDraftResponse` type is already exported from `src/contracts/task.contract.ts`. Remove:
```typescript
import type { TaskDraftResponse, TaskPriority } from './dto/task-draft-response.dto';
```

Add:
```typescript
import type { TaskDraftResponse } from '../contracts';
type TaskPriority = 'low' | 'normal' | 'high';
```

- [ ] **Step 3: Delete task DTO files**

```bash
ls /home/ssf/Documents/Github/ai-microservice/src/task/dto/
```

Delete all class-validator DTOs in the task dto directory:

```bash
rm -rf /home/ssf/Documents/Github/ai-microservice/src/task/dto/
```

- [ ] **Step 4: TypeScript build check**

```bash
cd /home/ssf/Documents/Github/ai-microservice
npx tsc --noEmit 2>&1 | head -30
```

Expected: No errors.

- [ ] **Step 5: Run full test suite**

```bash
cd /home/ssf/Documents/Github/ai-microservice
npx jest --no-coverage 2>&1 | tail -20
```

Expected: All tests PASS.

---

## Task 4: Migrate voice controller from class-validator to ZodValidationPipe

`TranscribeRequestSchema` already exists in `src/contracts/voice.contract.ts`.

**Files:**
- Modify: `src/voice/voice.controller.ts`
- Modify: `src/voice/voice.service.ts`
- Delete: `src/voice/dto/transcribe.dto.ts`

- [ ] **Step 1: Replace controller body validation**

Replace `src/voice/voice.controller.ts` with:

```typescript
import { Controller, Post, Body, HttpCode, UsePipes } from '@nestjs/common';
import { VoiceService } from './voice.service';
import { ZodValidationPipe } from '../contracts/zod-validation.pipe';
import { parseOrThrow, TranscribeRequestSchema, TranscribeResponseSchema } from '../contracts';
import type { TranscribeRequest, TranscribeResponse } from '../contracts';

@Controller('voice')
export class VoiceController {
  constructor(private readonly voiceService: VoiceService) {}

  @Post('transcribe')
  @HttpCode(200)
  @UsePipes(new ZodValidationPipe(TranscribeRequestSchema))
  async transcribe(@Body() dto: TranscribeRequest): Promise<TranscribeResponse> {
    const result = await this.voiceService.transcribe(dto);
    return parseOrThrow(TranscribeResponseSchema, result, 'voice.transcribe.response');
  }
}
```

- [ ] **Step 2: Update voice.service.ts param type**

In `src/voice/voice.service.ts`:

Remove:
```typescript
import { TranscribeDto } from './dto/transcribe.dto';
```

Add:
```typescript
import type { TranscribeRequest } from '../contracts';
```

Change method signature from:
```typescript
async transcribe(dto: TranscribeDto): Promise<{ transcript: string }> {
```
To:
```typescript
async transcribe(dto: TranscribeRequest): Promise<{ transcript: string }> {
```

- [ ] **Step 3: Delete voice DTO file**

```bash
rm /home/ssf/Documents/Github/ai-microservice/src/voice/dto/transcribe.dto.ts
rmdir /home/ssf/Documents/Github/ai-microservice/src/voice/dto/ 2>/dev/null || true
```

- [ ] **Step 4: TypeScript build check**

```bash
cd /home/ssf/Documents/Github/ai-microservice
npx tsc --noEmit 2>&1 | head -30
```

Expected: No errors.

- [ ] **Step 5: Run full test suite**

```bash
cd /home/ssf/Documents/Github/ai-microservice
npx jest --no-coverage 2>&1 | tail -20
```

Expected: All tests PASS.

---

## Task 5: Migrate claude-code controller + add NotFoundResponseSchema

`ExecuteCodeRequestSchema` already exists in `src/contracts/claude-code-job.contract.ts`. The GET endpoint also has an uncontracted 404 branch.

**Files:**
- Modify: `src/claude-code/claude-code.controller.ts`
- Modify: `src/claude-code/claude-code.service.ts`
- Delete: `src/claude-code/dto/execute-code.dto.ts`
- Modify: `src/contracts/http-responses.contract.ts`
- Modify: `src/contracts/index.ts` (if `NotFoundResponseSchema` not already exported)
- Modify: `src/contracts/contracts.spec.ts`

- [ ] **Step 1: Add NotFoundResponseSchema to http-responses.contract.ts**

Append to `src/contracts/http-responses.contract.ts`:

```typescript
export const NotFoundResponseSchema = z.object({
  error: z.literal('Job not found'),
});
export type NotFoundResponse = z.infer<typeof NotFoundResponseSchema>;
```

- [ ] **Step 2: Add test for NotFoundResponseSchema**

Append to `src/contracts/contracts.spec.ts`:

```typescript
import { NotFoundResponseSchema } from './http-responses.contract';

describe('NotFoundResponseSchema', () => {
  it('accepts valid not-found response', () => {
    expect(NotFoundResponseSchema.safeParse({ error: 'Job not found' }).success).toBe(true);
  });
  it('rejects wrong error string', () => {
    expect(NotFoundResponseSchema.safeParse({ error: 'Not found' }).success).toBe(false);
  });
  it('rejects missing error field', () => {
    expect(NotFoundResponseSchema.safeParse({}).success).toBe(false);
  });
});
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /home/ssf/Documents/Github/ai-microservice
npx jest src/contracts/contracts.spec.ts --no-coverage 2>&1 | tail -20
```

Expected: FAIL — `NotFoundResponseSchema` not found yet.

- [ ] **Step 4: Replace claude-code controller**

Replace `src/claude-code/claude-code.controller.ts` with:

```typescript
import { Controller, Post, Get, Body, Param, HttpCode, UsePipes } from '@nestjs/common';
import { ClaudeCodeService } from './claude-code.service';
import { ZodValidationPipe } from '../contracts/zod-validation.pipe';
import {
  parseOrThrow,
  ExecuteCodeRequestSchema,
  JobEnqueueResponseSchema,
  JobStatusResponseSchema,
  NotFoundResponseSchema,
} from '../contracts';
import type { ExecuteCodeRequest } from '../contracts';

@Controller('ai/claude-code-execute')
export class ClaudeCodeController {
  constructor(private service: ClaudeCodeService) {}

  @Post()
  @HttpCode(201)
  @UsePipes(new ZodValidationPipe(ExecuteCodeRequestSchema))
  async executeCode(@Body() dto: ExecuteCodeRequest) {
    const result = await this.service.enqueueJob(dto);
    return parseOrThrow(JobEnqueueResponseSchema, result, 'claude-code.enqueue.response');
  }

  @Get(':jobId')
  async getStatus(@Param('jobId') jobId: string) {
    const status = await this.service.getJobStatus(jobId);
    if (!status) {
      return parseOrThrow(NotFoundResponseSchema, { error: 'Job not found' }, 'claude-code.status.not-found');
    }
    return parseOrThrow(JobStatusResponseSchema, status, 'claude-code.status.response');
  }
}
```

- [ ] **Step 5: Update claude-code.service.ts param type**

In `src/claude-code/claude-code.service.ts`:

Remove:
```typescript
import { ExecuteCodeDto } from './dto/execute-code.dto';
```

Add (with other imports):
```typescript
import type { ExecuteCodeRequest } from '../contracts';
```

Change `enqueueJob` method signature from:
```typescript
async enqueueJob(dto: ExecuteCodeDto): Promise<JobEnqueueResponseDto> {
```
To:
```typescript
async enqueueJob(dto: ExecuteCodeRequest): Promise<JobEnqueueResponseDto> {
```

- [ ] **Step 6: Delete claude-code DTO files**

```bash
ls /home/ssf/Documents/Github/ai-microservice/src/claude-code/dto/
rm /home/ssf/Documents/Github/ai-microservice/src/claude-code/dto/execute-code.dto.ts
```

Note: `job-result.dto.ts`, `job-enqueue-response.dto.ts`, and `job-status-response.dto.ts` may remain if they are still referenced by the service. Check:

```bash
grep -r "job-enqueue-response\|job-status-response\|job-result" /home/ssf/Documents/Github/ai-microservice/src --include="*.ts"
```

If only referenced in `claude-code.service.ts`, update those imports to use `JobEnqueueResponse` and `JobStatusResponse` from contracts. If complex, leave them for a follow-up.

- [ ] **Step 7: Run contract tests**

```bash
cd /home/ssf/Documents/Github/ai-microservice
npx jest src/contracts/contracts.spec.ts --no-coverage 2>&1 | tail -20
```

Expected: All tests PASS including new `NotFoundResponseSchema` tests.

- [ ] **Step 8: TypeScript build check**

```bash
cd /home/ssf/Documents/Github/ai-microservice
npx tsc --noEmit 2>&1 | head -30
```

Expected: No errors.

- [ ] **Step 9: Run full test suite**

```bash
cd /home/ssf/Documents/Github/ai-microservice
npx jest --no-coverage 2>&1 | tail -30
```

Expected: All tests PASS.

---

## Task 6: Final verification and issue close

- [ ] **Step 1: Full test suite**

```bash
cd /home/ssf/Documents/Github/ai-microservice
npx jest --no-coverage 2>&1 | tail -20
```

Expected: All tests PASS.

- [ ] **Step 2: TypeScript build clean**

```bash
cd /home/ssf/Documents/Github/ai-microservice
npx tsc --noEmit 2>&1
```

Expected: No output (zero errors).

- [ ] **Step 3: Verify no remaining class-validator DTOs on controllers**

```bash
grep -rn "class-validator" /home/ssf/Documents/Github/ai-microservice/src --include="*.ts" | grep -v "node_modules"
```

Expected: No hits in controller files. Any remaining hits should be in non-controller files (e.g. if there are still internal-only DTOs not on HTTP boundaries — those are acceptable to leave).

- [ ] **Step 4: Verify all controllers use ZodValidationPipe or parseOrThrow on responses**

```bash
grep -rn "ZodValidationPipe\|parseOrThrow" /home/ssf/Documents/Github/ai-microservice/src --include="*.controller.ts"
```

Expected: Every controller file appears in the results.

- [ ] **Step 5: Update compliance table in shared CONTRACT_STANDARD.md**

In `shared/docs/CONTRACT_STANDARD.md`, the reference implementations table already lists ai-microservice. Verify it's still accurate — no change needed.

- [ ] **Step 6: Post completion comment and close Issue #2**

```bash
gh issue comment 2 --repo speakASAP/ai-microservice --body "## Completed

**What was done:**
- Added 6 response schemas to shop-assistant.contract.ts (ShopTranscribeResponseSchema, ShopRefineQueryResponseSchema, ShopSearchResponseSchema, ShopPresentationResponseSchema, ShopComparePricesResponseSchema, ShopExtractLocationResponseSchema)
- Wired parseOrThrow on all 6 shop-assistant controller responses
- Migrated ai, task, voice, claude-code controllers from class-validator DTOs to ZodValidationPipe + Zod schemas
- Removed 4 class-validator DTO files (complete-request.dto.ts, task-draft.dto.ts, transcribe.dto.ts, execute-code.dto.ts)
- Added NotFoundResponseSchema to http-responses.contract.ts; wired it in claude-code GET endpoint
- All new schemas covered by unit tests in contracts.spec.ts

**Files changed:**
- src/contracts/shop-assistant.contract.ts (added 6 response schemas)
- src/contracts/http-responses.contract.ts (added NotFoundResponseSchema)
- src/contracts/contracts.spec.ts (added tests for all new schemas)
- src/ai/ai.controller.ts, src/task/task.controller.ts, src/voice/voice.controller.ts, src/claude-code/claude-code.controller.ts (ZodValidationPipe on inputs)
- src/ai/ai.service.ts, src/task/task.service.ts, src/voice/voice.service.ts, src/claude-code/claude-code.service.ts (updated param types)
- src/shop-assistant/shop-assistant.controller.ts (parseOrThrow on all responses)
- Deleted: src/ai/dto/complete-request.dto.ts, src/task/dto/task-draft.dto.ts, src/voice/dto/transcribe.dto.ts, src/claude-code/dto/execute-code.dto.ts

**Outcome:**
All HTTP boundaries in ai-microservice now use Zod for both input and output validation. Full parity with business-orchestrator contract enforcement standard. TypeScript builds clean, full test suite passes."

gh issue close 2 --repo speakASAP/ai-microservice
```
