# RTK + Caveman Token Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install RTK + Caveman globally on the server and add token-savings telemetry to `ai-microservice` so token costs are reduced 60–90% with zero changes to any API contracts or calling services.

**Architecture:** RTK installs as a Rust binary and hooks into Claude Code `PreToolUse` to compress Bash output before it enters the model. Caveman installs as a Claude Code plugin and compresses model prose at `lite` intensity. Both are inherited by any Claude Code subprocess. Telemetry is logged via the existing `LoggingClient` in `ai.service.ts`.

**Tech Stack:** Rust binary (RTK), Node.js plugin (Caveman), NestJS (`ai-microservice`), existing `LoggingClient`

---

## File Map

| File | Change |
|---|---|
| `~/.claude/settings.json` | RTK writes PreToolUse hook here via `rtk init -g` |
| `~/.claude/plugins/caveman/` | Caveman installer creates this |
| `/home/ssf/Documents/Github/shared/AGENTS.md` | Add Caveman `lite` default |
| `ai-microservice/src/ai/ai.module.ts` | Inject `LoggingClient` into `AiModule` |
| `ai-microservice/src/ai/ai.service.ts` | Accept `LoggingClient`, emit telemetry log after each completion |
| `ai-microservice/src/ai/ai.service.spec.ts` | Add test: telemetry log called with correct shape |

---

### Task 1: Install RTK

**Files:**
- Modifies: `~/.claude/settings.json` (via `rtk init -g`)

- [ ] **Step 1: Install RTK binary**

```bash
curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh | sh
```

Expected output: RTK installed to `~/.local/bin/rtk` (or similar). Confirm with:
```bash
rtk --version
```

- [ ] **Step 2: Wire RTK into Claude Code PreToolUse hook**

```bash
rtk init -g
```

Expected: command succeeds, no error. This writes a `PreToolUse` hook entry into `~/.claude/settings.json`.

- [ ] **Step 3: Verify hook was written**

```bash
grep -A5 "PreToolUse\|rtk" ~/.claude/settings.json
```

Expected: a hook block referencing `rtk-rewrite.sh` or similar appears in the JSON.

- [ ] **Step 4: Smoke-test RTK compression**

```bash
rtk git status
```

Expected: output is shorter/compressed compared to `git status` raw output. If you see similar output to raw `git status`, RTK is working (it only compresses noise; clean repos show little difference). No error = pass.

- [ ] **Step 5: Commit note (no git action — document result)**

Record in `STATE.json` at repo root of `ai-microservice`:
```json
{ "rtk": "installed", "rtk_hook": "PreToolUse wired via rtk init -g" }
```

---

### Task 2: Install Caveman

**Files:**
- Creates: `~/.claude/plugins/caveman/` (via installer)

- [ ] **Step 1: Install Caveman plugin**

```bash
curl -fsSL https://raw.githubusercontent.com/JuliusBrussee/caveman/main/install.sh | bash
```

Expected: installer auto-detects Claude Code, installs plugin to `~/.claude/plugins/caveman/`. No errors.

- [ ] **Step 2: Verify plugin installed**

```bash
ls ~/.claude/plugins/caveman/
```

Expected: directory exists with plugin files (e.g., `manifest.json` or similar).

- [ ] **Step 3: Verify Caveman appears in Claude Code**

Start a new Claude Code session and run:
```
/caveman lite
```

Expected: Claude Code acknowledges the compression level change. If the command is not recognized, check `~/.claude/plugins/installed_plugins.json` to confirm caveman is listed.

- [ ] **Step 4: Set lite as ecosystem default in shared/AGENTS.md**

Open `/home/ssf/Documents/Github/shared/AGENTS.md` and add this block after the existing `## Knowledge Retrieval` section:

```markdown
## Token Compression

Caveman is active at `lite` intensity by default across all sessions and subagents.
- `lite` — removes filler words only; code, paths, JSON, file content stay byte-perfect
- Override per-session: `/caveman full` (telegraphic) or `/caveman ultra` (maximum brevity)
- Do not set higher than `lite` for human-facing output (notifications, emails, UI text)
```

---

### Task 3: Inject LoggingClient into AiModule

**Files:**
- Modify: `ai-microservice/src/ai/ai.module.ts`
- Modify: `ai-microservice/src/ai/ai.service.ts` (constructor only in this task)

Context: `LoggingClient` already exists at `src/claude-code/logging.client.ts` and is used by other modules. The `LOGGING_URL_TOKEN` and `LOGGING_FETCH_TOKEN` injection tokens are exported from that file.

- [ ] **Step 1: Write failing test for telemetry emission**

Open `ai-microservice/src/ai/ai.service.spec.ts`. Add this test (keep any existing tests):

```typescript
import { AiService } from './ai.service';
import { LoggingClient } from '../claude-code/logging.client';

describe('AiService telemetry', () => {
  let service: AiService;
  let loggingClient: jest.Mocked<LoggingClient>;

  beforeEach(() => {
    loggingClient = { log: jest.fn().mockResolvedValue(undefined) } as any;
    service = new AiService(loggingClient);
  });

  it('emits ai_complete log with compression metadata after successful completion', async () => {
    // Mock the CC CLI subprocess to return a valid envelope
    jest.spyOn(service as any, 'spawnCcCli').mockResolvedValue(
      JSON.stringify({
        result: 'hello world',
        usage: { input_tokens: 100, output_tokens: 20 },
      }),
    );

    await service.complete({
      model_tier: 'smart',
      user_prompt: 'say hello',
      correlation_id: 'test-corr-123',
    });

    expect(loggingClient.log).toHaveBeenCalledWith(
      'info',
      'ai_complete',
      expect.objectContaining({
        correlation_id: 'test-corr-123',
        inputTokens: 100,
        outputTokens: 20,
        compression: { rtk: true, caveman: 'lite' },
      }),
    );
  });

  it('emits ai_complete log with zero tokens on CLI failure', async () => {
    jest.spyOn(service as any, 'spawnCcCli').mockRejectedValue(
      new Error('claude CLI failed: timeout'),
    );

    await service.complete({
      model_tier: 'free',
      user_prompt: 'say hello',
      correlation_id: 'test-corr-456',
    });

    expect(loggingClient.log).toHaveBeenCalledWith(
      'info',
      'ai_complete',
      expect.objectContaining({
        correlation_id: 'test-corr-456',
        inputTokens: 0,
        outputTokens: 0,
        compression: { rtk: true, caveman: 'lite' },
      }),
    );
  });
});
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /home/ssf/Documents/Github/ai-microservice && npx jest src/ai/ai.service.spec.ts -t "telemetry" --no-coverage
```

Expected: FAIL — `AiService` constructor does not accept `LoggingClient` yet.

- [ ] **Step 3: Update AiModule to provide LoggingClient**

Replace `ai-microservice/src/ai/ai.module.ts` with:

```typescript
import { Module } from '@nestjs/common';
import { AiController } from './ai.controller';
import { AiService } from './ai.service';
import { LoggingClient } from '../claude-code/logging.client';

@Module({
  controllers: [AiController],
  providers: [AiService, LoggingClient],
  exports: [AiService],
})
export class AiModule {}
```

- [ ] **Step 4: Update AiService constructor to accept LoggingClient**

In `ai-microservice/src/ai/ai.service.ts`, change the class declaration and constructor (do NOT change any other logic):

```typescript
// Add to imports at top of file:
import { LoggingClient } from '../claude-code/logging.client';

// Replace the class opening and constructor:
@Injectable()
export class AiService {
  private readonly logger = new Logger(AiService.name);
  private activeProcesses = 0;

  constructor(private readonly loggingClient: LoggingClient) {}
```

---

### Task 4: Emit Telemetry in ai.service.ts

**Files:**
- Modify: `ai-microservice/src/ai/ai.service.ts`

This task adds the telemetry `log()` call. The subprocess logic is NOT touched — only the return paths.

- [ ] **Step 1: Extract spawn logic into private method (enables test mocking)**

In `ai.service.ts`, wrap the existing `spawn` Promise block into a private method `spawnCcCli(tmpFile: string): Promise<string>`. The method contains exactly the existing `new Promise<string>((resolve, reject) => { ... })` block (the part that spawns the child process and sets up the timer). Return type is `Promise<string>` (the raw stdout).

The `complete()` method calls `await this.spawnCcCli(tmpFile)` where it previously had the inline Promise.

This is a pure refactor — behavior is identical.

- [ ] **Step 2: Add telemetry emit helper**

Add this private method to `AiService` after `spawnCcCli`:

```typescript
private emitTelemetry(
  correlationId: string | undefined,
  modelUsed: string,
  inputTokens: number,
  outputTokens: number,
): void {
  this.loggingClient
    .log('info', 'ai_complete', {
      correlation_id: correlationId,
      model_used: modelUsed,
      inputTokens,
      outputTokens,
      token_usage_estimate: inputTokens + outputTokens,
      compression: { rtk: true, caveman: 'lite' },
    })
    .catch(() => {/* logging must never crash the service */});
}
```

- [ ] **Step 3: Call emitTelemetry at all return points in complete()**

There are four return paths in `complete()`. Add `this.emitTelemetry(...)` before each `return`:

**Path 1 — concurrency limit reached** (returns `cliFailureResult`):
```typescript
this.emitTelemetry(dto.correlation_id, `claude-${model}`, 0, 0);
return cliFailureResult(model, `claude CLI concurrency limit reached (${CC_MAX_CONCURRENT} active)`);
```

**Path 2 — CLI error caught in catch block** (returns `cliFailureResult`):
```typescript
this.emitTelemetry(dto.correlation_id, `claude-${model}`, 0, 0);
return cliFailureResult(model, `claude CLI failed: ${detail}`);
```

**Path 3 — CC API error envelope** (returns `ccApiErrorResult`):
```typescript
this.emitTelemetry(dto.correlation_id, `claude-${model}`, 0, 0);
return ccApiErrorResult(model, ccResult);
```

**Path 4 — success** (returns the spread result object at the bottom of `complete()`). Replace the final `return { ...parsedData, text: rawText, ... }` with:
```typescript
this.emitTelemetry(dto.correlation_id, `claude-${model}`, inputTokens, outputTokens);
return {
  ...parsedData,
  text: rawText,
  model_used: `claude-${model}`,
  inputTokens,
  outputTokens,
  token_usage_estimate: inputTokens + outputTokens,
};
```

- [ ] **Step 4: Run the telemetry tests**

```bash
cd /home/ssf/Documents/Github/ai-microservice && npx jest src/ai/ai.service.spec.ts -t "telemetry" --no-coverage
```

Expected: both tests PASS.

- [ ] **Step 5: Run the full test suite**

```bash
cd /home/ssf/Documents/Github/ai-microservice && npm test -- --no-coverage
```

Expected: all existing tests still pass (no regressions).

- [ ] **Step 6: TypeScript build check**

```bash
cd /home/ssf/Documents/Github/ai-microservice && npm run build
```

Expected: exits 0, no type errors.

---

### Task 5: Deploy and Verify

**Files:** None (deploy only)

- [ ] **Step 1: Deploy ai-microservice to K8s**

```bash
cd /home/ssf/Documents/Github/ai-microservice && ./scripts/deploy.sh
```

Expected: image builds, pushes to registry, rollout completes. Watch with:
```bash
kubectl rollout status deployment/ai-microservice -n statex-apps
```

- [ ] **Step 2: Smoke-test /ai/complete**

```bash
curl -s -X POST https://ai.alfares.cz/ai/complete \
  -H "Content-Type: application/json" \
  -d '{"schemaVersion":"1.0","model_tier":"free","user_prompt":"say hello in one word","correlation_id":"rtk-smoke-test"}' \
  | jq '{text, inputTokens, outputTokens, token_usage_estimate}'
```

Expected: JSON response with non-zero `inputTokens` and `outputTokens`.

- [ ] **Step 3: Verify telemetry log was emitted**

```bash
curl -s "http://localhost:3367/api/logs?service=ai-microservice&limit=5" \
  | jq '.[] | select(.message == "ai_complete") | .metadata'
```

Expected: log entry with `compression: { rtk: true, caveman: "lite" }` and non-zero token counts.

- [ ] **Step 4: Verify RTK hook survives new Claude Code session**

Open a new Claude Code session and run:
```bash
grep "rtk" ~/.claude/settings.json
```

Expected: RTK PreToolUse hook still present (confirm `rtk init -g` persisted to settings).

---

## Self-Review

**Spec coverage:**
- ✅ RTK install + PreToolUse hook — Task 1
- ✅ Caveman install + lite default in AGENTS.md — Task 2
- ✅ LoggingClient injected into AiModule — Task 3
- ✅ Telemetry emitted at all return paths — Task 4
- ✅ Deploy + verify — Task 5
- ✅ No AiCompleteRequestSchema changes — confirmed by plan (not touched)
- ✅ Zero changes to other microservices — confirmed

**Placeholder scan:** None found.

**Type consistency:**
- `LoggingClient` constructor signature matches existing `src/claude-code/logging.client.ts` (no injection tokens needed — plain class injection via NestJS DI)
- `emitTelemetry` uses `string | undefined` for `correlationId` matching `dto.correlation_id?: string` in the contract
- `spawnCcCli` returns `Promise<string>` matching existing Promise block's resolution type
