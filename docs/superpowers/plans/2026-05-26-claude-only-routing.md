# Claude-Only LLM Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken LiteLLM/Ollama multi-tier routing with a direct Anthropic API call using `claude-sonnet-4-6` for every request, while keeping the `model_tier` field in the request contract so business-orchestrator needs zero changes.

**Architecture:** `AiService.complete()` calls the Anthropic API directly via `fetch` using the `ANTHROPIC_API_KEY` already in Vault. The `model_tier` field in `CompleteRequestDto` is accepted but ignored — every call goes to `claude-sonnet-4-6`. LiteLLM and Ollama K8s containers are removed from the deployment. Token counting is preserved from the Anthropic usage response.

**Tech Stack:** NestJS, TypeScript, Anthropic Messages API (`https://api.anthropic.com/v1/messages`), K8s/Vault (existing)

**GitHub Issue:** https://github.com/speakASAP/ai-microservice/issues/1

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/ai/ai.service.ts` | Replace LiteLLM fetch with Anthropic Messages API fetch |
| Modify | `src/ai/ai.controller.ts` | No change needed — contract preserved |
| Modify | `src/ai/dto/complete-request.dto.ts` | Add comment that model_tier is ignored |
| Modify | `.env.example` | Mark LITELLM_* as deprecated, ANTHROPIC_API_KEY as required |
| Modify | `k8s/deployment.yaml` | Remove litellm + ollama containers/init-containers |
| Delete | `litellm_config.yaml` | No longer used |

---

### Task 1: Rewrite AiService to call Anthropic directly

**Files:**
- Modify: `src/ai/ai.service.ts`

- [ ] **Step 1: Write the failing test (verify it currently calls LiteLLM)**

Create `src/ai/ai.service.spec.ts`:

```typescript
import { AiService } from './ai.service';
import { Test } from '@nestjs/testing';

describe('AiService - Claude direct', () => {
  let service: AiService;

  beforeEach(async () => {
    const module = await Test.createTestingModule({ providers: [AiService] }).compile();
    service = module.get(AiService);
  });

  it('calls Anthropic API when ANTHROPIC_API_KEY is set', async () => {
    const mockFetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        content: [{ type: 'text', text: '{"result":"ok"}' }],
        usage: { input_tokens: 10, output_tokens: 5 },
        model: 'claude-sonnet-4-6-20251001',
      }),
    });
    global.fetch = mockFetch as any;
    process.env.ANTHROPIC_API_KEY = 'test-key';

    const result = await service.complete({
      model_tier: 'free', // ignored — always uses Claude
      user_prompt: 'hello',
    });

    expect(mockFetch).toHaveBeenCalledWith(
      'https://api.anthropic.com/v1/messages',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({ 'x-api-key': 'test-key' }),
      }),
    );
    expect(result.inputTokens).toBe(10);
    expect(result.outputTokens).toBe(5);
    expect(result.token_usage_estimate).toBe(15);
    expect(result.model_used).toBe('claude-sonnet-4-6');
  });

  it('throws when ANTHROPIC_API_KEY is missing', async () => {
    delete process.env.ANTHROPIC_API_KEY;
    await expect(service.complete({ model_tier: 'free', user_prompt: 'hi' }))
      .rejects.toThrow('ANTHROPIC_API_KEY');
  });
});
```

- [ ] **Step 2: Run test to confirm it fails (current impl uses LiteLLM)**

```bash
cd /home/ssf/Documents/Github/ai-microservice
npx jest src/ai/ai.service.spec.ts --no-coverage
```

Expected: FAIL — `mockFetch` not called with `api.anthropic.com`

- [ ] **Step 3: Rewrite ai.service.ts**

Replace the entire content of `src/ai/ai.service.ts` with:

```typescript
import { Injectable, InternalServerErrorException } from '@nestjs/common';
import type { CompleteRequestDto } from './dto/complete-request.dto';

const CLAUDE_MODEL = 'claude-sonnet-4-6-20251001';
const ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages';
const ANTHROPIC_API_VERSION = '2023-06-01';

interface AnthropicResponse {
  content: Array<{ type: string; text: string }>;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
  };
  model?: string;
}

/**
 * The response from /ai/complete is a flat object that merges:
 *   - metadata fields (model_used, inputTokens, outputTokens, token_usage_estimate, text)
 *   - the parsed JSON payload fields spread at the top level so callers can access
 *     e.g. response.output_ref, response.passed, response.new_tasks directly.
 */
export type AiCompleteResult = Record<string, unknown> & {
  text: string;
  model_used: string;
  inputTokens: number;
  outputTokens: number;
  token_usage_estimate: number;
};

@Injectable()
export class AiService {
  async complete(dto: CompleteRequestDto): Promise<AiCompleteResult> {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      throw new InternalServerErrorException('ANTHROPIC_API_KEY is not configured');
    }

    // Build messages — system_prompt goes in the Anthropic `system` field
    const messages: Array<{ role: string; content: string }> = [
      { role: 'user', content: dto.user_prompt },
    ];

    const requestBody: Record<string, unknown> = {
      model: CLAUDE_MODEL,
      max_tokens: dto.max_tokens ?? 1024,
      messages,
    };

    if (dto.system_prompt) {
      requestBody['system'] = dto.system_prompt;
    }

    // Request JSON output when output_schema is provided
    if (dto.output_schema) {
      const currentSystem = (requestBody['system'] as string | undefined) ?? '';
      requestBody['system'] = `${currentSystem}\nRespond with valid JSON only. No markdown fences.`.trim();
    }

    const res = await fetch(ANTHROPIC_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': ANTHROPIC_API_VERSION,
      },
      body: JSON.stringify(requestBody),
      signal: AbortSignal.timeout(120_000),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new InternalServerErrorException(`Anthropic API error ${res.status}: ${errText}`);
    }

    const body = (await res.json()) as AnthropicResponse;

    const rawText = body.content?.find((c) => c.type === 'text')?.text ?? '';
    const inputTokens = body.usage?.input_tokens ?? 0;
    const outputTokens = body.usage?.output_tokens ?? 0;
    const token_usage_estimate = inputTokens + outputTokens;

    // Attempt JSON parse and spread fields at top level (same contract as before)
    let parsedData: Record<string, unknown> = {};
    const trimmed = rawText.trim();
    if (trimmed.startsWith('{') || trimmed.startsWith('[') || dto.output_schema) {
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

    return {
      ...parsedData,
      text: rawText,
      model_used: 'claude-sonnet-4-6',
      inputTokens,
      outputTokens,
      token_usage_estimate,
    };
  }
}
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
npx jest src/ai/ai.service.spec.ts --no-coverage
```

Expected: PASS — both tests green

- [ ] **Step 5: Commit**

```bash
git add src/ai/ai.service.ts src/ai/ai.service.spec.ts
git commit -m "feat: replace LiteLLM routing with direct Anthropic claude-sonnet-4-6 calls"
```

---

### Task 2: Update .env.example and deprecate LiteLLM vars

**Files:**
- Modify: `.env.example`
- Delete: `litellm_config.yaml`

- [ ] **Step 1: Update .env.example**

In `.env.example`, find the `LITELLM_MASTER_KEY` and `LITELLM_BASE_URL` lines and add deprecation comments:

```bash
# DEPRECATED: LiteLLM routing replaced by direct Anthropic API (see issue #1)
# LITELLM_MASTER_KEY=
# LITELLM_BASE_URL=
```

Also ensure this line is present and uncommented:

```bash
# Required: Claude API key (stored in Vault secret/prod/ai-microservice)
ANTHROPIC_API_KEY=
```

- [ ] **Step 2: Delete litellm_config.yaml**

```bash
cd /home/ssf/Documents/Github/ai-microservice
git rm litellm_config.yaml
```

- [ ] **Step 3: Commit**

```bash
git add .env.example
git commit -m "chore: deprecate LiteLLM env vars, remove litellm_config.yaml"
```

---

### Task 3: Remove LiteLLM and Ollama from K8s deployment

**Files:**
- Modify: `k8s/deployment.yaml`

- [ ] **Step 1: Identify containers to remove**

```bash
grep -n "litellm\|ollama\|name:" /home/ssf/Documents/Github/ai-microservice/k8s/deployment.yaml | head -40
```

Note the container names for litellm and ollama sidecars/init-containers.

- [ ] **Step 2: Remove litellm and ollama container blocks**

Edit `k8s/deployment.yaml` to remove:
- Any container block with `name: litellm` or `image:` containing `litellm`
- Any container block with `name: ollama` or `image:` containing `ollama`
- Any init-container that pulls ollama models
- Any `LITELLM_BASE_URL` and `LITELLM_MASTER_KEY` env vars from the main container

Keep: the main NestJS container block, all other env vars, health probes, volume mounts for non-LiteLLM purposes.

- [ ] **Step 3: Verify the YAML is valid**

```bash
kubectl apply --dry-run=client -f /home/ssf/Documents/Github/ai-microservice/k8s/deployment.yaml
```

Expected: `deployment.apps/ai-microservice configured (dry run)`

- [ ] **Step 4: Commit**

```bash
git add k8s/deployment.yaml
git commit -m "chore: remove litellm and ollama containers from K8s deployment"
```

---

### Task 4: TypeScript build check

- [ ] **Step 1: Full build**

```bash
cd /home/ssf/Documents/Github/ai-microservice
npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 2: Full test suite**

```bash
npx jest --no-coverage
```

Expected: all tests pass

---

### Task 5: Deploy and verify end-to-end

- [ ] **Step 1: Deploy ai-microservice**

```bash
cd /home/ssf/Documents/Github/ai-microservice
bash scripts/deploy.sh
```

- [ ] **Step 2: Wait for rollout**

```bash
kubectl rollout status deployment/ai-microservice -n statex-apps --timeout=120s
```

Expected: `deployment "ai-microservice" successfully rolled out`

- [ ] **Step 3: Health check**

```bash
kubectl exec -n statex-apps deployment/ai-microservice -- wget -qO- http://localhost:3380/health
```

Expected: `{"status":"ok"}`

- [ ] **Step 4: Live smoke test — POST /ai/complete**

```bash
# Get JWT first
TOKEN=$(kubectl exec -n statex-apps deployment/ai-microservice -- \
  node -e "const j=require('./dist/service-identity/jwt.util'); console.log(j.JwtUtil.sign('test-client', process.env.JWT_SECRET, 60))" 2>/dev/null)

kubectl exec -n statex-apps deployment/ai-microservice -- \
  wget -qO- --post-data='{"model_tier":"free","user_prompt":"Reply with the single word: hello"}' \
  --header="Content-Type: application/json" \
  --header="Authorization: Bearer $TOKEN" \
  http://localhost:3380/ai/complete
```

Expected: JSON response with `"model_used":"claude-sonnet-4-6"` and non-zero `token_usage_estimate`

- [ ] **Step 5: Close GitHub issue**

```bash
gh issue comment 1 --repo speakASAP/ai-microservice --body "## Completed

**What was done:**
- Rewrote AiService.complete() to call Anthropic Messages API directly (claude-sonnet-4-6-20251001)
- model_tier field preserved in contract but ignored at runtime
- Token counting preserved: inputTokens/outputTokens from Anthropic usage response
- Deprecated LITELLM_BASE_URL and LITELLM_MASTER_KEY env vars
- Removed litellm_config.yaml
- Removed litellm + ollama containers from k8s/deployment.yaml

**Files changed:**
- src/ai/ai.service.ts (rewritten)
- src/ai/ai.service.spec.ts (new)
- .env.example (deprecated LiteLLM vars)
- litellm_config.yaml (deleted)
- k8s/deployment.yaml (removed litellm/ollama containers)

**Outcome:** All AI calls go to claude-sonnet-4-6. Pod starts without LiteLLM or Ollama. Smoke test confirms end-to-end response."

gh issue close 1 --repo speakASAP/ai-microservice
```
