# Fix ai.service.ts Working-Tree Drift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore `ai.service.ts` to the correct Anthropic-fetch implementation (already committed at HEAD), commit the legitimate Dockerfile and deployment.yaml improvements, rebuild the Docker image from clean source, and redeploy so the running pod matches git HEAD.

**Architecture:** Three uncommitted working-tree changes exist. `src/ai/ai.service.ts` was incorrectly overwritten with a claude-CLI subprocess version — it must be reverted to HEAD (Anthropic Messages API via `fetch`). The other two files (`Dockerfile`: `node:20-slim` base, `k8s/deployment.yaml`: security context added, hostPath volumes **removed**) are legitimate improvements and should be committed. Once source is clean, rebuild the Docker image and redeploy via `./scripts/deploy.sh`.

**Tech Stack:** NestJS · TypeScript · Docker (`node:20-slim`) · Kubernetes (kubectl) · Jest

---

## File Map

| File | Action | Why |
|---|---|---|
| `src/ai/ai.service.ts` | Revert to HEAD | Contains wrong claude-CLI implementation; HEAD has correct Anthropic fetch |
| `Dockerfile` | Commit as-is | `node:20-slim` (glibc) is correct; `node:20-alpine` (musl) was the original |
| `k8s/deployment.yaml` | Commit as-is | Adds `securityContext`; critically **removes** the hostPath volumes that caused ELF failures |
| `src/ai/ai.service.spec.ts` | No change needed | Already tests the Anthropic fetch path; tests will pass once `ai.service.ts` is restored |

---

### Task 1: Revert `ai.service.ts` to HEAD

**Files:**
- Modify: `src/ai/ai.service.ts` (revert to committed version)

- [ ] **Step 1: Verify the working-tree drift**

```bash
git -C /home/ssf/Documents/Github/ai-microservice diff HEAD -- src/ai/ai.service.ts | head -20
```

Expected: shows removal of `const CLAUDE_MODEL`, `ANTHROPIC_API_URL`, `fetch(...)` and addition of `execAsync`, `CC_CLI`, etc.

- [ ] **Step 2: Revert only `ai.service.ts` to HEAD**

```bash
git -C /home/ssf/Documents/Github/ai-microservice checkout HEAD -- src/ai/ai.service.ts
```

Expected: no output (success).

- [ ] **Step 3: Confirm revert**

```bash
git -C /home/ssf/Documents/Github/ai-microservice diff HEAD -- src/ai/ai.service.ts
```

Expected: empty output (no diff).

- [ ] **Step 4: Confirm the correct implementation is on disk**

```bash
grep -n "ANTHROPIC_API_URL\|fetch(\|execAsync\|CC_CLI" /home/ssf/Documents/Github/ai-microservice/src/ai/ai.service.ts
```

Expected: lines with `ANTHROPIC_API_URL` and `fetch(` — no `execAsync` or `CC_CLI`.

---

### Task 2: Run tests to confirm correctness

**Files:**
- Test: `src/ai/ai.service.spec.ts`

- [ ] **Step 1: Run the unit tests**

```bash
cd /home/ssf/Documents/Github/ai-microservice && npm test -- --testPathPattern=ai.service.spec.ts --no-coverage 2>&1
```

Expected: `Tests: 2 passed, 2 total` — both tests green.

The two tests verify:
1. `complete()` calls `https://api.anthropic.com/v1/messages` with `x-api-key` header when `ANTHROPIC_API_KEY` is set
2. `complete()` throws with `'ANTHROPIC_API_KEY'` in the message when the key is missing

- [ ] **Step 2: If tests fail — diagnose**

Failure here means `ai.service.ts` was not fully reverted. Re-run Task 1 Step 2 and check for any remaining `execAsync` references:

```bash
grep "execAsync\|CC_CLI\|claude CLI" /home/ssf/Documents/Github/ai-microservice/src/ai/ai.service.ts
```

Expected: no output. If anything prints, the revert did not apply cleanly.

---

### Task 3: Commit Dockerfile and deployment.yaml improvements

**Files:**
- Modify: `Dockerfile` (commit existing working-tree state — `node:20-slim`)
- Modify: `k8s/deployment.yaml` (commit existing working-tree state — adds `securityContext`, removes hostPath volumes)

The hostPath volume removal is the most important part: the volumes mounted `/home/ssf/.local` and `/home/ssf/.claude` into the pod so the container could find the host's `claude` CLI binary. That binary is a Node.js ELF executable that cannot run inside the container (missing interpreter). Removing the volumes eliminates this failure mode entirely — the correct `ai.service.ts` never touches the CLI anyway.

- [ ] **Step 1: Verify the two files are still modified**

```bash
git -C /home/ssf/Documents/Github/ai-microservice status
```

Expected: `modified: Dockerfile` and `modified: k8s/deployment.yaml` listed under "Changes not staged for commit". `src/ai/ai.service.ts` should NOT appear (reverted in Task 1).

- [ ] **Step 2: Confirm deployment.yaml has no hostPath volumes**

```bash
grep "hostPath\|local-bin\|claude-config" /home/ssf/Documents/Github/ai-microservice/k8s/deployment.yaml
```

Expected: no output — volumes are absent from the working-tree file.

- [ ] **Step 3: Confirm Dockerfile uses node:20-slim**

```bash
grep "^FROM" /home/ssf/Documents/Github/ai-microservice/Dockerfile
```

Expected:
```
FROM node:20-slim AS builder
FROM node:20-slim AS runner
```

- [ ] **Step 4: Stage and commit**

```bash
git -C /home/ssf/Documents/Github/ai-microservice add Dockerfile k8s/deployment.yaml
git -C /home/ssf/Documents/Github/ai-microservice commit -m "$(cat <<'EOF'
chore: use node:20-slim base image and remove hostPath volume mounts

- Switch Dockerfile from node:20-alpine to node:20-slim (glibc, avoids
  ELF interpreter mismatch when mounting host binaries)
- Remove hostPath volumes for /home/ssf/.local and /home/ssf/.claude
  from deployment.yaml — these were added for a claude-CLI subprocess
  approach that was superseded by direct Anthropic API calls
- Add securityContext (runAsUser/Group/fsGroup 1000) to deployment

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

Expected: commit hash printed, e.g. `[main abc1234] chore: use node:20-slim...`

- [ ] **Step 5: Verify clean working tree**

```bash
git -C /home/ssf/Documents/Github/ai-microservice status
```

Expected: `nothing to commit, working tree clean`

---

### Task 4: Rebuild Docker image and redeploy

**Files:**
- No source changes — this task builds from the now-clean working tree

- [ ] **Step 1: Run deploy.sh**

```bash
cd /home/ssf/Documents/Github/ai-microservice && ./scripts/deploy.sh 2>&1
```

This script:
1. Builds Docker image tagged `<git-short-sha>` and `:latest` from the current working tree
2. Pushes both tags to `localhost:5000`
3. Applies `k8s/configmap.yaml`, `k8s/external-secret.yaml`, `k8s/deployment.yaml`, `k8s/service.yaml`, `k8s/ingress.yaml`
4. Runs `kubectl rollout restart deployment/ai-microservice -n statex-apps`
5. Waits for rollout status (timeout 120s)

Expected: ends with something like:
```
deployment.apps/ai-microservice successfully rolled out
```

If it times out, run `kubectl rollout status deployment/ai-microservice -n statex-apps` manually.

- [ ] **Step 2: Confirm pod is running**

```bash
kubectl get pod -n statex-apps -l app=ai-microservice
```

Expected: one pod with `STATUS=Running` and `READY=1/1`.

- [ ] **Step 3: Confirm the running image was built from the correct source**

```bash
kubectl exec -n statex-apps deployment/ai-microservice -- \
  grep -c "ANTHROPIC_API_URL\|api.anthropic.com" /app/dist/ai/ai.service.js
```

Expected: `2` (two matches — the constant and the fetch call).

Also confirm no claude-CLI code in the dist:

```bash
kubectl exec -n statex-apps deployment/ai-microservice -- \
  grep -c "execAsync\|CC_CLI\|claude CLI" /app/dist/ai/ai.service.js 2>&1 || echo "0"
```

Expected: `0` or `grep: exit 1` (no matches).

- [ ] **Step 4: Confirm no hostPath volumes are mounted**

```bash
kubectl get pod -n statex-apps -l app=ai-microservice -o jsonpath='{.items[0].spec.volumes}' | python3 -m json.tool
```

Expected: output shows only the service-account token volume (`kube-api-access-*`), no `local-bin` or `claude-config`.

---

### Task 5: End-to-end smoke test

**Files:** None — runtime validation only.

- [ ] **Step 1: Port-forward the service**

```bash
kubectl port-forward -n statex-apps svc/ai-microservice 13380:3380 &
sleep 2
```

- [ ] **Step 2: Hit /health**

```bash
curl -s http://localhost:13380/health
```

Expected: `{"status":"ok"}`

- [ ] **Step 3: Generate a test JWT**

```bash
JWT_SECRET=$(kubectl get secret -n statex-apps ai-microservice-secret \
  -o jsonpath='{.data.JWT_SECRET}' | base64 -d)

TEST_JWT=$(node -e "
const crypto = require('crypto');
const secret = '$JWT_SECRET';
const header = Buffer.from(JSON.stringify({alg:'HS256',typ:'JWT'})).toString('base64url');
const payload = Buffer.from(JSON.stringify({iss:'ai-microservice',sub:'smoke-test',iat:Math.floor(Date.now()/1000),exp:Math.floor(Date.now()/1000)+3600})).toString('base64url');
const sig = crypto.createHmac('sha256',secret).update(header+'.'+payload).digest('base64url');
console.log(header+'.'+payload+'.'+sig);
")
echo "JWT generated"
```

- [ ] **Step 4: Call /ai/complete with model_tier=free**

```bash
curl -s -X POST http://localhost:13380/ai/complete \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TEST_JWT" \
  -d '{"model_tier":"free","user_prompt":"Reply with the single word: ok"}' \
  | python3 -m json.tool
```

Expected: JSON response with:
- `"text": "ok"` (or similar single-word reply)
- `"model_used": "claude-sonnet-4-6"`
- `"inputTokens"` > 0
- `"outputTokens"` > 0
- No `"error"` key

- [ ] **Step 5: Call /ai/complete with model_tier=smart (tier should be ignored)**

```bash
curl -s -X POST http://localhost:13380/ai/complete \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TEST_JWT" \
  -d '{"model_tier":"smart","user_prompt":"Reply with the single word: ok"}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print('model_used:', r.get('model_used')); print('inputTokens:', r.get('inputTokens'))"
```

Expected:
```
model_used: claude-sonnet-4-6
inputTokens: <non-zero number>
```

- [ ] **Step 6: Kill the port-forward**

```bash
kill %1 2>/dev/null; true
```

---

## Definition of Done

- [ ] `git status` in `ai-microservice` shows clean working tree
- [ ] Running pod dist contains `ANTHROPIC_API_URL` / `api.anthropic.com`, no `execAsync`
- [ ] Pod has no hostPath volumes
- [ ] `POST /ai/complete` returns 200 with `model_used: claude-sonnet-4-6` and non-zero token counts
- [ ] All 4 DoD items from issue #1 satisfied:
  - ✅ `POST /ai/complete` works with any `model_tier`, always calls Claude
  - ✅ Ollama/LiteLLM containers not required
  - ✅ K8s pod starts and passes `/health`
  - ✅ End-to-end: request completes via Claude with real token counts
