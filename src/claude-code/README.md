# Claude Code Executor Module

Async job system for executing Claude Code CLI tasks in isolated git worktrees.

## Endpoints

### POST /ai/claude-code-execute

Submit a code execution job.

**Request:**

```json
{
  "taskId": "uuid",
  "repoPath": "/home/ssf/Documents/Github/beauty",
  "branch": "main",
  "instructions": "List all files in src/ directory",
  "expectedOutcome": "File listing complete",
  "timeoutSeconds": 300,
  "validationScript": "scripts/validate.sh"
}
```

**Response (201):**

```json
{
  "jobId": "job-<uuid>",
  "taskId": "uuid",
  "status": "queued",
  "createdAt": "2026-04-19T12:00:00.000Z"
}
```

### GET /ai/claude-code-execute/:jobId

Poll job status and results.

**Response (executing):**

```json
{
  "jobId": "job-<uuid>",
  "status": "executing",
  "startedAt": "2026-04-19T12:00:01.000Z"
}
```

**Response (terminal):**

```json
{
  "jobId": "job-<uuid>",
  "status": "success",
  "exitCode": 0,
  "stdout": "...",
  "stderr": "",
  "gitDiff": "diff --git ...",
  "validationPassed": true,
  "validationOutput": "",
  "startedAt": "2026-04-19T12:00:01.000Z",
  "completedAt": "2026-04-19T12:05:00.000Z"
}
```

## Job Lifecycle (Phase 2)

```text
queued → executing → success
                   → failed        (non-retryable or max retries exceeded)
                   → retrying      (transient error, scheduled for re-execution)
                       └→ executing (retry attempt)
                   → timeout
```

Retry backoff: attempt 1 → 30s, attempt 2 → 90s, attempt 3 → 270s. Default max retries: 3 (per job).

## Logging Microservice Integration

Key events are posted fire-and-forget to `LOGGING_SERVICE_URL/api/logs`:

| Event | Level |
| ----- | ----- |
| Claude Code Job Executing | info |
| Claude Code Job Completed | info |
| Claude Code Job Retry Scheduled | warn |
| Claude Code Job Retry Recovery | warn |
| Claude Code Job Failed | error |

**Configuration:** Set environment variable:

```bash
export LOGGING_SERVICE_URL=http://logging-microservice:3367
```

The logging client is fire-and-forget: if the logging-microservice is unavailable, job execution continues normally. Never blocks.

## Architecture

1. **Enqueue** — POST creates a DB record (status=queued) and publishes to `claude-code-exchange` / routing key `claude-code.execute`
2. **Consumer** — `ClaudeCodeConsumer` dequeues, creates a git worktree at `/tmp/worktree-<jobId>`, runs `claude code` CLI, captures stdout/stderr/git diff, optionally runs validation script
3. **Poll** — Orchestrator GETs until status is terminal (success/failed/timeout)

## Testing

```bash
# Unit tests
npm test

# E2E tests (mocked DB + RabbitMQ)
npm run test:e2e

# Manual live test (requires running service)
SERVICE_URL=http://localhost:3380 ./scripts/test-claude-code-executor.sh
```

## Integration with business-orchestrator

The orchestrator coordinator calls POST to enqueue, then polls GET every 5s until terminal:

```typescript
const { jobId } = await axios.post(`${AI_URL}/ai/claude-code-execute`, payload);
let result;
while (!result || ['queued','executing'].includes(result.status)) {
  await sleep(5000);
  result = (await axios.get(`${AI_URL}/ai/claude-code-execute/${jobId}`)).data;
}
```
