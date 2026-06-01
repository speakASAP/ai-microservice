# RTK + Caveman Token Compression — Spec

**Date:** 2026-06-01  
**Status:** Approved

---

## Goal

Reduce LLM token costs across the entire ecosystem by 60–90% by installing RTK (input compression) and Caveman (output compression) as a proxy layer at the Claude Code level, with savings telemetry added to `ai-microservice`.

---

## Architecture

Two tools installed once on the server, both operating at the Claude Code layer:

- **RTK** — Rust binary, hooks into Claude Code `PreToolUse` event via `~/.claude/settings.json`. Intercepts every Bash command output before it enters the model context. Reduces input tokens 60–90%.
- **Caveman** — Node.js plugin, installs to `~/.claude/plugins/caveman/`. Compresses model prose output at `lite` intensity (filler words only — code, paths, JSON stay byte-perfect). Reduces output tokens ~22–50% at lite.

Both tools are inherited automatically by any Claude Code subprocess, including the Claude CLI spawned by `ai-microservice`.

**No changes to any other microservice.** All callers of `POST /ai/complete` are unaffected — the `AiCompleteRequestSchema` contract is unchanged.

---

## Coverage

| Workload | RTK | Caveman |
|---|---|---|
| Claude Code sessions (developer) | ✅ PreToolUse hook | ✅ Plugin auto-activates |
| `ai-microservice` subprocess (CC CLI) | ✅ Inherits hook | ✅ Inherits plugin |
| `business-orchestrator` agents | ✅ via ai-microservice | ✅ via ai-microservice |
| `agentic-email-processing-system` | ✅ via ai-microservice | ✅ via ai-microservice |

---

## Telemetry

`ai-microservice/src/ai/ai.service.ts` already extracts `inputTokens` + `outputTokens` from the CC JSON envelope. The change: after parsing those values, emit a structured `info` log via the existing `LoggingClient` pattern with compression metadata attached.

Log shape:
```json
{
  "service": "ai-microservice",
  "level": "info",
  "message": "ai_complete",
  "metadata": {
    "correlation_id": "<uuid>",
    "model_used": "claude-sonnet",
    "inputTokens": 420,
    "outputTokens": 85,
    "token_usage_estimate": 505,
    "compression": { "rtk": true, "caveman": "lite" }
  }
}
```

`LoggingClient` is injected into `AiService` via `AiModule`. No DB writes, no schema changes.

---

## Caveman Intensity

Default: `lite` — set once in `/home/ssf/Documents/Github/shared/AGENTS.md` so all Claude Code sessions and subagents inherit it. Can be overridden per-session with `/caveman full` or `/caveman ultra`.

---

## Out of Scope

- Per-caller `compression_level` field in `AiCompleteRequestSchema` (deferred — gather data first)
- Grafana dashboard for token savings (existing Grafana can query logging-microservice directly)
- Any changes to services other than `ai-microservice` and `shared/AGENTS.md`
