# Intent Preservation System

AI-microservice goals must carry an explicit intent block from intake through execution and validation. The intent is the stable user objective; implementation details can change, but agents must preserve the objective unless a newer user instruction explicitly changes it.

## Required intent fields

Every goal or implementation job targeting AI microservice should include:

```json
{
  "intent": "Plain-language objective and success condition.",
  "intentChecksum": "sha256 of the normalized intent text"
}
```

If the caller omits `intentChecksum`, AI microservice computes it when enqueueing an implementation job. Callers should store the returned checksum and compare it during status polling, validation, review, and follow-up jobs.

## Methodology

1. Capture intent before planning. State what outcome must remain true, what must not be changed, and how completion will be recognized.
2. Attach intent to all AI-microservice execution requests. For `/ai/claude-code-execute`, use `intent` and optionally `intentChecksum`.
3. Preserve intent through provider routing. `implementationProvider` can be `claude-code` or `codex`; provider selection must not rewrite the objective.
4. Validate against intent, not only tests. A job can pass build/tests and still fail if it changes the stated objective or ignores constraints.
5. Update intent only on explicit user change. New instructions replace or amend the intent; implicit implementation discoveries do not.

## Implementation provider routing

`/ai/claude-code-execute` remains the backward-compatible endpoint for code execution jobs. It now accepts:

```json
{
  "implementationProvider": "claude-code",
  "intent": "Add Codex as an implementation engine without breaking Claude Code jobs."
}
```

Supported providers:

| Provider | Runtime command | Notes |
| --- | --- | --- |
| `claude-code` | `CC_CLI_PATH --print ...` | Existing default behavior. |
| `codex` | `CODEX_CLI_PATH exec --cd <repo> --sandbox <mode> --ask-for-approval never -` | Requires Codex CLI and auth/config on the runtime host or pod. |

The execution result includes `implementationProvider` and `intentChecksum` so orchestrators can audit provider choice and intent continuity.
