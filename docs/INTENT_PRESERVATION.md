# Intent Preservation System

```yaml
id: DOC-AI-INTENT-PRESERVATION
status: reviewed
owner: orchestrator
created: 2026-06-12
last_updated: 2026-06-12
completeness_level: complete
upstream:
  - AGENTS.md
  - /Users/Sergej.Stasok/Documents/Gitlab/intent-preservation-system/README.md
  - /Users/Sergej.Stasok/Documents/Gitlab/intent-preservation-system/17_governance/AI_AGENT_RULES.md
  - /Users/Sergej.Stasok/Documents/Gitlab/intent-preservation-system/23_documentation_contracts/OPERATIONAL_GATE_STANDARD.md
downstream:
  - docs/IMPLEMENTATION_ORCHESTRATOR.md
  - docs/process/OPERATIONAL_GATES.md
  - implementation-goals/
```

AI-microservice goals must carry an explicit intent block from intake through planning, execution, validation, review, and state update. The intent is the stable user objective. Implementation details may change when evidence requires it, but agents must preserve the objective unless a newer user instruction explicitly changes it.

## Source Standard

This project implements the local subset of the company Intent Preservation System from `/Users/Sergej.Stasok/Documents/Gitlab/intent-preservation-system`.

The full source chain is:

```text
Constitution -> Vision -> Business Case -> Systems -> Subsystems -> Architecture/ADRs
-> Roadmap/Milestones -> Features -> Tasks -> Goal Impact -> Execution Plan
-> Context Package -> Coding Prompt -> Code -> Validation Report -> Audit
```

AI microservice uses a project-local chain:

```text
User Intent -> Project Invariants -> Implementation Goal -> Execution Plan
-> Context Package -> Coding Prompt -> Code -> Validation Report
-> docs/IMPLEMENTATION_STATE.md
```

The local chain is intentionally smaller than the reference repository, but it preserves the same controls: explicit intent, upstream traceability, bounded execution scope, operational gates, validation evidence, and stateful handoff.

## Required Intent Fields

Every goal, execution plan, context package, coding prompt, validation report, and implementation job targeting AI microservice must include:

```json
{
  "intent": "Plain-language objective and success condition.",
  "intentChecksum": "sha256 of the normalized intent text"
}
```

`intentChecksum` is required when a caller already has one. If the caller omits it for `/ai/claude-code-execute`, AI microservice computes it when enqueueing the implementation job. Callers must store the returned checksum and compare it during status polling, validation, review, and follow-up jobs.

## Normalization Rule

Normalize intent before computing or comparing a checksum:

1. Trim leading and trailing whitespace.
2. Convert repeated whitespace to a single space.
3. Preserve wording and case after whitespace normalization.
4. Hash the normalized text with SHA-256.

Do not rewrite intent to make a checksum match. If the objective changes, record the newer user instruction and produce a new checksum.

## Methodology

1. Capture intent before planning. State the outcome that must remain true, boundaries that must not be crossed, and how completion will be recognized.
2. Trace intent to an implementation goal in `implementation-goals/`.
3. Create or update an execution plan before code changes.
4. Create or update a context package that contains only the context needed for the selected goal.
5. Generate a coding prompt from the approved execution plan and context package.
6. Run the pre-coding gate before editing code.
7. Attach intent to all AI-microservice execution requests. For `/ai/claude-code-execute`, use `intent` and optionally `intentChecksum`.
8. Preserve intent through provider routing. `implementationProvider` can be `claude-code` or `codex`; provider selection must not rewrite the objective.
9. Validate against intent, not only tests. A job can pass build/tests and still fail if it changes the stated objective or ignores constraints.
10. Update `docs/IMPLEMENTATION_STATE.md` with validation evidence, changed files, risks, and next action.

## Before-Coding Requirements

No coding work may start until these artifacts exist and contain no execution-critical `[MISSING: ...]` markers:

- selected `implementation-goals/GOAL-XX-*.md`;
- `implementation-goals/GOAL-XX-*.execution-plan.md`;
- `implementation-goals/GOAL-XX-*.context-package.md`;
- `implementation-goals/GOAL-XX-*.coding-prompt.md`;
- project invariants in `docs/governance/PROJECT_INVARIANTS.md`;
- explicit validation commands and expected evidence.

Run:

```bash
python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-XX-name.md
```

The gate must fail closed on missing intent, missing traceability, unresolved required markers, unclear validation, sensitive-data uncertainty, or missing required artifacts.

## Validation Requirements

Each completed coding task must produce a validation report using `implementation-goals/templates/VALIDATION_REPORT.md`. The report must include:

- preserved intent and checksum evidence when available;
- commands run and result summary;
- gate evidence;
- invariant evidence;
- sensitive-data handling evidence;
- contract/schema impact evidence;
- failed or skipped checks with reasons;
- pass, fail, or blocked decision.

Build success alone is not sufficient. The validator must explicitly state whether the resulting behavior still satisfies the preserved intent.

## Implementation Provider Routing

`/ai/claude-code-execute` remains the backward-compatible endpoint for code execution jobs. It accepts:

```json
{
  "implementationProvider": "claude-code",
  "intent": "Add Codex as an implementation engine without breaking Claude Code jobs.",
  "intentChecksum": "optional-known-checksum"
}
```

Supported providers:

| Provider | Runtime command | Notes |
| --- | --- | --- |
| `claude-code` | `CC_CLI_PATH --print ...` | Existing default behavior. |
| `codex` | `CODEX_CLI_PATH exec --cd <repo> --sandbox <mode> --ask-for-approval never -` | Requires Codex CLI and auth/config on the runtime host or pod. |

The execution result includes `implementationProvider` and `intentChecksum` so orchestrators can audit provider choice and intent continuity.

## Change Control

Intent may be changed only by a newer user instruction. Implementation discoveries can change scope, sequencing, or design, but they cannot silently change the objective. If a discovered constraint makes the original intent impossible, mark the goal blocked in `docs/IMPLEMENTATION_STATE.md` and ask for a new instruction.
