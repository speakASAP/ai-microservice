# Agent Orchestration

## Core Requirement

AI Microservice must be coordinated by one master implementation orchestrator for repository work, while runtime services continue to serve inference and implementation execution requests.

The orchestrator should be able to:

- understand current repository goals and task state;
- choose the next goal from `implementation-goals/`;
- split selected goals into plans and executable tasks;
- route implementation tasks to Codex, Claude Code, or a future executor only after intent and gates are ready;
- preserve intent checksums where callers provide them;
- record decisions, validation evidence, changed files, blockers, and next actions.

## Runtime Boundary

This repository is an infrastructure AI service, not the autonomous master coordinator for all projects.

The master orchestrator described here is the working model for developing this repository. Runtime endpoints must stay focused on inference, AI-agent registry, and implementation job execution unless a selected goal explicitly expands that scope.

## Executor Model

External implementation executors should be treated as adapters behind a common contract:

```ts
interface ImplementationExecutor {
  id: string;
  provider: 'claude-code' | 'codex' | 'internal';
  capabilities: string[];
  canRun(task: ImplementationTask): Promise<boolean>;
  run(task: ImplementationTask): Promise<ImplementationResult>;
}
```

Routing inputs:

- task type;
- required capabilities;
- project root;
- risk level;
- user preference;
- `implementationProvider`;
- approved intent and `intentChecksum`;
- execution plan, context package, coding prompt, and validation criteria.

Routing output:

- selected executor;
- reason;
- fallback executor if allowed;
- approval requirement;
- blocked reason if intent or gate evidence is missing.

## Agent Registry

The admin AI-agent registry stores reusable agent definitions. It should remain compatible with these orchestration concepts:

- stable agent id or slug;
- display name;
- status;
- prompt and system prompt;
- model tier;
- provider override;
- token and temperature settings;
- output schema;
- tags and metadata;
- service scope and route path.

Registry entries are configuration; they do not replace the orchestrator's responsibility to validate intent and plan evidence before implementation work.

## Implementation Job Rules

For `/ai/claude-code-execute` jobs:

- include `intent`;
- include `intentChecksum` when the caller already has one;
- include `implementationProvider` when caller wants Codex instead of default Claude Code;
- persist provider choice and checksum in status responses;
- validate output against user intent, not only process exit status.

## Progress Reporting

Long-running implementation jobs should expose concise progress:

- task started;
- executor selected;
- current step;
- elapsed time;
- latest meaningful log summary;
- approval or user-input blockers;
- completion or failure report.

Do not expose raw secrets or full verbose logs through user-facing summaries.
