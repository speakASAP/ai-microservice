# Agent Gap Filling Rules

```yaml
id: DOC-AI-AGENT-GAP-FILLING
status: reviewed
owner: orchestrator
created: 2026-06-12
last_updated: 2026-06-12
completeness_level: complete
upstream:
  - docs/INTENT_PRESERVATION.md
  - docs/process/DOCUMENTATION_COMPLETENESS_STANDARD.md
  - /Users/Sergej.Stasok/Documents/Gitlab/intent-preservation-system/23_documentation_contracts/AGENT_GAP_FILLING_RULES.md
downstream:
  - docs/IMPLEMENTATION_ORCHESTRATOR.md
  - implementation-goals/
```

Agents may fill documentation gaps only when doing so preserves the user's intent and keeps the selected goal bounded. Incomplete documentation must never be hidden inside vague prose.

## Primary Rule

Do not proceed through incomplete implementation documentation silently. Either fill the gap from approved project sources or mark it explicitly with `[MISSING: ...]` or `[UNKNOWN: ...]`.

## Allowed

- Add missing execution-plan detail that directly follows from the selected goal.
- Add context-package and coding-prompt details derived from the goal, project invariants, and current implementation state.
- Choose local conventions already used in this repository.
- Add narrow validation steps for the files changed.
- Split a large goal into smaller tasks when ownership or validation is clearer.
- Add missing required sections to mutable process, goal, plan, prompt, and validation documents.
- Add `[MISSING: ...]` or `[UNKNOWN: ...]` markers while drafting.

## Not Allowed

- Expand product scope beyond the selected goal.
- Change runtime contracts without explicit goal scope.
- Bypass auth or intent-preservation requirements.
- Invent business goals, approval status, deployment evidence, or validation evidence.
- Mark incomplete documents as validated.
- Convert a task into code without the required execution plan, context package, coding prompt, and pre-coding gate.
- Delete or revert unrelated dirty worktree changes.
- Put secrets, raw tokens, production credentials, confidential identifiers, raw production samples, or unredacted sensitive logs into documentation, prompts, tests, screenshots, or reports.

## Gap Remediation Process

1. Identify the document type.
2. Load the required section list from `docs/process/DOCUMENTATION_COMPLETENESS_STANDARD.md`.
3. Compare existing headings to required headings.
4. Fill missing sections from approved upstream sources when possible.
5. Add `[MISSING: ...]` when information is required but cannot be derived.
6. Add `[UNKNOWN: ...]` when the missing information has a known discovery path.
7. Keep `completeness_level` accurate when the document has metadata.
8. Record remaining gaps in the completion report.

## When To Ask

Ask the user when:

- the goal has conflicting acceptance criteria;
- production behavior would change outside documented scope;
- a destructive action is required;
- credentials, approvals, or business policy decisions are missing;
- intent appears to conflict with project invariants.
