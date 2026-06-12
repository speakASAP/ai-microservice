# Branch Workflow

Default branch prefix for local work:

```text
codex/
```

Recommended goal branch names:

```text
codex/ai-goal-03-cost-tracking
codex/ai-goal-04-implementation-job-observability
codex/ai-goal-05-agent-registry-routing
codex/ai-goal-06-deployment-hardening
```

## Rules

- One active coding goal per branch unless the user asks for a combined change.
- Parallel goals require disjoint file ownership or separate branches.
- Merge only after each goal has validation evidence and an intent compliance report.
- Do not commit unrelated dirty files.
- Do not deploy from a branch without an explicit deployment step.
