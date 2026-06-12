# AI Microservice Implementation State

This file is the master orchestrator checkpoint. Update it before ending any implementation session.

## Current State

Stage: production

Primary role: central AI inference gateway and implementation-job executor for the Statex ecosystem.

Runtime constraints:

- Preserve existing AI runtime endpoints and deployment flow.
- Preserve intent fields on `/ai/claude-code-execute` jobs.
- Keep `implementationProvider=claude-code` and `implementationProvider=codex` compatible.
- Keep model-tier routing canonical in `litellm_config.yaml`.
- Do not route premium models without human approval.

## Active Goal

None.

## Completed Goals

- `GOAL-01-admin-agent-registry` - done. Added the admin frontend and persisted AI agent registry; production verified at `https://ai.alfares.cz/admin`.
- `GOAL-02-orchestrator-operating-model` - done. Added Goalkeeper-style orchestration docs, goal queue, process gates, templates, and resume helper.
- `GOAL-03-cost-tracking` - done. Added optional business-level accounting metadata to `/ai/complete` inference logs; migration applied and production deployment verified.
- `GOAL-04-implementation-job-observability` - done. Added optional observability metadata, redacted summaries, lifecycle/audit status fields, validation evidence, production migration, and deployment for implementation jobs.
- `GOAL-05-agent-registry-routing` - done. Added explicit `/ai/complete` active-agent registry routing with audit metadata while preserving normal model-tier routing.
- `GOAL-06-deployment-hardening` - done. Added project-specific deployment readiness checks, production-safe smoke coverage, rollback evidence, and deploy-time operator handoff.

## Goal Queue

| Goal | Status | Dependencies | Summary |
| --- | --- | --- | --- |
| `GOAL-03-cost-tracking` | done | 01, 02 | Add cost tracking per `business_id` to inference logs. |
| `GOAL-04-implementation-job-observability` | done | 02 | Improve execution status, logs, and audit summaries for `claude-code` and `codex` jobs. |
| `GOAL-05-agent-registry-routing` | done | 01, 02 | Use persisted admin agent definitions for controlled routing where appropriate. |
| `GOAL-06-deployment-hardening` | done | 03, 04 | Strengthen deployment, smoke checks, and rollback evidence. |

## Execution Waves

Wave 1: operating model and existing admin registry.

- `GOAL-01-admin-agent-registry` complete.
- `GOAL-02-orchestrator-operating-model` complete.
- `GOAL-03-cost-tracking` implemented, validated, migrated, and deployed.

Wave 2: operational accounting and executor observability.

- `GOAL-03-cost-tracking` complete; continue with `GOAL-04-implementation-job-observability`.
- `GOAL-04-implementation-job-observability` implemented, validated, migrated, and deployed after explicit user approval.

Wave 3: registry-driven routing and hardening.

- `GOAL-05-agent-registry-routing` implemented and validated; not deployed.
- `GOAL-06-deployment-hardening` implemented, validated, and deployed.

## Validation Evidence

2026-06-12:

- Intent Preservation System documentation aligned with the company reference at `/Users/Sergej.Stasok/Documents/Gitlab/intent-preservation-system`.
- Pre-coding documentation requirements now include execution plan, context package, coding prompt, operational gates, validation report evidence, and state update before closure.
- Admin registry work previously built and deployed.
- Production health check passed at `https://ai.alfares.cz/health`.
- Production admin UI available at `https://ai.alfares.cz/admin`.
- Authenticated CRUD smoke test passed for temporary admin agent.
- `./scripts/next_goal.sh` prints the current next action.
- `python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-03-cost-tracking.md` passed.
- `python3 scripts/deployment_readiness_gate.py --root .` passed.
- `python3 -m py_compile scripts/pre_coding_gate.py scripts/deployment_readiness_gate.py` passed.
- `npm run build` was attempted for validation, but local dependencies are not installed: `tsc: command not found`.
- `GOAL-03-cost-tracking`: reconciled the `src/app.module.ts` references by confirming the tracked `src/ai`, `src/service-identity`, and `InferenceLog` files are present in the remote checkout.
- `GOAL-03-cost-tracking`: `python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-03-cost-tracking.md` passed on `alfares`.
- `GOAL-03-cost-tracking`: `npm run build` passed on `alfares`.
- `GOAL-03-cost-tracking`: `npm test -- --runTestsByPath src/ai/ai.service.spec.ts` passed on `alfares` with 9 tests.
- `GOAL-03-cost-tracking`: `npm test` passed on `alfares` with 14 suites and 124 tests.
- `GOAL-03-cost-tracking`: production migration `src/database/migrations/005-inference-log-business-cost.sql` applied; six columns and `idx_inference_business_id` verified.
- `GOAL-03-cost-tracking`: deployed with image `localhost:5000/ai-microservice:goal-03-cost-tracking-20260612113516` (digest `sha256:c7e308f94fd3aeccc95148c47ce358ac0db0de4ad3d3f72cdc4b872e952891b3`).
- `GOAL-03-cost-tracking`: rollout completed with one ready pod and zero restarts; `https://ai.alfares.cz/health` returned `status=ok`.
- `GOAL-04-implementation-job-observability`: RAG lookup could not run because no `JWT_TOKEN` was available on `alfares`; fallback to repository docs was recorded in the context package.
- `GOAL-04-implementation-job-observability`: `python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-04-implementation-job-observability.md` passed on `alfares`.
- `GOAL-04-implementation-job-observability`: `npm run build` passed on `alfares`.
- `GOAL-04-implementation-job-observability`: `npm test -- --runTestsByPath test/claude-code/claude-code.controller.spec.ts test/claude-code/claude-code.e2e.spec.ts` passed on `alfares` with 2 suites and 8 tests.
- `GOAL-04-implementation-job-observability`: `npm test -- --runTestsByPath test/claude-code/claude-code.service.spec.ts` passed on `alfares` with 1 suite and 14 tests.
- `GOAL-04-implementation-job-observability`: `npm test` passed on `alfares` with 14 suites and 124 tests.
- `GOAL-04-implementation-job-observability`: explicit deployment/migration approval received from the user.
- `GOAL-04-implementation-job-observability`: pre-deploy `python3 scripts/deployment_readiness_gate.py --root .`, `npm run build`, and focused Claude Code tests passed on `alfares` with 3 suites and 22 tests.
- `GOAL-04-implementation-job-observability`: production migration `src/database/migrations/006-claude-code-job-observability.sql` applied; 8 observability columns and indexes `idx_claude_code_jobs_lifecycle_stage`, `idx_claude_code_jobs_last_observed_at` verified.
- `GOAL-04-implementation-job-observability`: deployed with image tag `localhost:5000/ai-microservice:goal-04-observability-20260612` and digest `sha256:73888fe494b1d87e32b533c392d9be43c0785d75358c9e918a6be3b998e44310`.
- `GOAL-04-implementation-job-observability`: rollout completed in namespace `statex-apps`; final pod `ai-microservice-6cf6978ff4-bt5fb` was `1/1 Running` with 0 restarts.
- `GOAL-04-implementation-job-observability`: external `https://ai.alfares.cz/health` and in-pod `http://localhost:3380/health` returned `{"status":"ok","service":"ai-microservice"}`.
- `GOAL-05-agent-registry-routing`: RAG lookup could not run because no `JWT_TOKEN` was available on `alfares`; fallback to repository docs was recorded in the context package.
- `GOAL-05-agent-registry-routing`: `python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-05-agent-registry-routing.md` passed on `alfares`.
- `GOAL-05-agent-registry-routing`: `npm run build` passed on `alfares`.
- `GOAL-05-agent-registry-routing`: `npm test -- --runTestsByPath src/ai/ai.service.spec.ts` passed on `alfares` with 1 suite and 13 tests.
- `GOAL-05-agent-registry-routing`: `npm test` passed on `alfares` with 14 suites and 128 tests.
- `GOAL-05-agent-registry-routing`: final `python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-05-agent-registry-routing.md` and `python3 scripts/deployment_readiness_gate.py --root .` passed on `alfares`.
- `GOAL-05-agent-registry-routing`: deployment was not performed because the selected goal did not include deployment and the user did not request it.
- `GOAL-06-deployment-hardening`: RAG lookup could not run because no `JWT_TOKEN` was available on `alfares`; fallback to repository docs was used.
- `GOAL-06-deployment-hardening`: deployment readiness hardening adds project-specific checks for Kubernetes manifests, deploy script phases, smoke coverage, rollback evidence, package scripts, and GOAL-06 orchestration artifacts.
- `GOAL-06-deployment-hardening`: smoke checks cover `/health`, `/ai/complete` premium approval blocking, and `/ai/claude-code-execute` invalid payload validation. `/ai/complete` missing agent routing is opt-in with `AI_SMOKE_CHECK_AGENT_ROUTING=true` until GOAL-05 is deployed. Live model inference is opt-in with `AI_SMOKE_RUN_LIVE_AI=true`.
- `GOAL-06-deployment-hardening`: rollback path is documented in the execution plan and printed by `scripts/deploy.sh` using captured previous image and rollout revision.
- `GOAL-06-deployment-hardening`: secret exposure review: scripts print synthetic payloads, image/revision metadata, and `LITELLM_BASE_URL`; they do not print provider keys, database passwords, JWTs, or raw implementation-job output.
- `GOAL-06-deployment-hardening`: `python3 scripts/pre_coding_gate.py --root . --goal implementation-goals/GOAL-06-deployment-hardening.md` passed on `alfares`.
- `GOAL-06-deployment-hardening`: `python3 -m py_compile scripts/deployment_readiness_gate.py scripts/pre_coding_gate.py` passed on `alfares`.
- `GOAL-06-deployment-hardening`: `bash -n scripts/deploy.sh scripts/smoke-unified-llm.sh` passed on `alfares`.
- `GOAL-06-deployment-hardening`: `python3 scripts/deployment_readiness_gate.py --root .` passed on `alfares`.
- `GOAL-06-deployment-hardening`: `npm run build` passed on `alfares`.
- `GOAL-06-deployment-hardening`: `npm test` passed on `alfares` with 14 suites and 128 tests.
- `GOAL-06-deployment-hardening`: authenticated production-safe smoke against `https://ai.alfares.cz` passed for `/health`, premium approval block, and invalid `/ai/claude-code-execute` payload validation. Agent-routing smoke and live inference were intentionally skipped by default.
- `GOAL-06-deployment-hardening`: deployed with image tag `localhost:5000/ai-microservice:goal-06-deployment-hardening-20260612` and running image digest `sha256:da24fd454caf4336f3f28d3931ea5c554b210046e62eb9b6373095a7f3e526a5`.
- `GOAL-06-deployment-hardening`: deployment readiness gate, preflight, rollback context capture, image build, registry push, Kubernetes apply, rollout wait, health check, and authenticated smoke checks completed successfully in `160.38s`.
- `GOAL-06-deployment-hardening`: rollout completed for `deployment/ai-microservice` in namespace `statex-apps`; final pod `ai-microservice-9bbc4d546-l6s4v` was `1/1 Running` with 0 restarts.
- `GOAL-06-deployment-hardening`: external `https://ai.alfares.cz/health` returned `{"status":"ok","service":"ai-microservice"}` after deployment.
- `GOAL-06-deployment-hardening`: deploy script rollback evidence was patched after deployment to prefer captured pod image digest when Kubernetes revision history only contains mutable `latest`; `bash -n scripts/deploy.sh scripts/smoke-unified-llm.sh` and `python3 scripts/deployment_readiness_gate.py --root .` passed after the patch.

## Risks And Follow-Ups

- The local working tree currently contains many untracked files. Treat them as existing project state unless a selected goal explicitly owns them.
- RAG lookup may fail locally without cluster DNS or `JWT_TOKEN`; record fallback when it happens.
- Local `npm run build` requires dependencies to be installed before TypeScript validation can run.
- `estimated_cost_usd` is nullable; current gateway responses primarily provide token usage for audit accounting unless future providers return cost metadata.
- Deployment must use the project runbook and should not be performed without an explicit deployment goal or user approval.
- `GOAL-04-implementation-job-observability` is deployed. Raw compatibility fields may still contain sensitive output for authorized callers; new summary fields are redacted/truncated operator surfaces.
- `GOAL-05-agent-registry-routing` is validated but not deployed. Agent prompt templates support simple `{{user_prompt}}`, `{{input}}`, and `{{prompt}}` substitution only.

## Changed Files In Last Orchestrator Update

- `docs/INTENT_PRESERVATION.md`
- `docs/process/DOCUMENTATION_COMPLETENESS_STANDARD.md`
- `docs/process/OPERATIONAL_GATES.md`
- `docs/process/AGENT_GAP_FILLING_RULES.md`
- `docs/IMPLEMENTATION_ORCHESTRATOR.md`
- `docs/IMPLEMENTATION_STATE.md`
- `implementation-goals/README.md`
- `implementation-goals/GOAL-03-cost-tracking.execution-plan.md`
- `implementation-goals/GOAL-03-cost-tracking.context-package.md`
- `implementation-goals/GOAL-03-cost-tracking.coding-prompt.md`
- `implementation-goals/templates/EXECUTION_PLAN.md`
- `implementation-goals/templates/CONTEXT_PACKAGE.md`
- `implementation-goals/templates/CODING_PROMPT.md`
- `implementation-goals/templates/VALIDATION_REPORT.md`
- `scripts/pre_coding_gate.py`
- `implementation-goals/GOAL-03-cost-tracking.md`
- `implementation-goals/GOAL-03-cost-tracking.validation-report.md`
- `src/contracts/ai-complete.contract.ts`
- `src/ai/ai.service.ts`
- `src/ai/ai.service.spec.ts`
- `src/database/entities/inference-log.entity.ts`
- `src/database/migrations/005-inference-log-business-cost.sql`
- `src/service-identity/inference-log.interceptor.ts`
- `implementation-goals/GOAL-04-implementation-job-observability.md`
- `implementation-goals/GOAL-04-implementation-job-observability.execution-plan.md`
- `implementation-goals/GOAL-04-implementation-job-observability.context-package.md`
- `implementation-goals/GOAL-04-implementation-job-observability.coding-prompt.md`
- `implementation-goals/GOAL-04-implementation-job-observability.validation-report.md`
- `implementation-goals/GOAL-05-agent-registry-routing.md`
- `implementation-goals/GOAL-05-agent-registry-routing.execution-plan.md`
- `implementation-goals/GOAL-05-agent-registry-routing.context-package.md`
- `implementation-goals/GOAL-05-agent-registry-routing.coding-prompt.md`
- `implementation-goals/GOAL-05-agent-registry-routing.validation-report.md`
- `src/ai/ai.module.ts`
- `src/ai/ai.service.ts`
- `src/ai/ai.service.spec.ts`
- `src/contracts/ai-complete.contract.ts`
- `src/claude-code/claude-code.service.ts`
- `src/claude-code/claude-code.consumer.ts`
- `src/claude-code/dto/job-enqueue-response.dto.ts`
- `src/claude-code/dto/job-status-response.dto.ts`
- `src/contracts/claude-code-job.contract.ts`
- `src/database/entities/claude-code-job.entity.ts`
- `src/database/migrations/006-claude-code-job-observability.sql`
- `test/claude-code/claude-code.controller.spec.ts`
- `test/claude-code/claude-code.e2e.spec.ts`
- `test/claude-code/claude-code.service.spec.ts`

## Next Action

No active implementation goal is queued in this state file.
