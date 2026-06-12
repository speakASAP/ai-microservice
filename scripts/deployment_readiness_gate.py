#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


class Gate:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.failures.append(message)

    def require_file(self, path: Path) -> str:
        if not path.is_file():
            self.failures.append(f"missing {path}")
            return ""
        return path.read_text(encoding="utf-8")


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def require_markers(gate: Gate, text: str, markers: list[str], source: str) -> None:
    missing = [marker for marker in markers if marker not in text]
    gate.require(not missing, f"{source} missing markers: {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AI Microservice deployment readiness documentation.")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument(
        "--goal",
        default="implementation-goals/GOAL-06-deployment-hardening.md",
        help="Selected deployment goal path, relative to root",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    gate = Gate()

    state_text = gate.require_file(root / "docs/IMPLEMENTATION_STATE.md")
    gates_text = gate.require_file(root / "docs/process/OPERATIONAL_GATES.md")
    goal_text = gate.require_file(root / args.goal)
    deploy_text = gate.require_file(root / "scripts/deploy.sh")
    smoke_text = gate.require_file(root / "scripts/smoke-unified-llm.sh")
    deployment_text = gate.require_file(root / "k8s/deployment.yaml")
    configmap_text = gate.require_file(root / "k8s/configmap.yaml")
    service_text = gate.require_file(root / "k8s/service.yaml")
    ingress_text = gate.require_file(root / "k8s/ingress.yaml")
    package_text = gate.require_file(root / "package.json")

    require_markers(gate, state_text, ["## Validation Evidence", "## Risks And Follow-Ups", "## Next Action"], "implementation state")
    require_markers(gate, gates_text, ["## Deployment Readiness Gate", "rollback note", "no known secret exposure"], "operational gates")
    require_markers(gate, goal_text, ["## Intent", "## Scope", "## Acceptance Criteria"], "selected goal")

    for suffix in ["execution-plan.md", "context-package.md", "coding-prompt.md", "validation-report.md"]:
        gate.require((root / f"implementation-goals/GOAL-06-deployment-hardening.{suffix}").is_file(), f"missing GOAL-06 {suffix}")

    require_markers(
        gate,
        deploy_text,
        [
            "deployment_readiness_gate.py",
            "smoke-unified-llm.sh",
            "kubectl rollout history",
            "kubectl rollout undo",
            "LITELLM_BASE_URL",
            "deploy_timing_run_phase",
        ],
        "deploy script",
    )
    require_markers(
        gate,
        smoke_text,
        [
            "/health",
            "/ai/complete",
            "Premium tier requires explicit human approval",
            "AGENT_NOT_AVAILABLE",
            "/ai/claude-code-execute",
            "AI_SMOKE_RUN_LIVE_AI",
        ],
        "smoke script",
    )
    require_markers(
        gate,
        deployment_text,
        [
            "maxUnavailable: 0",
            "startupProbe:",
            "livenessProbe:",
            "readinessProbe:",
            "runAsUser: 1000",
            "resources:",
        ],
        "k8s deployment",
    )
    require_markers(
        gate,
        configmap_text,
        [
            'NODE_ENV: "production"',
            'AI_COMPLETE_ROUTER: "litellm"',
            "LITELLM_BASE_URL",
            'DB_SYNC: "false"',
            "CLAUDE_CODE_RATE_LIMIT_FALLBACK_PROVIDER",
        ],
        "k8s configmap",
    )
    require_markers(gate, service_text, ["type: ClusterIP", "targetPort: 3380"], "k8s service")
    require_markers(gate, ingress_text, ["host: ai.alfares.cz", "secretName: ai-microservice-tls"], "k8s ingress")
    require_markers(gate, package_text, ['"build": "tsc -p tsconfig.build.json"', '"test": "jest"'], "package scripts")

    lower_state = state_text.lower()
    gate.require("rollback" in lower_state, "implementation state must include rollback evidence or rollback note")
    gate.require("smoke" in lower_state, "implementation state must include smoke-check evidence")
    gate.require("secret" in lower_state, "implementation state must include secret exposure review evidence")

    if gate.failures:
        fail("deployment readiness gate failed:\n- " + "\n- ".join(gate.failures))

    print("PASS: deployment readiness gate")


if __name__ == "__main__":
    main()
