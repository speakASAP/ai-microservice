#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


REQUIRED_DOCS = [
    "AGENTS.md",
    "README.md",
    "SYSTEM.md",
    "TASKS.md",
    "docs/INTENT_PRESERVATION.md",
    "docs/IMPLEMENTATION_STATE.md",
    "docs/IMPLEMENTATION_ORCHESTRATOR.md",
    "docs/AGENT_ORCHESTRATION.md",
    "docs/governance/PROJECT_INVARIANTS.md",
    "docs/process/DOCUMENTATION_COMPLETENESS_STANDARD.md",
    "docs/process/OPERATIONAL_GATES.md",
    "docs/process/AGENT_GAP_FILLING_RULES.md",
]


REQUIRED_GOAL_SECTIONS = [
    "## Intent",
    "## Scope",
    "## Non-Goals",
    "## Acceptance Criteria",
    "## Required Artifacts Before Coding",
    "## Validation Commands",
]


ARTIFACT_REQUIREMENTS = {
    "execution-plan": [
        "## Metadata",
        "## Upstream Traceability",
        "## Goal Impact",
        "## Project Invariants",
        "## Sensitive-Data Handling",
        "## Contract/Schema Impact",
        "## Replay/Determinism Impact",
        "## Scope",
        "## Non-Goals",
        "## Implementation Steps",
        "## Validation Plan",
        "## Gate Commands",
        "## Rollback Plan",
        "## Agent Handoff Prompt",
    ],
    "context-package": [
        "## Intent",
        "## Current State",
        "## Relevant Contracts",
        "## Files To Read First",
        "## Constraints",
        "## Sensitive-Data Rules",
        "## Validation Evidence Required",
    ],
    "coding-prompt": [
        "## Execution Plan",
        "## Goal",
        "## Intent",
        "## Required Context",
        "## Allowed Changes",
        "## Forbidden Changes",
        "## Validation",
        "## Acceptance Criteria",
        "## Completion Report",
    ],
}


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def artifact_paths(goal: Path) -> dict[str, Path]:
    stem = goal.with_suffix("")
    return {
        "execution-plan": stem.with_name(stem.name + ".execution-plan.md"),
        "context-package": stem.with_name(stem.name + ".context-package.md"),
        "coding-prompt": stem.with_name(stem.name + ".coding-prompt.md"),
    }


def require_sections(path: Path, sections: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    absent = [section for section in sections if section not in text]
    if absent:
        fail(f"{path.relative_to(path.parents[1])} is missing sections: {', '.join(absent)}")
    if "[MISSING:" in text:
        fail(f"{path.relative_to(path.parents[1])} contains unresolved [MISSING:] markers")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AI Microservice pre-coding orchestration artifacts.")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--goal", required=True, help="Selected implementation goal file")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    missing = [path for path in REQUIRED_DOCS if not (root / path).is_file()]
    if missing:
        fail("missing required docs: " + ", ".join(missing))

    goal = (root / args.goal).resolve()
    if not goal.is_file():
        fail(f"missing selected goal: {args.goal}")

    require_sections(goal, REQUIRED_GOAL_SECTIONS)

    artifacts = artifact_paths(goal)
    missing_artifacts = [str(path.relative_to(root)) for path in artifacts.values() if not path.is_file()]
    if missing_artifacts:
        fail("missing required pre-coding artifacts: " + ", ".join(missing_artifacts))

    for artifact_type, path in artifacts.items():
        require_sections(path, ARTIFACT_REQUIREMENTS[artifact_type])

    print("PASS: pre-coding gate")


if __name__ == "__main__":
    main()
