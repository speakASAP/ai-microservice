#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AI Microservice deployment readiness documentation.")
    parser.add_argument("--root", default=".", help="Repository root")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    state = root / "docs/IMPLEMENTATION_STATE.md"
    gates = root / "docs/process/OPERATIONAL_GATES.md"

    if not state.is_file():
        fail("missing docs/IMPLEMENTATION_STATE.md")
    if not gates.is_file():
        fail("missing docs/process/OPERATIONAL_GATES.md")

    state_text = state.read_text(encoding="utf-8")
    required_markers = ["## Validation Evidence", "## Risks And Follow-Ups", "## Next Action"]
    missing = [marker for marker in required_markers if marker not in state_text]
    if missing:
        fail("implementation state missing sections: " + ", ".join(missing))

    if "Deployment must use" not in state_text and "deployment" not in state_text.lower():
        fail("deployment or rollback readiness is not documented in implementation state")

    print("PASS: deployment readiness gate")


if __name__ == "__main__":
    main()
