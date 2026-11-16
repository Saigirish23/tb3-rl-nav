#!/usr/bin/env python3
"""ROS 2 package entry point delegating to the canonical benchmark runner.

Canonical source: scripts/evaluate_benchmark.py
This wrapper ensures clean ROS 2 installation without duplicating 700+ lines of runner logic.
"""

import sys
from pathlib import Path


def _find_repo_root() -> Path:
    """Traverse parents to find the repository root containing scripts/evaluate_benchmark.py."""
    curr = Path(__file__).resolve().parent
    for _ in range(6):
        if (curr / "scripts" / "evaluate_benchmark.py").exists() and (curr / "rl").exists():
            return curr
        curr = curr.parent
    raise FileNotFoundError("Could not locate repository root with scripts/evaluate_benchmark.py")


def main():
    repo_root = _find_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Import and delegate to the canonical benchmark runner
    from scripts.evaluate_benchmark import main as canonical_main
    canonical_main()


if __name__ == "__main__":
    main()
