import argparse
import os
import subprocess
import sys
from typing import List


def build_command(policy: str, passthrough: List[str]) -> List[str]:
    if policy == "no_store":
        script = os.path.join("privacy_test", "retention", "evaluate_retention_worlds.py")
    elif policy == "deletion":
        script = os.path.join("privacy_test", "deletion", "evaluate_deletion_worlds.py")
    else:
        raise ValueError(f"Unsupported policy: {policy}")
    return [sys.executable, script] + passthrough


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Unified evaluator entrypoint. "
            "Use --policy no_store|deletion, remaining args are passed through."
        ),
        add_help=True,
    )
    parser.add_argument("--policy", type=str, required=True, choices=["no_store", "deletion"])
    args, passthrough = parser.parse_known_args()

    cmd = build_command(args.policy, passthrough)
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd)
    raise SystemExit(res.returncode)


if __name__ == "__main__":
    main()
