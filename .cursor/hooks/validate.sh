#!/usr/bin/env bash
# On agent stop: run CI-equivalent checks and request a follow-up if they fail.
set -u

python3 - <<'PY'
import json
import os
import subprocess
import sys

try:
    data = json.load(sys.stdin)
except json.JSONDecodeError:
    data = {}

status = str(data.get("status") or data.get("stop_reason") or "completed").lower()
loop_count = int(data.get("loop_count") or 0)

# Skip validation when the turn was aborted or we already retried enough.
if status in {"aborted", "cancelled", "canceled", "error"}:
    print("{}")
    raise SystemExit(0)

if loop_count >= 3:
    print("{}")
    raise SystemExit(0)

root = os.getcwd()
commands = [
    ["uv", "run", "ruff", "check", "."],
    ["uv", "run", "ruff", "format", "--check", "."],
    ["uv", "run", "mypy", "src"],
    ["uv", "run", "pytest"],
]

failures: list[str] = []
for cmd in commands:
    try:
        result = subprocess.run(
            cmd,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=240,
        )
    except FileNotFoundError:
        print("{}")
        raise SystemExit(0)
    except subprocess.TimeoutExpired:
        failures.append(f"$ {' '.join(cmd)}\nTimed out after 240s")
        continue

    if result.returncode != 0:
        output = (result.stdout or "") + (result.stderr or "")
        output = output.strip()
        if len(output) > 4000:
            output = output[:4000] + "\n...[truncated]..."
        failures.append(f"$ {' '.join(cmd)}  (exit {result.returncode})\n{output}")

if not failures:
    print("{}")
    raise SystemExit(0)

body = (
    "Validation failed after this agent turn. Fix the failures, "
    "then re-run the same checks before finishing.\n\n"
    + "\n\n".join(failures)
)
print(json.dumps({"followup_message": body}))
PY
