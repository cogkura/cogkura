#!/usr/bin/env bash
# Format Python files after agent edits. Fail open on unexpected errors.
set -euo pipefail

input="$(cat)"
file_path="$(
  printf '%s' "${input}" | python3 -c '
import json, sys
data = json.load(sys.stdin)
for key in ("file_path", "path", "file", "uri"):
    value = data.get(key)
    if isinstance(value, str) and value:
        print(value)
        break
' 2>/dev/null || true
)"

# Some payloads nest edits under "edits" or similar.
if [[ -z "${file_path}" ]]; then
  file_path="$(
    printf '%s' "${input}" | python3 -c '
import json, sys
data = json.load(sys.stdin)
edits = data.get("edits") or data.get("files") or []
if isinstance(edits, list) and edits:
    first = edits[0]
    if isinstance(first, dict):
        for key in ("file_path", "path", "file"):
            value = first.get(key)
            if isinstance(value, str) and value:
                print(value)
                break
' 2>/dev/null || true
  )"
fi

if [[ -z "${file_path}" || "${file_path}" != *.py ]]; then
  exit 0
fi

# Strip file:// prefix if present
file_path="${file_path#file://}"

if [[ ! -f "${file_path}" ]]; then
  exit 0
fi

if command -v uv >/dev/null 2>&1; then
  uv run ruff format "${file_path}" >/dev/null 2>&1 || true
  uv run ruff check --fix "${file_path}" >/dev/null 2>&1 || true
fi

exit 0
