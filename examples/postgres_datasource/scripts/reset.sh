#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${EXAMPLE_DIR}"
docker compose down -v
docker compose up -d
docker compose exec -T postgres pg_isready -U postgres -d postgres
