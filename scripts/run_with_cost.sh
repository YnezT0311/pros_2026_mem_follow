#!/usr/bin/env bash
# Usage: bash run_with_cost.sh <runner.sh> [args...]
#
# Wraps any runner script, snapshots OpenRouter account usage before + after,
# and reports the cost of this run. Works for any model/system test as long
# as it hits OpenRouter under the same API key.
#
# The cost line is printed to stderr so it doesn't pollute model output piped
# to other tools, but is still captured in `2>&1` redirects.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEY_FILE="${OPENROUTER_KEY_FILE:-${SCRIPT_DIR}/../keys/openrouter_key.txt}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "$KEY_FILE" ]]; then
  echo "[cost] WARN: OpenRouter key file not found at $KEY_FILE; skipping cost tracking" >&2
  exec "$@"
fi

KEY=$(cat "$KEY_FILE")

snapshot_usage() {
  curl -s --max-time 15 -H "Authorization: Bearer $KEY" 'https://openrouter.ai/api/v1/key' \
    | "$PYTHON_BIN" -c "import sys,json; d=json.load(sys.stdin)['data']; print(d.get('usage', 0))" \
    2>/dev/null || echo "0"
}

USAGE_START=$(snapshot_usage)
START_TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "[cost] start=\$$USAGE_START  at $START_TS  cmd=$*" >&2

# Run the wrapped command, preserve its exit status
set +e
"$@"
EXIT_CODE=$?
set -e

USAGE_END=$(snapshot_usage)
END_TS=$(date '+%Y-%m-%d %H:%M:%S')
DELTA=$("$PYTHON_BIN" -c "import sys; print(f'{float(sys.argv[1]) - float(sys.argv[2]):.4f}')" "$USAGE_END" "$USAGE_START" 2>/dev/null || echo "?")

echo "[cost] end=\$$USAGE_END  at $END_TS  delta=\$$DELTA  exit=$EXIT_CODE  cmd=$*" >&2
exit $EXIT_CODE
