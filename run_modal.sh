#!/usr/bin/env bash
# One-command launcher for modal_app.py.
#   ./run_modal.sh                      -> task "check" (selfcheck + smoke test) on an L40S
#   ./run_modal.sh --task harness       -> Week 1 latency sweep
#   ./run_modal.sh --task bench_q03     -> Week 3 remainder (long; runs detached)
# First run: creates .venv, installs the Modal client, and opens `modal setup` in your browser.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q "modal>=1.6,<2"
fi
MODAL=.venv/bin/modal

if ! "$MODAL" token info >/dev/null 2>&1 && [ ! -f "$HOME/.modal.toml" ]; then
  echo "== Logging in to Modal (a browser window will open) =="
  "$MODAL" setup
fi

# Long Week 3 job: detach so closing the laptop doesn't kill it.
if [[ " $* " == *" bench_q03 "* ]]; then
  exec "$MODAL" run --detach modal_app.py "$@"
fi
exec "$MODAL" run modal_app.py "$@"
