#!/usr/bin/env bash
# One-command launcher for modal_app.py.
#   ./run_modal.sh                      -> task "check" (selfcheck + smoke test) on an L40S
#   ./run_modal.sh --task harness       -> Week 1 latency sweep
#   ./run_modal.sh --task bench_q03     -> Week 3 remainder (long; runs detached)
# First run: creates .venv, installs the Modal client, and opens `modal setup` in your browser.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/modal ]; then
  [ -x .venv/bin/python ] || python3 -m venv .venv
  echo "== Installing the Modal client into .venv =="
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install "modal>=1.6,<2"
fi
MODAL=.venv/bin/modal

if ! "$MODAL" token info >/dev/null 2>&1 && [ ! -f "$HOME/.modal.toml" ]; then
  echo "== Logging in to Modal (a browser window will open) =="
  "$MODAL" setup
fi

# Long jobs (Week 3 bench, Week 4 LoopProf): detach so closing the laptop doesn't kill them.
# Results stay on the Modal volume; fetch them later with --run-id <id> --pull.
if [[ " $* " == *" bench_q03 "* || " $* " == *" loopprof "* ]] && [[ " $* " != *" --pull "* ]]; then
  exec "$MODAL" run --detach modal_app.py "$@"
fi
exec "$MODAL" run modal_app.py "$@"
