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

# Long jobs: detach AND spawn, so the run lives entirely on Modal. (--detach alone did not survive this
# laptop sleeping on 2026-11-17: three runs were cancelled when the client went quiet.) The launcher exits
# at once; results stay on the Modal volume; fetch them later with --run-id <id> --pull.
task=""; prev=""
for a in "$@"; do [[ "$prev" == "--task" ]] && task="$a"; prev="$a"; done
if [[ "$task" == bench_q03 || "$task" == loopprof || "$task" == delta_* || "$task" == int8_probe || "$task" == week6 || "$task" == loopspec || "$task" == w6b_prep || "$task" == w6b_[a-z]*[a-z0-9] && "$task" != w6b_smoke ]] && [[ " $* " != *" --pull "* ]]; then
  exec "$MODAL" run --detach modal_app.py "$@" --spawn
fi
exec "$MODAL" run modal_app.py "$@"
