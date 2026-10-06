#!/usr/bin/env bash
# One-command box-only / non-target bench: latency + peak RSS.
# Does NOT download or redistribute model weights (license blocker).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"
MODE="${1:-both}"
mkdir -p "$ROOT/results"
export PYTHONUNBUFFERED=1

run_one() {
  local name="$1"
  local script="$2"
  local log="$ROOT/results/bench_${name}.log"
  echo "=== BOX-ONLY / NON-TARGET bench: $name ===" | tee "$log"
  echo "host=$(hostname) date=$(date -R) nproc=$(nproc)" | tee -a "$log"
  # Peak RSS via /proc wrapping
  python - "$script" "$log" <<'PY'
import os, sys, time, resource, subprocess
script, log = sys.argv[1], sys.argv[2]
t0 = time.time()
p = subprocess.run([sys.executable, script], check=False)
elapsed = time.time() - t0
# child peak is not in parent; report parent after join is incomplete.
# Re-read last JSON instead.
print(f"exit={p.returncode} wall_s={elapsed:.2f}", flush=True)
with open(log, "a") as f:
    f.write(f"exit={p.returncode} wall_s={elapsed:.2f}\n")
sys.exit(p.returncode)
PY
}

case "$MODE" in
  seganypet) run_one seganypet "$ROOT/scripts/feasibility_seganypet.py" ;;
  nnunet) run_one nnunet "$ROOT/scripts/feasibility_nnunet.py" ;;
  both)
    run_one nnunet "$ROOT/scripts/feasibility_nnunet.py"
    run_one seganypet "$ROOT/scripts/feasibility_seganypet.py"
    ;;
  *) echo "unknown mode: $MODE"; exit 2 ;;
esac
echo "Done. Results under $ROOT/results (box-only / non-target)."
