#!/bin/bash
# Keep re-entering run_chain.py until every stage reports complete.
#
# WHY: the chain itself is resumable, but nothing restarts it if the chain
# PROCESS dies -- an OOM kill (CLAUDE.md 16), a login-node reboot, a dropped
# shell. Each re-entry is cheap when there is nothing to do: a stage whose
# JSONL already holds `expect` rows is skipped without loading any weights, so
# the loop costs one manifest read per completed stage.
#
# It is NOT a retry-forever loop: MAX_ROUNDS bounds it, and a round that makes
# no progress twice in a row stops, so a genuinely broken stage surfaces
# instead of spinning.
#
#   STAGES=omnipro_qwen3 GPUS=0,1,2,3 nohup setsid \
#     bash scripts/run_chain_supervised.sh > runs/chain_qwen3.log 2>&1 &

set -uo pipefail
REPO="${REPO:-/iopsstor/scratch/cscs/dbartaula/querystream}"
STAGES="${STAGES:?set STAGES=comma,separated,stage,names}"
GPUS="${GPUS:-0,1,2,3}"
MAX_ROUNDS="${MAX_ROUNDS:-8}"
BUDGET="${BUDGET:-0}"
PY="${PY:-$(command -v python3)}"

export HF_HOME="${HF_HOME:-/iopsstor/scratch/cscs/dbartaula/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OPENCV_FFMPEG_CAPTURE_OPTIONS="timeout;15000000"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

cd "$REPO" || exit 1
stall=0
for round in $(seq 1 "$MAX_ROUNDS"); do
  echo "================ round $round/$MAX_ROUNDS  $(date -Is) ================"
  before=$("$PY" run_chain.py --stages "$STAGES" --progress 2>/dev/null | tail -1)
  "$PY" run_chain.py --stages "$STAGES" --gpus "$GPUS" \
       ${BUDGET:+--time-budget-sec "$BUDGET"} --continue-on-incomplete
  rc=$?
  after=$("$PY" run_chain.py --stages "$STAGES" --progress 2>/dev/null | tail -1)
  echo "[supervisor] round $round rc=$rc rows $before -> $after"
  if [ "$rc" -eq 0 ]; then
    echo "[supervisor] chain reports all stages complete"; break
  fi
  if [ "$after" -le "$before" ]; then
    stall=$((stall + 1))
    echo "[supervisor] no progress this round (stall=$stall)"
    [ "$stall" -ge 2 ] && { echo "[supervisor] two stalled rounds; stopping"; break; }
  else
    stall=0
  fi
done
echo "[supervisor] finished $(date -Is)"
