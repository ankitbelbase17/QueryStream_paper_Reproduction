#!/bin/bash
# Poll the submitted judge batch until it lands, then fold it into the report.
#
# The Batch API's contractual window is 24 h. It is usually minutes, but the run
# must not depend on it: exact-match letter parsing is the official
# StreamingBench metric and is already complete. This just adds the cross-check
# when it arrives, and regenerates the PDF so the judge section fills in.
#
#   export OPENAI_API_KEY=...
#   nohup bash scripts/collect_judge_and_report.sh runs/sb_subset &
set -uo pipefail
RUN=${1:-runs/sb_subset}
REPO=${REPO:-/iopsstor/scratch/cscs/dbartaula/querystream}
MAX_HOURS=${MAX_HOURS:-24}
INTERVAL=${INTERVAL:-120}

cd "$REPO"
deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if python judge_batch.py "$RUN" --collect --wait 30 >> "$RUN/judge.log" 2>&1; then
    echo "[collect] judge landed $(date)" >> "$RUN/judge.log"
    python make_report.py "$RUN" \
      --out "$RUN/QueryStream_StreamingBench_report.pdf" >> "$RUN/judge.log" 2>&1
    echo "[collect] report regenerated with judge section" >> "$RUN/judge.log"
    exit 0
  fi
  sleep "$INTERVAL"
done
echo "[collect] gave up after ${MAX_HOURS}h; exact-match results stand" >> "$RUN/judge.log"
exit 1
