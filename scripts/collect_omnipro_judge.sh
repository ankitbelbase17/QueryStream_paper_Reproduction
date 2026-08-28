#!/usr/bin/env bash
# Poll the OmniPro content-judge batch, then re-score and rebuild the PDF.
#
#   export OPENAI_API_KEY=...            # environment only; never written to disk
#   nohup bash scripts/collect_omnipro_judge.sh runs/op_full > runs/op_judge.log 2>&1 &
#
# Safe to run at any point: `score_omnipro.py` and `make_omnipro_report.py` are
# pure functions of the shard JSONLs, so the PDF that already exists stays valid
# and is simply replaced by a more complete one when the batch lands. Nothing
# here can change a timing metric -- the judge only decides content on the two
# open-ended sub-tasks.
#
# THIS SCRIPT DIES WITH ITS SHELL. If it is gone and the batch has landed, the
# manual equivalent is the last three lines of the loop below.
set -uo pipefail

RUN=${1:?usage: collect_omnipro_judge.sh <run_dir>}
REPO=${REPO:-/iopsstor/scratch/cscs/dbartaula/querystream}
PYTHON=${PYTHON:-/users/dbartaula/miniforge3/bin/python3}
INTERVAL=${INTERVAL:-120}
MAX_HOURS=${MAX_HOURS:-24}          # the Batch API's completion window

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[judge] OPENAI_API_KEY is not set; nothing to collect." >&2
  exit 2
fi

deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
while [[ $(date +%s) -lt $deadline ]]; do
  if "$PYTHON" "$REPO/judge_omnipro.py" "$RUN" --collect --wait 60; then
    echo "[judge] verdicts landed; re-scoring and rebuilding the report"
    "$PYTHON" "$REPO/score_omnipro.py" "$RUN" | tee "$RUN/score.txt"
    "$PYTHON" "$REPO/make_omnipro_report.py" "$RUN"
    exit 0
  fi
  sleep "$INTERVAL"
done
echo "[judge] batch did not complete within ${MAX_HOURS}h. The run's timing "
echo "        metrics and its seven exact-match sub-tasks are unaffected and "
echo "        already in $RUN/omnipro_metrics.json; only the two open-ended "
echo "        sub-tasks stay withheld."
exit 1
