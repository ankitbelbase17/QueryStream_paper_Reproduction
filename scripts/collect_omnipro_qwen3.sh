#!/bin/bash
# Poll the OmniPro content-judge batch for runs/op_qwen3 until it lands, then
# rescore and rebuild the PDF. Holds OPENAI_API_KEY in its ENVIRONMENT ONLY --
# the key is never written to this file, to a log, or to any JSONL.
#
#   OPENAI_API_KEY=... nohup setsid bash scripts/collect_omnipro_qwen3.sh \
#     > runs/op_qwen3_judge.log 2>&1 &
#
# Safe to run more than once and safe to kill: --collect is idempotent, and
# score/report are pure functions of the JSONL plus judge_output.jsonl.
set -uo pipefail
REPO="${REPO:-/iopsstor/scratch/cscs/dbartaula/querystream}"
RUN="${RUN:-runs/op_qwen3}"
MAX_HOURS="${MAX_HOURS:-26}"      # batch SLA is 24h; give it a little margin
cd "$REPO" || exit 1
: "${OPENAI_API_KEY:?export OPENAI_API_KEY before running}"
export HF_HOME="${HF_HOME:-/iopsstor/scratch/cscs/dbartaula/hf_cache}"
export PYTHONUNBUFFERED=1

deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  python3 judge_omnipro.py "$RUN" --collect --wait 900 2>&1 | grep -v '^\s*$'
  if [ -s "$RUN/judge_output.jsonl" ]; then
    echo "[collector] verdicts landed $(date -Is); rescoring"
    python3 score_omnipro.py "$RUN"      2>&1 | tail -5
    python3 make_omnipro_report.py "$RUN" 2>&1 | tail -3
    echo "[collector] done $(date -Is)"
    exit 0
  fi
  echo "[collector] still queued $(date -Is)"
done
echo "[collector] gave up after ${MAX_HOURS}h; timing metrics are unaffected and"
echo "[collector] the PDF already published content LOWER BOUNDS, never zeros."
exit 1
