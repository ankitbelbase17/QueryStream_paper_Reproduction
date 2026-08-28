#!/bin/bash
#SBATCH --job-name=qs_ovo
#SBATCH --account=a168
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --output=/iopsstor/scratch/cscs/dbartaula/querystream/runs/ovo_%j.log
#
# QueryStream on OVO-Bench with the paper's own backbone, TimeChat-Online-7B.
#
#   FRACTION=0.2 sbatch scripts/submit_ovo.sh          # 20% stratified subset
#   RUN_NAME=ovo_dense KEEP_MODE=dense sbatch scripts/submit_ovo.sh
#   BACKBONE=Qwen/Qwen2.5-VL-7B-Instruct RUN_NAME=ovo_qwen sbatch scripts/submit_ovo.sh
#
# THREE NODES, NOT FOUR: the debug QOS caps a job at `node=90` TRES-minutes, so
# 4 x 30 = 120 is rejected outright and the job pends forever with
# Reason=QOSMaxJobsPerUserLimit. `normal` is unusable for account a168 -- every
# association in it carries the `stop` QOS (MaxJobsPU=0). MaxJobsPU=1 /
# MaxSubmitPU=2 on debug, so submit twice: the second job resumes the first.
#
# DECODE RESIDENCY IS THE FAILURE MODE HERE. OVO clips run to 1920x1440 and CRR
# averages 753 s, so one decoded clip can exceed 6 GB. Four workers per node
# with a deep prefetch queue is what OOM-killed StreamingBench workers
# (CLAUDE.md 16). DECODE_WORKERS=1 / PREFETCH_DEPTH=1 caps residency at ~2-3
# clips per worker; raise it only on short-clip task subsets.

set -uo pipefail

REPO=${REPO:-/iopsstor/scratch/cscs/dbartaula/querystream}
RUN_NAME=${RUN_NAME:-ovo_20pct}
OUT=${OUT:-$REPO/runs/$RUN_NAME}

BACKBONE=${BACKBONE:-wyccccc/TimeChatOnline-7B}
FRACTION=${FRACTION:-0.2}
SEED=${SEED:-0}
TASKS=${TASKS:-}
MASK_SPACE=${MASK_SPACE:-score}
POOL=${POOL:-mean}
KEEP_MODE=${KEEP_MODE:-qdp}
GATE_MODE=${GATE_MODE:-full}
RELEVANCE_MODE=${RELEVANCE_MODE:-running_minmax}
TAU_TEMP=${TAU_TEMP:-0.90}
TAU_REL=${TAU_REL:-0.60}
TAU_DEN=${TAU_DEN:-0.15}
ALPHA=${ALPHA:-0.1}
CLIP_BATCH=${CLIP_BATCH:-8}
VISION_BATCH_PLANES=${VISION_BATCH_PLANES:-8}
DECODE_WORKERS=${DECODE_WORKERS:-1}
PREFETCH_DEPTH=${PREFETCH_DEPTH:-1}
MAX_CLIP_SECONDS=${MAX_CLIP_SECONDS:-0}
TIME_BUDGET=${TIME_BUDGET:-1500}

PYTHON=${PYTHON:-/users/dbartaula/miniforge3/bin/python3}
[[ -x "$PYTHON" ]] || { echo "[fatal] no interpreter at $PYTHON" >&2; exit 127; }

export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/dbartaula/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OPENCV_FFMPEG_CAPTURE_OPTIONS=${OPENCV_FFMPEG_CAPTURE_OPTIONS:-"timeout;15000000"}

mkdir -p "$OUT"
EXTRA=()
[[ -n "$TASKS" ]] && EXTRA+=(--tasks "$TASKS")
[[ "$MAX_CLIP_SECONDS" != "0" ]] && EXTRA+=(--max-clip-seconds "$MAX_CLIP_SECONDS")

run_worker() {
  local shard=$1 nshards=$2 gpu=$3
  echo "[worker $shard/$nshards] host=$(hostname) gpu=$gpu"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$REPO/eval_ovo.py" \
      --out "$OUT" --run-name "$RUN_NAME" --device cuda:0 \
      --backbone "$BACKBONE" \
      --fraction "$FRACTION" --seed "$SEED" \
      --num-shards "$nshards" --shard-id "$shard" --resume \
      --time-budget-sec "$TIME_BUDGET" \
      --mask-space "$MASK_SPACE" --pool "$POOL" --keep-mode "$KEEP_MODE" \
      --gate-mode "$GATE_MODE" --relevance-mode "$RELEVANCE_MODE" \
      --alpha "$ALPHA" --tau-temp "$TAU_TEMP" --tau-rel "$TAU_REL" \
      --tau-den "$TAU_DEN" \
      --clip-batch "$CLIP_BATCH" --vision-batch-planes "$VISION_BATCH_PLANES" \
      --decode-workers "$DECODE_WORKERS" --prefetch-depth "$PREFETCH_DEPTH" \
      "${EXTRA[@]}" \
      > "$OUT/worker$(printf '%02d' "$shard").log" 2>&1
  echo "[worker $shard] exit=$?"
}

# `$0` is SLURM's per-node spool copy and exists only on the batch node; re-exec
# the shared-scratch copy or every other node dies with exit code 127.
SELF=${SELF:-$REPO/scripts/submit_ovo.sh}
[[ -r "$SELF" ]] || { echo "[fatal] cannot read $SELF from the nodes" >&2; exit 127; }

if [[ -n "${QS_WORKER:-}" ]]; then
  run_worker "$SLURM_PROCID" "${SLURM_NTASKS:-1}" "${SLURM_LOCALID:-0}"
  exit $?
fi

echo "[job ${SLURM_JOB_ID:-local}] backbone=$BACKBONE fraction=$FRACTION out=$OUT"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun --kill-on-bad-exit=0 --export=ALL,QS_WORKER=1 bash "$SELF"
else
  N=${N:-$(nvidia-smi -L | wc -l)}
  for i in $(seq 0 $((N-1))); do run_worker "$i" "$N" "$i" & done
  wait
fi

echo "[score] aggregating shards ..."
"$PYTHON" "$REPO/score_ovo.py" "$OUT" | tee "$OUT/score.txt"
echo "[next] python make_ovo_report.py $OUT"
