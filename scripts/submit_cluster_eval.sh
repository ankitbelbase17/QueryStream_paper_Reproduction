#!/bin/bash
#SBATCH --job-name=qs_sb_eval
#SBATCH --account=a168
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --output=/iopsstor/scratch/cscs/dbartaula/querystream/runs/slurm_%j.log
#
# QueryStream -- full StreamingBench evaluation (paper Table 1).
#
#   sbatch scripts/submit_cluster_eval.sh
#   RUN_NAME=aggressive POOL=max sbatch scripts/submit_cluster_eval.sh
#   NODES=4 sbatch --nodes=4 scripts/submit_cluster_eval.sh
#
# ONE TASK PER GPU. Each task takes a strided slice of the 500 videos
# (`--shard-id $SLURM_PROCID --num-shards $SLURM_NTASKS`) and writes its own
# shard directory; `scripts/merge_shards.py` unions them at the end. Striding
# rather than blocking because neighbouring videos in the manifest have similar
# durations, so contiguous blocks finish at very different times.
#
# WHY `normal` AND NOT `debug`. Frame decoding dominates: 246 GB of video, ~252k
# frames at 1 FPS, and cv2 must walk every source frame to reach each sampled
# one. The debug QOS caps a job at 90 node-minutes (4 nodes x 22 min), which is
# not enough for the full split on one node. Use debug only for the dry run:
#   LIMIT_VIDEOS=5 ASSERT_KEEP=45,58 RUN_NAME=sb_dryrun \
#     sbatch --partition=debug --time=00:30:00 --nodes=1 --ntasks-per-node=1 \
#            --gres=gpu:1 scripts/submit_cluster_eval.sh
#
# RESUMABLE. Every task passes --resume, so a job that hits the wall clock can be
# resubmitted unchanged and will skip the question_ids already in its JSONL.
#
# OPENAI_API_KEY is NOT stored in this file. Export it in the submitting shell if
# you want the parse-failure judge (`JUDGE=openai`); sbatch forwards the
# environment by default. Without it, --judge none is used and unparseable
# responses are scored zero, which is the official StreamingBench protocol.

set -uo pipefail

# ---- configuration (override via environment at submit time) ------------- #
REPO=${REPO:-/iopsstor/scratch/cscs/dbartaula/querystream}
DATA=${DATA:-/iopsstor/scratch/cscs/dbartaula/eval_7_10_8_19/simplestreamingeval/data/streamingbench}
ANNO=${ANNO:-$DATA/questions_real.json}
VIDEOS=${VIDEOS:-$DATA/videos}

RUN_NAME=${RUN_NAME:-sb_score_mean}
OUT_ROOT=${OUT_ROOT:-$REPO/runs/$RUN_NAME}

MASK_SPACE=${MASK_SPACE:-score}
POOL=${POOL:-mean}
RELEVANCE_MODE=${RELEVANCE_MODE:-running_minmax}
KEEP_MODE=${KEEP_MODE:-qdp}
TAU_TEMP=${TAU_TEMP:-0.90}
TAU_REL=${TAU_REL:-0.60}
TAU_DEN=${TAU_DEN:-0.15}
ALPHA=${ALPHA:-0.1}
FPS=${FPS:-1.0}
MAX_PIXELS=${MAX_PIXELS:-200704}          # 448*448
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16}
CLIP_BATCH=${CLIP_BATCH:-1}
DECODE_WORKERS=${DECODE_WORKERS:-2}
# Stop cleanly BEFORE the wall clock so the merge step still runs and no
# question is lost mid-write. Needed on `debug`, whose 90 node-minute QOS cap
# means the full split takes more than one job -- every worker passes --resume,
# so a resubmission finishes what is left.
TIME_BUDGET=${TIME_BUDGET:-0}
PREFETCH_DEPTH=${PREFETCH_DEPTH:-2}
VISION_BATCH_PLANES=${VISION_BATCH_PLANES:-8}
JUDGE=${JUDGE:-none}
JUDGE_MODEL=${JUDGE_MODEL:-gpt-4o-mini}
LIMIT_VIDEOS=${LIMIT_VIDEOS:-}
SHUFFLE_SEED=${SHUFFLE_SEED:-}
ASSERT_KEEP=${ASSERT_KEEP:-}
TASKS=${TASKS:-}
EXPECTED_Q=${EXPECTED_Q:-2500}

# Absolute interpreter: the bare name is not guaranteed on the
# compute nodes even though it resolves on the login node.
PYTHON=${PYTHON:-/users/dbartaula/miniforge3/bin/python3}

# ---- environment --------------------------------------------------------- #
export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/dbartaula/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}     # weights are already cached
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
# Concurrent cv2 readers on Lustre hit ffmpeg's default stream timeout at
# 42-60 s and killed the M3b subset run outright. Raising the timeout is the
# proven fix; DECODE_WORKERS bounds how many readers exist per task (4 tasks
# per node, so the node-wide total is 4 x DECODE_WORKERS).
export OPENCV_FFMPEG_CAPTURE_OPTIONS=${OPENCV_FFMPEG_CAPTURE_OPTIONS:-"timeout;15000000"}

mkdir -p "$OUT_ROOT"
echo "[job ${SLURM_JOB_ID:-local}] host=$(hostname) nodes=${SLURM_NNODES:-1} tasks=${SLURM_NTASKS:-1}"
echo "[job] out=$OUT_ROOT  mask_space=$MASK_SPACE pool=$POOL relevance=$RELEVANCE_MODE keep=$KEEP_MODE"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

EXTRA=()
[[ -n "$LIMIT_VIDEOS" ]] && EXTRA+=(--limit-videos "$LIMIT_VIDEOS")
[[ -n "$TASKS" ]]        && EXTRA+=(--tasks "$TASKS")
[[ -n "$SHUFFLE_SEED" ]] && EXTRA+=(--shuffle-seed "$SHUFFLE_SEED")
[[ -n "$ASSERT_KEEP" ]]  && EXTRA+=(--dry-run-assert-keep "$ASSERT_KEEP")
[[ "$TIME_BUDGET" != "0" ]] && EXTRA+=(--time-budget-sec "$TIME_BUDGET")

# ---- one worker per GPU --------------------------------------------------- #
run_worker() {
  local shard_id=$1 num_shards=$2 local_gpu=$3
  local out="$OUT_ROOT/shard${shard_id}"
  mkdir -p "$out"
  echo "[worker $shard_id/$num_shards] gpu=$local_gpu -> $out"
  CUDA_VISIBLE_DEVICES=$local_gpu "$PYTHON" "$REPO/eval_streaming.py" \
      --anno "$ANNO" --video-dir "$VIDEOS" --output-dir "$out" \
      --device cuda:0 \
      --num-shards "$num_shards" --shard-id "$shard_id" --resume \
      --mask-space "$MASK_SPACE" --pool "$POOL" \
      --relevance-mode "$RELEVANCE_MODE" --keep-mode "$KEEP_MODE" \
      --alpha "$ALPHA" --tau-temp "$TAU_TEMP" --tau-rel "$TAU_REL" --tau-den "$TAU_DEN" \
      --fps "$FPS" --max-pixels "$MAX_PIXELS" --max-new-tokens "$MAX_NEW_TOKENS" \
      --clip-batch "$CLIP_BATCH" --vision-batch-planes "$VISION_BATCH_PLANES" \
      --decode-workers "$DECODE_WORKERS" --prefetch-depth "$PREFETCH_DEPTH" \
      --judge "$JUDGE" --judge-model "$JUDGE_MODEL" \
      --progress-every 5 \
      "${EXTRA[@]}" \
      > "$out/worker.log" 2>&1
  echo "[worker $shard_id] exit=$? -> $out/worker.log"
}

# SLURM sets SLURM_PROCID=0 in the BATCH step too, so it cannot distinguish
# "I am an srun task" from "I am the batch script". Without an explicit sentinel
# the batch script runs itself as worker 0 and never fans out -- a 4-task job
# would silently evaluate one quarter of the split and skip the merge. Verified
# on job 3162894, which did exactly that.
if [[ -n "${QS_WORKER:-}" ]]; then
  run_worker "$SLURM_PROCID" "${SLURM_NTASKS:-1}" "${SLURM_LOCALID:-0}"
  exit $?
fi

# `$0` inside a batch script is SLURM's per-node spool copy
# (/var/spool/slurmd/job*/slurm_script), which exists ONLY on the node running
# the batch step. `srun bash "$0"` therefore succeeds on that one node and dies
# with `exit code 127` on every other -- it reads as a partial hardware failure
# rather than a missing file. Re-exec the copy on shared scratch instead.
# This script had that bug and was safe only at --nodes=1; found on the OmniPro
# arm (job 3175916 lost 2 of its 3 nodes exactly this way).
SELF=${SELF:-$REPO/scripts/submit_cluster_eval.sh}
if [[ ! -r "$SELF" ]]; then
  echo "[fatal] cannot read $SELF from the compute nodes" >&2
  exit 127
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun --kill-on-bad-exit=0 --export=ALL,QS_WORKER=1 bash "$SELF"
else
  # Interactive fallback: fan out over the GPUs visible on this machine.
  N=${N:-$(nvidia-smi -L | wc -l)}
  echo "[local] no SLURM allocation; running $N workers on this host"
  for i in $(seq 0 $((N - 1))); do run_worker "$i" "$N" "$i" & done
  wait
fi

# ---- merge ---------------------------------------------------------------- #
echo "[merge] unioning shards ..."
"$PYTHON" "$REPO/scripts/merge_shards.py" "$OUT_ROOT" --expected "$EXPECTED_Q" \
    | tee "$OUT_ROOT/merged_summary.txt"
echo "[job] done: $OUT_ROOT"
