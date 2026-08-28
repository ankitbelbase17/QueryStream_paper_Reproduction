#!/bin/bash
#SBATCH --job-name=qs_omnipro
#SBATCH --account=a168
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --output=/iopsstor/scratch/cscs/dbartaula/querystream/runs/omnipro_%j.log
#
# QueryStream -- OmniPro Online-mode evaluation (vision only).
#
#   sbatch scripts/submit_omnipro.sh
#   RUN_NAME=op_dense KEEP_MODE=dense sbatch scripts/submit_omnipro.sh
#   RUN_NAME=op_relonly GATE_MODE=relevance_only sbatch scripts/submit_omnipro.sh
#
# THREE NODES, NOT FOUR. The debug QOS caps a job at `node=90` TRES-minutes
# (`sacctmgr show qos debug-qos format=MaxTRESMinsPJ`), so 4 nodes x 30 min = 120
# node-minutes is rejected outright: the job sits PENDING forever with
# Reason=QOSMaxNodeMinutesPerJob. 3 x 30 = 90 fits exactly and gives 12 GPUs;
# 4 nodes would have to drop to 22 minutes, which buys less useful time once the
# ~90 s model load is amortised over more GPUs. `debug` also has MaxNodes=4 and
# MaxJobsPU=1 / MaxSubmitPU=2, so at most two jobs can be queued at once --
# submit a second identical job to resume whatever the first does not finish.
#
# ONE TASK PER GPU. Each task takes a strided slice of
# the 1,262 videos and writes its own shard JSONL into the SHARED run directory;
# `score_omnipro.py` unions every `omnipro_raw_logs*.jsonl` it finds there.
# Sharding is BY VIDEO so one decode+encode pass serves all of a video's
# samples, and strided rather than blocked because neighbouring videos have
# correlated durations.
#
# WHY debug WORKS HERE, WHERE IT DID NOT FOR StreamingBench. OmniPro's videos
# average 189 s at 720p against StreamingBench's 504 s at 1080p, so the whole
# 2,700-sample benchmark is ~238k frames rather than ~252k much more expensive
# ones -- roughly 20 minutes across 16 GPUs. `--time-budget-sec` still makes
# each worker stop cleanly before the wall clock rather than being killed
# mid-sample, and every worker passes --resume, so a job that runs out of time
# can be resubmitted unchanged and will finish what is left.
#
# OPENAI_API_KEY IS NOT STORED IN THIS FILE and is not needed by it: the judge
# runs afterwards, off the cluster, over the Batch API (judge_omnipro.py).

set -uo pipefail

REPO=${REPO:-/iopsstor/scratch/cscs/dbartaula/querystream}
RUN_NAME=${RUN_NAME:-op_full}
OUT=${OUT:-$REPO/runs/$RUN_NAME}

BENCH=${BENCH:-/iopsstor/scratch/cscs/dbartaula/omnipro_data/benchmark.json}
DATASET_DIR=${DATASET_DIR:-/iopsstor/scratch/cscs/dbartaula/omnipro_data}

MASK_SPACE=${MASK_SPACE:-score}
POOL=${POOL:-mean}
RELEVANCE_MODE=${RELEVANCE_MODE:-running_minmax}
KEEP_MODE=${KEEP_MODE:-qdp}
GATE_MODE=${GATE_MODE:-full}
TAU_TEMP=${TAU_TEMP:-0.90}
TAU_REL=${TAU_REL:-0.60}
TAU_DEN=${TAU_DEN:-0.15}
ALPHA=${ALPHA:-0.1}
EVENT_GAP=${EVENT_GAP:-3.0}
MAX_EMITS=${MAX_EMITS:-64}
CLIP_BATCH=${CLIP_BATCH:-8}
VISION_BATCH_PLANES=${VISION_BATCH_PLANES:-8}
DECODE_WORKERS=${DECODE_WORKERS:-3}
PREFETCH_DEPTH=${PREFETCH_DEPTH:-2}
TIME_BUDGET=${TIME_BUDGET:-1500}
N_SAMPLES=${N_SAMPLES:-0}
PREFER_SHORT=${PREFER_SHORT:-0.0}
AUDIO=${AUDIO:-none,helpful,required}
TASKS=${TASKS:-}

# ABSOLUTE interpreter path, not the bare name. `python` resolves on the login
# node through the user's miniforge PATH, but that PATH is not guaranteed on
# every compute node: job 3175916 had 2 of 3 nodes die with `exit code 127`
# while the third ran fine, which reads as a mysterious partial failure rather
# than "command not found".
PYTHON=${PYTHON:-/users/dbartaula/miniforge3/bin/python3}
if [[ ! -x "$PYTHON" ]]; then
  echo "[fatal] interpreter not executable: $PYTHON" >&2
  exit 127
fi

export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/dbartaula/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
# Six concurrent cv2 readers on Lustre hit ffmpeg's default stream timeout at
# 42 s and killed a StreamingBench run mid-flight; the explicit timeout plus
# three decode threads ran clean.
export OPENCV_FFMPEG_CAPTURE_OPTIONS=${OPENCV_FFMPEG_CAPTURE_OPTIONS:-"timeout;15000000"}

mkdir -p "$OUT"
EXTRA=()
[[ -n "$TASKS" ]] && EXTRA+=(--tasks "$TASKS")
[[ "$N_SAMPLES" != "0" ]] && EXTRA+=(--n-samples "$N_SAMPLES" --prefer-short "$PREFER_SHORT")

run_worker() {
  local shard_id=$1 num_shards=$2 local_gpu=$3
  echo "[worker $shard_id/$num_shards] host=$(hostname) gpu=$local_gpu"
  CUDA_VISIBLE_DEVICES=$local_gpu "$PYTHON" "$REPO/eval_omnipro.py" \
      --benchmark-json "$BENCH" --dataset-dir "$DATASET_DIR" \
      --out "$OUT" --run-name "$RUN_NAME" --device cuda:0 \
      --num-shards "$num_shards" --shard-id "$shard_id" --resume \
      --time-budget-sec "$TIME_BUDGET" \
      --audio "$AUDIO" \
      --mask-space "$MASK_SPACE" --pool "$POOL" \
      --relevance-mode "$RELEVANCE_MODE" --keep-mode "$KEEP_MODE" \
      --gate-mode "$GATE_MODE" \
      --alpha "$ALPHA" --tau-temp "$TAU_TEMP" --tau-rel "$TAU_REL" \
      --tau-den "$TAU_DEN" --event-gap-s "$EVENT_GAP" --max-emits "$MAX_EMITS" \
      --clip-batch "$CLIP_BATCH" --vision-batch-planes "$VISION_BATCH_PLANES" \
      --decode-workers "$DECODE_WORKERS" --prefetch-depth "$PREFETCH_DEPTH" \
      "${EXTRA[@]}" \
      > "$OUT/worker$(printf '%02d' "$shard_id").log" 2>&1
  echo "[worker $shard_id] exit=$?"
}

# SLURM sets SLURM_PROCID=0 in the BATCH step as well as in srun tasks, so
# branching on it makes the batch script run itself as worker 0 and never fan
# out. The explicit sentinel is the fix (verified the hard way on job 3162894).
if [[ -n "${QS_WORKER:-}" ]]; then
  run_worker "$SLURM_PROCID" "${SLURM_NTASKS:-1}" "${SLURM_LOCALID:-0}"
  exit $?
fi

echo "[job ${SLURM_JOB_ID:-local}] nodes=${SLURM_NNODES:-1} tasks=${SLURM_NTASKS:-1} out=$OUT"
echo "[job] mask_space=$MASK_SPACE pool=$POOL relevance=$RELEVANCE_MODE gate=$GATE_MODE gap=$EVENT_GAP"

# `$0` inside a batch script is SLURM's per-node spool copy
# (/var/spool/slurmd/job*/slurm_script), which exists ONLY on the node running
# the batch step. `srun bash "$0"` therefore succeeds on that one node and dies
# with `exit code 127` on every other -- job 3175916 lost 2 of 3 nodes exactly
# this way, and it reads as a partial hardware failure rather than a missing
# file. Re-exec the copy on shared scratch instead. StreamingBench never hit
# this because it only ever ran --nodes=1.
SELF=${SELF:-$REPO/scripts/submit_omnipro.sh}
if [[ ! -r "$SELF" ]]; then
  echo "[fatal] cannot read $SELF from the compute nodes" >&2
  exit 127
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun --kill-on-bad-exit=0 --export=ALL,QS_WORKER=1 bash "$SELF"
else
  N=${N:-$(nvidia-smi -L | wc -l)}
  for i in $(seq 0 $((N - 1))); do run_worker "$i" "$N" "$i" & done
  wait
fi

# ---- score what landed ---------------------------------------------------- #
echo "[score] aggregating shards ..."
"$PYTHON" "$REPO/score_omnipro.py" "$OUT" | tee "$OUT/score.txt"
echo "[job] done: $OUT"
echo "[next] judge:  python judge_omnipro.py $OUT --submit"
echo "[next] then:   python judge_omnipro.py $OUT --collect && \\"
echo "               python score_omnipro.py $OUT && \\"
echo "               python make_omnipro_report.py $OUT"
