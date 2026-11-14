#!/usr/bin/env bash
set -euo pipefail
ROOT=/iopsstor/scratch/cscs/dbartaula/querystream_omnipro_20261114
export QS_ROOT="$ROOT"
export HF_HOME="$ROOT/hf"
export HF_HUB_CACHE=/iopsstor/scratch/cscs/dbartaula/hf_cache/hub
export XDG_CACHE_HOME="$ROOT/cache"
mkdir -p "$XDG_CACHE_HOME/torch/kernels"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$1"
STAGE="$2"
if [[ "$STAGE" == validation ]]; then
  exec python3 "$ROOT/validate.py"
elif [[ "$STAGE" == features ]]; then
  exec python3 "$ROOT/experiment.py" features "${@:3}"
else
  exec python3 "$ROOT/experiment.py" infer --stage "$STAGE" "${@:3}"
fi
