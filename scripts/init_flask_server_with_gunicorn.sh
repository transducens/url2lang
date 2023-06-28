#!/bin/bash

DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

pushd "$DIR"

MODEL="$1"
BATCH_SIZE="$2"
TOTAL_GPUS="$3"

if [[ -z "$MODEL" ]] || [[ ! -f "$MODEL" ]] || [[ -z "$BATCH_SIZE" ]] || [[ -z "$TOTAL_GPUS" ]]; then
  >&2 echo "Syntax: absolute_path_to_model batch_size total_gpus"
  exit 1
fi

echo "BE AWARE: if reverse connection are being used (e.g. 'ssh -N -R 8000:localhost:8000'-like), you might need to use 'ulimit -n'"

GUNICORN_THREADS_DIFF_BATCH_SIZE="10"
GUNICORN_THREADS=$((BATCH_SIZE > GUNICORN_THREADS_DIFF_BATCH_SIZE ? BATCH_SIZE - GUNICORN_THREADS_DIFF_BATCH_SIZE : 1))

srun --gres=gpu:$TOTAL_GPUS --cpus-per-task=2 --mem-per-cpu=6G \
  gunicorn --timeout 0 -w $TOTAL_GPUS --threads 10 --worker-class gthread \
    "flask_server_wrapper:init('$MODEL', $BATCH_SIZE, 0.2, 'language-identification')"

popd
