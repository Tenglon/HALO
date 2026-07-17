#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <hydra_train_config_name> <output_dir>" >&2
  echo "Example: $0 clip_vit_b32_cc3m_lorentz_lora_negdist outputs/halo_b32" >&2
  exit 2
fi

CONFIG_NAME="$1"
OUTPUT_DIR="$2"

NUM_PROCESSES="${NUM_PROCESSES:-8}"
NUM_MACHINES="${NUM_MACHINES:-1}"
PER_GPU_BATCH="${PER_GPU_BATCH:-256}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-2048}"
MAX_STEPS="${MAX_STEPS:-5000}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-10}"
LR="${LR:-2e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
SCHEDULER="${SCHEDULER:-cosine}"
WARMUP="${WARMUP:-500}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
TRAIN_LOGIT_SCALE="${TRAIN_LOGIT_SCALE:-1}"
LEARN_CURV="${LEARN_CURV:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"
DYNAMO_BACKEND="${DYNAMO_BACKEND:-no}"

GRAD_ACCUM=$((TARGET_GLOBAL_BATCH / (PER_GPU_BATCH * NUM_PROCESSES)))
if [ "${GRAD_ACCUM}" -lt 1 ]; then
  GRAD_ACCUM=1
fi

mkdir -p "${OUTPUT_DIR}"

cmd=(
  accelerate launch
  --multi_gpu
  --num_processes "${NUM_PROCESSES}"
  --num_machines "${NUM_MACHINES}"
  --mixed_precision "${MIXED_PRECISION}"
  --dynamo_backend "${DYNAMO_BACKEND}"
  --main_process_port "${MAIN_PROCESS_PORT}"
  -m vlm.cli.train
  --config-name "${CONFIG_NAME}"
  hydra.run.dir=.
  hydra.output_subdir=null
  "train.batch_size=${PER_GPU_BATCH}"
  "train.grad_accum=${GRAD_ACCUM}"
  "train.lr=${LR}"
  "train.wd=${WEIGHT_DECAY}"
  "train.scheduler=${SCHEDULER}"
  "train.warmup=${WARMUP}"
  "train.min_lr_ratio=${MIN_LR_RATIO}"
  "train.train_logit_scale=${TRAIN_LOGIT_SCALE}"
  "train.schedule_total_steps=${MAX_STEPS}"
  "model.lorentz.learn_curv=${LEARN_CURV}"
  "logging.log_every=${LOG_EVERY}"
  "train.max_steps=${MAX_STEPS}"
  "logging.save_every=${SAVE_EVERY}"
  "logging.output_dir=${OUTPUT_DIR}"
)

if [ -n "${RESUME_FROM:-}" ]; then
  cmd+=("train.resume_from=${RESUME_FROM}")
fi

echo "[halo_train] config=${CONFIG_NAME} output_dir=${OUTPUT_DIR}"
echo "[halo_train] num_processes=${NUM_PROCESSES} per_gpu_batch=${PER_GPU_BATCH} grad_accum=${GRAD_ACCUM} max_steps=${MAX_STEPS}"
"${cmd[@]}"
