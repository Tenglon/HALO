#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <train_config_path> <checkpoint_path> <output_root>" >&2
  echo "Example: $0 vlm/configs/clip_vit_b32_cc3m_lorentz_lora_negdist.yaml outputs/halo_b32/checkpoint_step_5000.pt outputs/eval/halo_b32" >&2
  exit 2
fi

TRAIN_CONFIG="$1"
CHECKPOINT="$2"
OUTPUT_ROOT="$3"

DATASET="${DATASET:-both}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

run_eval() {
  local name="$1"
  local config_name="$2"
  local out_dir="${OUTPUT_ROOT}/${name}"
  mkdir -p "${out_dir}"

  local -a cmd=(
    "${PYTHON_BIN}" -m vlm.cli.eval
    --config-name "${config_name}"
    "hydra.run.dir=${out_dir}"
    hydra.output_subdir=null
    eval.model_source=vlm_checkpoint
    "eval.vlm_train_config=${TRAIN_CONFIG}"
    "eval.vlm_checkpoint=${CHECKPOINT}"
    "eval.max_samples=${MAX_SAMPLES}"
    "eval.device=${DEVICE}"
  )

  if [ -n "${BATCH_SIZE}" ]; then
    cmd+=("eval.batch_size=${BATCH_SIZE}")
  fi

  echo "[halo_eval] dataset=${name} config=${config_name} output=${out_dir}"
  "${cmd[@]}" 2>&1 | tee "${out_dir}/run.log"
}

case "${DATASET}" in
  coco)
    run_eval coco eval_coco_retrieval_vlm_checkpoint
    ;;
  flickr|flickr30k)
    run_eval flickr eval_flickr30k_retrieval_vlm_checkpoint
    ;;
  both)
    run_eval coco eval_coco_retrieval_vlm_checkpoint
    run_eval flickr eval_flickr30k_retrieval_vlm_checkpoint
    ;;
  *)
    echo "Unknown DATASET=${DATASET}. Use coco, flickr, or both." >&2
    exit 2
    ;;
esac
