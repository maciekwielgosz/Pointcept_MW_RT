#!/usr/bin/env bash
set -Eeuo pipefail

CFG="configs/ptv3_forest_segmentation/ptv3_decoder_sem_seg_dual_head.py"
WEIGHT="exp/ptv3_forest_segmentation/ptv3_decoder_sem_seg_dual_head/model/model_last.pth"
OUT_DIR="output/inference_results"
NUM_GPUS=8

export PYTHONPATH=./
python3 tools/test.py \
  --config-file "${CFG}" \
  --num-gpus ${NUM_GPUS} \
  --options save_path="${OUT_DIR}" weight="${WEIGHT}"

