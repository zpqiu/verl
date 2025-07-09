#!/usr/bin/env bash
set -uxo pipefail

export TRAIN_FILE=${TRAIN_FILE:-"${HOME}/data/dapo-math-17k.parquet"}
export TEST_FILE=${TEST_FILE:-"${HOME}/data/aime-2024.parquet"}
export OVERWRITE=${OVERWRITE:-0}

if [ ! -f "${TRAIN_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TRAIN_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
fi

if [ ! -f "${TEST_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TEST_FILE}" "https://huggingface.co/datasets/pe-nlp/DAPO-AIME-2024/resolve/main/data/train-00000-of-00001.parquet?download=true"
fi
