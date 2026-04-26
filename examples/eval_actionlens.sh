#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ActionLens Evaluation Example
#
# Runs the full ActionLens benchmark (D1–D5) on a chosen model.
# Videos are downloaded automatically from HuggingFace on first run.
#
# Usage:
#   bash examples/eval_actionlens.sh                     # default: Qwen2.5-VL-3B
#   MODEL=Qwen/Qwen2.5-VL-72B-Instruct bash examples/eval_actionlens.sh
#   TASKS=d1_transition_sensitivity bash examples/eval_actionlens.sh  # single diagnostic
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL="${MODEL:-qwen2_5_vl}"
MODEL_PRETRAINED="${MODEL_PRETRAINED:-Qwen/Qwen2.5-VL-3B-Instruct}"
TASKS="${TASKS:-actionlens}"          # or e.g. d1_transition_sensitivity
BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/actionlens}"
LIMIT="${LIMIT:-10}"                    # set to e.g. 10 for a quick smoke-test

# ── Build optional flags ──────────────────────────────────────────────────────
LIMIT_FLAG=""
if [[ -n "${LIMIT}" ]]; then
    LIMIT_FLAG="--limit ${LIMIT}"
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo "=== ActionLens Evaluation ==="
echo "  Model      : ${MODEL_PRETRAINED}"
echo "  Tasks      : ${TASKS}"
echo "  Device     : ${DEVICE}"
echo "  Output dir : ${OUTPUT_DIR}"
echo ""

python -m lmms_eval \
    --model "${MODEL}" \
    --model_args "pretrained=${MODEL_PRETRAINED}" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --output_path "${OUTPUT_DIR}" \
    --log_samples \
    ${LIMIT_FLAG}

echo ""
echo "Results saved to ${OUTPUT_DIR}"
