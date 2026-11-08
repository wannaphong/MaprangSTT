#!/bin/bash
set -e

DATASET_DIR="/workspace/whistle-thai/dataset_whistle_combined"
OUTPUT_DIR="/workspace/whistle-thai/output_combined"
mkdir -p "$OUTPUT_DIR"

cd "$DATASET_DIR"

echo "Starting cactus whistle finetuning from zero on COMBINED Thai Dataset (594,231 clips, 1,022.78 hours)..."

unset WHISTLE_RESUME
unset WHISTLE_START_EPOCH

PYTHONUNBUFFERED=1 cactus whistle finetune \
    train.jsonl \
    validation.jsonl \
    test.jsonl \
    --language th \
    --tokens tokens.txt \
    --epochs 3 \
    --batch-size 16 \
    --lr 1e-3 \
    --checkpoint-dir "$OUTPUT_DIR/checkpoints" \
    --out "$OUTPUT_DIR/adapter.safetensors"

echo "Finetuning complete! Building standalone .cact model..."
cactus whistle build \
    --lora "$OUTPUT_DIR/adapter.safetensors" \
    --out "$OUTPUT_DIR/maprang.cact"

echo "Model build complete: $OUTPUT_DIR/maprang.cact"
