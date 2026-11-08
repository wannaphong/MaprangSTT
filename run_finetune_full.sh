#!/bin/bash
set -e

DATASET_DIR="/workspace/whistle-thai/dataset_whistle_full"
OUTPUT_DIR="/workspace/whistle-thai/output_full"
mkdir -p "$OUTPUT_DIR"

cd "$DATASET_DIR"

echo "Starting cactus whistle finetuning on FULL Thai Common Voice (178h)..."

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
    --out "$OUTPUT_DIR/whistle-thai-full.cact"

echo "Model build complete: $OUTPUT_DIR/whistle-thai-full.cact"
