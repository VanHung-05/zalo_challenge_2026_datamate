#!/usr/bin/env bash
set -e

# DataMate entrypoint script for Zalo AI Challenge 2026
# Usage:
#   ./predict.sh [/path/to/data_dir] [/path/to/output_file]

DATA_DIR=${1:-/data}
OUTPUT_FILE=${2:-/result/submission.jsonl}
CONFIG_FILE=${CONFIG:-configs/prod.yaml}

# If running locally and DATA_DIR does not exist, fallback to public_test
if [ ! -d "$DATA_DIR" ]; then
    if [ -d "public_test" ]; then
        DATA_DIR="public_test"
    elif [ -d "train" ]; then
        DATA_DIR="train"
    fi
fi

if [ -f "configs/dev.yaml" ] && [ ! -f "$CONFIG_FILE" ]; then
    CONFIG_FILE="configs/dev.yaml"
fi

echo "=========================================="
echo "Starting DataMate Agent"
echo "Data Directory: $DATA_DIR"
echo "Output File:    $OUTPUT_FILE"
echo "Config File:    $CONFIG_FILE"
echo "=========================================="

# Use virtualenv python if available, else system python3
PYTHON_EXEC="python3"
if [ -f ".venv/bin/python3" ]; then
    PYTHON_EXEC=".venv/bin/python3"
fi

$PYTHON_EXEC src/datamate/runner.py \
    --config "$CONFIG_FILE" \
    --data-dir "$DATA_DIR" \
    --output "$OUTPUT_FILE"

echo "DataMate processing completed successfully."
