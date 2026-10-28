#!/usr/bin/env bash
set -e

# DataMate entrypoint script for Zalo AI Challenge 2026
# Usage:
#   ./predict.sh [/path/to/data_dir] [/path/to/output_file]

DATA_DIR=${1:-/data}
OUTPUT_FILE=${2:-/result/submission.jsonl}
# Default config selection:
# If CONFIG is not set, use dev.yaml for local Mac, prod.yaml for container /data
if [ -z "$CONFIG" ]; then
    if [ "$DATA_DIR" = "/data" ] && [ -f "configs/prod.yaml" ]; then
        CONFIG_FILE="configs/prod.yaml"
    elif [ -f "configs/dev.yaml" ]; then
        CONFIG_FILE="configs/dev.yaml"
    else
        CONFIG_FILE="configs/prod.yaml"
    fi
else
    CONFIG_FILE="$CONFIG"
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
