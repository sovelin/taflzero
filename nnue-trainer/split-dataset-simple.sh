#!/bin/bash

# Check if at least one argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <file_name> [number_of_lines]"
    exit 1
fi

# Input file
INPUT_FILE="$1"

# Check if the file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "File '$INPUT_FILE' does not exist."
    exit 1
fi

# Determine total lines to use
if [ $# -eq 2 ]; then
    TOTAL_LINES_TO_USE=$2
    FILE_LINES=$(wc -l < "$INPUT_FILE")
    if [ "$TOTAL_LINES_TO_USE" -gt "$FILE_LINES" ]; then
        echo "The file has only $FILE_LINES lines, but you requested $TOTAL_LINES_TO_USE."
        exit 1
    fi
else
    TOTAL_LINES_TO_USE=$(wc -l < "$INPUT_FILE")
fi

# Generate output file names
TRAIN_FILE="train_$(basename "$INPUT_FILE")"
VALIDATE_FILE="validate_$(basename "$INPUT_FILE")"

# Calculate lines for training (90%) and validation (10%)
TRAIN_LINES=$((TOTAL_LINES_TO_USE * 90 / 100))
VALIDATE_LINES=$((TOTAL_LINES_TO_USE - TRAIN_LINES))

# Split the file
head -n "$TOTAL_LINES_TO_USE" "$INPUT_FILE" | head -n "$TRAIN_LINES" > "$TRAIN_FILE"
head -n "$TOTAL_LINES_TO_USE" "$INPUT_FILE" | tail -n "$VALIDATE_LINES" > "$VALIDATE_FILE"

# Print summary
echo "File split completed:"
echo "Training dataset: $TRAIN_FILE ($TRAIN_LINES lines)"
echo "Validation dataset: $VALIDATE_FILE ($VALIDATE_LINES lines)"
