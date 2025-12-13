#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <file>"
  exit 1
fi

FILE="$1"

if [ ! -f "$FILE" ]; then
  echo "File not found: $FILE"
  exit 1
fi

BASENAME=$(basename "$FILE")
TRAIN_FILE="train_$BASENAME"
VALIDATE_FILE="validate_$BASENAME"

echo "Counting lines (this will take time)..."
TOTAL_LINES=$(LC_ALL=C wc -l < "$FILE")

TRAIN_LINES=$((TOTAL_LINES * 90 / 100))
VALIDATE_LINES=$((TOTAL_LINES - TRAIN_LINES))

FILE_SIZE=$(stat -f%z "$FILE")

echo "Total lines:      $TOTAL_LINES"
echo "Training lines:   $TRAIN_LINES"
echo "Validation lines: $VALIDATE_LINES"
echo

echo "Writing train file (90%)..."
pv -s "$FILE_SIZE" "$FILE" \
  | head -n "$TRAIN_LINES" \
  > "$TRAIN_FILE"

echo
echo "Writing validation file (10%)..."
pv -s "$FILE_SIZE" "$FILE" \
  | tail -n "$VALIDATE_LINES" \
  > "$VALIDATE_FILE"

echo
echo "Done."
echo "Train:    $TRAIN_FILE"
echo "Validate: $VALIDATE_FILE"
