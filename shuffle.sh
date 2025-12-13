#!/usr/bin/env bash
set -euo pipefail

INPUT="$1"
CHUNK_LINES=10000000
TMP_DIR=./tmp_shuffle
OUT="${INPUT%.fen}.shuffled.fen"

if [ ! -f "$INPUT" ]; then
  echo "File not found: $INPUT"
  exit 1
fi

mkdir -p "$TMP_DIR"

echo "[1/4] Splitting $INPUT..."
split -l "$CHUNK_LINES" "$INPUT" "$TMP_DIR/chunk_"

echo "[2/4] Shuffling chunks..."
for f in "$TMP_DIR"/chunk_*; do
  shuf "$f" > "$f.shuf"
  rm "$f"
done

echo "[3/4] Shuffling chunk order..."
ls "$TMP_DIR"/chunk_*.shuf | shuf > "$TMP_DIR/chunks.list"

echo "[4/4] Merging into $OUT..."
> "$OUT"
while read -r f; do
  cat "$f" >> "$OUT"
done < "$TMP_DIR/chunks.list"

echo "Done: $OUT"
