#!/usr/bin/env bash
#
# prepare_datasets.sh
# Downloads & sets up:
#  - COCO 2017 (train/val images + captions annotations)
#  - SGP-Gen-70k (svg-gen-70k.jsonl)
#  - SGP-Object (SGP-Object.json)
#
# Usage:
#   bash prepare_datasets.sh [-c /path/to/coco] [-s /path/to/svg] [--env-file datasets.env]
#
# You can also pre-set env vars:
#   export COCO_DIR=/data/coco2017
#   export SVG_DIR=/data/svg
#
# After running, you can `source datasets.env` to export COCO_DIR/SVG_DIR in your shell.

set -euo pipefail

# -----------------------------
# Defaults & CLI parsing
# -----------------------------
export COCO_DIR="/your/coco/dir"
export SVG_DIR="/your/svg/dir"
COCO_DIR_DEFAULT="${COCO_DIR:-$PWD}"
SVG_DIR_DEFAULT="${SVG_DIR:-$PWD}"
ENV_FILE="datasets.env"

COCO_DIR="$COCO_DIR_DEFAULT"
SVG_DIR="$SVG_DIR_DEFAULT"

# Allow overriding COCO base URL (useful if you have a mirror)
COCO_BASE_URL="${COCO_BASE_URL:-http://images.cocodataset.org}"
# Parallel downloader preference; will auto-detect below
DOWNLOADER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--coco-dir)
      COCO_DIR="$2"; shift 2;;
    -s|--svg-dir)
      SVG_DIR="$2"; shift 2;;
    --env-file)
      ENV_FILE="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 [-c COCO_DIR] [-s SVG_DIR] [--env-file FILE]"
      echo "Optional: COCO_BASE_URL env var to use a mirror."
      exit 0;;
    *)
      echo "Unknown arg: $1"; exit 1;;
  esac
done

mkdir -p "$COCO_DIR" "$SVG_DIR"

# -----------------------------
# Helpers
# -----------------------------
need_cmd() { command -v "$1" >/dev/null 2>&1; }

pick_downloader() {
#   if need_cmd aria2c; then
#     DOWNLOADER="aria2c"
  if need_cmd wget; then
    DOWNLOADER="wget"
  elif need_cmd curl; then
    DOWNLOADER="curl"
  else
    echo "Error: need one of {aria2c,wget,curl} installed." >&2
    exit 1
  fi
}

fetch() {
  # fetch URL to OUT file (resumable)
  local url="$1"
  local out="$2"
  local tmp="${out}.part"

  case "$DOWNLOADER" in
    aria2c)
      # fast, parallel, resumable
      aria2c -x16 -s16 -k1M -c -o "$(basename "$out")" -d "$(dirname "$out")" "$url"
      ;;
    wget)
      wget -c --retry-connrefused --tries=20 --timeout=30 -O "$tmp" "$url"
      mv -f "$tmp" "$out"
      ;;
    curl)
      curl -L --retry 20 --retry-delay 2 --fail --retry-connrefused -C - -o "$tmp" "$url"
      mv -f "$tmp" "$out"
      ;;
  esac
}

unzip_if_needed() {
  local zip="$1"
  local expect_dir="$2"   # directory that should exist after unzip
  local target_root="$3"  # where to unzip into

  if [[ -d "$expect_dir" ]]; then
    echo "✓ Exists, skipping unzip: $expect_dir"
    return 0
  fi

  echo "→ Unzipping $(basename "$zip") ..."
  if need_cmd unzip; then
    unzip -q "$zip" -d "$target_root"
  elif need_cmd bsdtar; then
    bsdtar -xf "$zip" -C "$target_root"
  else
    echo "Error: need 'unzip' or 'bsdtar' to extract $zip" >&2
    exit 1
  fi
  echo "✓ Unzipped to $expect_dir"
}

# -----------------------------
# Start
# -----------------------------
echo "COCO_DIR: $COCO_DIR"
echo "SVG_DIR : $SVG_DIR"
pick_downloader
echo "Downloader: $DOWNLOADER"

# -----------------------------
# 1) Setup COCO 2017
# -----------------------------
echo ""
echo "=== [1/3] Downloading COCO 2017 ==="
pushd "$COCO_DIR" >/dev/null

TRAIN_ZIP="$COCO_DIR/train2017.zip"
VAL_ZIP="$COCO_DIR/val2017.zip"
ANN_ZIP="$COCO_DIR/annotations_trainval2017.zip"

TRAIN_URL="$COCO_BASE_URL/zips/train2017.zip"
VAL_URL="$COCO_BASE_URL/zips/val2017.zip"
ANN_URL="$COCO_BASE_URL/annotations/annotations_trainval2017.zip"

if [[ ! -f "$TRAIN_ZIP" ]]; then
  echo "→ Fetching train2017.zip ..."
  fetch "$TRAIN_URL" "$TRAIN_ZIP"
else
  echo "✓ Found $TRAIN_ZIP (skipping download)"
fi

if [[ ! -f "$VAL_ZIP" ]]; then
  echo "→ Fetching val2017.zip ..."
  fetch "$VAL_URL" "$VAL_ZIP"
else
  echo "✓ Found $VAL_ZIP (skipping download)"
fi

if [[ ! -f "$ANN_ZIP" ]]; then
  echo "→ Fetching annotations_trainval2017.zip ..."
  fetch "$ANN_URL" "$ANN_ZIP"
else
  echo "✓ Found $ANN_ZIP (skipping download)"
fi

# unzip_if_needed "$TRAIN_ZIP" "$COCO_DIR/train2017" "$COCO_DIR"
unzip_if_needed "$VAL_ZIP"   "$COCO_DIR/val2017"   "$COCO_DIR"
unzip_if_needed "$ANN_ZIP"   "$COCO_DIR/annotations" "$COCO_DIR"

quick sanity check for captions files
for f in "annotations/captions_train2017.json" "annotations/captions_val2017.json"; do
  if [[ ! -f "$COCO_DIR/$f" ]]; then
    echo "Warning: expected $COCO_DIR/$f to exist after extraction."
  fi
done

popd >/dev/null
echo "✓ COCO 2017 ready at $COCO_DIR"

# -----------------------------
# 2) Setup SVG training data (SGP-Gen-70k)
# -----------------------------
echo ""
echo "=== [2/3] Downloading MMSVG-Illustration-40k (MMSVG-Illustration-40k.jsonl) ==="
SGP_GEN_URL="https://docs.google.com/uc?export=download&id=1eStLujbYZ3bMm8K5m6IIsQZt_kFvLf-s"
SGP_GEN_OUT="$SVG_DIR/MMSVG-Illustration-40k.jsonl"

if [[ ! -f "$SGP_GEN_OUT" ]]; then
  echo "→ Fetching MMSVG-Illustration-40k.jsonl ..."
  fetch "$SGP_GEN_URL" "$SGP_GEN_OUT"
else
  echo "✓ Found $SGP_GEN_OUT (skipping download)"
fi
echo "✓ MMSVG-Illustration-40k ready at $SGP_GEN_OUT"

# -----------------------------
# 3) Prepare SGP-Single-9k eval (SGP-Object)
# -----------------------------
echo ""
echo "=== [3/3] Downloading SGP-Object (SGP-Object.json) ==="
SGP_OBJ_URL="https://docs.google.com/uc?export=download&id=1Pe3gkG6OD_m20grAV5b8cqLWdYPYuf4b"
SGP_OBJ_OUT="$SVG_DIR/SGP-Object.json"

if [[ ! -f "$SGP_OBJ_OUT" ]]; then
  echo "→ Fetching SGP-Object.json ..."
  fetch "$SGP_OBJ_URL" "$SGP_OBJ_OUT"
else
  echo "✓ Found $SGP_OBJ_OUT (skipping download)"
fi
echo "✓ SGP-Object ready at $SGP_OBJ_OUT"

# -----------------------------
# 4) Write env file
# -----------------------------
cat > "$ENV_FILE" <<EOF
# Generated by prepare_datasets.sh
export COCO_DIR="$COCO_DIR"
export SVG_DIR="$SVG_DIR"
EOF

echo ""
echo "==============================================="
echo "All done!"
echo "COCO folders:  $COCO_DIR/{train2017,val2017,annotations}"
echo "SVG train:     $SGP_GEN_OUT"
echo "SGP-Object:    $SGP_OBJ_OUT"
echo ""
echo "To export env vars in your shell, run:"
echo "  source $ENV_FILE"
echo "==============================================="
