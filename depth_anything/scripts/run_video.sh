#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_video.sh <input_video> [export_format]
#   <input_video>   : path to input video (relative or absolute)
#   [export_format] : da3 export format (default: glb)
#                     e.g. glb, glb-feat_vis, ply, ...

INPUT="${1:?usage: run_video.sh <input_video> [export_format]}"
FORMAT="${2:-glb}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="$(basename "${INPUT%.*}")"
OUT_DIR="${ROOT}/outputs/${NAME}"
mkdir -p "${OUT_DIR}"

da3 video "${INPUT}" \
    --fps 15 \
    --use-backend \
    --export-dir "${OUT_DIR}" \
    --export-format "${FORMAT}" \
    --process-res-method lower_bound_resize \
    --model-dir depth-anything/DA3-LARGE-1.1
