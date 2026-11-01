#!/usr/bin/env bash
# One-command demo: Adesanya vs Pereira 2 (UFC 287), 8:40–10:10 of the free fight upload.
#
#   scripts/run_demo.sh                 # download slice + full pipeline
#   scripts/run_demo.sh --max-frames 300   # quick 10-second test first
#
# Red = fighter on the LEFT in the first second of the slice (Adesanya in this one).
# Override with env vars:  RED=X BLUE=Y scripts/run_demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

URL="https://www.youtube.com/watch?v=vx3vs0p6TEs"
NAME="adesanya_pereira_2"
START="00:08:40"; END="00:10:10"
RED="${RED:-ADESANYA}"; BLUE="${BLUE:-PEREIRA}"

if [[ ! -f "data/clips/${NAME}.mp4" ]]; then
  scripts/download_clip.sh "$URL" "$NAME" "$START" "$END"
fi

python -m mma_vision.pipeline run "data/clips/${NAME}.mp4" --red "$RED" --blue "$BLUE" "$@"
echo
echo "open outputs/${NAME}/annotated.mp4 outputs/${NAME}/summary.png"
