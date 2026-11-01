#!/usr/bin/env bash
# Download a fight clip (or a time slice of one) with yt-dlp.
#   scripts/download_clip.sh "<youtube url>" fight1            # whole video
#   scripts/download_clip.sh "<youtube url>" fight1 00:12:30 00:13:30   # one-minute slice
#
# Best sources: UFC's own "Free Fight" uploads on YouTube, regional promotions
# (LFA, Cage Warriors, PFL). Pick a clip with a wide, mostly static camera —
# every hard cut resets the tracker, and close-ups lose the second fighter.
set -euo pipefail
URL="${1:?youtube url}"; NAME="${2:-clip}"; START="${3:-}"; END="${4:-}"
mkdir -p data/clips
command -v yt-dlp >/dev/null || pip install yt-dlp
ARGS=(-f "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]" --merge-output-format mp4 -o "data/clips/${NAME}.%(ext)s")
if [[ -n "$START" && -n "$END" ]]; then
  ARGS+=(--download-sections "*${START}-${END}" --force-keyframes-at-cuts)
fi
yt-dlp "${ARGS[@]}" "$URL"
echo "-> data/clips/${NAME}.mp4"
