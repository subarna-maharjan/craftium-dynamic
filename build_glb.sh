#!/usr/bin/env bash
# Export an animated GLB (voxel terrain + rigged, skinned mobs + player) for a
# generated level.
#
#   ./build_glb.sh                              # first level of datasets/dynamic
#   NAME=dyn2 LEVEL=774252441 ./build_glb.sh    # a specific level
#   FRAMES=50 ./build_glb.sh                    # only the first 50 frames
#   EXTRA="--highlight_player --show_los" ./build_glb.sh
#
# Default output: datasets/<NAME>/raw/<ENV>/<LEVEL>/clip.glb
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python

ENV="${ENV_ID:-OpenWorldCreative-v0}"
NAME="${NAME:-dynamic}"
ROOT="datasets/$NAME/raw/$ENV"
LEVEL="${LEVEL:-$(ls "$ROOT" | head -1)}"
OUT="${OUT:-$ROOT/$LEVEL/clip.glb}"

ARGS=(--dataset_dir datasets --dataset_name "$NAME" --env_id "$ENV" --seed "$LEVEL" --out "$OUT")
[ -n "$FRAMES" ] && ARGS+=(--frames "$FRAMES")

echo "==> GLB for $NAME/$LEVEL -> $OUT"
# shellcheck disable=SC2086
$PY tools/make_clip_glb.py "${ARGS[@]}" $EXTRA
ls -lh "$OUT"
