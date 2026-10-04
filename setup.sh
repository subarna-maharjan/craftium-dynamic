#!/usr/bin/env bash
# One-time setup: clone the Craftium fork (engine + Lua mods + VoxeLibre) and
# build this folder's own venv. The engine is compiled from source during
# `uv sync`, so this takes a while (~20-40 min on 4 cores).
#
#   cd ~/craftium-dynamic && ./setup.sh
#
# Re-running is safe: an existing gym_envs/craftium checkout is kept.
set -e
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"

FORK=https://github.com/subarnatamu2026-hub/craftium
BRANCH=claude/craftium-dynamic-agents-cauneh
COMMIT="${CRAFTIUM_COMMIT:-418355b61}"   # commit the PERSIST datasets were built with

if [ ! -d gym_envs/craftium/.git ]; then
  echo "==> Cloning $FORK ($BRANCH)"
  git clone --recursive -b "$BRANCH" "$FORK" gym_envs/craftium
  ( cd gym_envs/craftium && git checkout "$COMMIT" && git submodule update --init --recursive )
fi
echo "==> craftium at $(cd gym_envs/craftium && git log --oneline -1)"

echo "==> uv sync (compiles the Luanti engine)"
uv sync --group cu --group env

echo "==> Done. Try: ./generate.sh   then   ./build_glb.sh"
