#!/usr/bin/env bash
# Fast redeploy of ONLY the craftium_env Lua mods into this folder's venv - no
# recompile. Use after a .lua change in gym_envs/craftium/craftium-envs/.
# Python-only changes need nothing (they run from this folder).
set -e
cd "$(dirname "$0")"

SRC=gym_envs/craftium/craftium-envs
[ -d "$SRC" ] || { echo "ERROR: $SRC not found. Run ./setup.sh first." >&2; exit 1; }

count=0
while IFS= read -r dst; do
  env=$(echo "$dst" | grep -o 'openworld-[a-z]*')
  srcf="$SRC/$env/mods/craftium_env/$(basename "$dst")"
  if [ -f "$srcf" ]; then cp "$srcf" "$dst"; count=$((count + 1)); fi
done < <(find .venv -path '*openworld-*/mods/craftium_env/*.lua')
echo "Redeployed $count mod file(s) into .venv."
