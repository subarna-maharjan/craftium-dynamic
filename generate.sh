#!/usr/bin/env bash
# Generate a dynamic-agent dataset (default: ONE level, 600 frames).
#
#   ./generate.sh                               # datasets/dynamic/ , 1 level
#   NAME=dyn2 N=5 SEED=7 ./generate.sh          # 5 levels, different terrain
#   POOL=unseen NAME=zeroshot ./generate.sh     # mobs from the unseen 30% pool
#
# Output per level: datasets/<NAME>/raw/<ENV>/<level_seed>/
#   rgb.mp4, data.npz (player+camera+voxels), data_dynamic.npz (mob ground truth),
#   level_metadata.json
set -e
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
PY=.venv/bin/python

FRAMES="${FRAMES:-600}"
ENV="${ENV_ID:-OpenWorldCreative-v0}"
NAME="${NAME:-dynamic}"
N="${N:-1}"
SEED="${SEED:-1}"
POOL="${POOL:-seen}"   # seen | unseen

echo "==> $NAME: $N level(s), $FRAMES frames, seed=$SEED, pool=$POOL"

# Deterministic 70/30 split of the mob pool (same split as PERSIST).
$PY tools/agent_splits.py --seen_ratio 0.7 --seed 0 --out_dir datasets/agent_split
ENTS=$(cat "datasets/agent_split/$POOL.txt")

# Phase 1: init (level seeds + dataset params).
$PY dataset_toolkits/generate_raw_data.py \
  --dataset_dir datasets --dataset_name "$NAME" --env_id "$ENV" \
  --ep_timesteps "$FRAMES" --seed "$SEED" --init --overwrite_init --num_levels "$N" \
  --dynamic_agent_entities "$ENTS"

# Phase 2: render + record (re-renders existing levels so nothing is stale).
$PY dataset_toolkits/generate_raw_data.py \
  --dataset_dir datasets --dataset_name "$NAME" --env_id "$ENV" \
  --disable_commit_check --ep_timesteps "$FRAMES" --overwrite_leveldata \
  --dynamic_agent_entities "$ENTS"

echo "==> Mob-in-frame coverage:"
$PY tools/check_seen.py "datasets/$NAME/raw/$ENV/*" || true
echo "==> Done: datasets/$NAME/raw/$ENV/"
