# craftium-dynamic

A standalone version of the dynamic-agent dataset pipeline from `~/PERSIST`. It holds only what's needed to:

1. generate Craftium / VoxeLibre levels with roaming mobs (RGB video, player and camera data, voxels, and mob ground truth), and
2. export a level as an animated `.glb` (voxel terrain with rigged, skinned mobs).

## Setup (one time)

```bash
cd ~/craftium-dynamic
./setup.sh
```

The script clones the Craftium fork (`claude/craftium-dynamic-agents-cauneh` at `418355b61`) into `gym_envs/craftium` and builds `.venv` with `uv sync`. The sync compiles the Luanti engine from source.

You also need the system packages that the PERSIST build already uses: `cmake`, a C/C++ toolchain, `Xvfb`, and `ffmpeg`.

## Generate a dataset

```bash
./generate.sh                            # 1 level, 600 frames -> datasets/dynamic/
NAME=dyn2 N=5 SEED=7 ./generate.sh       # more levels / other terrain
POOL=unseen NAME=zeroshot ./generate.sh  # zero-shot mob pool
```

Each level is written to `datasets/<NAME>/raw/OpenWorldCreative-v0/<level_seed>/`:

| file | contents |
|---|---|
| `rgb.mp4` | first-person video |
| `data.npz` | player pose, camera, actions, voxel observations |
| `data_dynamic.npz` | per-frame mob ground truth (position, yaw, animation, bones) |
| `level_metadata.json` | level settings |

When a run finishes, `tools/check_seen.py` reports which mobs were actually on camera for at least 10 frames. It accounts for terrain blocking the view.

## Build the GLB

```bash
./build_glb.sh                                  # first level -> .../<level>/clip.glb
NAME=dyn2 LEVEL=<seed> FRAMES=50 ./build_glb.sh
EXTRA="--highlight_player --show_los" ./build_glb.sh
```

Textures and `.b3d` models are read from `gym_envs/craftium/craftium-envs/common_games/VoxeLibre`.

## Layout

```
dataset_toolkits/generate_raw_data.py   main generator (tyro CLI; see Args for all knobs)
dataset_toolkits/dynamic_data.py        mob log -> data_dynamic.npz alignment
dataset_toolkits/guided_nav.py          observation tour / 360 spin navigator
dataset_toolkits/follow_nav.py          follow-one-mob navigator
tools/agent_splits.py                   deterministic 70/30 seen/unseen mob split
tools/check_seen.py                     mob on-camera coverage check
tools/make_clip_glb.py                  animated GLB exporter
tools/make_frame_glb.py, tools/b3d_rig.py   mesh/texture/rig helpers for the exporter
utils/                                  seeding, hashing, action wrapper
redeploy_mods.sh                        copy edited Lua mods into .venv without recompiling
```
