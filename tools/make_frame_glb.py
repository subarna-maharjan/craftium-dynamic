#!/usr/bin/env python3
"""Build a single GLB of one dataset frame: voxel terrain + sheep + player.

The terrain is meshed with real VoxeLibre block textures and the sheep/player are
the game's own `.b3d` models, posed with the recorded position and yaw. Entities
are *not* part of `obs_voxel_mt` (that grid holds blocks only), so they come from
`data_dynamic.npz`.

All dataset positions are ENU (East, North, Up). glTF is Y-up / right-handed, so
vertices are remapped ENU -> (X=E, Y=U, Z=-N) after subtracting the voxel-grid
center for numerical stability.

Example:
    python tools/make_frame_glb.py --dataset_name sheep_test3 --frame 0 --out frame0.glb
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import trimesh
from trimesh.visual import ColorVisuals, TextureVisuals
from trimesh.visual.material import PBRMaterial

try:
    from PIL import Image
except ImportError:  # textures need Pillow; fall back to flat colors.
    Image = None

# Node ids treated as empty (air / ignore) in the Minetest voxel observation.
EMPTY_NODE_IDS = {126, 127}
# Node ids above this are considered corrupt (same cutoff as process_mt_data).
MAX_VALID_NODE_ID = 8192

DEFAULT_GAME_DIR = Path("gym_envs/craftium/craftium-envs/common_games/VoxeLibre")

# Minetest models are authored at 10 units per node.
B3D_UNITS_PER_NODE = 10.0

# Box faces for 8 corners numbered c with x=c&1, y=(c>>1)&1, z=(c>>2)&1.
# Wound counter-clockwise seen from outside so normals point outward; the
# opposite winding makes every box render inside-out (it looks hollow).
BOX_FACES = np.array(
    [
        [0, 3, 1],
        [0, 2, 3],  # z = 0
        [4, 7, 6],
        [4, 5, 7],  # z = 1
        [0, 5, 4],
        [0, 1, 5],  # y = 0
        [2, 7, 3],
        [2, 6, 7],  # y = 1
        [0, 6, 2],
        [0, 4, 6],  # x = 0
        [1, 7, 5],
        [1, 3, 7],  # x = 1
    ],
    dtype=np.int64,
)

# `dyn_obb_corners` numbers corners in Minetest axes (bit1 -> up, bit2 -> north)
# while the values are already ENU, so two axes are swapped relative to
# BOX_FACES. That mirroring flips the winding, hence the reversed triangles.
OBB_FACES = np.ascontiguousarray(BOX_FACES[:, ::-1])

# One entry per cube face: (name, axis, sign, 4 corner offsets CCW from outside,
# matching UV corners in tile space with v=0 at the top of the image).
_H = 0.5
FACE_DEFS = (
    (
        "top",
        2,
        +1,
        np.array([[-_H, -_H, _H], [_H, -_H, _H], [_H, _H, _H], [-_H, _H, _H]]),
        np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]),
    ),
    (
        "bottom",
        2,
        -1,
        np.array([[-_H, -_H, -_H], [-_H, _H, -_H], [_H, _H, -_H], [_H, -_H, -_H]]),
        np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
    ),
    (
        "side",
        0,
        +1,
        np.array([[_H, -_H, -_H], [_H, _H, -_H], [_H, _H, _H], [_H, -_H, _H]]),
        np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]),
    ),
    (
        "side",
        0,
        -1,
        np.array([[-_H, -_H, -_H], [-_H, -_H, _H], [-_H, _H, _H], [-_H, _H, -_H]]),
        np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]),
    ),
    (
        "side",
        1,
        +1,
        np.array([[-_H, _H, -_H], [-_H, _H, _H], [_H, _H, _H], [_H, _H, -_H]]),
        np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]),
    ),
    (
        "side",
        1,
        -1,
        np.array([[-_H, -_H, -_H], [_H, -_H, -_H], [_H, -_H, _H], [-_H, -_H, _H]]),
        np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]),
    ),
)

QUAD_TRIS = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

# Biome tint that VoxeLibre applies to grass/leaves (mcl_core_palette_grass.png).
GRASS_TINT = "#6DC173"

# Material -> per-face Minetest texture spec, using the real game textures.
MATERIALS = {
    "grass": {
        "top": f"mcl_core_grass_block_top.png^[multiply:{GRASS_TINT}",
        "side": f"default_dirt.png^(mcl_core_grass_block_side_overlay.png^[multiply:{GRASS_TINT})",
        "bottom": "default_dirt.png",
    },
    "dirt": {"all": "default_dirt.png"},
    "stone": {"all": "default_stone.png"},
    "andesite": {"all": "mcl_core_andesite.png"},
    "granite": {"all": "mcl_core_granite.png"},
    "diorite": {"all": "mcl_core_diorite.png"},
    "gravel": {"all": "default_gravel.png"},
    "sand": {"all": "default_sand.png"},
    "coal_ore": {"all": "mcl_core_coal_ore.png"},
    "bedrock": {"all": "mcl_core_bedrock.png"},
    "leaves": {"all": f"mcl_core_leaves_big_oak.png^[multiply:{GRASS_TINT}"},
    "log": {
        "top": "default_tree_top.png",
        "bottom": "default_tree_top.png",
        "side": "default_tree.png",
    },
    "water": {"all": "mcl_core_water_source_animation.png^[multiply:#3D6E9A"},
}

# Fallback flat colors (also used when Pillow is missing).
FALLBACK_RGB = {
    "grass": (109, 193, 115),
    "dirt": (108, 83, 70),
    "stone": (130, 122, 118),
    "andesite": (104, 110, 107),
    "granite": (153, 121, 110),
    "diorite": (153, 149, 146),
    "gravel": (112, 105, 99),
    "sand": (220, 170, 127),
    "coal_ore": (109, 101, 99),
    "bedrock": (91, 77, 66),
    "leaves": (43, 89, 66),
    "log": (95, 75, 57),
    "water": (61, 110, 154),
}

WATER_ALPHA = 165
# Cycled through for underground ids we cannot identify, so distinct block types
# stay visually distinct without looking like a rainbow.
UNDERGROUND_CYCLE = ("dirt", "andesite", "granite", "diorite", "gravel", "coal_ore")

# Wool colorize values straight out of VoxeLibre's sheep.lua.
SHEEP_WOOL_COLORIZE = {
    "unicolor_white": "#FFFFFF00",
    "unicolor_dark_orange": "#502A00D0",
    "unicolor_grey": "#5B5B5BD0",
    "unicolor_darkgrey": "#303030D0",
    "unicolor_blue": "#0000CCD0",
    "unicolor_dark_green": "#005000D0",
    "unicolor_green": "#50CC00D0",
    "unicolor_violet": "#5000CCD0",
    "unicolor_light_red": "#FF5050D0",
    "unicolor_yellow": "#CCCC00D0",
    "unicolor_orange": "#CC5000D0",
    "unicolor_red": "#CC0000D0",
    "unicolor_cyan": "#00CCCCD0",
    "unicolor_red_violet": "#CC0050D0",
    "unicolor_black": "#000000D0",
    "unicolor_light_blue": "#5050FFD0",
}

PLAYER_MODEL = "mcl_armor_character.b3d"
# Fallback heights, used only when the capture has no collision box.
SHEEP_HEIGHT = 1.3
PLAYER_HEIGHT = 1.8

# species -> (mesh, texture list in surface order, visual_size).
# Taken from the mobs_mc definitions; only the horse sets a visual_size, and
# textures use each mob's default variant (first entry of its texture table).
MOB_MODELS: dict[str, tuple[str, list[str], float]] = {
    "mobs_mc:sheep": ("mobs_mc_sheepfur.b3d", ["mobs_mc_sheep_fur.png", "mobs_mc_sheep.png"], 1.0),
    "mobs_mc:pig": ("mobs_mc_pig.b3d", ["mobs_mc_pig.png", "blank.png"], 1.0),
    "mobs_mc:cow": ("mobs_mc_cow.b3d", ["mobs_mc_cow.png", "blank.png"], 1.0),
    "mobs_mc:mooshroom": (
        "mobs_mc_cow.b3d",
        ["mobs_mc_mooshroom.png", "mobs_mc_mushroom_red.png"],
        1.0,
    ),
    "mobs_mc:chicken": ("mobs_mc_chicken.b3d", ["mobs_mc_chicken.png"], 1.0),
    "mobs_mc:rabbit": ("mobs_mc_rabbit.b3d", ["mobs_mc_rabbit_brown.png"], 1.0),
    "mobs_mc:llama": (
        "mobs_mc_llama.b3d",
        ["blank.png", "blank.png", "mobs_mc_llama_brown.png"],
        1.0,
    ),
    "mobs_mc:horse": (
        "mobs_mc_horse.b3d",
        ["blank.png", "mobs_mc_horse_brown.png", "blank.png"],
        3.0,
    ),
    "mobs_mc:mule": (
        "mobs_mc_horse.b3d",
        ["blank.png", "mobs_mc_mule.png", "blank.png"],
        2.82,
    ),
    "mobs_mc:donkey": (
        "mobs_mc_horse.b3d",
        ["blank.png", "mobs_mc_donkey.png", "blank.png"],
        2.58,
    ),
    "mobs_mc:skeleton_horse": (
        "mobs_mc_horse.b3d",
        ["blank.png", "mobs_mc_horse_skeleton.png", "blank.png"],
        3.0,
    ),
    "mobs_mc:cat": ("mobs_mc_cat.b3d", ["mobs_mc_cat_black.png"], 1.0),
    "mobs_mc:vindicator": (
        "mobs_mc_vindicator.b3d",
        ["mobs_mc_vindicator.png", "blank.png", "default_tool_steelaxe.png"],
        2.75,
    ),
    "mobs_mc:cave_spider": (
        "mobs_mc_spider.b3d",
        ["mobs_mc_cave_spider.png^(mobs_mc_spider_eyes.png^[makealpha:0,0,0)"],
        0.5,
    ),
    "mobs_mc:zombie": (
        "mobs_mc_zombie.b3d",
        ["mobs_mc_empty.png", "mobs_mc_zombie.png"],
        1.0,
    ),
    "mobs_mc:baby_zombie": (
        "mobs_mc_zombie.b3d",
        ["mobs_mc_empty.png", "mobs_mc_zombie.png"],
        0.5,
    ),
    "mobs_mc:husk": (
        "mobs_mc_zombie.b3d",
        ["mobs_mc_empty.png", "mobs_mc_husk.png"],
        1.0,
    ),
    "mobs_mc:baby_husk": (
        "mobs_mc_zombie.b3d",
        ["mobs_mc_empty.png", "mobs_mc_husk.png"],
        0.5,
    ),
    "mobs_mc:skeleton": (
        "mobs_mc_skeleton.b3d",
        ["mobs_mc_empty.png", "mobs_mc_skeleton.png", "mcl_bows_bow_0.png"],
        1.0,
    ),
    "mobs_mc:ocelot": ("mobs_mc_cat.b3d", ["mobs_mc_cat_ocelot.png"], 1.0),
    "mobs_mc:wolf": ("mobs_mc_wolf.b3d", ["mobs_mc_wolf.png"], 1.0),
    "mobs_mc:polar_bear": ("mobs_mc_polarbear.b3d", ["mobs_mc_polarbear.png"], 3.0),
    "mobs_mc:iron_golem": ("mobs_mc_iron_golem.b3d", ["mobs_mc_iron_golem.png"], 3.0),
    "mobs_mc:snowman": (
        "mobs_mc_snowman.b3d",
        [
            "mobs_mc_snowman.png",
            "farming_pumpkin_side.png",
            "farming_pumpkin_top.png",
            "farming_pumpkin_face.png",
            "farming_pumpkin_side.png",
            "farming_pumpkin_side.png",
            "farming_pumpkin_top.png",
        ],
        3.0,
    ),
    "mobs_mc:pillager": (
        "mobs_mc_pillager.b3d",
        ["mobs_mc_pillager.png", "mcl_bows_crossbow_3.png"],
        2.75,
    ),
    "mobs_mc:baby_pigman": (
        "mobs_mc_zombie_pigman.b3d",
        ["mobs_mc_zombie_pigman.png", "default_tool_goldsword.png", "mobs_mc_zombie_pigman.png"],
        1.5,
    ),
    "mobs_mc:piglin_brute": (
        "extra_mobs_sword_piglin.b3d",
        ["extra_mobs_piglin_brute.png", "default_tool_goldaxe.png"],
        1.0,
    ),
    "mobs_mc:killer_bunny": (
        "mobs_mc_rabbit.b3d",
        ["mobs_mc_rabbit_caerbannog.png"],
        1.0,
    ),
    # A cube mesh scaled way up, which is why slimes need the 6.25 factor.
    "mobs_mc:slime_big": ("mobs_mc_slime.b3d", ["mobs_mc_slime.png"] * 2, 12.5),
    "mobs_mc:slime_small": ("mobs_mc_slime.b3d", ["mobs_mc_slime.png"] * 2, 6.25),
    "mobs_mc:slime_tiny": ("mobs_mc_slime.b3d", ["mobs_mc_slime.png"] * 2, 3.125),
    "mobs_mc:spider": ("mobs_mc_spider.b3d", ["mobs_mc_spider.png"], 1.0),
    "mobs_mc:villager": (
        "mobs_mc_villager.b3d",
        ["mobs_mc_villager.png", "mobs_mc_villager.png"],
        1.0,
    ),
    "mobs_mc:villager_zombie": (
        "mobs_mc_villager_zombie.b3d",
        ["mobs_mc_zombie_villager.png"],
        2.75,
    ),
    "mobs_mc:witherskeleton": (
        "mobs_mc_witherskeleton.b3d",
        ["mobs_mc_empty.png", "default_tool_stonesword.png", "mobs_mc_wither_skeleton.png"],
        1.2,
    ),
    "mobs_mc:zoglin": ("extra_mobs_hoglin.b3d", ["extra_mobs_zoglin.png"], 3.0),
    "mobs_mc:hoglin": ("extra_mobs_hoglin.b3d", ["extra_mobs_hoglin.png"], 3.0),
    "mobs_mc:baby_hoglin": ("extra_mobs_hoglin.b3d", ["extra_mobs_hoglin.png"], 0.75),
    "mobs_mc:zombified_piglin": (
        "mobs_mc_zombie_pigman.b3d",
        ["blank.png", "default_tool_goldsword.png", "mobs_mc_zombie_pigman.png"],
        3.0,
    ),
    "mobs_mc:baby_zombified_piglin": (
        "mobs_mc_zombie_pigman.b3d",
        ["mobs_mc_zombie_pigman.png", "default_tool_goldsword.png", "mobs_mc_zombie_pigman.png"],
        1.5,
    ),
    "mobs_mc:sword_piglin": (
        "extra_mobs_sword_piglin.b3d",
        ["extra_mobs_piglin.png", "default_tool_goldsword.png"],
        1.0,
    ),
    "mobs_mc:magma_cube_big": (
        "mobs_mc_magmacube.b3d",
        ["mobs_mc_magmacube.png", "mobs_mc_magmacube.png"],
        12.5,
    ),
    "mobs_mc:magma_cube_small": (
        "mobs_mc_magmacube.b3d",
        ["mobs_mc_magmacube.png", "mobs_mc_magmacube.png"],
        6.25,
    ),
    "mobs_mc:magma_cube_tiny": (
        "mobs_mc_magmacube.b3d",
        ["mobs_mc_magmacube.png", "mobs_mc_magmacube.png"],
        3.125,
    ),
}

# Texture names that stand for "draw nothing" (unused armor/saddle/chest slots).
EMPTY_TEXTURES = ("blank.png", "mobs_mc_empty.png", "extra_mobs_trans.png")

SHEEP_RGBA = np.array([220, 40, 40, 110], dtype=np.uint8)
PLAYER_RGBA = np.array([40, 200, 70, 110], dtype=np.uint8)
# Body-frame gizmo, REP-103 style: X forward, Y left, Z up.
AXIS_FORWARD_RGBA = np.array([230, 45, 45, 255], dtype=np.uint8)
AXIS_LEFT_RGBA = np.array([60, 210, 80, 255], dtype=np.uint8)
AXIS_UP_RGBA = np.array([70, 130, 255, 255], dtype=np.uint8)
# Magenta so the camera ray never reads as one of the blue up axes.
CAMERA_RGBA = np.array([255, 60, 220, 255], dtype=np.uint8)


# --------------------------------------------------------------------------- #
# coordinates
# --------------------------------------------------------------------------- #
def enu_to_gltf(points: np.ndarray, origin_enu: np.ndarray) -> np.ndarray:
    """Subtract origin then map ENU (E, N, U) -> glTF (X=E, Y=U, Z=-N)."""
    p = np.asarray(points, dtype=np.float64) - np.asarray(origin_enu, dtype=np.float64)
    out = np.empty_like(p, dtype=np.float64)
    out[..., 0] = p[..., 0]
    out[..., 1] = p[..., 2]
    out[..., 2] = -p[..., 1]
    return out


def to_gltf_uv(uv: np.ndarray) -> np.ndarray:
    """Pre-flip V because trimesh's glTF exporter writes `1 - v`.

    Without this the exported file samples every texture upside down (the
    player's head ends up wearing the shirt pixels).
    """
    out = np.array(uv, dtype=np.float64, copy=True)
    out[:, 1] = 1.0 - out[:, 1]
    return out


def mt_to_enu(points_mt: np.ndarray) -> np.ndarray:
    """Minetest (x=east, y=up, z=north) -> ENU (east, north, up)."""
    p = np.asarray(points_mt, dtype=np.float64)
    return np.stack([p[..., 0], p[..., 2], p[..., 1]], axis=-1)


def yaw_rotate_mt(points_mt: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate Minetest-axis points about the up (y) axis by `yaw` radians."""
    c, s = np.cos(yaw), np.sin(yaw)
    p = np.asarray(points_mt, dtype=np.float64)
    return np.stack(
        [p[..., 0] * c + p[..., 2] * (-s), p[..., 1], p[..., 0] * s + p[..., 2] * c],
        axis=-1,
    )


# --------------------------------------------------------------------------- #
# Blitz3D (.b3d) model loading
# --------------------------------------------------------------------------- #
class _Chunks:
    """Cursor over a BB3D chunk stream."""

    def __init__(self, data: bytes, pos: int, end: int):
        self.d = data
        self.p = pos
        self.end = end

    def i(self) -> int:
        v = struct.unpack_from("<i", self.d, self.p)[0]
        self.p += 4
        return v

    def f(self) -> float:
        v = struct.unpack_from("<f", self.d, self.p)[0]
        self.p += 4
        return v

    def floats(self, n: int) -> list[float]:
        v = list(struct.unpack_from(f"<{n}f", self.d, self.p))
        self.p += 4 * n
        return v

    def s(self) -> str:
        z = self.d.index(b"\0", self.p)
        v = self.d[self.p : z].decode("latin-1")
        self.p = z + 1
        return v

    def next(self) -> tuple[str, "_Chunks"]:
        tag = self.d[self.p : self.p + 4].decode("latin-1")
        length = struct.unpack_from("<i", self.d, self.p + 4)[0]
        body = _Chunks(self.d, self.p + 8, self.p + 8 + length)
        self.p += 8 + length
        return tag, body

    def more(self) -> bool:
        return self.p < self.end


def _quat_matrix(q: list[float]) -> np.ndarray:
    """BB3D stores quaternions w-first."""
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def load_b3d(path: Path) -> dict:
    """Parse a .b3d into surfaces (one per TRIS chunk) and node world positions.

    Returns {"surfaces": [{"vertices", "uv", "faces"}], "nodes": {name: pos}}.
    Only the rest/bind pose is read; bone animation is ignored, which is what a
    single captured frame needs.
    """
    data = path.read_bytes()
    if data[:4] != b"BB3D":
        raise ValueError(f"{path} is not a BB3D file")
    total = struct.unpack_from("<i", data, 4)[0]

    surfaces: list[dict] = []
    nodes: dict[str, np.ndarray] = {}

    def read_vrts(c: _Chunks) -> tuple[np.ndarray, np.ndarray]:
        flags, n_sets, set_size = c.i(), c.i(), c.i()
        per = 3 + (3 if flags & 1 else 0) + (4 if flags & 2 else 0) + n_sets * set_size
        n = (c.end - c.p) // (4 * per)
        raw = np.frombuffer(c.d, dtype="<f4", count=n * per, offset=c.p).reshape(n, per)
        c.p += 4 * n * per
        verts = raw[:, 0:3].astype(np.float64)
        off = 3 + (3 if flags & 1 else 0) + (4 if flags & 2 else 0)
        uv = (
            raw[:, off : off + 2].astype(np.float64)
            if n_sets > 0 and set_size >= 2
            else np.zeros((n, 2))
        )
        return verts, uv

    def read_node(c: _Chunks, parent: np.ndarray):
        name = c.s()
        pos = np.array(c.floats(3))
        scale = np.array(c.floats(3))
        quat = c.floats(4)
        local = np.eye(4)
        local[:3, :3] = _quat_matrix(quat) * scale[None, :]
        local[:3, 3] = pos
        world = parent @ local
        nodes.setdefault(name, world[:3, 3].copy())

        verts = uv = None
        while c.more():
            tag, body = c.next()
            if tag == "MESH":
                body.i()  # default brush id, unused (per-TRIS brush wins)
                while body.more():
                    mtag, mbody = body.next()
                    if mtag == "VRTS":
                        verts, uv = read_vrts(mbody)
                    elif mtag == "TRIS":
                        mbody.i()  # brush id
                        n = (mbody.end - mbody.p) // 12
                        faces = np.frombuffer(
                            mbody.d, dtype="<i4", count=n * 3, offset=mbody.p
                        ).reshape(n, 3)
                        if verts is not None:
                            # Every TRIS in a MESH indexes the same VRTS block, so
                            # keep only the vertices this surface actually uses.
                            # Otherwise a surface inherits its siblings' extent,
                            # which throws off bounding boxes and orient_outward.
                            used, remap = np.unique(faces, return_inverse=True)
                            xyz = (world[:3, :3] @ verts[used].T).T + world[:3, 3]
                            surfaces.append(
                                {
                                    "vertices": xyz,
                                    "uv": uv[used].copy(),
                                    "faces": remap.reshape(-1, 3).astype(np.int64),
                                }
                            )
            elif tag == "NODE":
                read_node(body, world)

    root = _Chunks(data, 12, 8 + total)
    while root.more():
        tag, body = root.next()
        if tag == "NODE":
            read_node(body, np.eye(4))

    if not surfaces:
        raise ValueError(f"no mesh surfaces found in {path}")
    return {"surfaces": surfaces, "nodes": nodes}


def orient_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Flip winding if most triangles face the mesh centroid (inside-out mesh)."""
    if len(faces) == 0:
        return faces
    v = vertices
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    centroids = (a + b + c) / 3.0
    outward = centroids - v.mean(axis=0)
    votes = np.einsum("ij,ij->i", normals, outward)
    if float(np.sum(votes)) < 0.0:
        return np.ascontiguousarray(faces[:, ::-1])
    return faces


# --------------------------------------------------------------------------- #
# Minetest texture specs
# --------------------------------------------------------------------------- #
class TextureLibrary:
    """Resolves Minetest texture strings against the game's texture files.

    Supports the subset used here: `a.png^b.png` overlays, parenthesised
    sub-expressions, `[colorize:#RRGGBBAA[:ratio]` and `[multiply:#RRGGBB`.
    """

    def __init__(self, game_dir: Path):
        self.files: dict[str, Path] = {}
        if game_dir.is_dir():
            for p in game_dir.rglob("*.png"):
                self.files.setdefault(p.name, p)
        self._cache: dict[str, "Image.Image"] = {}
        self.missing: set[str] = set()

    @staticmethod
    def _split_top(spec: str) -> list[str]:
        parts, depth, cur = [], 0, []
        for ch in spec:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == "^" and depth == 0:
                parts.append("".join(cur))
                cur = []
            else:
                cur.append(ch)
        parts.append("".join(cur))
        return [p for p in parts if p]

    @staticmethod
    def _parse_hex(text: str) -> tuple[int, int, int, int]:
        t = text.lstrip("#")
        if len(t) == 6:
            t += "FF"
        return tuple(int(t[i : i + 2], 16) for i in (0, 2, 4, 6))  # type: ignore[return-value]

    def _load_file(self, name: str) -> "Image.Image":
        path = self.files.get(name)
        if path is None:
            self.missing.add(name)
            return Image.new("RGBA", (16, 16), (255, 0, 255, 255))
        img = Image.open(path).convert("RGBA")
        # Animation strips are stored as a vertical run of frames; use frame 0.
        w, h = img.size
        if h > w and h % w == 0:
            img = img.crop((0, 0, w, w))
        return img

    def get(self, spec: str) -> "Image.Image":
        if Image is None:
            raise RuntimeError("Pillow is required for textures")
        if spec in self._cache:
            return self._cache[spec]

        out: "Image.Image" | None = None
        for token in self._split_top(spec):
            if token.startswith("(") and token.endswith(")"):
                layer = self.get(token[1:-1])
                out = layer.copy() if out is None else self._overlay(out, layer)
            elif token.startswith("[colorize:"):
                args = token[len("[colorize:") :].split(":")
                r, g, b, a = self._parse_hex(args[0])
                if len(args) > 1 and args[1] == "alpha":
                    # Take the color's RGB, keep the texture's alpha (scaled).
                    out = self._colorize_alpha(out, (r, g, b), a / 255.0)
                else:
                    ratio = float(args[1]) / 255.0 if len(args) > 1 else a / 255.0
                    out = self._colorize(out, (r, g, b), ratio)
            elif token.startswith("[multiply:"):
                r, g, b, _ = self._parse_hex(token[len("[multiply:") :])
                out = self._multiply(out, (r, g, b))
            elif token.startswith("["):
                pass  # unsupported modifier: leave the image untouched
            else:
                layer = self._load_file(token)
                out = layer.copy() if out is None else self._overlay(out, layer)

        if out is None:
            out = Image.new("RGBA", (16, 16), (255, 0, 255, 255))
        self._cache[spec] = out
        return out

    @staticmethod
    def _overlay(base: "Image.Image", layer: "Image.Image") -> "Image.Image":
        if layer.size != base.size:
            layer = layer.resize(base.size, Image.NEAREST)
        return Image.alpha_composite(base, layer)

    @staticmethod
    def _colorize(base, rgb, ratio):
        arr = np.asarray(base, dtype=np.float64)
        tint = np.array(rgb, dtype=np.float64)
        arr[..., :3] = arr[..., :3] * (1.0 - ratio) + tint[None, None, :] * ratio
        return Image.fromarray(arr.round().clip(0, 255).astype(np.uint8), "RGBA")

    @staticmethod
    def _colorize_alpha(base, rgb, color_alpha):
        arr = np.asarray(base, dtype=np.float64).copy()
        arr[..., 0], arr[..., 1], arr[..., 2] = rgb
        arr[..., 3] = arr[..., 3] * color_alpha
        return Image.fromarray(arr.round().clip(0, 255).astype(np.uint8), "RGBA")

    @staticmethod
    def _multiply(base, rgb):
        arr = np.asarray(base, dtype=np.float64)
        tint = np.array(rgb, dtype=np.float64) / 255.0
        arr[..., :3] = arr[..., :3] * tint[None, None, :]
        return Image.fromarray(arr.round().clip(0, 255).astype(np.uint8), "RGBA")

    def flatten(self, spec: str, tile: int = 16) -> np.ndarray:
        """RGB tile with alpha composited over the image's own mean color."""
        img = self.get(spec).resize((tile, tile), Image.NEAREST)
        arr = np.asarray(img, dtype=np.float64)
        rgb, alpha = arr[..., :3], arr[..., 3:4] / 255.0
        solid = alpha[..., 0] > 0.5
        base = rgb[solid].mean(axis=0) if solid.any() else np.array([128.0, 128.0, 128.0])
        return (rgb * alpha + base[None, None, :] * (1.0 - alpha)).round().clip(0, 255).astype(
            np.uint8
        )


# --------------------------------------------------------------------------- #
# node id -> material classification
# --------------------------------------------------------------------------- #
def occupied_mask(node_ids: np.ndarray) -> np.ndarray:
    return (
        ~np.isin(node_ids, list(EMPTY_NODE_IDS))
        & (node_ids >= 0)
        & (node_ids <= MAX_VALID_NODE_ID)
    )


def support_node_ids(
    positions_enu: np.ndarray,
    node_ids: np.ndarray,
    occupied: np.ndarray,
    voxel_center: np.ndarray,
    origin_idx: np.ndarray,
) -> set[int]:
    """Node ids that a clear majority of entities are standing on top of.

    Only counts entities whose feet voxel is air, so the block below really is
    carrying them. Players and mobs cannot stand on a liquid surface, which
    makes this a useful constraint on the material guess. A lone mob bobbing at
    a water surface satisfies the same test though, so an id has to carry
    several entities before we trust it as solid ground.
    """
    votes: dict[int, int] = {}
    total = 0
    dims = occupied.shape
    for pos in np.atleast_2d(np.asarray(positions_enu, dtype=np.float64)):
        i, j, k = voxel_index(pos, voxel_center, origin_idx)
        if not (0 <= i < dims[0] and 0 <= j < dims[1] and 1 <= k < dims[2]):
            continue
        if occupied[i, j, k] or not occupied[i, j, k - 1]:
            continue  # buried or airborne: tells us nothing about the surface
        votes[int(node_ids[i, j, k - 1])] = votes.get(int(node_ids[i, j, k - 1]), 0) + 1
        total += 1
    threshold = max(3, int(np.ceil(0.5 * total)))
    return {nid for nid, n in votes.items() if n >= threshold}


def classify_node_ids(
    node_ids: np.ndarray,
    occupied: np.ndarray,
    ground_ids: set[int] | None = None,
) -> dict[int, str]:
    """Guess a block material per node_id from grid topology.

    The dataset stores raw Minetest content ids with no id->name table, so this
    infers materials from where each id sits: what fraction of its voxels have
    air above (surface), how exposed it is, and how its height is distributed.
    `ground_ids` are ids that entities stand on, which rules out liquids.
    """
    if not np.any(occupied):
        return {}
    ground_ids = set() if ground_ids is None else {int(x) for x in ground_ids}

    air_above = np.ones_like(occupied, dtype=bool)
    air_above[:, :, :-1] = ~occupied[:, :, 1:]
    top = occupied & air_above

    pad = np.pad(occupied, 1, constant_values=False)
    n_air = (
        (~pad[2:, 1:-1, 1:-1]).astype(np.int16)
        + (~pad[:-2, 1:-1, 1:-1]).astype(np.int16)
        + (~pad[1:-1, 2:, 1:-1]).astype(np.int16)
        + (~pad[1:-1, :-2, 1:-1]).astype(np.int16)
        + (~pad[1:-1, 1:-1, 2:]).astype(np.int16)
        + (~pad[1:-1, 1:-1, :-2]).astype(np.int16)
    )
    exposed = occupied & (n_air > 0)
    ground_u = float(np.median(np.nonzero(top)[2])) if np.any(top) else 24.0

    unique, counts = np.unique(node_ids[occupied], return_counts=True)
    feats = []
    for nid, count in zip(unique.tolist(), counts.tolist()):
        m = node_ids == nid
        uu = np.nonzero(m)[2].astype(np.float64)
        feats.append(
            {
                "id": int(nid),
                "count": int(count),
                "top": float(top[m].mean()),
                "exposed": float(exposed[m].mean()),
                "mean_u": float(uu.mean()),
                "std_u": float(uu.std()),
            }
        )

    labels: dict[int, str] = {}
    flat: list[dict] = []
    rest: list[dict] = []
    for f in feats:
        if f["top"] > 0.45:
            labels[f["id"]] = "grass" if f["mean_u"] <= ground_u + 6.0 else "leaves"
        elif f["id"] in ground_ids:
            # Walkable but not grass-topped: a bare rock plain, never a liquid.
            labels[f["id"]] = "stone"
        elif f["exposed"] > 0.6 and f["mean_u"] > ground_u + 5.0:
            labels[f["id"]] = "leaves" if f["count"] > 30 else "log"
        elif f["std_u"] < 2.6 and f["top"] < 0.25 and f["mean_u"] <= ground_u + 3.0:
            flat.append(f)  # thin level band near ground height: water or its bed
        else:
            rest.append(f)

    # Only the biggest flat bands are water; thinner ones are more likely the
    # sand/gravel bed underneath it.
    flat.sort(key=lambda x: -x["count"])
    for i, f in enumerate(flat):
        labels[f["id"]] = "water" if i < 2 else ("sand" if i % 2 else "gravel")

    rest.sort(key=lambda x: -x["count"])
    for i, f in enumerate(rest):
        if i == 0:
            labels[f["id"]] = "stone"  # the bulk underground id
        elif f["count"] < 200 and f["mean_u"] < ground_u - 8.0:
            labels[f["id"]] = "bedrock"
        elif i == 1:
            labels[f["id"]] = "dirt"
        else:
            labels[f["id"]] = UNDERGROUND_CYCLE[i % len(UNDERGROUND_CYCLE)]
    return labels


def material_face_spec(material: str, face: str) -> str:
    spec = MATERIALS.get(material, MATERIALS["stone"])
    return spec.get(face) or spec.get("all") or spec["side"]


# --------------------------------------------------------------------------- #
# terrain meshing
# --------------------------------------------------------------------------- #
def build_atlas(
    materials: list[str], library: TextureLibrary, tile: int = 16
) -> tuple["Image.Image", dict[tuple[str, str], int], int, int]:
    """Pack every (material, face) tile into one atlas image."""
    keys = [(m, f) for m in materials for f in ("top", "bottom", "side")]
    cols = int(np.ceil(np.sqrt(len(keys))))
    rows = int(np.ceil(len(keys) / cols))
    atlas = np.zeros((rows * tile, cols * tile, 3), dtype=np.uint8)
    index: dict[tuple[str, str], int] = {}
    for n, key in enumerate(keys):
        r, c = divmod(n, cols)
        atlas[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = library.flatten(
            material_face_spec(*key), tile
        )
        index[key] = n
    return Image.fromarray(atlas, "RGB"), index, cols, rows


def _tile_uv_bounds(n: int, cols: int, rows: int, inset: float) -> tuple[float, float, float, float]:
    r, c = divmod(n, cols)
    u0, v0 = c / cols, r / rows
    return u0 + inset / cols, v0 + inset / rows, (1.0 - 2 * inset) / cols, (1.0 - 2 * inset) / rows


def terrain_meshes(
    node_ids: np.ndarray,
    voxel_center: np.ndarray,
    origin_idx: np.ndarray,
    origin_enu: np.ndarray,
    library: TextureLibrary | None,
    full_volume: bool,
    open_cut: bool = False,
    ground_ids: set[int] | None = None,
) -> tuple[list[tuple[str, trimesh.Trimesh]], dict]:
    """Mesh the voxel grid, emitting only faces that are actually visible.

    Faces between two solid voxels are skipped. Faces on the grid boundary are
    kept (out-of-bounds counts as air) so the terrain is closed where the
    observation cube slices through it; 41% of columns in a typical level reach
    the ceiling, and dropping those faces leaves see-through holes.

    With `open_cut` the boundary faces are dropped instead, which opens the cube
    up so you can look inside at the cost of those holes.
    """
    occupied = occupied_mask(node_ids)
    stats = {"occupied": int(occupied.sum()), "faces": 0, "materials": {}}
    if stats["occupied"] == 0:
        return [], stats

    labels = classify_node_ids(node_ids, occupied, ground_ids)
    for nid, name in labels.items():
        stats["materials"][name] = stats["materials"].get(name, 0) + int(
            (node_ids == nid).sum()
        )
    materials = sorted(set(labels.values()))

    is_water = occupied & np.isin(node_ids, [n for n, m in labels.items() if m == "water"])
    is_solid = occupied & ~is_water

    # node_id -> material lookup tables, one per face kind.
    lut = {}
    if library is not None:
        atlas_img, tile_index, cols, rows = build_atlas(materials, library)
        for face in ("top", "bottom", "side"):
            table = np.zeros(MAX_VALID_NODE_ID + 1, dtype=np.int64)
            for nid, name in labels.items():
                table[nid] = tile_index[(name, face)]
            lut[face] = table
    else:
        atlas_img = None
        color_table = np.zeros((MAX_VALID_NODE_ID + 1, 3), dtype=np.uint8)
        for nid, name in labels.items():
            color_table[nid] = FALLBACK_RGB.get(name, (130, 122, 118))

    def collect(mask: np.ndarray, blockers: np.ndarray):
        """Visible quads of `mask`, hidden where the neighbor is in `blockers`."""
        verts, uvs, colors, tris = [], [], [], []
        base = 0
        for face_kind, axis, sign, corners, uv_corners in FACE_DEFS:
            # Neighbor along +-axis; out of bounds is air unless open_cut.
            neighbor = np.full_like(blockers, bool(open_cut))
            src = [slice(None)] * 3
            dst = [slice(None)] * 3
            if sign > 0:
                dst[axis] = slice(0, -1)
                src[axis] = slice(1, None)
            else:
                dst[axis] = slice(1, None)
                src[axis] = slice(0, -1)
            neighbor[tuple(dst)] = blockers[tuple(src)]
            visible = mask & ~neighbor
            ii, jj, kk = np.nonzero(visible)
            n = int(ii.size)
            if n == 0:
                continue
            centers = voxel_center[None, :] + (
                np.stack([ii, jj, kk], axis=1).astype(np.float64) - origin_idx[None, :]
            )
            verts.append((centers[:, None, :] + corners[None, :, :]).reshape(-1, 3))
            ids = node_ids[ii, jj, kk]
            if atlas_img is not None:
                tiles = lut[face_kind][ids]
                bounds = np.array(
                    [_tile_uv_bounds(int(t), cols, rows, 0.25) for t in tiles]
                )
                uv = np.empty((n, 4, 2))
                uv[..., 0] = bounds[:, 0:1] + uv_corners[None, :, 0] * bounds[:, 2:3]
                uv[..., 1] = bounds[:, 1:2] + uv_corners[None, :, 1] * bounds[:, 3:4]
                uvs.append(uv.reshape(-1, 2))
            else:
                colors.append(np.repeat(color_table[ids], 4, axis=0))
            tris.append(
                (QUAD_TRIS[None, :, :] + (base + np.arange(n) * 4)[:, None, None]).reshape(-1, 3)
            )
            base += n * 4
        if base == 0:
            return None
        return (
            np.concatenate(verts),
            np.concatenate(uvs) if uvs else None,
            np.concatenate(colors) if colors else None,
            np.concatenate(tris),
        )

    out: list[tuple[str, trimesh.Trimesh]] = []

    def emit(name: str, packed, alpha: int):
        if packed is None:
            return
        verts, uv, colors, faces = packed
        stats["faces"] += len(faces) // 2
        mesh = trimesh.Trimesh(
            vertices=enu_to_gltf(verts, origin_enu).astype(np.float32),
            faces=faces,
            process=False,
        )
        if uv is not None:
            mesh.visual = TextureVisuals(
                uv=to_gltf_uv(uv),
                material=PBRMaterial(
                    name=name,
                    baseColorTexture=atlas_img,
                    baseColorFactor=[1.0, 1.0, 1.0, alpha / 255.0],
                    alphaMode="BLEND" if alpha < 255 else "OPAQUE",
                    metallicFactor=0.0,
                    roughnessFactor=1.0,
                ),
            )
        else:
            rgba = np.concatenate(
                [colors, np.full((len(colors), 1), alpha, dtype=np.uint8)], axis=1
            )
            mesh.visual = ColorVisuals(mesh=mesh, vertex_colors=rgba)
            mesh.visual.material = PBRMaterial(
                name=name,
                baseColorFactor=[1.0, 1.0, 1.0, alpha / 255.0],
                alphaMode="BLEND" if alpha < 255 else "OPAQUE",
                metallicFactor=0.0,
                roughnessFactor=1.0,
            )
        out.append((name, mesh))

    if full_volume:
        nothing = np.zeros_like(occupied)
        emit("terrain", collect(is_solid, nothing), 255)
        emit("water", collect(is_water, nothing), WATER_ALPHA)
    else:
        # Solid blocks hide behind solid blocks only, so the lake bed stays
        # visible through the water surface.
        emit("terrain", collect(is_solid, is_solid), 255)
        emit("water", collect(is_water, occupied), WATER_ALPHA)
    return out, stats


# --------------------------------------------------------------------------- #
# entity meshes
# --------------------------------------------------------------------------- #
def voxel_index(pos_enu: np.ndarray, voxel_center: np.ndarray, origin_idx: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(pos_enu, dtype=np.float64) - voxel_center + origin_idx).astype(int)


# A moving entity legitimately overlaps the block it stands on by a few
# centimetres for one frame; only deeper overlaps mean it is stuck in terrain.
BURIAL_TOLERANCE = 0.25


def collisionbox_heights(box: np.ndarray, fallback: float) -> np.ndarray:
    """Entity heights from recorded collision boxes (y2 - y1, Minetest axes).

    Species differ by almost 4x here (rabbit 0.5 to llama 1.87), so the
    recorded box beats any per-species constant.
    """
    box = np.atleast_2d(np.asarray(box, dtype=np.float64))
    height = box[:, 4] - box[:, 1]
    return np.where(height > 1e-3, height, fallback)


def ground_offset(
    pos_enu: np.ndarray,
    occupied: np.ndarray,
    voxel_center: np.ndarray,
    origin_idx: np.ndarray,
    height: float,
) -> tuple[float, float]:
    """How far to lift an entity so its feet rest on the surface.

    Returns (delta_u, depth). `depth` is how far the feet sit below the top face
    of the solid voxel they are in, so a fast-moving entity clipping a block
    corner by a few centimetres is not confused with one buried a whole block.
    """
    i, j, k = voxel_index(pos_enu, voxel_center, origin_idx)
    dims = occupied.shape
    if not (0 <= i < dims[0] and 0 <= j < dims[1] and 0 <= k < dims[2]):
        return 0.0, 0.0
    if not occupied[i, j, k]:
        return 0.0, 0.0

    feet_voxel_top = voxel_center[2] + (k - origin_idx[2]) + 0.5
    depth = float(feet_voxel_top - pos_enu[2])

    need = max(1, int(np.ceil(height)))
    for kk in range(k, dims[2] - need):
        if occupied[i, j, kk]:
            continue
        if np.any(occupied[i, j, kk : kk + need]):
            continue
        if kk > 0 and not occupied[i, j, kk - 1]:
            continue
        # Feet sit 0.01 above the block top, matching Minetest's collision box.
        feet_u = voxel_center[2] + (kk - origin_idx[2]) - 0.5 + 0.01
        return float(feet_u - pos_enu[2]), depth
    return 0.0, depth


# Correction between a model's bind-pose facing and dyn_yaw: there is none.
# Verified by posing a cow at a sweep of yaws and photographing it from due
# south: the face appears at yaw 180, and the head sits on the camera's left at
# yaw 90 (west) and its right at yaw 270 (east), which is exactly what
# yaw_basis_enu says. Independently, walking mobs' velocity agrees with that same
# basis in 96% of frames across every capture.
#
# Do not reintroduce a per-model guess based on the head node. Head nodes are
# pivots at the neck, so their offset points backwards on quadrupeds and is zero
# on humanoids; using it turns every quadruped around.
MODEL_FACES_YAW = 0.0


def posed_surface(
    surface: dict,
    pos_enu: np.ndarray,
    yaw: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Model-space surface -> world ENU vertices at `pos_enu` facing `yaw`."""
    v_mt = np.asarray(surface["vertices"], dtype=np.float64) * scale
    v_mt = yaw_rotate_mt(v_mt, yaw)
    v_enu = mt_to_enu(v_mt) + np.asarray(pos_enu, dtype=np.float64)[None, :]
    faces = orient_outward(v_enu, np.asarray(surface["faces"], dtype=np.int64))
    return v_enu, np.asarray(surface["uv"], dtype=np.float64), faces


def textured_mesh(
    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    origin_enu: np.ndarray,
    image,
    name: str,
) -> trimesh.Trimesh:
    verts, uvs, faces, base = [], [], [], 0
    for v, uv, f in groups:
        verts.append(v)
        uvs.append(uv)
        faces.append(f + base)
        base += len(v)
    mesh = trimesh.Trimesh(
        vertices=enu_to_gltf(np.concatenate(verts), origin_enu).astype(np.float32),
        faces=np.concatenate(faces),
        process=False,
    )
    mesh.visual = TextureVisuals(
        uv=to_gltf_uv(np.concatenate(uvs)),
        material=PBRMaterial(
            name=name,
            baseColorTexture=image,
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            alphaMode="MASK",
            alphaCutoff=0.5,
            metallicFactor=0.0,
            roughnessFactor=1.0,
            doubleSided=True,
        ),
    )
    return mesh


def sheep_texture_specs(color: str, sheared: bool) -> list[str]:
    """sheep.lua: textures = { colorized fur, sheep skin } in surface order."""
    wool = SHEEP_WOOL_COLORIZE.get(color or "unicolor_white", "#FFFFFF00")
    fur = "blank.png" if sheared else f"mobs_mc_sheep_fur.png^[colorize:{wool}"
    return [fur, "mobs_mc_sheep.png"]


def mob_geometry(
    dyn,
    frame: int,
    origin_enu: np.ndarray,
    library: TextureLibrary,
    game_dir: Path,
    positions: np.ndarray | None = None,
) -> tuple[list[tuple[str, trimesh.Trimesh]], dict]:
    """Pose every present agent with its own species model and textures.

    Newer captures mix species in one level and leave the per-slot `dyn_mesh`
    and `dyn_textures` metadata empty, so the model and texture list come from
    `MOB_MODELS` keyed on `dyn_names`. Surfaces are merged per (model, texture)
    pair, which keeps the GLB small no matter how many mobs share a species.
    """
    present = np.asarray(dyn["dyn_present"][frame]).astype(np.int8)
    pos = (
        np.asarray(dyn["dyn_pos"][frame], dtype=np.float64)
        if positions is None
        else np.asarray(positions, dtype=np.float64)
    )
    yaw = np.asarray(dyn["dyn_yaw"][frame], dtype=np.float64)
    names = [str(x) for x in np.atleast_1d(dyn["dyn_names"])]
    colors = dyn["dyn_color"][frame] if "dyn_color" in dyn.files else None
    baby = np.asarray(dyn["dyn_baby"][frame]).astype(np.int8) if "dyn_baby" in dyn.files else None
    sheared = (
        np.asarray(dyn["dyn_sheared"][frame]).astype(np.int8)
        if "dyn_sheared" in dyn.files
        else None
    )

    models: dict[str, dict] = {}
    buckets: dict[tuple[str, str], list] = {}
    stats = {"drawn": {}, "unknown": {}, "missing_model": {}}
    for slot in np.flatnonzero(present == 1).tolist():
        species = names[slot] if slot < len(names) else ""
        entry = MOB_MODELS.get(species)
        if entry is None:
            stats["unknown"][species] = stats["unknown"].get(species, 0) + 1
            continue
        mesh_file, specs, visual_size = entry
        if mesh_file not in models:
            path = find_model(game_dir, mesh_file)
            models[mesh_file] = load_b3d(path) if path is not None else None
        model = models[mesh_file]
        if model is None:
            stats["missing_model"][mesh_file] = stats["missing_model"].get(mesh_file, 0) + 1
            continue

        if species == "mobs_mc:sheep":
            specs = sheep_texture_specs(
                str(colors[slot]) if colors is not None else "",
                bool(sheared[slot]) if sheared is not None else False,
            )
        scale = (visual_size / B3D_UNITS_PER_NODE) * (
            0.5 if baby is not None and baby[slot] else 1.0
        )
        # Align the model's head with the recorded yaw direction.
        angle = float(yaw[slot]) + MODEL_FACES_YAW
        for si, surface in enumerate(model["surfaces"]):
            spec = specs[si] if si < len(specs) else specs[-1]
            if any(empty in spec for empty in EMPTY_TEXTURES):
                continue  # unused layer: saddle, chest, armor
            buckets.setdefault((mesh_file, spec), []).append(
                posed_surface(surface, pos[slot], angle, scale)
            )
        stats["drawn"][species] = stats["drawn"].get(species, 0) + 1

    out = []
    for i, ((mesh_file, spec), groups) in enumerate(buckets.items()):
        name = f"mob_{i}_{Path(mesh_file).stem}"
        out.append((name, textured_mesh(groups, origin_enu, library.get(spec), name)))
    return out, stats


def player_geometry(
    dyn,
    frame: int,
    origin_enu: np.ndarray,
    model: dict,
    library: TextureLibrary,
    position: np.ndarray | None = None,
) -> list[tuple[str, trimesh.Trimesh]]:
    if "dyn_player_present" not in dyn.files or int(dyn["dyn_player_present"][frame]) != 1:
        return []
    pos = (
        np.asarray(dyn["dyn_player_pos"][frame], dtype=np.float64)
        if position is None
        else np.asarray(position, dtype=np.float64)
    )
    yaw = float(np.asarray(dyn["dyn_player_rotation"][frame], dtype=np.float64)[1])
    specs = [str(s) for s in np.atleast_1d(dyn["dyn_player_textures"])]

    angle = yaw + MODEL_FACES_YAW
    out = []
    for si, surface in enumerate(model["surfaces"]):
        spec = specs[si] if si < len(specs) else specs[0]
        if "blank.png" in spec:
            continue  # empty armor layer
        groups = [posed_surface(surface, pos, angle, 1.0 / B3D_UNITS_PER_NODE)]
        out.append(
            (
                f"player_{si}",
                textured_mesh(groups, origin_enu, library.get(spec), f"player_{si}"),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# debug boxes and markers
# --------------------------------------------------------------------------- #
def mesh_from_boxes(
    corners_enu: np.ndarray, origin_enu: np.ndarray, rgba: np.ndarray, name: str
) -> trimesh.Trimesh:
    """Build one mesh from (B, 8, 3) ENU corners with a uniform RGBA color."""
    n = int(corners_enu.shape[0])
    verts = enu_to_gltf(corners_enu.reshape(-1, 3), origin_enu).astype(np.float32)
    faces = (OBB_FACES[None, :, :] + (np.arange(n, dtype=np.int64) * 8)[:, None, None]).reshape(
        -1, 3
    )
    colors = np.broadcast_to(np.asarray(rgba, dtype=np.uint8), (verts.shape[0], 4)).copy()
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual = ColorVisuals(mesh=mesh, vertex_colors=colors)
    alpha = float(rgba[3]) / 255.0
    mesh.visual.material = PBRMaterial(
        name=name,
        baseColorFactor=[1.0, 1.0, 1.0, alpha],
        alphaMode="BLEND" if alpha < 1.0 else "OPAQUE",
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    return mesh


def paint_solid(mesh: trimesh.Trimesh, rgba: np.ndarray, name: str) -> trimesh.Trimesh:
    colors = np.broadcast_to(rgba, (len(mesh.vertices), 4)).copy()
    mesh.visual = ColorVisuals(mesh=mesh, vertex_colors=colors)
    mesh.visual.material = PBRMaterial(
        name=name, baseColorFactor=[1.0, 1.0, 1.0, 1.0], metallicFactor=0.0
    )
    return mesh


def yaw_basis_enu(yaw: float) -> np.ndarray:
    """Rows (forward, left, up) in ENU for a Minetest yaw.

    Forward matches where `posed_surface` puts the model's head: yaw 0 faces
    north, and left is up x forward so the triple is right-handed.
    """
    s, c = np.sin(yaw), np.cos(yaw)
    return np.array([[-s, c, 0.0], [-c, -s, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def arrow_mesh(
    start_enu: np.ndarray,
    direction: np.ndarray,
    length: float,
    radius: float,
    origin_enu: np.ndarray,
) -> trimesh.Trimesh:
    """Shaft plus arrowhead from `start_enu` along `direction`, in glTF coords."""
    head_len = min(0.32 * length, 4.0 * radius)
    shaft_len = length - head_len
    rotate = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], direction)

    shaft = trimesh.creation.cylinder(radius=radius, height=shaft_len, sections=10)
    shaft.apply_transform(rotate)
    shaft.vertices = np.asarray(shaft.vertices) + direction * (shaft_len * 0.5)

    head = trimesh.creation.cone(radius=radius * 2.2, height=head_len, sections=12)
    head.apply_transform(rotate)
    head.vertices = np.asarray(head.vertices) + direction * shaft_len

    mesh = trimesh.util.concatenate([shaft, head])
    mesh.vertices = enu_to_gltf(np.asarray(mesh.vertices) + start_enu, origin_enu).astype(
        np.float32
    )
    return mesh


def axes_markers(
    positions_enu: np.ndarray,
    yaws: np.ndarray,
    origin_enu: np.ndarray,
    heights: np.ndarray,
    length: float,
    prefix: str,
) -> list[tuple[str, trimesh.Trimesh]]:
    """One merged arrow mesh per axis covering every entity in `positions_enu`.

    Merging by axis keeps the GLB at three extra primitives no matter how many
    sheep there are. Each gizmo starts at the entity's mid-height so the arrows
    emerge from the body instead of being swallowed by the ground.
    """
    positions_enu = np.atleast_2d(np.asarray(positions_enu, dtype=np.float64))
    yaws = np.atleast_1d(np.asarray(yaws, dtype=np.float64))
    heights = np.atleast_1d(np.asarray(heights, dtype=np.float64))
    if positions_enu.size == 0:
        return []

    radius = max(0.022, 0.03 * length)
    specs = [
        ("forward", 0, AXIS_FORWARD_RGBA),
        ("left", 1, AXIS_LEFT_RGBA),
        ("up", 2, AXIS_UP_RGBA),
    ]
    out = []
    for axis_name, row, rgba in specs:
        parts = []
        for pos, yaw, height in zip(positions_enu, yaws, heights):
            start = pos + np.array([0.0, 0.0, 0.5 * float(height)])
            parts.append(
                arrow_mesh(start, yaw_basis_enu(float(yaw))[row], length, radius, origin_enu)
            )
        name = f"{prefix}_axis_{axis_name}"
        out.append((name, paint_solid(trimesh.util.concatenate(parts), rgba, name)))
    return out


def camera_marker(
    cam_pos: np.ndarray,
    cam_dir: np.ndarray,
    origin_enu: np.ndarray,
    length: float = 2.5,
) -> list[trimesh.Trimesh]:
    """Sphere at the camera plus a thin shaft along cam_dir (both ENU)."""
    def paint(mesh):
        return paint_solid(mesh, CAMERA_RGBA, "camera")

    direction = np.asarray(cam_dir, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return []
    direction = direction / norm

    # Start clear of the player's head so the marker does not hide the model.
    start = np.asarray(cam_pos, dtype=np.float64) + direction * 0.55
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.10)
    sphere.vertices = enu_to_gltf(np.asarray(sphere.vertices) + start, origin_enu).astype(
        np.float32
    )
    shaft = trimesh.creation.cylinder(radius=0.05, height=length, sections=12)
    shaft.apply_transform(trimesh.geometry.align_vectors([0.0, 0.0, 1.0], direction))
    shaft.vertices = enu_to_gltf(
        np.asarray(shaft.vertices) + start + direction * (length * 0.5), origin_enu
    ).astype(np.float32)
    return [paint(sphere), paint(shaft)]


# --------------------------------------------------------------------------- #
# export helpers
# --------------------------------------------------------------------------- #
def patch_nearest_filtering(path: Path) -> None:
    """Force NEAREST texture sampling so 16px block art stays crisp."""
    data = path.read_bytes()
    if data[:4] != b"glTF":
        return
    json_len, json_type = struct.unpack_from("<II", data, 12)
    if json_type != 0x4E4F534A:
        return
    doc = json.loads(data[20 : 20 + json_len])
    if not doc.get("textures"):
        return
    samplers = doc.setdefault("samplers", [])
    samplers.append({"magFilter": 9728, "minFilter": 9728, "wrapS": 10497, "wrapT": 10497})
    idx = len(samplers) - 1
    for tex in doc["textures"]:
        tex["sampler"] = idx

    new_json = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    new_json += b" " * (-len(new_json) % 4)
    rest = data[20 + json_len :]
    out = bytearray(b"glTF")
    out += struct.pack("<II", 2, 12 + 8 + len(new_json) + len(rest))
    out += struct.pack("<II", len(new_json), 0x4E4F534A)
    out += new_json
    out += rest
    path.write_bytes(bytes(out))


def find_seed_dir(raw_env_dir: Path, seed: str | None) -> Path:
    if seed is not None:
        seed_dir = raw_env_dir / str(seed)
        if not seed_dir.is_dir():
            raise FileNotFoundError(f"Seed folder not found: {seed_dir}")
        return seed_dir
    candidates = sorted(
        p for p in raw_env_dir.iterdir() if p.is_dir() and (p / "data.npz").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No seed folders with data.npz under {raw_env_dir}")
    return candidates[0]


def find_model(game_dir: Path, filename: str) -> Path | None:
    for p in game_dir.rglob(filename):
        return p
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default="datasets", type=Path)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--env_id", default="OpenWorldCreative-v0")
    parser.add_argument("--seed", default=None)
    parser.add_argument("--frame", default=0, type=int)
    parser.add_argument("--out", default="frame0.glb", type=Path)
    parser.add_argument(
        "--game_dir",
        default=DEFAULT_GAME_DIR,
        type=Path,
        help="VoxeLibre game directory holding models and textures.",
    )
    parser.add_argument(
        "--full_volume",
        action="store_true",
        help="Also mesh buried voxels (makes the grid a closed cube).",
    )
    parser.add_argument(
        "--open_cut",
        action="store_true",
        help="Drop faces on the grid boundary so you can see inside the cube "
        "(leaves see-through holes where terrain reaches the edge).",
    )
    parser.add_argument(
        "--snap_to_ground",
        action="store_true",
        help="Lift entities whose feet are inside a block onto the local surface. "
        "This changes recorded positions, so it is a visualization aid only.",
    )
    parser.add_argument(
        "--boxes",
        action="store_true",
        help="Add the translucent ground-truth OBBs on top of the entity models.",
    )
    parser.add_argument(
        "--no_textures",
        action="store_true",
        help="Use flat palette colors instead of the game's textures.",
    )
    parser.add_argument(
        "--no_axes",
        action="store_true",
        help="Skip the per-entity orientation gizmos.",
    )
    parser.add_argument(
        "--axes_length",
        default=1.3,
        type=float,
        help="Arrow length in blocks for the orientation gizmos.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_dir / args.dataset_name
    params_path = dataset_root / "dataset_params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing {params_path}")

    with params_path.open() as f:
        params = json.load(f)
    vox_info = params.get("minetest_voxel_info") or params.get("voxel_info") or {}
    origin_idx = np.asarray(vox_info.get("origin_idx", [24, 24, 24]), dtype=np.float64)

    seed_dir = find_seed_dir(dataset_root / "raw" / args.env_id, args.seed)
    seed = seed_dir.name
    data_path, dyn_path = seed_dir / "data.npz", seed_dir / "data_dynamic.npz"
    for p in (data_path, dyn_path):
        if not p.exists():
            raise FileNotFoundError(p)

    data = np.load(data_path, allow_pickle=True)
    dyn = np.load(dyn_path, allow_pickle=True)

    frame = int(args.frame)
    t_max = int(data["obs_voxel_mt"].shape[0])
    if frame < 0 or frame >= t_max:
        raise IndexError(f"--frame {frame} out of range [0, {t_max})")

    # Channel 0 = node_id, channel 1 = param2.
    voxel_ids = np.asarray(data["obs_voxel_mt"][frame, ..., 0])
    voxel_center = np.asarray(data["obs_voxel_center"][frame], dtype=np.float64)
    origin_enu = voxel_center.copy()
    grid_dims = np.array(voxel_ids.shape, dtype=np.int64)

    use_textures = not args.no_textures and Image is not None
    library = TextureLibrary(args.game_dir) if use_textures else None
    if use_textures and not library.files:
        print(f"WARNING: no textures found under {args.game_dir}; using flat colors")
        library, use_textures = None, False

    scene = trimesh.Scene()

    present = np.asarray(dyn["dyn_present"][frame]).astype(np.int8)
    sheep_idx = np.flatnonzero(present == 1)
    dyn_pos = np.asarray(dyn["dyn_pos"][frame], dtype=np.float64)
    dyn_yaw = np.asarray(dyn["dyn_yaw"][frame], dtype=np.float64)
    player_pos = np.asarray(dyn["dyn_player_pos"][frame], dtype=np.float64)

    # Check every entity against the recorded grid: feet inside a solid voxel
    # means the level was captured with that agent stuck inside terrain.
    occupied = occupied_mask(voxel_ids)
    ground_ids = support_node_ids(
        np.vstack([dyn_pos[sheep_idx], player_pos[None, :]]) if sheep_idx.size else player_pos,
        voxel_ids,
        occupied,
        voxel_center,
        origin_idx,
    )

    terrain, tstats = terrain_meshes(
        voxel_ids,
        voxel_center,
        origin_idx,
        origin_enu,
        library,
        args.full_volume,
        args.open_cut,
        ground_ids,
    )
    for name, mesh in terrain:
        scene.add_geometry(mesh, node_name=name)

    sheep_heights = collisionbox_heights(dyn["dyn_collisionbox"][frame], SHEEP_HEIGHT)
    player_height = float(
        collisionbox_heights(dyn["dyn_player_collisionbox"][frame], PLAYER_HEIGHT)[0]
    )

    sheep_pos = dyn_pos.copy()
    buried: list[tuple[int, float, float]] = []
    for n in sheep_idx:
        delta, depth = ground_offset(
            dyn_pos[n], occupied, voxel_center, origin_idx, height=float(sheep_heights[n])
        )
        if depth > BURIAL_TOLERANCE:
            buried.append((int(n), delta, depth))
            if args.snap_to_ground:
                sheep_pos[n, 2] += delta

    player_draw_pos = player_pos.copy()
    player_delta, player_depth = ground_offset(
        player_pos, occupied, voxel_center, origin_idx, height=player_height
    )
    player_buried = player_depth > BURIAL_TOLERANCE
    if player_buried and args.snap_to_ground:
        player_draw_pos[2] += player_delta

    mob_stats: dict = {"drawn": {}, "unknown": {}, "missing_model": {}}
    player_model = None
    if use_textures:
        if sheep_idx.size:
            mobs, mob_stats = mob_geometry(
                dyn, frame, origin_enu, library, args.game_dir, sheep_pos
            )
            for name, mesh in mobs:
                scene.add_geometry(mesh, node_name=name)
        player_path = find_model(args.game_dir, PLAYER_MODEL)
        if player_path is not None:
            player_model = load_b3d(player_path)
            for name, mesh in player_geometry(
                dyn, frame, origin_enu, player_model, library, player_draw_pos
            ):
                scene.add_geometry(mesh, node_name=name)

    drew_models = bool(mob_stats["drawn"]) or player_model is not None
    if args.boxes or not drew_models:
        if sheep_idx.size:
            obb = np.asarray(dyn["dyn_obb_corners"][frame], dtype=np.float64)
            scene.add_geometry(
                mesh_from_boxes(obb[sheep_idx], origin_enu, SHEEP_RGBA, "sheep_obb"),
                node_name="sheep_obb",
            )
        if int(dyn["dyn_player_present"][frame]) == 1:
            pobb = np.asarray(dyn["dyn_player_obb_corners"][frame], dtype=np.float64)[None, ...]
            scene.add_geometry(
                mesh_from_boxes(pobb, origin_enu, PLAYER_RGBA, "player_obb"),
                node_name="player_obb",
            )

    if not args.no_axes:
        if sheep_idx.size:
            for name, mesh in axes_markers(
                sheep_pos[sheep_idx],
                dyn_yaw[sheep_idx],
                origin_enu,
                heights=sheep_heights[sheep_idx],
                length=args.axes_length,
                prefix="mob",
            ):
                scene.add_geometry(mesh, node_name=name)
        if int(dyn["dyn_player_present"][frame]) == 1:
            player_yaw = float(
                np.asarray(dyn["dyn_player_rotation"][frame], dtype=np.float64)[1]
            )
            for name, mesh in axes_markers(
                player_draw_pos,
                [player_yaw],
                origin_enu,
                heights=[player_height],
                length=args.axes_length,
                prefix="player",
            ):
                scene.add_geometry(mesh, node_name=name)

    if "cam_pos" in data.files and "cam_dir" in data.files:
        cam_pos = np.asarray(data["cam_pos"][frame], dtype=np.float64)
        cam_dir = np.asarray(data["cam_dir"][frame], dtype=np.float64)
        for i, geom in enumerate(camera_marker(cam_pos, cam_dir, origin_enu)):
            scene.add_geometry(geom, node_name=f"camera_{i}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out_path)
    patch_nearest_filtering(out_path)

    # --- summary / sanity ---
    grid_min = voxel_center - origin_idx - 0.5
    grid_max = voxel_center + (grid_dims.astype(np.float64) - 1.0 - origin_idx) + 0.5
    player_delta = player_pos - voxel_center
    print(f"seed:              {seed}")
    print(f"frame:             {frame}")
    print(f"voxel grid dims:   {tuple(int(x) for x in grid_dims)}")
    print(f"origin_idx:        {origin_idx.astype(int).tolist()}")
    print(f"obs_voxel_center:  {voxel_center.tolist()}")
    print(f"occupied voxels:   {tstats['occupied']}")
    print(f"drawn block faces: {tstats['faces']} (full_volume={args.full_volume})")
    print("materials:         " + ", ".join(
        f"{k}={v}" for k, v in sorted(tstats["materials"].items(), key=lambda x: -x[1])
    ))
    print(f"textures:          {'game textures' if use_textures else 'flat palette'}")
    if mob_stats["drawn"]:
        print("species drawn:     " + ", ".join(
            f"{k.split(':')[-1]}={v}"
            for k, v in sorted(mob_stats["drawn"].items(), key=lambda x: -x[1])
        ))
    if player_model is not None:
        h = max(s["vertices"][:, 1].max() for s in player_model["surfaces"]) / B3D_UNITS_PER_NODE
        print(f"player model:      {PLAYER_MODEL} height={h:.2f} nodes (collisionbox 1.80)")
    print(f"agents drawn:      {int(sheep_idx.size)}")
    buried_map = {n: (lift, depth) for n, lift, depth in buried}
    hp = np.asarray(dyn["dyn_hp"][frame], dtype=np.float64) if "dyn_hp" in dyn.files else None
    names = [str(x) for x in np.atleast_1d(dyn["dyn_names"])]
    for n in sheep_idx:
        n = int(n)
        note = ""
        if n in buried_map:
            lift, depth = buried_map[n]
            note = (
                f"  BURIED {depth:.2f} blocks deep (needs {lift:+.2f} U to reach surface)"
                if lift
                else f"  BURIED {depth:.2f} blocks deep (no free surface found in this column)"
            )
        hp_txt = f" hp={hp[n]:.0f}" if hp is not None else ""
        species = names[n].split(":")[-1] if n < len(names) else "?"
        print(
            f"  slot {n:2d}  {species:10s} dyn_pos={np.round(dyn_pos[n], 3).tolist()} "
            f"yaw={np.degrees(dyn_yaw[n]):7.1f} deg h={sheep_heights[n]:.2f}{hp_txt}{note}"
        )
    for species, count in sorted(mob_stats["unknown"].items()):
        print(f"WARNING: {count} agent(s) of unknown species {species!r} were not drawn.")
        print("         Add it to MOB_MODELS to render it.")
    for mesh_file, count in sorted(mob_stats["missing_model"].items()):
        print(f"WARNING: {count} agent(s) skipped, model {mesh_file} not found under {args.game_dir}")
    print(f"dyn_player_pos:    {np.round(player_pos, 3).tolist()}")
    if player_depth > 0.0:
        print(
            f"player overlap:    feet {player_depth:.2f} blocks into the voxel below"
            f"{' (BURIED)' if player_buried else ' (grounded, within tolerance)'}"
        )
    if buried:
        print(
            f"WARNING: {len(buried)}/{int(sheep_idx.size)} agents are more than "
            f"{BURIAL_TOLERANCE:.2f} blocks inside a solid voxel in the recorded grid."
        )
        print("         This is in the captured data, not the export.")
        print(
            "         Rendered as recorded; pass --snap_to_ground to lift them onto the surface."
            if not args.snap_to_ground
            else "         --snap_to_ground is on, so they were lifted onto the surface."
        )
    if not args.no_axes:
        print(
            f"axes gizmos:       {int(sheep_idx.size)} agents + player, "
            f"{args.axes_length:.2f} blocks long"
        )
        print("                   red=facing (yaw), green=left, blue=up, magenta=camera ray")
    print(f"output:            {out_path.resolve()}")
    if library is not None and library.missing:
        print(f"WARNING: missing textures: {sorted(library.missing)}")

    if np.linalg.norm(player_delta[:2]) > 2.0 or abs(player_delta[2]) > 3.0:
        print(
            f"WARNING: player is not near the voxel-grid center "
            f"(delta ENU={np.round(player_delta, 3).tolist()})"
        )
    else:
        print(
            f"sanity: player is near voxel-grid center "
            f"(delta ENU={np.round(player_delta, 3).tolist()})"
        )

    if "dyn_obb_corners" in dyn.files:
        obb_all = np.asarray(dyn["dyn_obb_corners"][frame], dtype=np.float64)
        for n in sheep_idx:
            centroid = obb_all[n].mean(axis=0)
            if np.any(centroid < grid_min - 5.0) or np.any(centroid > grid_max + 5.0):
                print(
                    f"WARNING: sheep slot {int(n)} centroid {np.round(centroid, 2).tolist()} "
                    f"is far outside the voxel grid "
                    f"[{np.round(grid_min, 1).tolist()}, {np.round(grid_max, 1).tolist()}]"
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
