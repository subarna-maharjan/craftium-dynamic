#!/usr/bin/env python3
"""Export a short animated GLB: voxel terrain plus rigged, skinned mobs.

Each entity keeps its .b3d armature. Per dataset frame the exporter samples
the playing clip (dyn_anim_range / dyn_anim_speed vs dt_minetest) and composes
dyn_bone_rot overrides, then writes those joint poses as a glTF animation.
The mesh is skinned, not baked to a static posed mesh.

Example:
    .venv/bin/python tools/make_clip_glb.py \\
        --dataset_name geval1 --seed 794921487 --frames 50 \\
        --out datasets/geval1/raw/OpenWorldCreative-v0/794921487/clip_f0-49.glb
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_frame_glb import (  # noqa: E402
    B3D_UNITS_PER_NODE,
    BURIAL_TOLERANCE,
    DEFAULT_GAME_DIR,
    EMPTY_TEXTURES,
    MOB_MODELS,
    MODEL_FACES_YAW,
    PLAYER_HEIGHT,
    PLAYER_MODEL,
    SHEEP_HEIGHT,
    TextureLibrary,
    collisionbox_heights,
    enu_to_gltf,
    find_model,
    find_seed_dir,
    ground_offset,
    occupied_mask,
    orient_outward,
    patch_nearest_filtering,
    sheep_texture_specs,
    support_node_ids,
    terrain_meshes,
    textured_mesh,
    to_gltf_uv,
)

from b3d_rig import (  # noqa: E402
    anim_frame,
    load_b3d_rig,
    mt_to_gltf_matrix,
    mt_vec_to_gltf,
    sample_local_pose,
    skin_weights,
    trs_mt_to_gltf,
)


def local_surface(surface: dict, scale: float):
    """Bind-pose surface at the origin, facing yaw 0, in ENU then glTF."""
    v_mt = np.asarray(surface["vertices"], dtype=np.float64) * scale
    v_enu = mt_to_enu(v_mt)
    faces = orient_outward(v_enu, np.asarray(surface["faces"], dtype=np.int64))
    return v_enu, np.asarray(surface["uv"], dtype=np.float64), faces


def yaw_to_quat(yaw: np.ndarray) -> np.ndarray:
    """glTF xyzw quaternions for a rotation about +Y by unwrapped `yaw`."""
    half = np.unwrap(np.asarray(yaw, dtype=np.float64)) * 0.5
    q = np.zeros((half.shape[0], 4), dtype=np.float32)
    q[:, 1] = np.sin(half)
    q[:, 3] = np.cos(half)
    return q


def enu_translation(pos: np.ndarray, origin_enu: np.ndarray) -> np.ndarray:
    return enu_to_gltf(np.asarray(pos, dtype=np.float64), origin_enu).astype(np.float32)


def union_voxel_grid(data, frames: np.ndarray, origin_idx: np.ndarray):
    """Pack occupied world cells into one grid. Later frames overwrite overlaps."""
    voxel = np.asarray(data["obs_voxel_mt"][..., 0])
    centers = np.asarray(data["obs_voxel_center"], dtype=np.float64)
    cube = np.array(voxel.shape[1:], dtype=np.float64)
    used_c = centers[frames]
    lo = np.floor((used_c - origin_idx).min(axis=0)).astype(np.int32) - 1
    hi = np.ceil((used_c - origin_idx + cube - 1).max(axis=0)).astype(np.int32) + 1
    shape = tuple(int(x) for x in (hi - lo + 1))
    grid = np.full(shape, 126, dtype=np.int16)
    origin = origin_idx.reshape(1, 3)
    for t in frames:
        ids = voxel[int(t)]
        occ = occupied_mask(ids)
        ii, jj, kk = np.nonzero(occ)
        if ii.size == 0:
            continue
        world = np.rint(
            centers[int(t)][None, :]
            + np.stack([ii, jj, kk], axis=1).astype(np.float64)
            - origin
        ).astype(np.int32)
        grid[
            world[:, 0] - lo[0],
            world[:, 1] - lo[1],
            world[:, 2] - lo[2],
        ] = ids[ii, jj, kk]
    packed_origin = np.zeros(3, dtype=np.float64)
    packed_center = lo.astype(np.float64)
    return grid, packed_center, packed_origin, lo, hi


def bind_pose_meshes(
    model: dict,
    specs: list[str],
    scale: float,
    library: TextureLibrary,
    name_prefix: str,
) -> list[tuple[str, trimesh.Trimesh]]:
    """Local glTF meshes at the origin, yaw 0. Parents carry the world pose."""
    out = []
    origin = np.zeros(3, dtype=np.float64)
    for si, surface in enumerate(model["surfaces"]):
        spec = specs[si] if si < len(specs) else specs[-1]
        if any(empty in spec for empty in EMPTY_TEXTURES):
            continue
        mesh = textured_mesh(
            [local_surface(surface, scale)],
            origin,
            library.get(spec),
            f"{name_prefix}_{si}",
        )
        out.append((f"{name_prefix}_{si}", mesh))
    return out


def species_scale(species: str, visual_size: float, is_baby: bool) -> float:
    """Baby variants that already bake the 1/2 scale into visual_size."""
    baked = species.startswith("mobs_mc:baby_")
    child = 0.5 if is_baby and not baked else 1.0
    return (visual_size / B3D_UNITS_PER_NODE) * child


def load_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    json_len, json_type = struct.unpack_from("<II", data, 12)
    if json_type != 0x4E4F534A:
        raise ValueError(f"{path} is not a GLB")
    doc = json.loads(data[20 : 20 + json_len])
    rest = data[20 + json_len :]
    bin_len = struct.unpack_from("<I", rest, 0)[0]
    blob = bytearray(rest[8 : 8 + bin_len])
    return doc, blob


PLAYER_HIGHLIGHT = (1.0, 0.15, 0.85)


def _annulus_mesh(inner: float, outer: float, y: float, segs: int = 32):
    """Flat ring in the XZ plane (glTF Y-up), CCW from above."""
    thetas = np.linspace(0.0, 2.0 * np.pi, segs, endpoint=False)
    c, s = np.cos(thetas), np.sin(thetas)
    inner_r = np.stack([inner * c, np.full(segs, y), inner * s], axis=1)
    outer_r = np.stack([outer * c, np.full(segs, y), outer * s], axis=1)
    verts = np.vstack([inner_r, outer_r]).astype(np.float32)
    faces = []
    for i in range(segs):
        j = (i + 1) % segs
        faces.append([i, segs + i, segs + j])
        faces.append([i, segs + j, j])
    return verts, np.asarray(faces, dtype=np.uint32)


def highlight_player_in_glb(path: Path) -> int:
    """Tint the player mesh and parent a magenta ring to the player root."""
    doc, blob = load_glb(path)
    blob = bytearray(blob)
    nodes = doc.setdefault("nodes", [])
    player_i = next((i for i, n in enumerate(nodes) if n.get("name") == "player"), None)
    if player_i is None:
        raise ValueError(f"{path} has no node named 'player'")
    already = any(n.get("name") == "player_highlight" for n in nodes)

    n_mats = 0
    for mat in doc.setdefault("materials", []):
        name = str(mat.get("name", ""))
        if not name.startswith("player_") or name == "player_highlight_ring":
            continue
        pbr = mat.setdefault("pbrMetallicRoughness", {})
        pbr["baseColorFactor"] = [1.0, 0.45, 1.0, 1.0]
        pbr["metallicFactor"] = 0.0
        pbr["roughnessFactor"] = 0.45
        mat["emissiveFactor"] = list(PLAYER_HIGHLIGHT)
        mat.setdefault("extensions", {})["KHR_materials_emissive_strength"] = {
            "emissiveStrength": 3.0
        }
        n_mats += 1
    doc.setdefault("extensionsUsed", [])
    if "KHR_materials_emissive_strength" not in doc["extensionsUsed"]:
        doc["extensionsUsed"].append("KHR_materials_emissive_strength")

    if not already:
        verts, faces = _annulus_mesh(inner=5.5, outer=8.0, y=2.0)
        pos_acc = _accessor(doc, blob, verts, "VEC3", 5126)
        idx_acc = _accessor(
            doc, blob, np.ascontiguousarray(faces.reshape(-1)), "SCALAR", 5125, minmax=False
        )
        mat_i = len(doc["materials"])
        doc["materials"].append(
            {
                "name": "player_highlight_ring",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [*PLAYER_HIGHLIGHT, 0.95],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.3,
                },
                "emissiveFactor": list(PLAYER_HIGHLIGHT),
                "alphaMode": "BLEND",
                "doubleSided": True,
                "extensions": {
                    "KHR_materials_unlit": {},
                    "KHR_materials_emissive_strength": {"emissiveStrength": 4.0},
                },
            }
        )
        if "KHR_materials_unlit" not in doc["extensionsUsed"]:
            doc["extensionsUsed"].append("KHR_materials_unlit")
        mesh_i = len(doc.setdefault("meshes", []))
        doc["meshes"].append(
            {
                "name": "player_highlight_ring",
                "primitives": [
                    {
                        "attributes": {"POSITION": pos_acc},
                        "indices": idx_acc,
                        "material": mat_i,
                        "mode": 4,
                    }
                ],
            }
        )
        ring_i = len(nodes)
        nodes.append(
            {
                "name": "player_highlight",
                "mesh": mesh_i,
                "translation": [0.0, 0.0, 0.0],
            }
        )
        nodes[player_i].setdefault("children", []).append(ring_i)
    save_glb(path, doc, blob)
    return n_mats


LOS_RGBA = (0.15, 0.95, 1.0)
LOS_DEFAULT_LENGTH = 10.0


def _unit_cylinder_y(segs: int = 10) -> tuple[np.ndarray, np.ndarray]:
    thetas = np.linspace(0.0, 2.0 * np.pi, segs, endpoint=False)
    c, s = np.cos(thetas), np.sin(thetas)
    bot = np.stack([c, np.zeros(segs), s], axis=1)
    top = np.stack([c, np.ones(segs), s], axis=1)
    verts = np.vstack([bot, top]).astype(np.float32)
    faces = []
    for i in range(segs):
        j = (i + 1) % segs
        faces.append([i, segs + i, segs + j])
        faces.append([i, segs + j, j])
    return verts, np.asarray(faces, dtype=np.uint32)


def _quat_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """xyzw quaternion rotating unit vector `a` onto unit vector `b`."""
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    c = float(np.dot(a, b))
    if c > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if c < -0.999999:
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 0.0, 1.0])
        axis = axis / np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float64)
    v = np.cross(a, b)
    q = np.array([v[0], v[1], v[2], 1.0 + c], dtype=np.float64)
    return q / np.linalg.norm(q)


def _los_trs(
    cam_pos: np.ndarray,
    cam_dir: np.ndarray,
    origin_enu: np.ndarray,
    length: float,
    radius: float = 0.045,
    start_offset: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame TRS for a unit +Y cylinder covering the look ray."""
    pos = np.asarray(cam_pos, dtype=np.float64)
    raw = np.asarray(cam_dir, dtype=np.float64)
    nrm = np.linalg.norm(raw, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-8)
    fwd = raw / nrm
    start = pos + fwd * start_offset
    trans = enu_to_gltf(start, origin_enu).astype(np.float32)
    # ENU (E,N,U) -> glTF (E, U, -N)
    d_gltf = np.stack([fwd[:, 0], fwd[:, 2], -fwd[:, 1]], axis=1)
    y_axis = np.array([0.0, 1.0, 0.0])
    rot = np.zeros((len(fwd), 4), dtype=np.float32)
    for i, d in enumerate(d_gltf):
        q = _quat_from_to(y_axis, d).astype(np.float32)
        if i and float(np.dot(rot[i - 1], q)) < 0.0:
            q = -q
        rot[i] = q
    scale = np.repeat([[radius, float(length), radius]], len(fwd), axis=0).astype(np.float32)
    return trans, rot, scale


def add_player_los_in_glb(
    path: Path,
    cam_pos: np.ndarray,
    cam_dir: np.ndarray,
    origin_enu: np.ndarray,
    length: float = LOS_DEFAULT_LENGTH,
) -> None:
    """Append an animated cyan ray along recorded cam_dir, from the eye."""
    doc, blob = load_glb(path)
    blob = bytearray(blob)
    nodes = doc.setdefault("nodes", [])
    if any(n.get("name") == "player_los" for n in nodes):
        return
    anims = doc.get("animations") or []
    if not anims or not anims[0].get("samplers"):
        raise ValueError(f"{path} has no clip animation to attach the look ray to")
    t_acc = int(anims[0]["samplers"][0]["input"])
    n_keys = int(doc["accessors"][t_acc]["count"])
    trans, rot, scale = _los_trs(cam_pos[:n_keys], cam_dir[:n_keys], origin_enu, length)

    verts, faces = _unit_cylinder_y()
    pos_acc = _accessor(doc, blob, verts, "VEC3", 5126)
    idx_acc = _accessor(
        doc, blob, np.ascontiguousarray(faces.reshape(-1)), "SCALAR", 5125, minmax=False
    )
    doc.setdefault("extensionsUsed", [])
    for ext in ("KHR_materials_unlit", "KHR_materials_emissive_strength"):
        if ext not in doc["extensionsUsed"]:
            doc["extensionsUsed"].append(ext)
    mat_i = len(doc.setdefault("materials", []))
    doc["materials"].append(
        {
            "name": "los_ray",
            "pbrMetallicRoughness": {
                "baseColorFactor": [*LOS_RGBA, 0.9],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.25,
            },
            "emissiveFactor": list(LOS_RGBA),
            "alphaMode": "BLEND",
            "doubleSided": True,
            "extensions": {
                "KHR_materials_unlit": {},
                "KHR_materials_emissive_strength": {"emissiveStrength": 3.0},
            },
        }
    )
    mesh_i = len(doc.setdefault("meshes", []))
    doc["meshes"].append(
        {
            "name": "player_los",
            "primitives": [
                {
                    "attributes": {"POSITION": pos_acc},
                    "indices": idx_acc,
                    "material": mat_i,
                    "mode": 4,
                }
            ],
        }
    )
    los_i = len(nodes)
    nodes.append(
        {
            "name": "player_los",
            "mesh": mesh_i,
            "translation": trans[0].tolist(),
            "rotation": rot[0].tolist(),
            "scale": scale[0].tolist(),
        }
    )
    doc["scenes"][0].setdefault("nodes", []).append(los_i)

    samplers = anims[0]["samplers"]
    channels = anims[0]["channels"]
    typ = {"translation": "VEC3", "rotation": "VEC4", "scale": "VEC3"}
    for path_name, arr in (("translation", trans), ("rotation", rot), ("scale", scale)):
        acc = _accessor(
            doc, blob, np.ascontiguousarray(arr.astype(np.float32)), typ[path_name], 5126
        )
        si = len(samplers)
        samplers.append({"input": t_acc, "output": acc, "interpolation": "LINEAR"})
        channels.append({"sampler": si, "target": {"node": los_i, "path": path_name}})
    save_glb(path, doc, blob)


def save_glb(path: Path, doc: dict, blob: bytes | bytearray) -> None:
    blob = bytes(blob)
    pad = (-len(blob)) % 4
    blob += b"\x00" * pad
    raw_json = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    raw_json += b" " * ((-len(raw_json)) % 4)
    total = 12 + 8 + len(raw_json) + 8 + len(blob)
    out = bytearray(b"glTF")
    out += struct.pack("<I", 2)
    out += struct.pack("<I", total)
    out += struct.pack("<II", len(raw_json), 0x4E4F534A)
    out += raw_json
    out += struct.pack("<II", len(blob), 0x004E4942)
    out += blob
    path.write_bytes(bytes(out))


def _accessor(
    doc: dict, blob: bytearray, array: np.ndarray, typ: str, component: int, minmax: bool = True
) -> int:
    arr = np.ascontiguousarray(array)
    offset = len(blob)
    blob += arr.tobytes()
    pad = (-len(blob)) % 4
    blob += b"\x00" * pad
    views = doc.setdefault("bufferViews", [])
    view_i = len(views)
    views.append({"buffer": 0, "byteOffset": offset, "byteLength": arr.nbytes})
    accessors = doc.setdefault("accessors", [])
    acc_i = len(accessors)
    acc: dict = {
        "bufferView": view_i,
        "componentType": component,
        "count": int(arr.shape[0]),
        "type": typ,
    }
    if minmax:
        flat = arr.reshape(arr.shape[0], -1)
        acc["min"] = flat.min(axis=0).astype(float).tolist()
        acc["max"] = flat.max(axis=0).astype(float).tolist()
    accessors.append(acc)
    doc["buffers"][0]["byteLength"] = len(blob)
    return acc_i


def inject_animation(path: Path, tracks: list[dict], times: np.ndarray) -> None:
    """Append a glTF animation targeting named nodes already in the file."""
    doc, blob = load_glb(path)
    nodes = doc.get("nodes", [])
    name_to_i = {n.get("name"): i for i, n in enumerate(nodes) if "name" in n}
    times = np.ascontiguousarray(times.astype(np.float32))
    t_acc = _accessor(doc, blob, times, "SCALAR", 5126)

    samplers = []
    channels = []
    for track in tracks:
        ni = name_to_i.get(track["name"])
        if ni is None:
            print(f"WARNING: node {track['name']!r} missing from GLB, skip animation")
            continue
        tr = np.ascontiguousarray(track["translation"].astype(np.float32))
        rot = np.ascontiguousarray(track["rotation"].astype(np.float32))
        sc = np.ascontiguousarray(track["scale"].astype(np.float32))
        for path_name, acc in (
            ("translation", _accessor(doc, blob, tr, "VEC3", 5126)),
            ("rotation", _accessor(doc, blob, rot, "VEC4", 5126)),
            ("scale", _accessor(doc, blob, sc, "VEC3", 5126)),
        ):
            si = len(samplers)
            samplers.append({"input": t_acc, "output": acc, "interpolation": "LINEAR"})
            channels.append({"sampler": si, "target": {"node": ni, "path": path_name}})
        # Animated nodes may only use TRS; a leftover matrix would win in some viewers.
        nodes[ni].pop("matrix", None)
        nodes[ni]["translation"] = tr[0].tolist()
        nodes[ni]["rotation"] = rot[0].tolist()
        nodes[ni]["scale"] = sc[0].tolist()

    doc["animations"] = [{"name": "clip", "samplers": samplers, "channels": channels}]
    save_glb(path, doc, blob)


def _png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGBA").save(buf, format="PNG")
    return buf.getvalue()


def write_rigged_clip(path: Path, actors: list[dict], times: np.ndarray) -> None:
    """Append skinned armatures + a clip animation onto a terrain-only GLB."""
    doc, blob = load_glb(path)
    blob = bytearray(blob)
    nodes = doc.setdefault("nodes", [])
    meshes = doc.setdefault("meshes", [])
    skins = doc.setdefault("skins", [])
    images = doc.setdefault("images", [])
    textures = doc.setdefault("textures", [])
    materials = doc.setdefault("materials", [])
    scene_roots = doc["scenes"][0].setdefault("nodes", [])

    times = np.ascontiguousarray(times.astype(np.float32))
    t_acc = _accessor(doc, blob, times, "SCALAR", 5126)
    samplers: list[dict] = []
    channels: list[dict] = []

    def add_trs_track(node_i: int, tr, rot, sc):
        tr = np.ascontiguousarray(tr.astype(np.float32))
        rot = np.ascontiguousarray(rot.astype(np.float32))
        sc = np.ascontiguousarray(sc.astype(np.float32))
        for path_name, acc in (
            ("translation", _accessor(doc, blob, tr, "VEC3", 5126)),
            ("rotation", _accessor(doc, blob, rot, "VEC4", 5126)),
            ("scale", _accessor(doc, blob, sc, "VEC3", 5126)),
        ):
            si = len(samplers)
            samplers.append({"input": t_acc, "output": acc, "interpolation": "LINEAR"})
            channels.append({"sampler": si, "target": {"node": node_i, "path": path_name}})
        nodes[node_i].pop("matrix", None)
        nodes[node_i]["translation"] = tr[0].tolist()
        nodes[node_i]["rotation"] = rot[0].tolist()
        nodes[node_i]["scale"] = sc[0].tolist()

    for actor in actors:
        rig = actor["rig"]
        n_b3d = len(rig["nodes"])
        agent_i = len(nodes)
        nodes.append({"name": actor["name"], "children": []})
        scene_roots.append(agent_i)

        b3d_index = []
        for bi, bnode in enumerate(rig["nodes"]):
            gi = len(nodes)
            b3d_index.append(gi)
            nodes.append({"name": f"{actor['name']}::{bi}_{bnode['name']}"})
        for bi, bnode in enumerate(rig["nodes"]):
            gi = b3d_index[bi]
            parent = bnode["parent"]
            if parent < 0:
                nodes[agent_i].setdefault("children", []).append(gi)
            else:
                nodes[b3d_index[parent]].setdefault("children", []).append(gi)
        for node in nodes[agent_i:]:
            if not node.get("children"):
                node.pop("children", None)

        has_weights = any(n["weights"] for n in rig["nodes"])
        skin_i = None
        if has_weights:
            ibm = np.stack(
                [
                    np.linalg.inv(mt_to_gltf_matrix(bnode["bind_world"])).T.reshape(16)
                    for bnode in rig["nodes"]
                ]
            ).astype(np.float32)
            ibm_acc = _accessor(doc, blob, ibm, "MAT4", 5126, minmax=False)
            skin_i = len(skins)
            skins.append(
                {
                    "name": actor["name"] + "_skin",
                    "joints": b3d_index,
                    "inverseBindMatrices": ibm_acc,
                    "skeleton": b3d_index[0],
                }
            )

        tex_cache: dict[str, int] = {}
        for mi, mesh in enumerate(rig["meshes"]):
            host = rig["nodes"][mesh["node"]]
            v_local = np.asarray(mesh["vertices"], dtype=np.float64)
            ones = np.ones((len(v_local), 1))
            v_model = (host["bind_world"] @ np.hstack([v_local, ones]).T).T[:, :3]
            v_gltf = mt_vec_to_gltf(v_model).astype(np.float32)
            uv = to_gltf_uv(np.asarray(mesh["uv"], dtype=np.float64)).astype(np.float32)
            jnt, wts = skin_weights(rig, mi)
            pos_acc = _accessor(doc, blob, v_gltf, "VEC3", 5126)
            uv_acc = _accessor(doc, blob, uv, "VEC2", 5126)
            attrs = {"POSITION": pos_acc, "TEXCOORD_0": uv_acc}
            if has_weights:
                attrs["JOINTS_0"] = _accessor(
                    doc, blob, np.ascontiguousarray(jnt), "VEC4", 5123, minmax=False
                )
                attrs["WEIGHTS_0"] = _accessor(
                    doc, blob, np.ascontiguousarray(wts), "VEC4", 5126, minmax=False
                )
            primitives = []
            specs = actor["specs"]
            for si, (_brush, faces) in enumerate(mesh["surfaces"]):
                spec = specs[si] if si < len(specs) else specs[-1]
                if any(empty in spec for empty in EMPTY_TEXTURES):
                    continue
                faces = np.asarray(faces, dtype=np.int32)
                faces = faces[:, ::-1]  # Z-flip winding
                idx_acc = _accessor(
                    doc, blob, np.ascontiguousarray(faces.reshape(-1).astype(np.uint32)),
                    "SCALAR", 5125, minmax=False,
                )
                if spec not in tex_cache:
                    img = actor["library"].get(spec)
                    png = _png_bytes(img)
                    off = len(blob)
                    blob += png
                    pad = (-len(blob)) % 4
                    blob += b"\x00" * pad
                    views = doc.setdefault("bufferViews", [])
                    view_i = len(views)
                    views.append({"buffer": 0, "byteOffset": off, "byteLength": len(png)})
                    img_i = len(images)
                    images.append({"bufferView": view_i, "mimeType": "image/png"})
                    tex_i = len(textures)
                    textures.append({"source": img_i})
                    tex_cache[spec] = tex_i
                    doc["buffers"][0]["byteLength"] = len(blob)
                mat_i = len(materials)
                materials.append(
                    {
                        "name": f"{actor['name']}_m{mi}_{si}",
                        "pbrMetallicRoughness": {
                            "baseColorTexture": {"index": tex_cache[spec]},
                            "baseColorFactor": [1, 1, 1, 1],
                            "metallicFactor": 0.0,
                            "roughnessFactor": 1.0,
                        },
                        "alphaMode": "MASK",
                        "alphaCutoff": 0.5,
                        "doubleSided": True,
                    }
                )
                prim = {"attributes": attrs, "indices": idx_acc, "material": mat_i, "mode": 4}
                if has_weights and skin_i is not None:
                    pass
                primitives.append(prim)
            if not primitives:
                continue
            mesh_i = len(meshes)
            mesh_def = {"name": f"{actor['name']}_mesh{mi}", "primitives": primitives}
            meshes.append(mesh_def)
            mesh_node = nodes[b3d_index[mesh["node"]]]
            mesh_node["mesh"] = mesh_i
            if has_weights and skin_i is not None:
                mesh_node["skin"] = skin_i

        add_trs_track(agent_i, actor["root_t"], actor["root_r"], actor["root_s"])
        for bi in range(n_b3d):
            add_trs_track(
                b3d_index[bi],
                actor["joint_t"][bi],
                actor["joint_r"][bi],
                actor["joint_s"][bi],
            )

    if skins:
        doc["skins"] = skins
    doc["animations"] = [{"name": "clip", "samplers": samplers, "channels": channels}]
    doc["buffers"][0]["byteLength"] = len(blob)
    save_glb(path, doc, blob)


def video_fps(seed_dir: Path, fallback: float = 24.0) -> float:
    """Read fps from `rgb.mp4` so the clip matches the recorded video."""
    mp4 = seed_dir / "rgb.mp4"
    if not mp4.exists():
        return fallback
    try:
        import imageio.v2 as iio

        reader = iio.get_reader(mp4)
        fps = float(reader.get_meta_data().get("fps") or fallback)
        reader.close()
        return fps if fps > 0 else fallback
    except Exception:
        return fallback


def clip_times(n: int, data, frames: np.ndarray, seed_dir: Path, clock: str) -> tuple[np.ndarray, str]:
    """Keyframe times in seconds. `video` matches rgb.mp4; `engine` uses dt_minetest."""
    if clock == "engine":
        dt = np.asarray(data["dt_minetest"][frames], dtype=np.float64)
        times = np.zeros(n, dtype=np.float32)
        if n > 1:
            times[1:] = np.cumsum(dt[:-1])
        return times, f"engine dt_minetest ({float(times[-1]):.2f} s)"
    fps = video_fps(seed_dir)
    times = (np.arange(n, dtype=np.float32) / fps)
    return times, f"video {fps:g} fps ({float(times[-1]):.2f} s, rgb.mp4)"


def sample_rig_clip(rig: dict, anim_frames: np.ndarray, overrides: list[dict | None]):
    """Sample every joint's local glTF TRS at each clip step."""
    n_nodes = len(rig["nodes"])
    n = len(anim_frames)
    jt = np.zeros((n_nodes, n, 3), dtype=np.float32)
    jr = np.zeros((n_nodes, n, 4), dtype=np.float32)
    js = np.ones((n_nodes, n, 3), dtype=np.float32)
    for bi, node in enumerate(rig["nodes"]):
        for i in range(n):
            ov = None
            if overrides[i]:
                ov = overrides[i].get(node["name"])
            pos, sc, quat = sample_local_pose(node, float(anim_frames[i]), ov)
            t, s, r = trs_mt_to_gltf(pos, sc, quat)
            jt[bi, i], js[bi, i], jr[bi, i] = t, s, r
    return jt, jr, js


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_dir", default="datasets", type=Path)
    p.add_argument("--dataset_name", default="geval1")
    p.add_argument("--env_id", default="OpenWorldCreative-v0")
    p.add_argument("--seed", default="794921487")
    p.add_argument("--start", default=0, type=int)
    p.add_argument(
        "--frames",
        default=None,
        type=int,
        help="Number of frames. Default: the entire capture (matches rgb.mp4).",
    )
    p.add_argument("--out", default="geval1_794921487_f0-49.glb", type=Path)
    p.add_argument("--game_dir", default=DEFAULT_GAME_DIR, type=Path)
    p.add_argument(
        "--clock",
        choices=("video", "engine"),
        default="video",
        help="video: one frame per rgb.mp4 tick (24 fps here). engine: dt_minetest wall clock.",
    )
    p.add_argument("--snap_to_ground", action="store_true")
    p.add_argument(
        "--highlight_player",
        action="store_true",
        help="Tint the player magenta and add a ring at their feet.",
    )
    p.add_argument(
        "--show_los",
        action="store_true",
        help="Draw an animated ray along the player's recorded look direction (cam_dir).",
    )
    p.add_argument(
        "--los_length",
        type=float,
        default=LOS_DEFAULT_LENGTH,
        help="Length of the look ray in blocks (default 10).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_dir / args.dataset_name
    params_path = dataset_root / "dataset_params.json"
    if not params_path.exists():
        raise FileNotFoundError(params_path)
    with params_path.open() as f:
        params = json.load(f)
    vox_info = params.get("minetest_voxel_info") or params.get("voxel_info") or {}
    origin_idx = np.asarray(vox_info.get("origin_idx", [24, 24, 24]), dtype=np.float64)

    seed_dir = find_seed_dir(dataset_root / "raw" / args.env_id, str(args.seed))
    data = np.load(seed_dir / "data.npz", allow_pickle=True)
    dyn = np.load(seed_dir / "data_dynamic.npz", allow_pickle=True)

    t0 = int(args.start)
    t_max = int(data["obs_voxel_mt"].shape[0])
    n = int(args.frames) if args.frames is not None else t_max - t0
    if n <= 0 or t0 < 0 or t0 + n > t_max:
        raise IndexError(f"frames [{t0}, {t0 + n}) out of range [0, {t_max})")
    frames = np.arange(t0, t0 + n)

    times, clock_label = clip_times(n, data, frames, seed_dir, args.clock)

    print(f"union {n} voxel frames...", flush=True)
    grid, packed_center, packed_origin, lo, hi = union_voxel_grid(data, frames, origin_idx)
    print(f"terrain bbox {lo.tolist()} .. {hi.tolist()}  {tuple(int(x) for x in (hi - lo + 1))}", flush=True)
    # Scene origin: voxel-grid centre of the first frame, same convention as a still.
    origin_enu = np.asarray(data["obs_voxel_center"][t0], dtype=np.float64)

    library = TextureLibrary(args.game_dir)
    if not library.files:
        raise FileNotFoundError(f"no textures under {args.game_dir}")

    # Ground ids from every entity that appears in the clip.
    feet = []
    for t in frames:
        present = np.asarray(dyn["dyn_present"][t]) == 1
        pos = np.asarray(dyn["dyn_pos"][t], dtype=np.float64)
        if present.any():
            feet.append(pos[present])
        if "dyn_player_present" in dyn.files and int(dyn["dyn_player_present"][t]) == 1:
            feet.append(np.asarray(dyn["dyn_player_pos"][t], dtype=np.float64)[None, :])
    feet_arr = np.concatenate(feet) if feet else origin_enu[None, :]
    # Classify against the first frame's grid (close enough for material IDs).
    ids0 = np.asarray(data["obs_voxel_mt"][t0, ..., 0])
    occ0 = occupied_mask(ids0)
    center0 = np.asarray(data["obs_voxel_center"][t0], dtype=np.float64)
    ground_ids = support_node_ids(feet_arr, ids0, occ0, center0, origin_idx)

    terrain, tstats = terrain_meshes(
        grid,
        packed_center,
        packed_origin,
        origin_enu,
        library,
        full_volume=False,
        open_cut=False,
        ground_ids=ground_ids,
    )

    scene = trimesh.Scene()
    for name, mesh in terrain:
        scene.add_geometry(mesh, node_name=name)

    names = [str(x) for x in np.atleast_1d(dyn["dyn_names"])]
    n_slots = int(dyn["dyn_present"].shape[1])
    present_all = np.asarray(dyn["dyn_present"][frames]).astype(np.int8)
    pos_all = np.asarray(dyn["dyn_pos"][frames], dtype=np.float64)
    yaw_all = np.asarray(dyn["dyn_yaw"][frames], dtype=np.float64)
    baby_all = (
        np.asarray(dyn["dyn_baby"][frames]).astype(np.int8)
        if "dyn_baby" in dyn.files
        else np.zeros_like(present_all)
    )
    sheared_all = (
        np.asarray(dyn["dyn_sheared"][frames]).astype(np.int8)
        if "dyn_sheared" in dyn.files
        else np.zeros_like(present_all)
    )
    colors = dyn["dyn_color"] if "dyn_color" in dyn.files else None
    heights = collisionbox_heights(dyn["dyn_collisionbox"][t0], SHEEP_HEIGHT)
    dt_engine = np.asarray(data["dt_minetest"][frames], dtype=np.float64)
    engine_cum = np.cumsum(dt_engine)
    anim_range = (
        np.asarray(dyn["dyn_anim_range"][frames], dtype=np.float64)
        if "dyn_anim_range" in dyn.files
        else np.zeros((n, n_slots, 2))
    )
    anim_speed = (
        np.asarray(dyn["dyn_anim_speed"][frames], dtype=np.float64)
        if "dyn_anim_speed" in dyn.files
        else np.zeros((n, n_slots))
    )
    bone_names = (
        [str(x) for x in np.atleast_1d(dyn["dyn_bone_names"])]
        if "dyn_bone_names" in dyn.files
        else []
    )
    bone_rot = dyn["dyn_bone_rot"][frames] if bone_names else None
    bone_present = dyn["dyn_bone_present"][frames] if bone_names else None

    rig_cache: dict[str, dict | None] = {}
    actors: list[dict] = []
    drawn = {}
    unknown = {}
    missing = {}

    model_index: dict[str, Path] = {}
    for p in args.game_dir.rglob("*.b3d"):
        model_index.setdefault(p.name, p)

    def get_rig(filename: str):
        if filename not in rig_cache:
            path = model_index.get(filename) or find_model(args.game_dir, filename)
            try:
                rig_cache[filename] = load_b3d_rig(path) if path is not None else None
            except Exception as exc:
                print(f"WARNING: failed to load rig {filename}: {exc}")
                rig_cache[filename] = None
        return rig_cache[filename]

    def slot_overrides(slot: int) -> list[dict | None]:
        out: list[dict | None] = []
        for i in range(n):
            ov: dict | None = None
            if bone_present is not None:
                for b, bname in enumerate(bone_names):
                    if int(bone_present[i, slot, b]) != 1:
                        continue
                    if ov is None:
                        ov = {}
                    ov[bname] = np.asarray(bone_rot[i, slot, b], dtype=np.float64)
            out.append(ov)
        return out

    def root_tracks(pos_seq, yaw_seq, present_seq, uniform_scale: float, height: float):
        trans = np.zeros((n, 3), dtype=np.float32)
        yaws = np.zeros(n, dtype=np.float64)
        scales = np.ones((n, 3), dtype=np.float32) * uniform_scale
        last_pos = np.asarray(pos_seq[0], dtype=np.float64).copy()
        last_yaw = float(yaw_seq[0])
        for i, t in enumerate(frames):
            if int(present_seq[i]) == 1:
                p = np.asarray(pos_seq[i], dtype=np.float64).copy()
                if args.snap_to_ground:
                    occ = occupied_mask(np.asarray(data["obs_voxel_mt"][t, ..., 0]))
                    vc = np.asarray(data["obs_voxel_center"][t], dtype=np.float64)
                    delta, depth = ground_offset(p, occ, vc, origin_idx, height=height)
                    if depth > BURIAL_TOLERANCE:
                        p[2] += delta
                last_pos = p
                last_yaw = float(yaw_seq[i]) + MODEL_FACES_YAW
            else:
                scales[i] = 0.0
            trans[i] = enu_translation(last_pos, origin_enu)
            yaws[i] = last_yaw
        return trans, yaw_to_quat(yaws), scales

    for slot in range(n_slots):
        if not present_all[:, slot].any():
            continue
        species = names[slot] if slot < len(names) else ""
        entry = MOB_MODELS.get(species)
        if entry is None and ":" not in species:
            entry = MOB_MODELS.get(f"mobs_mc:{species}")
        if entry is None:
            unknown[species] = unknown.get(species, 0) + 1
            continue
        mesh_file, specs, visual_size = entry
        if species == "mobs_mc:sheep":
            col = str(colors[t0, slot]) if colors is not None else ""
            specs = sheep_texture_specs(col, bool(sheared_all[0, slot]))
        rig = get_rig(mesh_file)
        if rig is None:
            missing[mesh_file] = missing.get(mesh_file, 0) + 1
            continue
        scale = species_scale(species, visual_size, bool(baby_all[0, slot]))
        parent = f"agent_{slot}_{species.split(':')[-1]}"
        aframes = np.array(
            [
                anim_frame(anim_range[i, slot, 0], anim_range[i, slot, 1], anim_speed[i, slot], engine_cum[i])
                for i in range(n)
            ],
            dtype=np.float64,
        )
        jt, jr, js = sample_rig_clip(rig, aframes, slot_overrides(slot))
        rt, rr, rs = root_tracks(
            pos_all[:, slot], yaw_all[:, slot], present_all[:, slot], scale, float(heights[slot])
        )
        actors.append(
            {
                "name": parent,
                "rig": rig,
                "specs": specs,
                "library": library,
                "root_t": rt,
                "root_r": rr,
                "root_s": rs,
                "joint_t": jt,
                "joint_r": jr,
                "joint_s": js,
            }
        )
        drawn[species] = drawn.get(species, 0) + 1

    if "dyn_player_present" in dyn.files and np.any(dyn["dyn_player_present"][frames] == 1):
        mesh_name = str(dyn["dyn_player_mesh"]) if "dyn_player_mesh" in dyn.files else PLAYER_MODEL
        if not mesh_name:
            mesh_name = PLAYER_MODEL
        prig = get_rig(mesh_name) or get_rig(PLAYER_MODEL)
        specs = [str(s) for s in np.atleast_1d(dyn["dyn_player_textures"])] if prig else []
        if prig is not None:
            p_range = (
                np.asarray(dyn["dyn_player_anim_range"][frames], dtype=np.float64)
                if "dyn_player_anim_range" in dyn.files
                else np.zeros((n, 2))
            )
            p_speed = (
                np.asarray(dyn["dyn_player_anim_speed"][frames], dtype=np.float64)
                if "dyn_player_anim_speed" in dyn.files
                else np.zeros(n)
            )
            p_bone_names = (
                [str(x) for x in np.atleast_1d(dyn["dyn_player_bone_names"])]
                if "dyn_player_bone_names" in dyn.files
                else []
            )
            p_overrides: list[dict | None] = []
            for i in range(n):
                ov = None
                if p_bone_names and "dyn_player_bone_present" in dyn.files:
                    present = dyn["dyn_player_bone_present"][frames[i]]
                    rot = dyn["dyn_player_bone_rot"][frames[i]]
                    for b, bname in enumerate(p_bone_names):
                        if int(present[b]) != 1:
                            continue
                        if ov is None:
                            ov = {}
                        ov[bname] = np.asarray(rot[b], dtype=np.float64)
                p_overrides.append(ov)
            aframes = np.array(
                [anim_frame(p_range[i, 0], p_range[i, 1], p_speed[i], engine_cum[i]) for i in range(n)],
                dtype=np.float64,
            )
            jt, jr, js = sample_rig_clip(prig, aframes, p_overrides)
            pheight = float(
                collisionbox_heights(dyn["dyn_player_collisionbox"][t0], PLAYER_HEIGHT)[0]
            )
            p_present = np.asarray(dyn["dyn_player_present"][frames])
            p_pos = np.asarray(dyn["dyn_player_pos"][frames], dtype=np.float64)
            p_yaw = np.asarray(dyn["dyn_player_rotation"][frames, 1], dtype=np.float64)
            rt, rr, rs = root_tracks(p_pos, p_yaw, p_present, 1.0 / B3D_UNITS_PER_NODE, pheight)
            actors.append(
                {
                    "name": "player",
                    "rig": prig,
                    "specs": specs,
                    "library": library,
                    "root_t": rt,
                    "root_r": rr,
                    "root_s": rs,
                    "joint_t": jt,
                    "joint_r": jr,
                    "joint_s": js,
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out_path)
    write_rigged_clip(out_path, actors, times)
    if args.highlight_player:
        n_hl = highlight_player_in_glb(out_path)
        print(f"player highlight:  {n_hl} material(s) tinted + feet ring")
    if args.show_los:
        if "cam_pos" not in data.files or "cam_dir" not in data.files:
            print("WARNING: no cam_pos/cam_dir in data.npz, skip look ray")
        else:
            add_player_los_in_glb(
                out_path,
                np.asarray(data["cam_pos"][frames], dtype=np.float64),
                np.asarray(data["cam_dir"][frames], dtype=np.float64),
                origin_enu,
                length=float(args.los_length),
            )
            print(f"player look ray:   {float(args.los_length):g} blocks along cam_dir")
    patch_nearest_filtering(out_path)

    print(f"seed:              {seed_dir.name}")
    print(f"frames:            {t0} .. {t0 + n - 1}  ({n} steps)")
    print(f"clock:             {clock_label}")
    print(f"terrain union:     world cells {lo.tolist()} .. {hi.tolist()}")
    print(f"occupied voxels:   {tstats['occupied']}")
    print(f"drawn block faces: {tstats['faces']}")
    print("species drawn:     " + ", ".join(f"{k.split(':')[-1]}={v}" for k, v in drawn.items()))
    for species, c in unknown.items():
        print(f"WARNING: {c} slot(s) of unknown species {species!r} were not drawn")
    for mesh_file, c in missing.items():
        print(f"WARNING: model {mesh_file} not found ({c} slot(s))")
    print(f"rigged actors:     {len(actors)}")
    print(f"output:            {out_path.resolve()}")
    if library.missing:
        print(f"WARNING: missing textures: {sorted(library.missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
