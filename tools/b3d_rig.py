#!/usr/bin/env python3
"""Load a rigged .b3d (armature + KEYS) and sample Minetest-style clip poses.

Vertices stay in bind pose. Joint local TRS is sampled from KEYS, then optional
bone-override Eulers are composed on top. Used by `make_clip_glb.py` so the
exported GLB keeps a skin instead of a static posed mesh.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from make_frame_glb import _Chunks, _quat_matrix


def _local_matrix(pos, scale, quat_wxyz) -> np.ndarray:
    """T @ R @ S in Minetest/B3D axes. quat is w,x,y,z."""
    m = np.eye(4)
    m[:3, :3] = _quat_matrix(list(quat_wxyz)) * np.asarray(scale, dtype=np.float64)[None, :]
    m[:3, 3] = np.asarray(pos, dtype=np.float64)
    return m


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        n = np.linalg.norm(q)
        return q / n if n > 1e-12 else q0
    th = np.arccos(min(1.0, dot))
    s = np.sin(th)
    return (np.sin((1.0 - t) * th) * q0 + np.sin(t * th) * q1) / s


def euler_xyz_to_quat(xyz: np.ndarray) -> np.ndarray:
    """Minetest Euler XYZ (radians) -> wxyz quaternion."""
    x, y, z = (float(v) for v in xyz)
    cx, sx = np.cos(x * 0.5), np.sin(x * 0.5)
    cy, sy = np.cos(y * 0.5), np.sin(y * 0.5)
    cz, sz = np.cos(z * 0.5), np.sin(z * 0.5)
    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        dtype=np.float64,
    )


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, wxyz. Applies `b` first, then `a`."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def anim_frame(start: float, end: float, speed: float, cumulative_time: float) -> float:
    """Minetest clip clock: start + (t * speed) mod (length + 1)."""
    length = float(end) - float(start)
    if length <= 1e-6 or abs(speed) < 1e-12:
        return float(start)
    span = length + 1.0
    return float(start) + float(np.fmod(float(cumulative_time) * float(speed), span))


def load_b3d_rig(path: Path) -> dict:
    """Parse NODE / MESH / BONE / KEYS. Key frames are stored 0-based (Irrlicht)."""
    data = path.read_bytes()
    if data[:4] != b"BB3D":
        raise ValueError(f"{path} is not a BB3D file")
    total = struct.unpack_from("<i", data, 4)[0]

    nodes: list[dict] = []
    meshes: list[dict] = []

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

    def read_node(c: _Chunks, parent: int) -> int:
        name = c.s()
        pos = np.array(c.floats(3), dtype=np.float64)
        scale = np.array(c.floats(3), dtype=np.float64)
        quat = np.array(c.floats(4), dtype=np.float64)
        idx = len(nodes)
        node = {
            "name": name,
            "parent": parent,
            "bind_pos": pos,
            "bind_scale": scale,
            "bind_quat": quat,
            "mesh": None,
            "weights": [],  # (vertex_id, weight) into node['mesh']
            "pos_keys": {},
            "scale_keys": {},
            "rot_keys": {},
        }
        nodes.append(node)
        while c.more():
            tag, body = c.next()
            if tag == "MESH":
                body.i()
                verts = uv = None
                surfaces: list[tuple[int, np.ndarray]] = []
                while body.more():
                    mtag, mbody = body.next()
                    if mtag == "VRTS":
                        verts, uv = read_vrts(mbody)
                    elif mtag == "TRIS":
                        brush = mbody.i()
                        n = (mbody.end - mbody.p) // 12
                        faces = np.frombuffer(
                            mbody.d, dtype="<i4", count=n * 3, offset=mbody.p
                        ).reshape(n, 3)
                        surfaces.append((brush, faces))
                if verts is None:
                    continue
                mesh_i = len(meshes)
                meshes.append(
                    {
                        "node": idx,
                        "vertices": verts,
                        "uv": uv,
                        "surfaces": surfaces,
                    }
                )
                node["mesh"] = mesh_i
            elif tag == "BONE":
                n = (body.end - body.p) // 8
                raw = np.frombuffer(
                    body.d, dtype="<i4", count=n * 2, offset=body.p
                ).reshape(n, 2)
                vid = raw[:, 0]
                wfloat = np.frombuffer(raw[:, 1].tobytes(), dtype="<f4")
                node["weights"] = list(zip(vid.tolist(), wfloat.tolist()))
                body.p = body.end
            elif tag == "KEYS":
                flags = body.i()
                while body.p < body.end:
                    frame_1 = body.i()
                    frame = max(frame_1, 1) - 1  # Irrlicht 0-based
                    if flags & 1:
                        node["pos_keys"][frame] = np.array(body.floats(3), dtype=np.float64)
                    if flags & 2:
                        node["scale_keys"][frame] = np.array(body.floats(3), dtype=np.float64)
                    if flags & 4:
                        node["rot_keys"][frame] = np.array(body.floats(4), dtype=np.float64)
            elif tag == "ANIM":
                body.i()
                body.i()
                body.f()
            elif tag == "NODE":
                read_node(body, idx)
        return idx

    root = _Chunks(data, 12, 8 + total)
    while root.more():
        tag, body = root.next()
        if tag == "NODE":
            read_node(body, -1)

    if not nodes:
        raise ValueError(f"no nodes in {path}")

    # Bind worlds in Minetest axes.
    worlds = [np.eye(4) for _ in nodes]
    for i, node in enumerate(nodes):
        local = _local_matrix(node["bind_pos"], node["bind_scale"], node["bind_quat"])
        p = node["parent"]
        worlds[i] = (worlds[p] @ local) if p >= 0 else local
        node["bind_world"] = worlds[i]

    # Each BONE weights vertices of the nearest ancestor MESH.
    for i, node in enumerate(nodes):
        j = i
        mesh_i = None
        while j >= 0:
            if nodes[j]["mesh"] is not None:
                mesh_i = nodes[j]["mesh"]
                break
            j = nodes[j]["parent"]
        node["mesh_for_weights"] = mesh_i

    return {"nodes": nodes, "meshes": meshes, "path": str(path)}


def _interp_keys(keys: dict, frame: float, default: np.ndarray, slerp: bool = False):
    if not keys:
        return default.copy()
    frames = sorted(keys)
    if frame <= frames[0]:
        return keys[frames[0]].copy()
    if frame >= frames[-1]:
        return keys[frames[-1]].copy()
    lo = 0
    hi = len(frames) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if frames[mid] <= frame:
            lo = mid
        else:
            hi = mid
    f0, f1 = frames[lo], frames[hi]
    t = 0.0 if f1 == f0 else (frame - f0) / (f1 - f0)
    a, b = keys[f0], keys[f1]
    if slerp:
        return _slerp(a, b, t)
    return a * (1.0 - t) + b * t


def sample_local_pose(
    node: dict,
    frame: float,
    override_euler: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local pos/scale/quat (wxyz) at `frame`, with optional relative Euler override."""
    pos = _interp_keys(node["pos_keys"], frame, node["bind_pos"])
    scale = _interp_keys(node["scale_keys"], frame, node["bind_scale"])
    quat = _interp_keys(node["rot_keys"], frame, node["bind_quat"], slerp=True)
    if override_euler is not None:
        quat = quat_mul(euler_xyz_to_quat(override_euler), quat)
    return pos, scale, quat


def skin_weights(rig: dict, mesh_i: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-4 joint indices (B3D node index) and weights per vertex."""
    mesh = rig["meshes"][mesh_i]
    n = len(mesh["vertices"])
    acc: list[list[tuple[float, int]]] = [[] for _ in range(n)]
    for ni, node in enumerate(rig["nodes"]):
        if node["mesh_for_weights"] != mesh_i:
            continue
        for vid, w in node["weights"]:
            if 0 <= vid < n and w > 1e-8:
                acc[vid].append((float(w), ni))
    joints = np.zeros((n, 4), dtype=np.uint16)
    weights = np.zeros((n, 4), dtype=np.float32)
    for v, pairs in enumerate(acc):
        pairs.sort(reverse=True)
        pairs = pairs[:4]
        if not pairs:
            weights[v, 0] = 1.0
            continue
        tot = sum(p[0] for p in pairs) or 1.0
        for k, (w, ji) in enumerate(pairs):
            joints[v, k] = ji
            weights[v, k] = w / tot
    return joints, weights


def mt_to_gltf_matrix(m: np.ndarray) -> np.ndarray:
    """Minetest (x,y,z) -> glTF (x, y, -z)."""
    p = np.diag([1.0, 1.0, -1.0, 1.0])
    return p @ m @ p


def mt_vec_to_gltf(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    out = np.empty_like(v)
    out[..., 0] = v[..., 0]
    out[..., 1] = v[..., 1]
    out[..., 2] = -v[..., 2]
    return out


def quat_wxyz_to_gltf_xyzw(q: np.ndarray) -> np.ndarray:
    """Reflect rotation through Z (MT -> glTF) and emit xyzw."""
    w, x, y, z = (float(v) for v in q)
    # Axis (x,y,z) -> (x,y,-z); w unchanged.
    return np.array([x, y, -z, w], dtype=np.float32)


def trs_mt_to_gltf(
    pos: np.ndarray, scale: np.ndarray, quat_wxyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = mt_to_gltf_matrix(_local_matrix(pos, scale, quat_wxyz))
    t = m[:3, 3].astype(np.float32)
    sx = np.linalg.norm(m[:3, 0])
    sy = np.linalg.norm(m[:3, 1])
    sz = np.linalg.norm(m[:3, 2])
    s = np.array([max(sx, 1e-8), max(sy, 1e-8), max(sz, 1e-8)], dtype=np.float32)
    rotm = np.column_stack([m[:3, 0] / s[0], m[:3, 1] / s[1], m[:3, 2] / s[2]])
    # Ensure right-handed; if det < 0 flip Z column.
    if np.linalg.det(rotm) < 0:
        rotm[:, 2] *= -1
        s[2] *= -1
    # quat xyzw from rotation matrix
    tr = float(np.trace(rotm))
    if tr > 0:
        r = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * r
        qx = (rotm[2, 1] - rotm[1, 2]) / r
        qy = (rotm[0, 2] - rotm[2, 0]) / r
        qz = (rotm[1, 0] - rotm[0, 1]) / r
    elif rotm[0, 0] > rotm[1, 1] and rotm[0, 0] > rotm[2, 2]:
        r = np.sqrt(1.0 + rotm[0, 0] - rotm[1, 1] - rotm[2, 2]) * 2.0
        qw = (rotm[2, 1] - rotm[1, 2]) / r
        qx = 0.25 * r
        qy = (rotm[0, 1] + rotm[1, 0]) / r
        qz = (rotm[0, 2] + rotm[2, 0]) / r
    elif rotm[1, 1] > rotm[2, 2]:
        r = np.sqrt(1.0 + rotm[1, 1] - rotm[0, 0] - rotm[2, 2]) * 2.0
        qw = (rotm[0, 2] - rotm[2, 0]) / r
        qx = (rotm[0, 1] + rotm[1, 0]) / r
        qy = 0.25 * r
        qz = (rotm[1, 2] + rotm[2, 1]) / r
    else:
        r = np.sqrt(1.0 + rotm[2, 2] - rotm[0, 0] - rotm[1, 1]) * 2.0
        qw = (rotm[1, 0] - rotm[0, 1]) / r
        qx = (rotm[0, 2] + rotm[2, 0]) / r
        qy = (rotm[1, 2] + rotm[2, 1]) / r
        qz = 0.25 * r
    q = np.array([qx, qy, qz, qw], dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n > 1e-8:
        q /= n
    return t, np.abs(s), q
