# -*- coding: utf-8 -*-
"""ReplacePrimitive core logic.

메시 형태를 분석하여 적합한 USD 프리미티브(Cube/Cylinder)로 교체합니다.
원본 메시는 active=false 처리하여 복구 가능하게 유지합니다.

형태 감지 (PCA/OBB 기반 — 회전된 메시 대응):
  1. flat     : 공면성 비율 <= flat_threshold  → UsdGeom.Cube (얇은 슬랩)
  2. box      : OBB 면 이탈 비율 <= box_threshold → UsdGeom.Cube (회전 포함)
  3. cylinder : PCA 축 기준 반지름 std/mean <= cyl_threshold → UsdGeom.Cylinder
  4. unknown  : 위 조건 불충족 → 스킵

교체 prim 경로: {원본경로}_prim
원본 메시:       active=false (복구 가능)
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Gf


# ──────────────────────────────── helpers ────────────────────────────────

def _get_local_points(prim: Usd.Prim) -> np.ndarray | None:
    mesh = UsdGeom.Mesh(prim)
    attr = mesh.GetPointsAttr()
    if not (attr and attr.HasValue()):
        return None
    pts = attr.Get()
    if not pts:
        return None
    return np.array([[float(p[0]), float(p[1]), float(p[2])] for p in pts])


# ──────────────────────────── flat detector ──────────────────────────────

def _flat_ratio(arr: np.ndarray) -> float | None:
    """공면성 비율. 낮을수록 flat에 가까움."""
    p0 = arr[0]
    normal = None
    for i in range(1, len(arr) - 1):
        v1 = arr[i]     - p0
        v2 = arr[i + 1] - p0
        cross  = np.cross(v1, v2)
        length = float(np.linalg.norm(cross))
        if length > 1e-9:
            normal = cross / length
            break
    if normal is None:
        return None
    d0       = float(np.dot(normal, p0))
    max_dist = float(np.max(np.abs(arr @ normal - d0)))
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    return max_dist / diag if diag > 1e-6 else None


# ─────────────────────── OBB analysis (PCA 기반) ─────────────────────────

def _obb_analysis(arr: np.ndarray) -> dict | None:
    """
    PCA로 주축 계산 → OBB 분석.
    반환: {ratio, vecs, lo, hi, center_world, projected}
      vecs    : (3,3) 행렬, 열 = 주축 (eigenvectors, ascending eigenvalue)
      lo/hi   : OBB 로컬 좌표계에서의 min/max
      center_world : 월드 좌표계에서의 OBB 중심
      projected    : (N,3) — 각 점의 OBB 로컬 좌표 투영값
    """
    centroid = arr.mean(axis=0)
    centered = arr - centroid

    cov        = (centered.T @ centered) / len(arr)
    _, vecs    = np.linalg.eigh(cov)          # 열 = 주축, 오름차순 정렬
    projected  = centered @ vecs              # OBB 로컬 공간 투영

    lo   = projected.min(axis=0)
    hi   = projected.max(axis=0)
    dims = hi - lo
    diag = float(np.linalg.norm(dims))

    if diag < 1e-6:
        return None

    # OBB 면 이탈 거리 (max of min-dist-to-face)
    max_d = 0.0
    for row in projected:
        d = min(
            abs(float(row[0]) - float(lo[0])), abs(float(row[0]) - float(hi[0])),
            abs(float(row[1]) - float(lo[1])), abs(float(row[1]) - float(hi[1])),
            abs(float(row[2]) - float(lo[2])), abs(float(row[2]) - float(hi[2])),
        )
        max_d = max(max_d, d)

    obb_center_local = (lo + hi) / 2
    center_world     = centroid + vecs @ obb_center_local

    return {
        "ratio":        max_d / diag,
        "vecs":         vecs,
        "lo":           lo,
        "hi":           hi,
        "center_world": center_world,
        "projected":    projected,
        "dims":         dims,
    }


# ─────────────────── cylinder detector (PCA 기반) ────────────────────────

def _cyl_pca_ratio(projected: np.ndarray) -> tuple[float, int]:
    """
    OBB 로컬 공간에서 각 축을 실린더 높이 축으로 가정,
    단면 반지름 std/mean으로 실린더 적합도 계산.
    반환: (best_ratio, best_axis_in_obb_space)
    """
    best_ratio = float("inf")
    best_axis  = 2

    for axis in range(3):
        r0_idx, r1_idx = [i for i in range(3) if i != axis]
        radii  = np.sqrt(projected[:, r0_idx] ** 2 + projected[:, r1_idx] ** 2)
        mean_r = float(radii.mean())
        if mean_r < 1e-9:
            continue
        std_r = float(radii.std())
        ratio = std_r / mean_r
        if ratio < best_ratio:
            best_ratio = ratio
            best_axis  = axis

    return best_ratio, best_axis


# ───────────────────────────── main detector ─────────────────────────────

def detect_shape(
    prim: Usd.Prim,
    flat_threshold: float = 0.005,
    box_threshold:  float = 0.01,
    cyl_threshold:  float = 0.05,
) -> tuple[str, dict]:
    """
    Mesh prim의 형태를 PCA/OBB 기반으로 분석 (회전된 메시 대응).
    Returns: (shape_type, meta)
      shape_type: "flat" | "box" | "cylinder" | "unknown"
    """
    arr = _get_local_points(prim)
    if arr is None or len(arr) < 4:
        return "unknown", {}

    # 1. flat 검사 (공면성)
    fr = _flat_ratio(arr)
    if fr is not None and fr <= flat_threshold:
        lo = arr.min(axis=0)
        hi = arr.max(axis=0)
        return "flat", {"lo": lo, "hi": hi, "obb": False}

    # 2. OBB 분석
    obb = _obb_analysis(arr)
    if obb is None:
        return "unknown", {}

    # 3. box 검사
    if obb["ratio"] <= box_threshold:
        return "box", {**obb, "obb": True}

    # 4. cylinder 검사 (PCA 공간)
    cyl_r, cyl_axis = _cyl_pca_ratio(obb["projected"])
    if cyl_r <= cyl_threshold:
        return "cylinder", {**obb, "obb": True, "cyl_axis": cyl_axis}

    return "unknown", {}


# ─────────────────────── rotation matrix → quaternion ────────────────────

def _rot_to_quatd(R: np.ndarray) -> Gf.Quatd:
    """3x3 회전 행렬(numpy) → Gf.Quatd (Shepperd method)."""
    m  = R
    tr = m[0,0] + m[1,1] + m[2,2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m[2,1] - m[1,2]) * s
        y = (m[0,2] - m[2,0]) * s
        z = (m[1,0] - m[0,1]) * s
    elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
        s = 2.0 * math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
        w = (m[2,1] - m[1,2]) / s
        x = 0.25 * s
        y = (m[0,1] + m[1,0]) / s
        z = (m[0,2] + m[2,0]) / s
    elif m[1,1] > m[2,2]:
        s = 2.0 * math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
        w = (m[0,2] - m[2,0]) / s
        x = (m[0,1] + m[1,0]) / s
        y = 0.25 * s
        z = (m[1,2] + m[2,1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
        w = (m[1,0] - m[0,1]) / s
        x = (m[0,2] + m[2,0]) / s
        y = (m[1,2] + m[2,1]) / s
        z = 0.25 * s
    return Gf.Quatd(w, Gf.Vec3d(x, y, z))


# ─────────────────────────── primitive factory ───────────────────────────

def _create_cube(stage: Usd.Stage, path: str, meta: dict) -> Usd.Prim:
    if meta.get("obb"):
        vecs   = meta["vecs"]
        lo     = meta["lo"]
        hi     = meta["hi"]
        center = meta["center_world"]
        dims   = hi - lo
        sx, sy, sz = (dims / 2).tolist()

        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(2.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        cube.AddOrientOp().Set(_rot_to_quatd(vecs))
        cube.AddScaleOp().Set(Gf.Vec3f(float(sx), float(sy), float(sz)))
    else:
        lo = meta["lo"]
        hi = meta["hi"]
        cx, cy, cz = ((lo + hi) / 2).tolist()
        sx, sy, sz = ((hi - lo) / 2).tolist()
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(2.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(float(cx), float(cy), float(cz)))
        cube.AddScaleOp().Set(Gf.Vec3f(float(sx), float(sy), float(sz)))
    return cube.GetPrim()


def _create_cylinder(stage: Usd.Stage, path: str, meta: dict) -> Usd.Prim:
    vecs      = meta["vecs"]
    lo        = meta["lo"]
    hi        = meta["hi"]
    center    = meta["center_world"]
    cyl_axis  = meta.get("cyl_axis", 2)  # index in OBB (PCA) space

    dims      = hi - lo
    height    = float(dims[cyl_axis])
    r_axes    = [i for i in range(3) if i != cyl_axis]
    radius    = float((dims[r_axes[0]] + dims[r_axes[1]]) / 4)

    # 실린더 기본 축은 Z. OBB의 cyl_axis 주축이 Z와 얼마나 다른지 반영
    # 회전: OBB 기준으로 cyl_axis 번 주축 → world Z로 맞춤
    # 간단하게 전체 OBB rotation 적용 후 axis attribute 사용
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.GetHeightAttr().Set(height)
    cyl.GetRadiusAttr().Set(radius)
    # cyl_axis가 OBB 공간의 몇 번 축인지 → "X"/"Y"/"Z" 매핑
    # OBB 주축 정렬: eigenvalue ascending → 0=가장 분산 작음, 2=가장 큼
    cyl.GetAxisAttr().Set(("X", "Y", "Z")[cyl_axis])
    cyl.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.AddOrientOp().Set(_rot_to_quatd(vecs))
    return cyl.GetPrim()


def _copy_material(src: Usd.Prim, dst: Usd.Prim) -> None:
    binding = UsdShade.MaterialBindingAPI(src).GetDirectBinding()
    mat     = binding.GetMaterial()
    if mat:
        UsdShade.MaterialBindingAPI.Apply(dst).Bind(mat)


# ─────────────────────────────── main API ────────────────────────────────

def process_stage(
    stage: Usd.Stage,
    flat_threshold: float = 0.005,
    box_threshold:  float = 0.01,
    cyl_threshold:  float = 0.05,
    skip_paths:     list[str] | None = None,
    log:            Callable[[str], None] | None = None,
) -> tuple[int, int]:
    """
    stage 내 Mesh prim을 PCA/OBB 형태 분석하여 USD 프리미티브로 교체.
    회전된 메시(splitMeshes 이후 등)도 올바르게 감지.

    Args:
        stage:          대상 USD Stage
        flat_threshold: 공면성 비율 임계값 (기본 0.005 = 0.5%)
        box_threshold:  OBB 면 이탈 비율 임계값 (기본 0.01 = 1%)
        cyl_threshold:  실린더 반지름 편차 비율 임계값 (기본 0.05 = 5%)
        skip_paths:     처리 제외할 prim 경로 접두어 목록
        log:            로그 콜백

    Returns:
        (replaced_count, skipped_count)
    """
    _log       = log or (lambda msg: None)
    skip_paths = skip_paths or []

    meshes = []
    for prim in stage.Traverse():
        if not prim.IsActive():
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path_str = prim.GetPath().pathString
        if any(path_str.startswith(sp) for sp in skip_paths):
            continue
        meshes.append(prim)

    _log(f"{len(meshes)} mesh(es) found")

    replaced = skipped = 0
    counts   = {"flat": 0, "box": 0, "cylinder": 0, "unknown": 0}

    for prim in meshes:
        path_str    = prim.GetPath().pathString
        shape, meta = detect_shape(prim, flat_threshold, box_threshold, cyl_threshold)
        counts[shape] = counts.get(shape, 0) + 1

        if shape == "unknown":
            skipped += 1
            continue

        prim_path = path_str + "_prim"

        try:
            if shape == "cylinder":
                new_prim = _create_cylinder(stage, prim_path, meta)
            else:
                new_prim = _create_cube(stage, prim_path, meta)

            _copy_material(prim, new_prim)
            prim.SetActive(False)
            _log(f"[{shape.upper()}] {path_str} → {prim_path}")
            replaced += 1
        except Exception as e:
            _log(f"[WARN] {path_str}: {e}")
            skipped += 1

    _log(
        f"replaced: {replaced} / skipped: {skipped} "
        f"(flat={counts['flat']}, box={counts['box']}, "
        f"cylinder={counts['cylinder']}, unknown={counts['unknown']})"
    )
    return replaced, skipped
