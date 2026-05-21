# -*- coding: utf-8 -*-
"""ReplacePrimitive core logic.

메시 형태를 분석하여 적합한 USD 프리미티브(Cube/Cylinder)로 교체합니다.
원본 메시는 active=false 처리하여 복구 가능하게 유지합니다.

형태 감지 (PCA/OBB 기반 — 회전/중공 실린더 대응):
  1. flat     : 공면성 비율 <= flat_threshold
               → UsdGeom.Cube (얇은 슬랩)
  2. cylinder : PCA cross-section 등방성(eccentricity) >= cyl_threshold
               → UsdGeom.Cylinder (solid/hollow 실린더, 파이프 포함)
  3. box      : OBB 면 이탈 비율 <= box_threshold
               → UsdGeom.Cube (회전 포함)
  4. unknown  : 위 조건 불충족 → 스킵

검사 순서: flat → cylinder → box (cylinder 먼저 → 파이프가 box로 오감지 방지)

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
        cross  = np.cross(arr[i] - p0, arr[i + 1] - p0)
        length = float(np.linalg.norm(cross))
        if length > 1e-9:
            normal = cross / length
            break
    if normal is None:
        return None
    d0       = float(np.dot(normal, p0))
    max_dist = float(np.max(np.abs(arr @ normal - d0)))
    diag     = float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)))
    return max_dist / diag if diag > 1e-6 else None


# ──────────────────── PCA analysis (공통) ────────────────────────────────

def _pca(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """PCA 분석. 반환: (eigenvalues, vecs, projected) — 오름차순 정렬."""
    centroid = arr.mean(axis=0)
    centered = arr - centroid
    cov      = (centered.T @ centered) / len(arr)
    try:
        eigenvalues, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    projected = centered @ vecs
    return eigenvalues, vecs, projected


# ─────────────────── cylinder detector (eccentricity 기반) ───────────────

def _cyl_eccentricity(eigenvalues: np.ndarray) -> tuple[float, int]:
    """
    PCA 고유값으로 cross-section 원형 대칭성(등방성) 검사.

    원형 단면 = 두 cross-section 고유값이 거의 같음 = eccentricity ≈ 1.0
    파이프(중공 실린더)도 동일하게 감지.

    반환: (best_eccentricity, height_axis_idx_in_pca_space)
    """
    best_ecc  = 0.0
    best_axis = 0

    for height_idx in range(3):
        cross_idx = [i for i in range(3) if i != height_idx]
        e0 = eigenvalues[cross_idx[0]]
        e1 = eigenvalues[cross_idx[1]]
        if e0 < 1e-9 or e1 < 1e-9:
            continue
        ecc = min(e0, e1) / max(e0, e1)   # 1.0 = 완전 원형
        if ecc > best_ecc:
            best_ecc  = ecc
            best_axis = height_idx

    return best_ecc, best_axis


# ─────────────────────── OBB analysis (box 감지) ─────────────────────────

def _obb_analysis(arr: np.ndarray) -> dict | None:
    """
    PCA 기반 OBB 분석.
    반환: {ratio, vecs, lo, hi, center_world, dims}
    """
    result = _pca(arr)
    if result is None:
        return None
    eigenvalues, vecs, projected = result

    lo   = projected.min(axis=0)
    hi   = projected.max(axis=0)
    dims = hi - lo
    diag = float(np.linalg.norm(dims))
    if diag < 1e-6:
        return None

    max_d = 0.0
    for row in projected:
        d = min(
            abs(float(row[0]) - float(lo[0])), abs(float(row[0]) - float(hi[0])),
            abs(float(row[1]) - float(lo[1])), abs(float(row[1]) - float(hi[1])),
            abs(float(row[2]) - float(lo[2])), abs(float(row[2]) - float(hi[2])),
        )
        max_d = max(max_d, d)

    centroid     = arr.mean(axis=0)
    center_world = centroid + vecs @ ((lo + hi) / 2)

    return {
        "ratio":         max_d / diag,
        "eigenvalues":   eigenvalues,
        "vecs":          vecs,
        "lo":            lo,
        "hi":            hi,
        "center_world":  center_world,
        "dims":          dims,
    }


# ───────────────────────────── main detector ─────────────────────────────

def detect_shape(
    prim: Usd.Prim,
    flat_threshold: float = 0.005,
    box_threshold:  float = 0.01,
    cyl_threshold:  float = 0.8,
) -> tuple[str, dict]:
    """
    Mesh prim 형태를 PCA 기반으로 분석 (회전·중공 실린더 대응).

    Args:
        flat_threshold: 공면성 비율 임계값 (기본 0.005 = 0.5%)
        box_threshold:  OBB 면 이탈 비율 임계값 (기본 0.01 = 1%)
        cyl_threshold:  cross-section eccentricity 하한 (기본 0.8, 범위 0~1)
                        높을수록 엄격 (1.0 = 완전한 원만 감지)

    Returns: (shape_type, meta)
      shape_type: "flat" | "cylinder" | "box" | "unknown"
    """
    arr = _get_local_points(prim)
    if arr is None or len(arr) < 4:
        return "unknown", {}

    # 1. flat 검사
    fr = _flat_ratio(arr)
    if fr is not None and fr <= flat_threshold:
        lo, hi = arr.min(axis=0), arr.max(axis=0)
        return "flat", {"lo": lo, "hi": hi, "obb": False}

    # OBB 공통 분석
    obb = _obb_analysis(arr)
    if obb is None:
        return "unknown", {}

    # 2. cylinder 검사 (box 전에 먼저 — 파이프가 box로 오감지 방지)
    ecc, cyl_height_axis = _cyl_eccentricity(obb["eigenvalues"])
    if ecc >= cyl_threshold:
        return "cylinder", {**obb, "obb": True, "cyl_axis": cyl_height_axis}

    # 3. box 검사
    if obb["ratio"] <= box_threshold:
        return "box", {**obb, "obb": True}

    return "unknown", {}


# ─────────────────────── rotation matrix → quaternion ────────────────────

def _rot_to_quatd(R: np.ndarray) -> Gf.Quatd:
    """3x3 회전 행렬(numpy) → Gf.Quatd (Shepperd method)."""
    m  = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s;                 z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
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
    cyl_axis  = meta.get("cyl_axis", 0)   # PCA 공간에서의 높이 축 인덱스

    dims     = hi - lo
    height   = float(dims[cyl_axis])
    r_axes   = [i for i in range(3) if i != cyl_axis]

    # 외측 반지름: 두 cross-section 축의 max extent 평균
    radius = float((dims[r_axes[0]] + dims[r_axes[1]]) / 4)

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.GetHeightAttr().Set(height)
    cyl.GetRadiusAttr().Set(radius)
    # PCA 높이 축 → USD cylinder axis 매핑
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
    cyl_threshold:  float = 0.8,
    skip_paths:     list[str] | None = None,
    log:            Callable[[str], None] | None = None,
) -> tuple[int, int]:
    """
    stage 내 Mesh prim을 형태 분석하여 USD 프리미티브로 교체.

    Args:
        stage:          대상 USD Stage
        flat_threshold: 공면성 비율 임계값 (기본 0.005 = 0.5%)
        box_threshold:  OBB 면 이탈 비율 임계값 (기본 0.01 = 1%)
        cyl_threshold:  cross-section eccentricity 하한 (기본 0.8)
                        0.8 = 80% 이상 원형이면 실린더 감지 (파이프 포함)
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
