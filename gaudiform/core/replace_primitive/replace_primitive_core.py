# -*- coding: utf-8 -*-
"""ReplacePrimitive core logic.

메시 형태를 분석하여 적합한 USD 프리미티브(Cube/Cylinder)로 교체합니다.
원본 메시는 active=false 처리하여 복구 가능하게 유지합니다.

형태 감지:
  1. flat     : 공면성 비율 <= flat_threshold  → UsdGeom.Cube (얇은 슬랩)
  2. box      : BBox 면 이탈 비율 <= box_threshold → UsdGeom.Cube
  3. cylinder : 반지름 std/mean <= cyl_threshold → UsdGeom.Cylinder
  4. unknown  : 위 조건 불충족 → 스킵

교체 prim 경로: {원본경로}_prim
원본 메시:       active=false (복구 가능)
"""

from __future__ import annotations

import math
from typing import Callable

from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt


# ──────────────────────────────── helpers ────────────────────────────────

def _get_local_points(prim: Usd.Prim) -> list[Gf.Vec3d] | None:
    mesh = UsdGeom.Mesh(prim)
    attr = mesh.GetPointsAttr()
    if not (attr and attr.HasValue()):
        return None
    pts = attr.Get()
    return [Gf.Vec3d(p) for p in pts] if pts else None


def _local_bbox(pts: list[Gf.Vec3d]) -> tuple[Gf.Vec3d, Gf.Vec3d]:
    lo = Gf.Vec3d(float("inf"),  float("inf"),  float("inf"))
    hi = Gf.Vec3d(float("-inf"), float("-inf"), float("-inf"))
    for p in pts:
        lo = Gf.Vec3d(min(lo[0], p[0]), min(lo[1], p[1]), min(lo[2], p[2]))
        hi = Gf.Vec3d(max(hi[0], p[0]), max(hi[1], p[1]), max(hi[2], p[2]))
    return lo, hi


def _diagonal(lo: Gf.Vec3d, hi: Gf.Vec3d) -> float:
    return math.sqrt(sum((hi[i] - lo[i]) ** 2 for i in range(3)))


# ──────────────────────────── shape detectors ────────────────────────────

def _flat_ratio(pts: list[Gf.Vec3d], diagonal: float) -> float | None:
    """공면성 비율. 낮을수록 flat에 가까움."""
    p0 = pts[0]
    normal = None
    for i in range(1, len(pts) - 1):
        v1    = pts[i]     - p0
        v2    = pts[i + 1] - p0
        cross = Gf.Cross(v1, v2)
        length = cross.GetLength()
        if length > 1e-9:
            normal = cross / length
            break
    if normal is None:
        return None
    d0       = Gf.Dot(normal, p0)
    max_dist = max(abs(Gf.Dot(normal, p) - d0) for p in pts)
    return max_dist / diagonal


def _box_ratio(pts: list[Gf.Vec3d], lo: Gf.Vec3d, hi: Gf.Vec3d, diagonal: float) -> float:
    """최대 BBox면 이탈 비율. 낮을수록 박스에 가까움."""
    max_d = 0.0
    for p in pts:
        d = min(
            abs(p[0] - lo[0]), abs(p[0] - hi[0]),
            abs(p[1] - lo[1]), abs(p[1] - hi[1]),
            abs(p[2] - lo[2]), abs(p[2] - hi[2]),
        )
        if d > max_d:
            max_d = d
    return max_d / diagonal


def _cylinder_ratio(
    pts: list[Gf.Vec3d],
    lo: Gf.Vec3d,
    hi: Gf.Vec3d,
) -> tuple[float, int]:
    """
    각 축을 실린더 축으로 가정할 때 반지름 std/mean 계산.
    반환: (best_ratio, best_axis)  axis: 0=X, 1=Y, 2=Z
    """
    best_ratio = float("inf")
    best_axis  = 2

    for axis in range(3):
        r0, r1 = [i for i in range(3) if i != axis]
        cx = (lo[r0] + hi[r0]) / 2
        cy = (lo[r1] + hi[r1]) / 2

        radii = [math.sqrt((p[r0] - cx) ** 2 + (p[r1] - cy) ** 2) for p in pts]
        mean_r = sum(radii) / len(radii)
        if mean_r < 1e-9:
            continue
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in radii) / len(radii))
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
    Mesh prim의 형태를 분석.
    Returns: (shape_type, meta)
      shape_type: "flat" | "box" | "cylinder" | "unknown"
      meta: {"lo", "hi", "diagonal", ...}
    """
    pts = _get_local_points(prim)
    if not pts or len(pts) < 4:
        return "unknown", {}

    lo, hi   = _local_bbox(pts)
    diag     = _diagonal(lo, hi)
    if diag < 1e-6:
        return "unknown", {}

    meta = {"lo": lo, "hi": hi, "diagonal": diag}

    fr = _flat_ratio(pts, diag)
    if fr is not None and fr <= flat_threshold:
        return "flat", meta

    if _box_ratio(pts, lo, hi, diag) <= box_threshold:
        return "box", meta

    cyl_r, cyl_axis = _cylinder_ratio(pts, lo, hi)
    if cyl_r <= cyl_threshold:
        meta["cyl_axis"] = cyl_axis
        return "cylinder", meta

    return "unknown", meta


# ─────────────────────────── primitive factory ───────────────────────────

def _create_cube(stage: Usd.Stage, path: str, lo: Gf.Vec3d, hi: Gf.Vec3d) -> Usd.Prim:
    cx = (lo[0] + hi[0]) / 2
    cy = (lo[1] + hi[1]) / 2
    cz = (lo[2] + hi[2]) / 2
    sx = (hi[0] - lo[0]) / 2
    sy = (hi[1] - lo[1]) / 2
    sz = (hi[2] - lo[2]) / 2

    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(2.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))
    cube.AddScaleOp().Set(Gf.Vec3f(float(sx), float(sy), float(sz)))
    return cube.GetPrim()


def _create_cylinder(
    stage: Usd.Stage,
    path: str,
    lo: Gf.Vec3d,
    hi: Gf.Vec3d,
    axis: int,
) -> Usd.Prim:
    cx = (lo[0] + hi[0]) / 2
    cy = (lo[1] + hi[1]) / 2
    cz = (lo[2] + hi[2]) / 2

    dims    = [hi[i] - lo[i] for i in range(3)]
    height  = dims[axis]
    r_axes  = [i for i in range(3) if i != axis]
    radius  = (dims[r_axes[0]] + dims[r_axes[1]]) / 4

    axis_token = ("X", "Y", "Z")[axis]
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.GetHeightAttr().Set(height)
    cyl.GetRadiusAttr().Set(radius)
    cyl.GetAxisAttr().Set(axis_token)
    cyl.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))
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
    stage 내 Mesh prim을 형태 분석하여 USD 프리미티브로 교체.

    Args:
        stage:          대상 USD Stage
        flat_threshold: 공면성 비율 임계값 (기본 0.005 = 0.5%)
        box_threshold:  BBox 면 이탈 비율 임계값 (기본 0.01 = 1%)
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
        path_str   = prim.GetPath().pathString
        shape, meta = detect_shape(prim, flat_threshold, box_threshold, cyl_threshold)
        counts[shape] = counts.get(shape, 0) + 1

        if shape == "unknown":
            skipped += 1
            continue

        lo, hi   = meta["lo"], meta["hi"]
        prim_path = path_str + "_prim"

        try:
            if shape == "cylinder":
                new_prim = _create_cylinder(stage, prim_path, lo, hi, meta["cyl_axis"])
            else:
                new_prim = _create_cube(stage, prim_path, lo, hi)

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
