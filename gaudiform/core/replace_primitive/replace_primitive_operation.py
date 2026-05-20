# -*- coding: utf-8 -*-
"""ReplacePrimitive post-processing operation.

phase = "per_file" — 메시 형태를 분석하여 Cube/Cylinder 프리미티브로 교체합니다.
원본 메시는 active=false 처리하여 복구 가능하게 유지합니다.

스케줄러 config.json 예시:
    {
      "post_processing": {
        "enable": true,
        "operations": [
          {
            "enable": true,
            "operation": "external",
            "script": "C:/path/to/gaudiform-replace-primitive/gaudiform/core/replace_primitive/replace_primitive_operation.py",
            "params": {
              "flat_threshold": 0.005,
              "box_threshold":  0.01,
              "cyl_threshold":  0.05,
              "skip_paths":     []
            }
          }
        ]
      }
    }

params:
    flat_threshold  (float,     default 0.005) — 공면성 비율 임계값 (flat 감지)
    box_threshold   (float,     default 0.01)  — BBox 면 이탈 비율 임계값 (box 감지)
    cyl_threshold   (float,     default 0.05)  — 실린더 반지름 편차 비율 임계값
    skip_paths      (list[str], default [])    — 처리 제외할 prim 경로 접두어
"""

from __future__ import annotations

from gaudiform.core.post_processing import PostProcessOperation, PostProcessContext
from gaudiform.core.replace_primitive.replace_primitive_core import process_stage

_TAG = "ReplacePrimitiveOperation"


class ReplacePrimitiveOperation(PostProcessOperation):
    """메시 형태 분석 기반 USD 프리미티브 교체 오퍼레이션."""

    phase            = "per_file"
    handles_own_save = True

    def execute(self, context: PostProcessContext) -> None:
        stage = context.stage
        if stage is None:
            context.on_warn(_TAG, "stage가 없습니다. 스킵합니다.")
            return

        p              = context.params
        flat_threshold = float(p.get("flat_threshold", 0.005))
        box_threshold  = float(p.get("box_threshold",  0.01))
        cyl_threshold  = float(p.get("cyl_threshold",  0.05))
        skip_paths     = p.get("skip_paths") or []

        context.on_info(_TAG, (
            f"프리미티브 교체 시작 "
            f"(flat={flat_threshold}, box={box_threshold}, cyl={cyl_threshold})"
        ))
        context.on_info(_TAG, f"file: {context.usd_file_path}")

        def _log(msg: str) -> None:
            if "[WARN]" in msg:
                context.on_warn(_TAG, msg.strip())
            else:
                context.on_info(_TAG, msg.strip())

        replaced, skipped = process_stage(
            stage=stage,
            flat_threshold=flat_threshold,
            box_threshold=box_threshold,
            cyl_threshold=cyl_threshold,
            skip_paths=skip_paths,
            log=_log,
        )

        stage.GetRootLayer().Save()
        context.on_info(_TAG, f"완료: {replaced}개 교체, {skipped}개 스킵 → 저장됨")
