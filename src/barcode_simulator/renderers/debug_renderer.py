"""
Visual Ground-Truth Debug Overlay Renderer.
Annotates frames with exact polygon outlines, bounding boxes, labels, visibility metrics, and trajectory vectors.
"""

from __future__ import annotations

from typing import List, Optional
import cv2
import numpy as np

from barcode_simulator.core.models import FrameAnnotation
from barcode_simulator.core.scene import SceneDescription


class DebugOverlayRenderer:
    """
    Renders visual ground truth overlays for quality assurance and debugging.
    """

    def render_debug_frame(
        self,
        base_frame: np.ndarray,
        scene: SceneDescription,
        annotations: List[FrameAnnotation],
    ) -> np.ndarray:
        """
        Draw debug overlays on a copy of the base frame.
        """
        debug_img = base_frame.copy()

        # 1. Draw Package outlines & Barcode Polygons
        for pkg_inst in scene.active_packages:
            # Package polygon (cyan)
            pkg_pts = pkg_inst.package_polygon.to_numpy().astype(np.int32)
            cv2.polylines(debug_img, [pkg_pts], isClosed=True, color=(20, 180, 240), thickness=2)

            # Center point & Velocity vector
            cx, cy = int(round(pkg_inst.current_position.x)), int(round(pkg_inst.current_position.y))
            cv2.circle(debug_img, (cx, cy), 4, (0, 255, 255), -1)
            vx = int(round(pkg_inst.current_velocity_px_s * 0.2))
            cv2.arrowedLine(debug_img, (cx, cy), (cx + vx, cy), (0, 255, 255), 2, tipLength=0.3)

        # 2. Draw Barcode Annotations
        for ann in annotations:
            if not ann.is_in_frame:
                continue

            # Choose color by visibility (Green = high, Yellow = partial, Red = heavily occluded)
            if ann.visibility > 0.8:
                color = (0, 255, 80)
            elif ann.visibility > 0.3:
                color = (0, 220, 255)
            else:
                color = (40, 40, 255)

            # Barcode 4-point polygon
            poly_pts = np.array(ann.polygon, dtype=np.int32)
            cv2.polylines(debug_img, [poly_pts], isClosed=True, color=color, thickness=2)

            # Bounding Box (dashed or thin yellow)
            bx1, by1, bx2, by2 = [int(round(c)) for c in ann.bbox]
            cv2.rectangle(debug_img, (bx1, by1), (bx2, by2), (255, 220, 0), 1)

            # Annotation text tag
            tag = f"{ann.barcode_id} [{ann.barcode_type}] '{ann.barcode_value}'"
            subtag = f"Vis: {ann.visibility:.2f} | Rot: {ann.rotation_degrees:.1f}deg | Blur: {ann.motion_blur_kernel}px"

            tx = max(10, min(debug_img.shape[1] - 300, bx1))
            ty = max(25, by1 - 10)

            # Text background box
            cv2.rectangle(debug_img, (tx - 2, ty - 22), (tx + 280, ty + 16), (20, 20, 20), -1)
            cv2.putText(debug_img, tag, (tx, ty - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(debug_img, subtag, (tx, ty + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        # 3. Global HUD header
        h, w = debug_img.shape[:2]
        hud = f"Frame: {scene.frame_index:04d} | Time: {scene.time_seconds:.2f}s | Active Pkgs: {len(scene.active_packages)}"
        cv2.rectangle(debug_img, (10, 10), (380, 36), (15, 15, 18), -1)
        cv2.putText(debug_img, hud, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1, cv2.LINE_AA)

        return debug_img
