"""
OpenCV / NumPy based 2D Renderer.
Implements the Renderer interface for high-performance synthetic 2D frame synthesis.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
import cv2
import numpy as np

from barcode_simulator.core.models import Point2D, Polygon2D
from barcode_simulator.core.randomization import SeededRNG
from barcode_simulator.core.scene import SceneDescription
from barcode_simulator.renderers.base import Renderer
from barcode_simulator.renderers.effects import (
    apply_gaussian_blur,
    apply_glare_spots,
    apply_jpeg_compression,
    apply_motion_blur,
    apply_photometric_adjustments,
    apply_sensor_noise,
)
from barcode_simulator.utils.geometry import apply_transform_to_points, get_perspective_matrix


class OpenCV2DRenderer(Renderer):
    """
    2D Synthetic Conveyor Belt Renderer.
    """

    def __init__(self, rng: Optional[SeededRNG] = None):
        self.rng = rng if rng is not None else SeededRNG(42)
        self.canvas_w: int = 1280
        self.canvas_h: int = 720
        self._background_cache: Optional[np.ndarray] = None

    def initialize(self, initial_scene: SceneDescription) -> None:
        self.canvas_w = initial_scene.camera.width
        self.canvas_h = initial_scene.camera.height
        self._background_cache = None

    def render_frame(self, scene: SceneDescription) -> np.ndarray:
        w = scene.camera.width
        h = scene.camera.height

        # 1. Render Conveyor Background
        frame = self._render_conveyor_background(scene, w, h)

        # 2. Render Packages (sorted by z_index / Y coordinate for proper occlusion layer order)
        sorted_packages = sorted(
            scene.active_packages,
            key=lambda p: (p.z_index, p.current_position.y)
        )

        for pkg_inst in sorted_packages:
            frame = self._render_package_instance(frame, pkg_inst, scene)

        # 3. Apply Environment & Lighting Effects
        # Glare spots
        if scene.lighting.glare_sources:
            frame = apply_glare_spots(frame, scene.lighting.glare_sources)

        # Photometric adjustments (brightness, contrast, gamma)
        frame = apply_photometric_adjustments(
            frame,
            brightness=scene.lighting.ambient_brightness,
            contrast=scene.lighting.contrast,
            gamma=scene.lighting.gamma,
        )

        # Sensor noise
        if scene.lighting.sensor_noise_stddev > 0.01:
            frame = apply_sensor_noise(frame, scene.lighting.sensor_noise_stddev, self.rng)

        # JPEG compression if specified in scene metadata
        jpeg_q = scene.metadata.get("jpeg_quality")
        if jpeg_q is not None and jpeg_q < 98:
            frame = apply_jpeg_compression(frame, jpeg_q)

        return frame

    def _render_conveyor_background(self, scene: SceneDescription, w: int, h: int) -> np.ndarray:
        """Render industrial factory environment and conveyor belt with 3D metallic rollers."""
        bg = np.zeros((h, w, 3), dtype=np.uint8)

        # Factory concrete floor with fine noise
        bg[:, :] = (78, 82, 86)
        grid_step = 90
        for gx in range(0, w, grid_step):
            cv2.line(bg, (gx, 0), (gx, h), (70, 74, 78), 1)
        for gy in range(0, h, grid_step):
            cv2.line(bg, (0, gy), (w, gy), (70, 74, 78), 1)

        # Conveyor belt region
        lane_y = int(round(scene.conveyor.lane_y))
        lane_h = int(round(scene.conveyor.lane_height))
        top_y = max(0, lane_y - lane_h // 2)
        bot_y = min(h, lane_y + lane_h // 2)

        # 3D Conveyor frame shadow onto floor
        shadow_rail = 12
        bg[max(0, top_y - shadow_rail - 16):max(0, top_y - 16), :] = (55, 58, 62)
        bg[min(h, bot_y + 16):min(h, bot_y + shadow_rail + 16), :] = (55, 58, 62)

        # Steel conveyor side rails with 3D chamfer
        rail_h = 18
        # Top rail (Metallic steel gradient)
        bg[max(0, top_y - rail_h):top_y, :] = (135, 140, 145)
        cv2.line(bg, (0, top_y - rail_h), (w, top_y - rail_h), (180, 185, 190), 2)  # Top highlight
        cv2.line(bg, (0, top_y), (w, top_y), (60, 62, 65), 2)  # Bottom shadow

        # Bottom rail
        bg[bot_y:min(h, bot_y + rail_h), :] = (135, 140, 145)
        cv2.line(bg, (0, bot_y), (w, bot_y), (180, 185, 190), 2)
        cv2.line(bg, (0, min(h - 1, bot_y + rail_h)), (w, min(h - 1, bot_y + rail_h)), (60, 62, 65), 2)

        # Conveyor belt surface (matte industrial rubber/composite)
        belt_color = scene.conveyor.belt_color
        bg[top_y:bot_y, :] = belt_color

        # 3D Moving Metallic Rollers / Slats
        roller_spacing = 50
        offset_x = int(round(scene.conveyor.current_offset_x)) % roller_spacing
        for sx in range(-roller_spacing, w + roller_spacing, roller_spacing):
            rx = sx + offset_x
            if -10 <= rx < w + 10:
                # Cylindrical roller shading (dark-bright-dark gradient)
                cv2.line(bg, (rx - 2, top_y), (rx - 2, bot_y), (25, 25, 28), 1)
                cv2.line(bg, (rx - 1, top_y), (rx - 1, bot_y), (65, 68, 72), 1)  # Specular highlight
                cv2.line(bg, (rx, top_y), (rx, bot_y), (50, 52, 55), 1)
                cv2.line(bg, (rx + 1, top_y), (rx + 1, bot_y), (20, 20, 22), 1)

        # Yellow safety hazard stripes on side rail edges
        stripe_step = 40
        for sx in range(0, w, stripe_step):
            pts_top = np.array([[sx, top_y - rail_h], [sx + 15, top_y - rail_h], [sx + 5, top_y], [sx - 10, top_y]])
            cv2.fillConvexPoly(bg, pts_top, (210, 180, 40))
            pts_bot = np.array([[sx, bot_y], [sx + 15, bot_y], [sx + 5, bot_y + rail_h], [sx - 10, bot_y + rail_h]])
            cv2.fillConvexPoly(bg, pts_bot, (210, 180, 40))

        return bg

    def _render_package_instance(
        self,
        canvas: np.ndarray,
        pkg_inst: ActivePackageInstance,
        scene: SceneDescription,
    ) -> np.ndarray:
        """
        Warp and composite a package onto the canvas using homography.
        """
        pkg_data = pkg_inst.package_data
        if pkg_data.texture_image is None:
            # Generate texture if not cached
            from barcode_simulator.products.package import Package
            p = Package(pkg_data, self.rng)
            p.generate_composite_texture()

        tex = pkg_data.texture_image
        tex_h, tex_w = tex.shape[:2]

        # Source rectangle corners [top-left, top-right, bottom-right, bottom-left]
        src_corners = np.array([
            [0.0, 0.0],
            [float(tex_w), 0.0],
            [float(tex_w), float(tex_h)],
            [0.0, float(tex_h)]
        ], dtype=np.float32)

        # Destination quadrilateral in screen coordinates
        dst_corners = pkg_inst.package_polygon.to_numpy()

        # Compute Homography Matrix
        h_matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)

        # 1. Package 3D Drop Shadow (cast from box base onto conveyor)
        shadow_base = pkg_inst.base_polygon.to_numpy() if pkg_inst.base_polygon else dst_corners
        shadow_offset_x = 8.0
        shadow_offset_y = 10.0
        shadow_pts = (shadow_base + np.array([shadow_offset_x, shadow_offset_y], dtype=np.float32))
        s_xmin = max(0, int(np.min(shadow_pts[:, 0])) - 20)
        s_ymin = max(0, int(np.min(shadow_pts[:, 1])) - 20)
        s_xmax = min(canvas.shape[1], int(np.max(shadow_pts[:, 0])) + 20)
        s_ymax = min(canvas.shape[0], int(np.max(shadow_pts[:, 1])) + 20)

        if s_xmax > s_xmin and s_ymax > s_ymin:
            roi_h = s_ymax - s_ymin
            roi_w = s_xmax - s_xmin
            local_shadow_pts = (shadow_pts - np.array([s_xmin, s_ymin], dtype=np.float32)).astype(np.int32)
            shadow_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            cv2.fillConvexPoly(shadow_mask, local_shadow_pts, 255)
            shadow_mask = cv2.GaussianBlur(shadow_mask, (25, 25), 11)

            alpha_shadow = (shadow_mask.astype(np.float32) / 255.0)[:, :, np.newaxis] * 0.45
            canvas_roi = canvas[s_ymin:s_ymax, s_xmin:s_xmax].astype(np.float32)
            canvas[s_ymin:s_ymax, s_xmin:s_xmax] = np.clip(canvas_roi * (1.0 - alpha_shadow), 0, 255).astype(np.uint8)

        # 2. Render 3D Side Face (if visible)
        base_color = np.array(pkg_data.color, dtype=np.float32)
        if pkg_inst.side_face_polygon and len(pkg_inst.side_face_polygon.vertices) >= 4:
            side_pts = pkg_inst.side_face_polygon.to_numpy().astype(np.int32)
            side_color = tuple(int(c * 0.58) for c in pkg_data.color)
            cv2.fillConvexPoly(canvas, side_pts, side_color)
            # Side crease / bevel outline
            cv2.polylines(canvas, [side_pts], isClosed=True, color=tuple(max(0, int(c * 0.4)) for c in pkg_data.color), thickness=1)

        # 3. Render 3D Front Face (if visible)
        if pkg_inst.front_face_polygon and len(pkg_inst.front_face_polygon.vertices) >= 4:
            front_pts = pkg_inst.front_face_polygon.to_numpy().astype(np.int32)
            front_color = tuple(int(c * 0.74) for c in pkg_data.color)
            cv2.fillConvexPoly(canvas, front_pts, front_color)
            # Front crease / bevel outline
            cv2.polylines(canvas, [front_pts], isClosed=True, color=tuple(max(0, int(c * 0.5)) for c in pkg_data.color), thickness=1)

        # 4. Warp package Top Face texture
        tex_rgba = np.dstack([tex, np.full((tex_h, tex_w), 255, dtype=np.uint8)])
        warped_rgba = cv2.warpPerspective(
            tex_rgba,
            h_matrix,
            (canvas.shape[1], canvas.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        warped_rgb = warped_rgba[:, :, :3]
        warped_alpha = warped_rgba[:, :, 3]

        # 5. Apply package-specific Motion Blur if active
        if pkg_inst.motion_blur_kernel > 1:
            warped_rgb, _ = apply_motion_blur(
                warped_rgb,
                velocity_px_s=pkg_inst.current_velocity_px_s,
                factor=0.04,
                min_kernel=pkg_inst.motion_blur_kernel,
                max_kernel=pkg_inst.motion_blur_kernel,
                angle_rad=0.0 if scene.conveyor.direction == "left_to_right" else math.pi,
            )

        # 6. Alpha blend warped Top Face onto canvas
        p_xmin = max(0, int(np.min(dst_corners[:, 0])) - 10)
        p_ymin = max(0, int(np.min(dst_corners[:, 1])) - 10)
        p_xmax = min(canvas.shape[1], int(np.max(dst_corners[:, 0])) + 10)
        p_ymax = min(canvas.shape[0], int(np.max(dst_corners[:, 1])) + 10)

        if p_xmax > p_xmin and p_ymax > p_ymin:
            mask_roi = (warped_alpha[p_ymin:p_ymax, p_xmin:p_xmax].astype(np.float32) / 255.0)[:, :, np.newaxis]
            canvas_roi = canvas[p_ymin:p_ymax, p_xmin:p_xmax].astype(np.float32)
            pkg_roi = warped_rgb[p_ymin:p_ymax, p_xmin:p_xmax].astype(np.float32)
            canvas[p_ymin:p_ymax, p_xmin:p_xmax] = np.clip(canvas_roi * (1.0 - mask_roi) + pkg_roi * mask_roi, 0, 255).astype(np.uint8)

        # Draw crisp 3D top perimeter outline
        top_pts_int = dst_corners.astype(np.int32)
        cv2.polylines(canvas, [top_pts_int], isClosed=True, color=(30, 30, 32), thickness=1)

        return canvas

    def shutdown(self) -> None:
        self._background_cache = None
