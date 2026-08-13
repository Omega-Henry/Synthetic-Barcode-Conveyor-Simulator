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
        """Render factory environment and conveyor belt with moving texture stripes."""
        bg = np.zeros((h, w, 3), dtype=np.uint8)

        # Factory floor background (industrial concrete gray)
        bg[:, :] = (85, 88, 92)
        # Floor tile grid lines
        grid_step = 80
        for gx in range(0, w, grid_step):
            cv2.line(bg, (gx, 0), (gx, h), (75, 78, 82), 1)
        for gy in range(0, h, grid_step):
            cv2.line(bg, (0, gy), (w, gy), (75, 78, 82), 1)

        # Conveyor belt region
        lane_y = int(round(scene.conveyor.lane_y))
        lane_h = int(round(scene.conveyor.lane_height))
        top_y = max(0, lane_y - lane_h // 2)
        bot_y = min(h, lane_y + lane_h // 2)

        # Steel conveyor side rails
        rail_h = 16
        # Top rail
        bg[max(0, top_y - rail_h):top_y, :] = (140, 145, 150)
        cv2.line(bg, (0, top_y), (w, top_y), (60, 60, 65), 2)
        # Bottom rail
        bg[bot_y:min(h, bot_y + rail_h), :] = (140, 145, 150)
        cv2.line(bg, (0, bot_y), (w, bot_y), (60, 60, 65), 2)

        # Conveyor belt surface
        belt_color = scene.conveyor.belt_color
        bg[top_y:bot_y, :] = belt_color

        # Conveyor moving texture markings (e.g. segmented rubber slats or roller lines)
        offset_x = int(round(scene.conveyor.current_offset_x)) % 60
        for sx in range(-60, w + 60, 60):
            line_x = sx + offset_x
            if 0 <= line_x < w:
                cv2.line(bg, (line_x, top_y), (line_x, bot_y), (30, 30, 32), 2)

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

        # 1. Package Drop Shadow (ROI optimized)
        shadow_offset_x = 10.0
        shadow_offset_y = 12.0
        shadow_pts = (dst_corners + np.array([shadow_offset_x, shadow_offset_y], dtype=np.float32))
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
            shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 9)

            alpha_shadow = (shadow_mask.astype(np.float32) / 255.0)[:, :, np.newaxis] * 0.4
            canvas_roi = canvas[s_ymin:s_ymax, s_xmin:s_xmax].astype(np.float32)
            canvas[s_ymin:s_ymax, s_xmin:s_xmax] = np.clip(canvas_roi * (1.0 - alpha_shadow), 0, 255).astype(np.uint8)

        # 2. Warp package texture
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

        # 3. Apply package-specific Motion Blur if active
        if pkg_inst.motion_blur_kernel > 1:
            warped_rgb, _ = apply_motion_blur(
                warped_rgb,
                velocity_px_s=pkg_inst.current_velocity_px_s,
                factor=0.04,
                min_kernel=pkg_inst.motion_blur_kernel,
                max_kernel=pkg_inst.motion_blur_kernel,
                angle_rad=0.0 if scene.conveyor.direction == "left_to_right" else math.pi,
            )

        # 4. Alpha blend warped package onto canvas (ROI bounded)
        p_xmin = max(0, int(np.min(dst_corners[:, 0])) - 10)
        p_ymin = max(0, int(np.min(dst_corners[:, 1])) - 10)
        p_xmax = min(canvas.shape[1], int(np.max(dst_corners[:, 0])) + 10)
        p_ymax = min(canvas.shape[0], int(np.max(dst_corners[:, 1])) + 10)

        if p_xmax > p_xmin and p_ymax > p_ymin:
            mask_roi = (warped_alpha[p_ymin:p_ymax, p_xmin:p_xmax].astype(np.float32) / 255.0)[:, :, np.newaxis]
            canvas_roi = canvas[p_ymin:p_ymax, p_xmin:p_xmax].astype(np.float32)
            pkg_roi = warped_rgb[p_ymin:p_ymax, p_xmin:p_xmax].astype(np.float32)
            canvas[p_ymin:p_ymax, p_xmin:p_xmax] = np.clip(canvas_roi * (1.0 - mask_roi) + pkg_roi * mask_roi, 0, 255).astype(np.uint8)

        return canvas

    def shutdown(self) -> None:
        self._background_cache = None
