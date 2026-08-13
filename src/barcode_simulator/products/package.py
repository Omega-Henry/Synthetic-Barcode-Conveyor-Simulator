"""
Package representation, barcode placement, and composite texture generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw

from barcode_simulator.core.models import BarcodeData, BoundingBox, PackageData, PackageMaterial, Point2D, Polygon2D
from barcode_simulator.core.randomization import SeededRNG
from barcode_simulator.products.textures import generate_package_texture


class Package:
    """
    Logical and visual package entity.
    """

    def __init__(
        self,
        data: PackageData,
        rng: SeededRNG,
    ):
        self.data = data
        self.rng = rng
        self._composite_image: Optional[np.ndarray] = None
        self._barcode_rect_on_package: Optional[BoundingBox] = None
        self._barcode_polygon_on_package: Optional[Polygon2D] = None

    @property
    def id(self) -> str:
        return self.data.package_id

    @property
    def width(self) -> float:
        return self.data.width

    @property
    def height(self) -> float:
        return self.data.height

    @property
    def barcode(self) -> Optional[BarcodeData]:
        return self.data.barcode_data

    @property
    def barcode_rect_on_package(self) -> BoundingBox:
        """Returns relative bounding box of barcode on the package surface in pixels [0, 0, W, H]."""
        if self._barcode_rect_on_package is None:
            self.generate_composite_texture()
        return self._barcode_rect_on_package

    @property
    def barcode_polygon_on_package(self) -> Polygon2D:
        """Returns 4-corner polygon of barcode on package surface [top-left, top-right, bottom-right, bottom-left]."""
        if self._barcode_polygon_on_package is None:
            self.generate_composite_texture()
        return self._barcode_polygon_on_package

    def generate_composite_texture(self) -> np.ndarray:
        """
        Generate and cache the complete package surface texture with attached barcode and labels.
        """
        if self._composite_image is not None:
            return self._composite_image

        w_px = int(round(self.data.width))
        h_px = int(round(self.data.height))

        # 1. Base procedural surface
        base_surface = generate_package_texture(
            width_px=w_px,
            height_px=h_px,
            material=self.data.material,
            rng=self.rng,
            base_color=self.data.color,
            add_tape=self.data.has_tape,
            add_labels=self.data.has_labels,
            add_text=self.data.has_text,
        )

        base_img = Image.fromarray(base_surface).convert("RGBA")

        # 2. Attach barcode if present
        if self.data.barcode_data is not None and self.data.barcode_data.image is not None:
            bc_raw = self.data.barcode_data.image
            bc_img = Image.fromarray(bc_raw).convert("RGBA")

            # Determine barcode dimensions on package
            bc_orig_w, bc_orig_h = bc_img.size
            aspect = bc_orig_w / max(1, bc_orig_h)

            if aspect >= 1.3:
                # 1D Barcode (wide rectangular)
                max_avail_w = w_px * 0.88
                max_avail_h = h_px * 0.70
                target_w = max(70.0, min(max_avail_w, max_avail_h * aspect) * self.data.barcode_rel_scale)
                target_h = max(30.0, target_w / aspect)
            else:
                # 2D Matrix / QR Code (square)
                min_dim = min(w_px, h_px)
                target_size = max(90.0, min_dim * 0.72 * max(0.85, self.data.barcode_rel_scale))
                target_w = target_size
                target_h = target_size

            # Resize barcode with high-quality resampling
            bc_resized = bc_img.resize((int(round(target_w)), int(round(target_h))), Image.Resampling.LANCZOS)

            # Placement center coordinates
            rel_x = float(np.clip(self.data.barcode_rel_x, 0.15, 0.85))
            rel_y = float(np.clip(self.data.barcode_rel_y, 0.15, 0.85))
            center_x = rel_x * w_px
            center_y = rel_y * h_px

            # Rotate barcode if specified
            rot_deg = self.data.barcode_rotation_deg
            if abs(rot_deg) > 0.01:
                bc_resized = bc_resized.rotate(rot_deg, expand=True, resample=Image.Resampling.BICUBIC)

            bc_rw, bc_rh = bc_resized.size
            paste_x = int(round(center_x - bc_rw / 2.0))
            paste_y = int(round(center_y - bc_rh / 2.0))

            # Store exact unrotated/rotated polygon relative to package (0,0)
            hw = target_w / 2.0
            hh = target_h / 2.0
            corners = np.array([
                [-hw, -hh],
                [hw, -hh],
                [hw, hh],
                [-hw, hh]
            ], dtype=np.float64)

            if abs(rot_deg) > 0.01:
                rad = math.radians(-rot_deg)  # Image rotate is counter-clockwise
                c_a, s_a = math.cos(rad), math.sin(rad)
                rot_m = np.array([[c_a, -s_a], [s_a, c_a]], dtype=np.float64)
                corners = corners @ rot_m.T

            poly_pts = corners + np.array([center_x, center_y], dtype=np.float64)
            self._barcode_polygon_on_package = Polygon2D([
                Point2D(float(p[0]), float(p[1])) for p in poly_pts
            ])
            self._barcode_rect_on_package = self._barcode_polygon_on_package.bounding_box

            # Draw white label background plate with quiet zone padding
            pad = int(round(self.data.barcode_data.quiet_zone_px))
            label_x1 = max(0, paste_x - pad)
            label_y1 = max(0, paste_y - pad)
            label_x2 = min(w_px - 1, paste_x + bc_rw + pad)
            label_y2 = min(h_px - 1, paste_y + bc_rh + pad)

            draw = ImageDraw.Draw(base_img, "RGBA")
            draw.rectangle([label_x1, label_y1, label_x2, label_y2], fill=(255, 255, 255, 255), outline=(200, 200, 200, 255), width=1)

            # Paste barcode onto package face
            base_img.paste(bc_resized, (paste_x, paste_y), bc_resized)
        else:
            self._barcode_rect_on_package = BoundingBox(0, 0, 0, 0)
            self._barcode_polygon_on_package = Polygon2D([])

        self._composite_image = np.array(base_img.convert("RGB"), dtype=np.uint8)
        self.data.texture_image = self._composite_image
        return self._composite_image
