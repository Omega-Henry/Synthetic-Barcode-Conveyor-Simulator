"""
Procedural texture synthesis for industrial packages: cardboard fiber, packing tape, labels, and text markings.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from barcode_simulator.core.models import PackageMaterial
from barcode_simulator.core.randomization import SeededRNG


def generate_package_texture(
    width_px: int,
    height_px: int,
    material: PackageMaterial,
    rng: SeededRNG,
    base_color: Optional[Tuple[int, int, int]] = None,
    add_tape: bool = True,
    add_labels: bool = True,
    add_text: bool = True,
) -> np.ndarray:
    """
    Generate an RGB image of the package surface texture.
    """
    w = max(40, int(width_px))
    h = max(30, int(height_px))

    # 1. Base surface color & material noise
    if material == PackageMaterial.CARDBOARD:
        if base_color is None:
            # Randomize Kraft / Cardboard brown tones
            r = rng.integer(180, 215)
            g = rng.integer(140, 175)
            b = rng.integer(95, 125)
            base_color = (r, g, b)
        img_np = np.zeros((h, w, 3), dtype=np.float32)
        img_np[:, :] = base_color

        # Add cardboard fiber noise
        noise = rng.normal(mean=0.0, stddev=6.0)
        grain = rng.np_rng.normal(0.0, 5.0, size=(h, w, 1))
        # Corrugation lines (subtle horizontal or vertical bands)
        corrugation = (np.sin(np.linspace(0, h * 0.15, h)).reshape((h, 1, 1)) * 3.5)
        img_np += grain + corrugation

    elif material == PackageMaterial.WHITE_CARTON:
        if base_color is None:
            c = rng.integer(235, 248)
            base_color = (c, c, c)
        img_np = np.zeros((h, w, 3), dtype=np.float32)
        img_np[:, :] = base_color
        grain = rng.np_rng.normal(0.0, 3.0, size=(h, w, 1))
        img_np += grain

    elif material == PackageMaterial.POLY_MAILER:
        if base_color is None:
            c = rng.choice([(230, 230, 235), (45, 45, 50), (220, 225, 230)])
            base_color = c
        img_np = np.zeros((h, w, 3), dtype=np.float32)
        img_np[:, :] = base_color
        # Plastic sheen gradient
        gradient = np.linspace(-6, 6, w).reshape((1, w, 1))
        grain = rng.np_rng.normal(0.0, 4.0, size=(h, w, 1))
        img_np += gradient + grain

    elif material == PackageMaterial.COLORED_CARTON:
        if base_color is None:
            # Industrial box colors (yellow, blue, dark green, orange)
            palette = [
                (220, 180, 50),
                (60, 110, 175),
                (190, 80, 55),
                (70, 130, 90),
                (160, 160, 165),
            ]
            base_color = rng.choice(palette)
        img_np = np.zeros((h, w, 3), dtype=np.float32)
        img_np[:, :] = base_color
        grain = rng.np_rng.normal(0.0, 4.0, size=(h, w, 1))
        img_np += grain

    else:
        # Default Kraft Paper
        if base_color is None:
            base_color = (200, 160, 115)
        img_np = np.zeros((h, w, 3), dtype=np.float32)
        img_np[:, :] = base_color

    # Add dark border crease around box edges
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img, "RGBA")

    # Dark edge bevel
    draw.rectangle([0, 0, w - 1, h - 1], outline=(0, 0, 0, 60), width=2)

    # 2. Add Packing Tape
    if add_tape and rng.boolean(0.8):
        tape_w = rng.integer(max(15, min(w, h) // 7), max(25, min(w, h) // 4))
        tape_type = rng.choice(["brown", "clear"])
        if tape_type == "brown":
            tape_color = (160, 110, 65, 180)
        else:
            tape_color = (220, 220, 200, 100)

        # Center seam tape
        if rng.boolean(0.6):
            # Horizontal tape
            y_mid = h // 2 + rng.integer(-5, 5)
            draw.rectangle([0, y_mid - tape_w // 2, w, y_mid + tape_w // 2], fill=tape_color)
            draw.line([(0, y_mid), (w, y_mid)], fill=(0, 0, 0, 40), width=1)
        else:
            # Vertical tape
            x_mid = w // 2 + rng.integer(-5, 5)
            draw.rectangle([x_mid - tape_w // 2, 0, x_mid + tape_w // 2, h], fill=tape_color)
            draw.line([(x_mid, 0), (x_mid, h)], fill=(0, 0, 0, 40), width=1)

    # 3. Add Non-barcode Text & Symbols
    if add_text and rng.boolean(0.85):
        _draw_package_markings(draw, w, h, rng)

    # 4. Add Generic Shipping Label (if requested)
    if add_labels and rng.boolean(0.7):
        _draw_shipping_label_background(draw, w, h, rng)

    return np.array(img.convert("RGB"), dtype=np.uint8)


def _draw_package_markings(draw: ImageDraw.ImageDraw, w: int, h: int, rng: SeededRNG) -> None:
    """Draw industrial handling icons, fragile text, warning symbols."""
    # Orientation arrows ^ ^
    if rng.boolean(0.6) and w > 80 and h > 60:
        ax = rng.integer(8, max(9, w // 4))
        ay = rng.integer(8, max(9, h // 4))
        arrow_color = (20, 20, 20, 140)
        # Draw 2 small arrows
        for off in [0, 12]:
            draw.line([(ax + off, ay + 14), (ax + off, ay)], fill=arrow_color, width=2)
            draw.line([(ax + off - 3, ay + 4), (ax + off, ay)], fill=arrow_color, width=2)
            draw.line([(ax + off + 3, ay + 4), (ax + off, ay)], fill=arrow_color, width=2)
        draw.line([(ax - 2, ay + 17), (ax + 16, ay + 17)], fill=arrow_color, width=2)

    # FRAGILE / HANDLE WITH CARE text stamp
    if rng.boolean(0.5) and w > 120 and h > 80:
        tx = rng.integer(w // 2, max(w // 2 + 1, w - 80))
        ty = rng.integer(8, max(9, h - 35))
        stamp_color = rng.choice([(180, 30, 30, 160), (30, 30, 30, 140)])
        draw.rectangle([tx, ty, tx + 65, ty + 20], outline=stamp_color, width=2)
        draw.text((tx + 6, ty + 4), "FRAGILE", fill=stamp_color)


def _draw_shipping_label_background(draw: ImageDraw.ImageDraw, w: int, h: int, rng: SeededRNG) -> None:
    """Draw a white address/routing shipping label backdrop."""
    lw = rng.integer(max(50, w // 3), max(60, w * 2 // 3))
    lh = rng.integer(max(40, h // 3), max(50, h * 2 // 3))
    lx = rng.integer(5, max(6, w - lw - 5))
    ly = rng.integer(5, max(6, h - lh - 5))

    # White label background with slight border
    draw.rectangle([lx, ly, lx + lw, ly + lh], fill=(250, 250, 252, 240), outline=(180, 180, 180, 200), width=1)

    # Simulated address lines
    num_lines = rng.integer(2, 4)
    for i in range(num_lines):
        line_y = ly + 8 + i * 8
        line_w = rng.integer(lw // 3, lw - 12)
        draw.line([(lx + 6, line_y), (lx + 6 + line_w, line_y)], fill=(40, 40, 40, 180), width=2)
