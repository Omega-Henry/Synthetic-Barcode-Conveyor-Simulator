"""
Visual domain randomization effects: motion blur, perspective warps, lighting, sensor noise, glare, and compression.
"""

from __future__ import annotations

import io
import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from barcode_simulator.core.models import Point2D, Polygon2D
from barcode_simulator.core.randomization import SeededRNG


def create_motion_blur_kernel(length: int, angle_rad: float = 0.0) -> np.ndarray:
    """
    Construct a normalized directional linear motion blur kernel.
    """
    length = max(1, int(length))
    if length <= 1:
        return np.array([[1.0]], dtype=np.float32)

    # Make kernel square of odd size
    k_size = length if (length % 2 == 1) else length + 1
    kernel = np.zeros((k_size, k_size), dtype=np.float32)
    center = k_size // 2

    # Draw line along motion direction
    dx = math.cos(angle_rad) * (length / 2.0)
    dy = math.sin(angle_rad) * (length / 2.0)

    x1 = int(round(center - dx))
    y1 = int(round(center - dy))
    x2 = int(round(center + dx))
    y2 = int(round(center + dy))

    cv2.line(kernel, (x1, y1), (x2, y2), 1.0, thickness=1)

    s = np.sum(kernel)
    if s > 0:
        kernel /= s
    else:
        kernel[center, center] = 1.0

    return kernel


def apply_motion_blur(
    image: np.ndarray,
    velocity_px_s: float,
    factor: float = 0.04,
    min_kernel: int = 1,
    max_kernel: int = 15,
    angle_rad: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """
    Apply directional motion blur proportional to velocity.
    Returns (blurred_image, effective_kernel_size).
    """
    k_len = int(round(abs(velocity_px_s) * factor))
    k_len = int(np.clip(k_len, min_kernel, max_kernel))

    if k_len <= 1:
        return image, 1

    kernel = create_motion_blur_kernel(k_len, angle_rad)
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred, k_len


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply isotropic Gaussian blur."""
    if sigma <= 0.1:
        return image
    k_size = int(math.ceil(sigma * 3)) * 2 + 1
    return cv2.GaussianBlur(image, (k_size, k_size), sigma)


def apply_sensor_noise(image: np.ndarray, stddev: float, rng: SeededRNG) -> np.ndarray:
    """Add Gaussian sensor noise."""
    if stddev <= 0.01:
        return image
    h, w, c = image.shape
    noise = rng.np_rng.normal(0.0, stddev, size=(h, w, c)).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_photometric_adjustments(
    image: np.ndarray,
    brightness: float = 1.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Adjust brightness, contrast, and gamma.
    """
    img_f = image.astype(np.float32)

    # Brightness & Contrast: (I - 128)*contrast + 128*brightness
    if abs(contrast - 1.0) > 0.01 or abs(brightness - 1.0) > 0.01:
        img_f = (img_f - 128.0) * contrast + 128.0 * brightness

    # Gamma correction: 255 * (I / 255) ^ (1 / gamma)
    if abs(gamma - 1.0) > 0.01 and gamma > 0.01:
        img_norm = np.clip(img_f / 255.0, 0.0, 1.0)
        img_f = 255.0 * np.power(img_norm, 1.0 / gamma)

    return np.clip(img_f, 0, 255).astype(np.uint8)


def apply_jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
    """Simulate lossy JPEG transmission artifacts."""
    q = int(np.clip(quality, 10, 100))
    if q >= 98:
        return image
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    # cv2 expects BGR
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode(".jpg", bgr, encode_param)
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)


def apply_glare_spots(image: np.ndarray, glare_sources: List[Dict[str, Any]]) -> np.ndarray:
    """Render bright industrial spotlight reflections."""
    if not glare_sources:
        return image

    img_f = image.astype(np.float32)
    h, w = image.shape[:2]

    for spot in glare_sources:
        cx = spot.get("x", w / 2)
        cy = spot.get("y", h / 2)
        intensity = spot.get("intensity", 0.5)
        radius = spot.get("radius", 60)

        # Create radial Gaussian glare mask
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        glare_mask = np.exp(-dist_sq / (2.0 * (radius / 2.0) ** 2)) * (255.0 * intensity)
        glare_mask = glare_mask[:, :, np.newaxis]

        img_f = np.clip(img_f + glare_mask, 0, 255)

    return img_f.astype(np.uint8)
