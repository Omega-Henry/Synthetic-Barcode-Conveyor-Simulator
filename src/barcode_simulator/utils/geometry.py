"""
Geometric utilities: 2D transforms, perspective projection, polygon clipping, and visibility analysis.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np

from barcode_simulator.core.models import BoundingBox, Point2D, Polygon2D


def order_quadrilateral_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 quadrilateral points into [top-left, top-right, bottom-right, bottom-left].
    """
    pts = np.asarray(pts, dtype=np.float32).reshape((4, 2))
    rect = np.zeros((4, 2), dtype=np.float32)

    # Top-left has smallest sum (x + y), bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has smallest diff (y - x), bottom-left has largest diff
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def get_perspective_matrix(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Compute 3x3 homography / perspective transformation matrix mapping src_pts to dst_pts.
    Uses Direct Linear Transform (DLT) with SVD, avoiding hard dependency on cv2 if needed.
    """
    src = np.asarray(src_pts, dtype=np.float64).reshape((4, 2))
    dst = np.asarray(dst_pts, dtype=np.float64).reshape((4, 2))

    a_matrix = []
    for i in range(4):
        x, y = src[i, 0], src[i, 1]
        u, v = dst[i, 0], dst[i, 1]
        a_matrix.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        a_matrix.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])

    a = np.array(a_matrix, dtype=np.float64)
    _, _, vh = np.linalg.svd(a)
    h = vh[-1].reshape((3, 3))
    return h / h[2, 2]


def apply_transform_to_points(points: Sequence[Union[Point2D, Tuple[float, float], Sequence[float]]], matrix: np.ndarray) -> List[Point2D]:
    """
    Apply a 3x3 transformation matrix to a sequence of 2D points.
    """
    pts_array = np.array([[p.x, p.y] if isinstance(p, Point2D) else [p[0], p[1]] for p in points], dtype=np.float64)
    n = len(pts_array)
    homo = np.hstack([pts_array, np.ones((n, 1), dtype=np.float64)])  # (N, 3)
    transformed = homo @ matrix.T  # (N, 3)
    # Normalize by homogeneous w
    w = transformed[:, 2:3]
    w[np.abs(w) < 1e-9] = 1e-9
    transformed_2d = transformed[:, :2] / w
    return [Point2D(float(transformed_2d[i, 0]), float(transformed_2d[i, 1])) for i in range(n)]


def rotate_point_around_center(point: Point2D, center: Point2D, angle_degrees: float) -> Point2D:
    """Rotate a point around a center point by angle_degrees."""
    rad = math.radians(angle_degrees)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    dx = point.x - center.x
    dy = point.y - center.y
    rx = dx * cos_a - dy * sin_a + center.x
    ry = dx * sin_a + dy * cos_a + center.y
    return Point2D(rx, ry)


def create_perspective_quad(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    rotation_deg: float = 0.0,
    tilt_x_deg: float = 0.0,
    tilt_y_deg: float = 0.0,
    skew_factor: float = 0.0,
) -> Polygon2D:
    """
    Construct a 4-corner polygon representing a rectangle with 3D-like tilt, rotation, and skew.
    """
    hw = width / 2.0
    hh = height / 2.0

    # Base flat rectangle corners centered at (0, 0)
    # [top-left, top-right, bottom-right, bottom-left]
    corners = np.array([
        [-hw, -hh],
        [hw, -hh],
        [hw, hh],
        [-hw, hh]
    ], dtype=np.float64)

    # 3D Tilt perspective simulation
    rad_x = math.radians(tilt_x_deg)
    rad_y = math.radians(tilt_y_deg)

    # Perspective factor along x and y axes
    factor_x = math.sin(rad_y) * 0.4
    factor_y = math.sin(rad_x) * 0.4

    # Apply perspective convergence
    # When tilted around Y (horizontal perspective), right or left side compresses
    corners[0, 1] *= (1.0 + factor_x)
    corners[3, 1] *= (1.0 + factor_x)
    corners[1, 1] *= (1.0 - factor_x)
    corners[2, 1] *= (1.0 - factor_x)

    # When tilted around X (vertical perspective), top or bottom side compresses
    corners[0, 0] *= (1.0 + factor_y)
    corners[1, 0] *= (1.0 + factor_y)
    corners[3, 0] *= (1.0 - factor_y)
    corners[2, 0] *= (1.0 - factor_y)

    # Skew
    if abs(skew_factor) > 1e-5:
        corners[:, 0] += corners[:, 1] * skew_factor

    # Rotation
    rad_rot = math.radians(rotation_deg)
    cos_r = math.cos(rad_rot)
    sin_r = math.sin(rad_rot)
    rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float64)
    rotated = corners @ rot_matrix.T

    # Translation to center
    final_pts = rotated + np.array([center_x, center_y], dtype=np.float64)

    return Polygon2D([Point2D(float(p[0]), float(p[1])) for p in final_pts])


def sutherland_hodgman_clip(subject_polygon: List[Point2D], clip_polygon: List[Point2D]) -> List[Point2D]:
    """
    Sutherland-Hodgman polygon clipping algorithm.
    Clips a subject polygon against an arbitrary convex clip polygon (counter-clockwise or clockwise).
    """
    if len(subject_polygon) < 3 or len(clip_polygon) < 3:
        return []

    def is_inside(cp1: Point2D, cp2: Point2D, p: Point2D) -> bool:
        # Cross product: (cp2.x - cp1.x)*(p.y - cp1.y) - (cp2.y - cp1.y)*(p.x - cp1.x)
        return (cp2.x - cp1.x) * (p.y - cp1.y) - (cp2.y - cp1.y) * (p.x - cp1.x) >= -1e-7

    def compute_intersection(cp1: Point2D, cp2: Point2D, s: Point2D, e: Point2D) -> Point2D:
        dc = Point2D(cp1.x - cp2.x, cp1.y - cp2.y)
        dp = Point2D(s.x - e.x, s.y - e.y)
        n1 = cp1.x * cp2.y - cp1.y * cp2.x
        n2 = s.x * e.y - s.y * e.x
        denom = dc.x * dp.y - dc.y * dp.x
        if abs(denom) < 1e-9:
            return s
        ix = (n1 * dp.x - n2 * dc.x) / denom
        iy = (n1 * dp.y - n2 * dc.y) / denom
        return Point2D(ix, iy)

    # Ensure clip polygon is oriented counter-clockwise
    poly = Polygon2D(clip_polygon)
    # Check orientation with shoelace
    area_signed = 0.0
    for i in range(len(clip_polygon)):
        j = (i + 1) % len(clip_polygon)
        area_signed += clip_polygon[i].x * clip_polygon[j].y - clip_polygon[j].x * clip_polygon[i].y
    if area_signed < 0:
        clip_poly = list(reversed(clip_polygon))
    else:
        clip_poly = list(clip_polygon)

    output_list = list(subject_polygon)

    for i in range(len(clip_poly)):
        input_list = output_list
        output_list = []
        if not input_list:
            break
        cp1 = clip_poly[i]
        cp2 = clip_poly[(i + 1) % len(clip_poly)]

        s = input_list[-1]
        for e in input_list:
            if is_inside(cp1, cp2, e):
                if is_inside(cp1, cp2, s):
                    output_list.append(e)
                else:
                    output_list.append(compute_intersection(cp1, cp2, s, e))
                    output_list.append(e)
            elif is_inside(cp1, cp2, s):
                output_list.append(compute_intersection(cp1, cp2, s, e))
            s = e

    return output_list


def clip_polygon_to_viewport(polygon: Polygon2D, width: float, height: float) -> Polygon2D:
    """Clips a polygon against the rectangular screen viewport [0, 0, width, height]."""
    viewport_quad = [
        Point2D(0.0, 0.0),
        Point2D(width, 0.0),
        Point2D(width, height),
        Point2D(0.0, height),
    ]
    clipped_pts = sutherland_hodgman_clip(polygon.vertices, viewport_quad)
    return Polygon2D(clipped_pts)


def compute_visibility(
    barcode_polygon: Polygon2D,
    viewport_width: float,
    viewport_height: float,
    occluder_polygons: Optional[List[Polygon2D]] = None,
) -> Tuple[float, float]:
    """
    Calculate exact visibility ratio in [0.0, 1.0] and occlusion ratio (1.0 - visibility).
    Considers both frame clipping and occlusion by other packages / objects.
    """
    original_area = barcode_polygon.area
    if original_area < 1e-5:
        return 0.0, 1.0

    # 1. Viewport clipping
    in_frame_poly = clip_polygon_to_viewport(barcode_polygon, viewport_width, viewport_height)
    in_frame_area = in_frame_poly.area
    if in_frame_area < 1e-5:
        return 0.0, 1.0

    # 2. Subtract occlusions
    visible_area = in_frame_area
    if occluder_polygons:
        for occluder in occluder_polygons:
            if occluder.area < 1e-5:
                continue
            # Clip the visible polygon against the occluder to find the occluded piece
            overlap_pts = sutherland_hodgman_clip(in_frame_poly.vertices, occluder.vertices)
            overlap_poly = Polygon2D(overlap_pts)
            visible_area = max(0.0, visible_area - overlap_poly.area)

    visibility = float(np.clip(visible_area / original_area, 0.0, 1.0))
    occlusion = float(np.clip(1.0 - visibility, 0.0, 1.0))
    return visibility, occlusion
