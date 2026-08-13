"""
Automated unit tests for geometry, perspective transforms, and polygon clipping.
"""

import numpy as np
import pytest

from barcode_simulator.core.models import BoundingBox, Point2D, Polygon2D
from barcode_simulator.utils.geometry import (
    apply_transform_to_points,
    clip_polygon_to_viewport,
    compute_visibility,
    create_perspective_quad,
    get_perspective_matrix,
    order_quadrilateral_points,
    sutherland_hodgman_clip,
)


def test_bounding_box_iou():
    box1 = BoundingBox(0, 0, 100, 100)
    box2 = BoundingBox(50, 0, 150, 100)
    # Intersection: 50x100 = 5000
    # Area1: 10000, Area2: 10000, Union: 15000 -> IoU = 1/3
    assert abs(box1.iou(box2) - (1.0 / 3.0)) < 1e-5

    # Non-overlapping
    box3 = BoundingBox(200, 200, 300, 300)
    assert box1.iou(box3) == 0.0

    # Identical
    assert box1.iou(box1) == 1.0


def test_quadrilateral_point_ordering():
    pts = np.array([
        [100, 100],
        [0, 0],
        [100, 0],
        [0, 100]
    ])
    ordered = order_quadrilateral_points(pts)
    # Top-left (0,0), Top-right (100,0), Bottom-right (100,100), Bottom-left (0,100)
    assert np.allclose(ordered[0], [0, 0])
    assert np.allclose(ordered[1], [100, 0])
    assert np.allclose(ordered[2], [100, 100])
    assert np.allclose(ordered[3], [0, 100])


def test_perspective_homography_exactness():
    src = np.array([[0, 0], [200, 0], [200, 100], [0, 100]], dtype=np.float64)
    dst = np.array([[50, 40], [240, 60], [220, 180], [30, 150]], dtype=np.float64)

    h_mat = get_perspective_matrix(src, dst)
    transformed_pts = apply_transform_to_points(src, h_mat)

    for i in range(4):
        assert abs(transformed_pts[i].x - dst[i, 0]) < 1e-3
        assert abs(transformed_pts[i].y - dst[i, 1]) < 1e-3


def test_sutherland_hodgman_polygon_clipping():
    # Subject square [50, 50, 150, 150] (area 10000)
    subject = [Point2D(50, 50), Point2D(150, 50), Point2D(150, 150), Point2D(50, 150)]
    # Viewport [0, 0, 100, 100]
    viewport = [Point2D(0, 0), Point2D(100, 0), Point2D(100, 100), Point2D(0, 100)]

    clipped = sutherland_hodgman_clip(subject, viewport)
    clipped_poly = Polygon2D(clipped)
    # Intersection square [50, 50, 100, 100] (area 2500)
    assert abs(clipped_poly.area - 2500.0) < 1e-2


def test_visibility_calculation():
    # Barcode completely inside viewport
    bc_inside = Polygon2D([Point2D(100, 100), Point2D(200, 100), Point2D(200, 150), Point2D(100, 150)])
    vis, occ = compute_visibility(bc_inside, viewport_width=1280, viewport_height=720)
    assert abs(vis - 1.0) < 1e-3
    assert abs(occ - 0.0) < 1e-3

    # Barcode completely outside viewport
    bc_outside = Polygon2D([Point2D(-200, 100), Point2D(-100, 100), Point2D(-100, 150), Point2D(-200, 150)])
    vis_out, occ_out = compute_visibility(bc_outside, viewport_width=1280, viewport_height=720)
    assert vis_out == 0.0
    assert occ_out == 1.0

    # Barcode half occluded by another box
    occluder = Polygon2D([Point2D(150, 80), Point2D(250, 80), Point2D(250, 180), Point2D(150, 180)])
    vis_occ, occ_occ = compute_visibility(bc_inside, viewport_width=1280, viewport_height=720, occluder_polygons=[occluder])
    # Half of the 100x50 barcode is covered by occluder (from x=150 to 200) -> 50% visible
    assert abs(vis_occ - 0.5) < 1e-2
    assert abs(occ_occ - 0.5) < 1e-2
