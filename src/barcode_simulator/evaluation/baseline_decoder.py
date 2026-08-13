"""
Baseline barcode scanner adapter using zxing-cpp, pyzbar, and OpenCV BarcodeDetector.
Serves as an out-of-the-box sanity check and baseline benchmark.
"""

from __future__ import annotations

import time
from typing import List, Optional
import cv2
import numpy as np

from barcode_simulator.core.models import BoundingBox, Point2D, Polygon2D
from barcode_simulator.evaluation.adapter import ScannerAdapter, ScanResult


class BaselineScanner(ScannerAdapter):
    """
    Baseline scanner leveraging available local libraries (zxing-cpp, pyzbar, OpenCV).
    """

    def __init__(self, engine_priority: str = "zxing"):
        self.engine_priority = engine_priority

    def scan(self, frame_rgb: np.ndarray) -> List[ScanResult]:
        start_t = time.perf_counter()
        results: List[ScanResult] = []

        # 1. Try zxing-cpp
        try:
            import zxingcpp
            zx_res = zxingcpp.read_barcodes(frame_rgb)
            for r in zx_res:
                # Convert position quadrilateral
                poly_pts: List[Point2D] = []
                if hasattr(r, "position"):
                    pos = r.position
                    poly_pts = [
                        Point2D(float(pos.top_left.x), float(pos.top_left.y)),
                        Point2D(float(pos.top_right.x), float(pos.top_right.y)),
                        Point2D(float(pos.bottom_right.x), float(pos.bottom_right.y)),
                        Point2D(float(pos.bottom_left.x), float(pos.bottom_left.y)),
                    ]
                poly = Polygon2D(poly_pts) if len(poly_pts) == 4 else None
                bbox = poly.bounding_box if poly else None

                results.append(
                    ScanResult(
                        value=r.text,
                        barcode_type=str(r.format).replace("BarcodeFormat.", ""),
                        bbox=bbox,
                        polygon=poly,
                        confidence=1.0,
                    )
                )
            if results:
                lat = (time.perf_counter() - start_t) * 1000.0
                for res in results:
                    res.latency_ms = lat
                return results
        except Exception:
            pass

        # 2. Try pyzbar
        try:
            import pyzbar.pyzbar as pyzbar
            pyz_res = pyzbar.decode(frame_rgb)
            for r in pyz_res:
                val = r.data.decode("utf-8", errors="ignore")
                poly_pts = [Point2D(float(p.x), float(p.y)) for p in r.polygon]
                poly = Polygon2D(poly_pts) if len(poly_pts) == 4 else None
                rect = r.rect
                bbox = BoundingBox(float(rect.left), float(rect.top), float(rect.left + rect.width), float(rect.top + rect.height))

                results.append(
                    ScanResult(
                        value=val,
                        barcode_type=r.type,
                        bbox=bbox,
                        polygon=poly,
                        confidence=1.0,
                    )
                )
            if results:
                lat = (time.perf_counter() - start_t) * 1000.0
                for res in results:
                    res.latency_ms = lat
                return results
        except Exception:
            pass

        # 3. Try OpenCV BarcodeDetector
        try:
            detector = cv2.barcode_BarcodeDetector()
            ok, decoded_info, decoded_type, corners = detector.detectAndDecode(frame_rgb)
            if ok and decoded_info:
                for i, val in enumerate(decoded_info):
                    if not val:
                        continue
                    btype = decoded_type[i] if i < len(decoded_type) else "UNKNOWN"
                    poly = None
                    bbox = None
                    if corners is not None and i < len(corners):
                        c = corners[i]
                        poly_pts = [Point2D(float(p[0]), float(p[1])) for p in c]
                        poly = Polygon2D(poly_pts)
                        bbox = poly.bounding_box
                    results.append(
                        ScanResult(
                            value=val,
                            barcode_type=btype,
                            bbox=bbox,
                            polygon=poly,
                            confidence=1.0,
                        )
                    )
        except Exception:
            pass

        lat = (time.perf_counter() - start_t) * 1000.0
        for res in results:
            res.latency_ms = lat
        return results
