"""
Automated unit tests for baseline scanner adapter and evaluation metrics.
"""

import numpy as np
import pytest

from barcode_simulator.core.models import FrameAnnotation
from barcode_simulator.evaluation.adapter import ScanResult
from barcode_simulator.evaluation.baseline_decoder import BaselineScanner
from barcode_simulator.evaluation.metrics import EvaluationMetrics


def test_evaluation_metrics_precision_recall():
    evaluator = EvaluationMetrics(iou_threshold=0.5, min_visibility=0.2)

    gt1 = FrameAnnotation(
        frame_id=1,
        timestamp=0.0,
        package_id="PKG_001",
        barcode_id="BC_001",
        barcode_type="CODE128",
        barcode_value="PKG-001",
        bbox=[100, 100, 200, 200],
        polygon=[[100, 100], [200, 100], [200, 200], [100, 200]],
        visibility=1.0,
        occlusion=0.0,
        rotation_degrees=0.0,
        motion_blur_kernel=1,
        velocity_px_s=100.0,
        is_in_frame=True,
        difficulty_tags=["normal"],
    )

    pred1 = ScanResult(value="PKG-001", barcode_type="CODE128", confidence=1.0)
    pred_fp = ScanResult(value="GHOST_VALUE", barcode_type="CODE128", confidence=0.8)

    evaluator.update(predictions=[pred1, pred_fp], ground_truths=[gt1])

    summary = evaluator.compute_summary()
    assert summary["true_positives"] == 1
    assert summary["false_positives"] == 1
    assert summary["false_negatives"] == 0
    assert summary["recall"] == 1.0
    assert summary["precision"] == 0.5
    assert summary["f1_score"] == pytest.approx(2.0 / 3.0, abs=1e-3)
