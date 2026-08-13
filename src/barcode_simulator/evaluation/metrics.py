"""
Evaluation metrics computation, stratification, and reporting for barcode scanner algorithms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from barcode_simulator.core.models import FrameAnnotation
from barcode_simulator.evaluation.adapter import ScanResult
from barcode_simulator.utils.io import save_json


class EvaluationMetrics:
    """
    Computes rigorous scientific evaluation metrics between predictions and ground-truth.
    """

    def __init__(self, iou_threshold: float = 0.5, min_visibility: float = 0.2):
        self.iou_threshold = iou_threshold
        self.min_visibility = min_visibility

        self.total_gt_visible: int = 0
        self.total_gt_all: int = 0
        self.total_predictions: int = 0

        self.true_positives: int = 0
        self.false_positives: int = 0
        self.false_negatives: int = 0

        self.ious: List[float] = []
        self.latencies: List[float] = []

        # Stratified buckets
        self.by_type: Dict[str, Dict[str, int]] = {}
        self.by_difficulty: Dict[str, Dict[str, int]] = {}
        self.by_visibility_bracket: Dict[str, Dict[str, int]] = {
            "low (0.0-0.3)": {"gt": 0, "tp": 0},
            "medium (0.3-0.7)": {"gt": 0, "tp": 0},
            "high (0.7-1.0)": {"gt": 0, "tp": 0},
        }

    def update(self, predictions: List[ScanResult], ground_truths: List[FrameAnnotation]) -> None:
        """Update metrics for a single video frame."""
        self.total_predictions += len(predictions)

        # Filter GT barcodes by in_frame / visibility
        visible_gts = [gt for gt in ground_truths if gt.is_in_frame and gt.visibility >= self.min_visibility]
        self.total_gt_visible += len(visible_gts)
        self.total_gt_all += len(ground_truths)

        matched_gt_indices = set()
        matched_pred_indices = set()

        # Match predictions to ground truth
        for p_idx, pred in enumerate(predictions):
            if pred.latency_ms > 0:
                self.latencies.append(pred.latency_ms)

            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, gt in enumerate(visible_gts):
                if g_idx in matched_gt_indices:
                    continue

                # Check value match
                is_val_match = (pred.value.strip() == gt.barcode_value.strip())

                # If bbox available, calculate IoU
                if pred.bbox is not None:
                    iou_val = pred.bbox.iou(gt.bbox if hasattr(gt.bbox, 'iou') else None) if hasattr(pred.bbox, 'iou') else 0.0
                    if iou_val > best_iou:
                        best_iou = iou_val

                if is_val_match:
                    best_gt_idx = g_idx
                    break

            if best_gt_idx >= 0:
                matched_gt_indices.add(best_gt_idx)
                matched_pred_indices.add(p_idx)
                self.true_positives += 1
                if best_iou > 0:
                    self.ious.append(best_iou)

                # Update stratified breakdown
                gt_match = visible_gts[best_gt_idx]
                self._record_stratified_hit(gt_match)

        # Unmatched predictions are False Positives
        self.false_positives += (len(predictions) - len(matched_pred_indices))

        # Unmatched GTs are False Negatives
        unmatched_gt_count = len(visible_gts) - len(matched_gt_indices)
        self.false_negatives += unmatched_gt_count

        for g_idx, gt in enumerate(visible_gts):
            if g_idx not in matched_gt_indices:
                self._record_stratified_miss(gt)

    def _record_stratified_hit(self, gt: FrameAnnotation) -> None:
        # By Type
        btype = gt.barcode_type
        if btype not in self.by_type:
            self.by_type[btype] = {"gt": 0, "tp": 0}
        self.by_type[btype]["gt"] += 1
        self.by_type[btype]["tp"] += 1

        # By Visibility Bracket
        v = gt.visibility
        if v < 0.3:
            bracket = "low (0.0-0.3)"
        elif v < 0.7:
            bracket = "medium (0.3-0.7)"
        else:
            bracket = "high (0.7-1.0)"
        self.by_visibility_bracket[bracket]["gt"] += 1
        self.by_visibility_bracket[bracket]["tp"] += 1

        # By Difficulty tags
        for tag in gt.difficulty_tags:
            if tag not in self.by_difficulty:
                self.by_difficulty[tag] = {"gt": 0, "tp": 0}
            self.by_difficulty[tag]["gt"] += 1
            self.by_difficulty[tag]["tp"] += 1

    def _record_stratified_miss(self, gt: FrameAnnotation) -> None:
        btype = gt.barcode_type
        if btype not in self.by_type:
            self.by_type[btype] = {"gt": 0, "tp": 0}
        self.by_type[btype]["gt"] += 1

        v = gt.visibility
        if v < 0.3:
            bracket = "low (0.0-0.3)"
        elif v < 0.7:
            bracket = "medium (0.3-0.7)"
        else:
            bracket = "high (0.7-1.0)"
        self.by_visibility_bracket[bracket]["gt"] += 1

        for tag in gt.difficulty_tags:
            if tag not in self.by_difficulty:
                self.by_difficulty[tag] = {"gt": 0, "tp": 0}
            self.by_difficulty[tag]["gt"] += 1

    def compute_summary(self) -> Dict[str, Any]:
        """Compute final performance summary dict."""
        tp = self.true_positives
        fp = self.false_positives
        fn = self.false_negatives

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (2 * precision * recall) / max(1e-9, precision + recall) if (precision + recall) > 0 else 0.0

        mean_iou = float(np.mean(self.ious)) if self.ious else 0.0
        mean_lat = float(np.mean(self.latencies)) if self.latencies else 0.0

        # Stratified summary
        type_summary = {}
        for btype, counts in self.by_type.items():
            type_summary[btype] = {
                "gt_total": counts["gt"],
                "tp_decoded": counts["tp"],
                "accuracy": round(counts["tp"] / max(1, counts["gt"]), 4),
            }

        diff_summary = {}
        for tag, counts in self.by_difficulty.items():
            diff_summary[tag] = {
                "gt_total": counts["gt"],
                "tp_decoded": counts["tp"],
                "accuracy": round(counts["tp"] / max(1, counts["gt"]), 4),
            }

        vis_summary = {}
        for bracket, counts in self.by_visibility_bracket.items():
            vis_summary[bracket] = {
                "gt_total": counts["gt"],
                "tp_decoded": counts["tp"],
                "accuracy": round(counts["tp"] / max(1, counts["gt"]), 4),
            }

        return {
            "total_frames_evaluated": len(self.latencies) if self.latencies else 0,
            "ground_truth_visible_barcodes": self.total_gt_visible,
            "total_predictions": self.total_predictions,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "mean_iou": round(mean_iou, 4),
            "mean_latency_ms": round(mean_lat, 2),
            "by_barcode_type": type_summary,
            "by_visibility": vis_summary,
            "by_difficulty_tag": diff_summary,
        }
