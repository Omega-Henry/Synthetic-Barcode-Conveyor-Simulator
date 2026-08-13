"""
Evaluation and benchmarking module for barcode scanners.
"""

from barcode_simulator.evaluation.adapter import ScannerAdapter, ScanResult
from barcode_simulator.evaluation.baseline_decoder import BaselineScanner
from barcode_simulator.evaluation.metrics import EvaluationMetrics

__all__ = [
    "ScannerAdapter",
    "ScanResult",
    "BaselineScanner",
    "EvaluationMetrics",
]
