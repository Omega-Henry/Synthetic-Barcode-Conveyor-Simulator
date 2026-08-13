"""
Annotation manager handling ground_truth.jsonl, COCO, and YOLO dataset creation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from barcode_simulator.annotations.coco import export_coco_dataset
from barcode_simulator.annotations.yolo import export_yolo_dataset
from barcode_simulator.core.models import FrameAnnotation
from barcode_simulator.utils.io import append_jsonl, ensure_dir, save_jsonl


class GroundTruthAnnotationManager:
    """
    Collects, indexes, and exports frame annotations to multiple standard CV formats.
    """

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.jsonl_path = self.output_dir / "ground_truth.jsonl"
        self.annotations_dir = self.output_dir / "annotations"
        ensure_dir(self.annotations_dir)

        self._all_frame_annotations: List[List[FrameAnnotation]] = []

    def record_frame(self, frame_annotations: List[FrameAnnotation]) -> None:
        """Record annotations for a frame and stream to ground_truth.jsonl."""
        self._all_frame_annotations.append(frame_annotations)
        for ann in frame_annotations:
            append_jsonl(ann.to_dict(), self.jsonl_path)

    def export_coco(self, width: int, height: int) -> Path:
        coco_path = self.annotations_dir / "coco_annotations.json"
        export_coco_dataset(
            all_annotations=self._all_frame_annotations,
            image_width=width,
            image_height=height,
            output_path=coco_path,
        )
        return coco_path

    def export_yolo(self, width: int, height: int) -> Path:
        yolo_dir = self.annotations_dir / "yolo"
        export_yolo_dataset(
            all_annotations=self._all_frame_annotations,
            image_width=width,
            image_height=height,
            output_dir=yolo_dir,
        )
        return yolo_dir
