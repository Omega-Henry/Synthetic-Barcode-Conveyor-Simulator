"""
YOLO format dataset exporter.
Exports frame annotations to normalized bounding box text files (classes.txt, frame_id.txt, dataset.yaml).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union
import numpy as np

from barcode_simulator.core.models import FrameAnnotation


def export_yolo_dataset(
    all_annotations: List[List[FrameAnnotation]],
    image_width: int,
    image_height: int,
    output_dir: Union[str, Path],
    category_mode: str = "single",  # "single" (0: barcode) or "by_type" (0: CODE128, 1: EAN13, etc.)
) -> None:
    """
    Export annotations to YOLO format text files.
    """
    out_p = Path(output_dir)
    labels_dir = out_p / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    if category_mode == "by_type":
        classes = ["CODE128", "EAN13", "EAN8", "UPCA", "QRCODE", "DATAMATRIX"]
        cat_map = {name: i for i, name in enumerate(classes)}
    else:
        classes = ["barcode"]
        cat_map = {"default": 0}

    # Save classes.txt
    with open(out_p / "classes.txt", "w", encoding="utf-8") as f:
        for c in classes:
            f.write(f"{c}\n")

    # Save dataset.yaml
    yaml_content = f"""# YOLOv8 / YOLOv5 Dataset configuration
path: {out_p.resolve()}
train: images
val: images

names:
"""
    for i, c in enumerate(classes):
        yaml_content += f"  {i}: {c}\n"

    with open(out_p / "dataset.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # Save frame label text files
    for frame_idx, frame_anns in enumerate(all_annotations):
        frame_id = frame_idx + 1
        txt_path = labels_dir / f"{frame_id:06d}.txt"

        lines: List[str] = []
        for ann in frame_anns:
            if not ann.is_in_frame or ann.visibility <= 0.001:
                continue

            bx1, by1, bx2, by2 = ann.bbox
            x_center = (bx1 + bx2) / (2.0 * image_width)
            y_center = (by1 + by2) / (2.0 * image_height)
            norm_w = max(0.0, bx2 - bx1) / image_width
            norm_h = max(0.0, by2 - by1) / image_height

            x_center = float(np.clip(x_center, 0.0, 1.0))
            y_center = float(np.clip(y_center, 0.0, 1.0))
            norm_w = float(np.clip(norm_w, 0.0, 1.0))
            norm_h = float(np.clip(norm_h, 0.0, 1.0))

            cls_id = cat_map.get(ann.barcode_type, 0) if category_mode == "by_type" else 0
            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        with open(txt_path, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")
