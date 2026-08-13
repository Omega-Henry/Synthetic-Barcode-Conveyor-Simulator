"""
COCO format dataset exporter.
Exports frame annotations to COCO Object Detection / Instance Segmentation JSON schema.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from barcode_simulator.core.models import FrameAnnotation
from barcode_simulator.utils.io import save_json


def export_coco_dataset(
    all_annotations: List[List[FrameAnnotation]],
    image_width: int,
    image_height: int,
    output_path: Union[str, Path],
    category_mode: str = "single",  # "single" (barcode=1) or "by_type" (CODE128=1, EAN13=2, etc.)
) -> Dict[str, Any]:
    """
    Convert simulation annotations into standard COCO JSON format and save to file.
    """
    categories: List[Dict[str, Any]] = []
    cat_map: Dict[str, int] = {}

    if category_mode == "by_type":
        types = ["CODE128", "EAN13", "EAN8", "UPCA", "QRCODE", "DATAMATRIX"]
        for idx, t in enumerate(types, start=1):
            categories.append({
                "id": idx,
                "name": t,
                "supercategory": "barcode",
            })
            cat_map[t] = idx
    else:
        categories.append({
            "id": 1,
            "name": "barcode",
            "supercategory": "object",
        })
        cat_map["default"] = 1

    images: List[Dict[str, Any]] = []
    annotations_list: List[Dict[str, Any]] = []
    ann_counter = 1

    for frame_idx, frame_anns in enumerate(all_annotations):
        frame_id = frame_idx + 1
        images.append({
            "id": frame_id,
            "file_name": f"frames/{frame_id:06d}.jpg",
            "width": image_width,
            "height": image_height,
            "date_captured": datetime.datetime.now().isoformat(),
        })

        for ann in frame_anns:
            # Skip annotations with 0 visibility or completely offscreen
            if not ann.is_in_frame or ann.visibility <= 0.001:
                continue

            bx1, by1, bx2, by2 = ann.bbox
            bw = max(1.0, bx2 - bx1)
            bh = max(1.0, by2 - by1)
            area = bw * bh

            cat_id = cat_map.get(ann.barcode_type, 1) if category_mode == "by_type" else 1

            # Flatten 4-point polygon
            flat_poly = []
            for pt in ann.polygon:
                flat_poly.extend([pt[0], pt[1]])

            annotations_list.append({
                "id": ann_counter,
                "image_id": frame_id,
                "category_id": cat_id,
                "bbox": [round(bx1, 2), round(by1, 2), round(bw, 2), round(bh, 2)],
                "segmentation": [flat_poly],
                "area": round(area, 2),
                "iscrowd": 0,
                "attributes": {
                    "package_id": ann.package_id,
                    "barcode_id": ann.barcode_id,
                    "barcode_type": ann.barcode_type,
                    "barcode_value": ann.barcode_value,
                    "visibility": round(ann.visibility, 3),
                    "occlusion": round(ann.occlusion, 3),
                    "rotation_degrees": round(ann.rotation_degrees, 2),
                    "motion_blur": ann.motion_blur_kernel,
                },
            })
            ann_counter += 1

    coco_doc = {
        "info": {
            "description": "Synthetic Industrial Barcode Dataset",
            "version": "1.0",
            "year": datetime.datetime.now().year,
            "date_created": datetime.datetime.now().isoformat(),
        },
        "licenses": [
            {
                "id": 1,
                "name": "MIT License",
                "url": "https://opensource.org/licenses/MIT",
            }
        ],
        "images": images,
        "annotations": annotations_list,
        "categories": categories,
    }

    save_json(coco_doc, output_path)
    return coco_doc
