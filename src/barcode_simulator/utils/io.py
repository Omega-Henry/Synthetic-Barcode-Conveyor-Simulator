"""
I/O utilities for file system management, JSON/JSONL/YAML serialization, and image operations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
import numpy as np
from PIL import Image


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Save object to JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: Union[str, Path]) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(records: Sequence[Dict[str, Any]], path: Union[str, Path]) -> None:
    """Save list of dicts to JSON Lines file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")


def append_jsonl(record: Dict[str, Any], path: Union[str, Path]) -> None:
    """Append a single record to a JSON Lines file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """Stream records from a JSON Lines file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_image(img_array: np.ndarray, path: Union[str, Path], format: Optional[str] = None, quality: int = 95) -> None:
    """
    Save RGB / RGBA numpy array to image file using Pillow.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if img_array.dtype != np.uint8:
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_array)
    img.save(p, format=format, quality=quality)


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load image from disk as RGB numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)
