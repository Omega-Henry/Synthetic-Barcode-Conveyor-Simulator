"""
Barcode types and specifications supported by the generator.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Set


class BarcodeFormat(str, Enum):
    CODE128 = "CODE128"
    EAN13 = "EAN13"
    EAN8 = "EAN8"
    UPCA = "UPCA"
    QRCODE = "QRCODE"
    DATAMATRIX = "DATAMATRIX"


SUPPORTED_FORMATS: Set[str] = {
    BarcodeFormat.CODE128.value,
    BarcodeFormat.EAN13.value,
    BarcodeFormat.EAN8.value,
    BarcodeFormat.UPCA.value,
    BarcodeFormat.QRCODE.value,
}
