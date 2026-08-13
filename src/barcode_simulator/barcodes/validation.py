"""
Structural and semantic validation of generated barcodes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from barcode_simulator.barcodes.generator import calculate_ean13_check_digit
from barcode_simulator.core.models import BarcodeData, BarcodeType


class BarcodeValidationError(Exception):
    """Raised when a generated barcode fails validation."""
    pass


def validate_barcode_data(
    barcode: BarcodeData,
    attempt_decode: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Validate a BarcodeData instance.
    Returns (is_valid, list_of_errors).
    """
    errors: List[str] = []

    # 1. Non-empty value and ID
    if not barcode.barcode_id:
        errors.append("Barcode ID is empty.")
    if not barcode.encoded_value:
        errors.append("Encoded barcode value is empty.")

    # 2. Checksum validation for EAN-13
    if barcode.barcode_type == BarcodeType.EAN13:
        val = barcode.encoded_value
        if len(val) != 13 or not val.isdigit():
            errors.append(f"EAN-13 value must be 13 digits, got '{val}' (len {len(val)})")
        else:
            base12 = val[:12]
            expected_check = calculate_ean13_check_digit(base12)
            actual_check = int(val[12])
            if expected_check != actual_check:
                errors.append(f"EAN-13 checksum mismatch: expected {expected_check}, got {actual_check}")

    # 3. Image validation
    if barcode.image is None:
        errors.append("Barcode image array is None.")
    else:
        if barcode.image.size == 0:
            errors.append("Barcode image array is empty (0 size).")
        h, w = barcode.image.shape[:2]
        if w < 20 or h < 20:
            errors.append(f"Barcode image dimensions too small: {w}x{h}")
        # Check contrast / not all black or all white
        if np.all(barcode.image == 255) or np.all(barcode.image == 0):
            errors.append("Barcode image is completely blank or uniform.")

    # 4. Optional decoding check
    if attempt_decode and barcode.image is not None and not errors:
        decoded_val = try_decode_barcode(barcode.image)
        if decoded_val is not None:
            # Check matching prefix/value
            if barcode.barcode_type == BarcodeType.EAN13:
                # EAN-13 values should match
                if decoded_val != barcode.encoded_value:
                    errors.append(f"Decoded EAN-13 '{decoded_val}' differs from encoded '{barcode.encoded_value}'")
            elif barcode.barcode_type == BarcodeType.CODE128:
                if decoded_val != barcode.encoded_value:
                    errors.append(f"Decoded Code128 '{decoded_val}' differs from encoded '{barcode.encoded_value}'")

    return (len(errors) == 0, errors)


def try_decode_barcode(image: np.ndarray) -> Optional[str]:
    """
    Attempt decoding a raw barcode image using zxing-cpp or pyzbar if available.
    Returns decoded text or None.
    """
    # Try zxing-cpp
    try:
        import zxingcpp
        results = zxingcpp.read_barcodes(image)
        if results:
            return results[0].text
    except Exception:
        pass

    # Try pyzbar
    try:
        import pyzbar.pyzbar as pyzbar
        results = pyzbar.decode(image)
        if results:
            return results[0].data.decode("utf-8")
    except Exception:
        pass

    # Try OpenCV BarcodeDetector
    try:
        import cv2
        detector = cv2.barcode_BarcodeDetector()
        ok, decoded_info, _, _ = detector.detectAndDecode(image)
        if ok and decoded_info:
            return decoded_info[0]
    except Exception:
        pass

    return None
