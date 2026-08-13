"""
Quick demo generation script.
Runs an easy and medium simulation to demonstrate the system immediately.
"""

import sys
from pathlib import Path

# Add src to python path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from barcode_simulator.cli import main

if __name__ == "__main__":
    sys.argv = ["barcode-simulator", "generate", "--config", "configs/easy.yaml", "--seed", "42", "--debug"]
    main()
