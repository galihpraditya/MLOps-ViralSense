#!/usr/bin/env python3
"""Root entrypoint untuk skrip pelatihan model dan integrasi MLflow - ViralSense LK-06."""

import sys
from pathlib import Path

# Pastikan root workspace terdaftar pada sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import main

if __name__ == "__main__":
    main()
