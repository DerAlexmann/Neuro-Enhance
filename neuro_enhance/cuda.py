"""
CuPy laden - an einer einzigen Stelle

Die CUDA-Bibliotheken kommen aus den pip-Paketen von NVIDIA (cuda-toolkit);
ein installiertes CUDA-Toolkit und damit CUDA_PATH braucht es nicht. CuPy
warnt trotzdem beim Import, dass CUDA_PATH fehlt - das wird hier ausgeblendet.

Ohne CuPy (Tests und CI ohne Grafikkarte) ist `cupy` None.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import warnings

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CUDA path could not be detected")
        import cupy
except ImportError:
    cupy = None
