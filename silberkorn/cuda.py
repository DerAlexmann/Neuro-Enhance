"""
CuPy laden - an einer einzigen Stelle

Die CUDA-Bibliotheken kommen aus den pip-Paketen von NVIDIA (cuda-toolkit);
ein installiertes CUDA-Toolkit und damit CUDA_PATH braucht es nicht. CuPy
warnt trotzdem beim Import, dass CUDA_PATH fehlt - das wird hier ausgeblendet.
In der EXE kommen sie aus dem beim ersten Start geladenen Ordner
(nvidia_laufzeit.py), der vor dem Import bekannt gemacht wird.

Ohne CuPy (Tests und CI ohne Grafikkarte) ist `cupy` None und `fehler` sagt,
woran der Import gescheitert ist.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import warnings

from . import nvidia_laufzeit

nvidia_laufzeit.bereitstellen()

fehler: str | None = None
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CUDA path could not be detected")
        import cupy
except ImportError as ausnahme:
    cupy = None
    fehler = f"{type(ausnahme).__name__}: {ausnahme}"
