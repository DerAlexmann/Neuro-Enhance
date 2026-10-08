# PyInstaller-Bauplan fuer Silberkorn - Aufruf ueber build.cmd (werkzeuge/exe_bauen.py)
#
# Ein Ordner (onedir) statt einer einzelnen EXE: Die Qt-Bibliotheken bleiben als
# eigene Dateien austauschbar, wie es die LGPL verlangt, und das Programm
# startet schneller. Ohne UPX - gepackte EXE-Dateien halten Virenscanner gern
# fuer verdaechtig.
#
# Licensed under MIT License
# Copyright 2026 Alexander Unverhau
# Created with assistance of Claude AI

import json
import os
import re

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo, StringFileInfo, StringStruct, StringTable, VarFileInfo, VarStruct,
    VSVersionInfo)

WURZEL = os.path.abspath(SPECPATH)
with open(os.path.join(WURZEL, "silberkorn", "__init__.py"), encoding="utf-8") as datei:
    VERSION = re.search(r'^VERSION = "([^"]+)"', datei.read(), re.M).group(1)

# Nie in die EXE: NVIDIA-Bibliotheken (laedt die EXE selbst), TensorRT,
# pillow-heif (libheif steht unter der LGPL und ist fest eingebaut) und was
# nur zum Entwickeln gebraucht wird.
AUSGESCHLOSSEN = ["nvidia", "tensorrt", "tensorrt_libs", "pillow_heif", "torch",
                  "tkinter", "_tkinter", "matplotlib", "scipy", "pandas", "IPython",
                  "pytest", "PyQt5", "PyQt6", "sympy"]

# imagecodecs bringt rund 70 Codecs mit, darunter libheif (LGPL). Gebraucht
# werden nur PNG mit 16 Bit und die gaengigen TIFF-Kompressionen.
IMAGECODECS = {"_shared", "_shared_cython", "_imcd", "_zlib", "_deflate", "_png",
               "_jpeg8", "_zstd", "_lzma", "_webp"}

# CuPy und imagecodecs laden ihre Module erst bei Bedarf - PyInstaller sieht
# sie nicht von selbst. CuPy braucht ausserdem seine Header (cupy/_core/include),
# um die Kernel beim ersten Gebrauch zu uebersetzen, und .data/_wheel.json.
VERSTECKT = (collect_submodules("cuda.pathfinder") + collect_submodules("cupy")
             + collect_submodules("cupy_backends") + collect_submodules("cupyx.scipy.ndimage")
             + ["imagecodecs." + m for m in sorted(IMAGECODECS)]
             # Standardmodule, die nur kompilierte Module (Cython) importieren
             + ["graphlib", "struct"])
DATEN = collect_data_files("cupy", include_py_files=False) + collect_data_files("cupyx")

a = Analysis(
    [os.path.join(WURZEL, "Silberkorn.pyw")],
    pathex=[WURZEL],
    hiddenimports=VERSTECKT,
    datas=DATEN,
    excludes=AUSGESCHLOSSEN,
)

# Ohne Nutzen fuer Silberkorn: die Bruecke von ONNX Runtime zu TensorRT und
# Qts Software-OpenGL (Silberkorn laeuft nur mit NVIDIA-Treiber)
UNNOETIG = ("onnxruntime_providers_tensorrt", "opengl32sw")


def gewollt(eintrag) -> bool:
    ziel = eintrag[0].replace("\\", "/")
    name = ziel.rsplit("/", 1)[-1]
    if ziel.startswith("imagecodecs/") and name.endswith(".pyd"):
        return name.split(".")[0] in IMAGECODECS
    return not name.lower().startswith(UNNOETIG)


a.binaries = [e for e in a.binaries if gewollt(e)]

# Fuer die Lizenzsammlung: welche Pakete tatsaechlich in der EXE stecken
module = {n.split(".")[0] for n, *_ in a.pure}
module |= {e[0].replace("\\", "/").split("/")[0].split(".")[0] for e in a.binaries + a.datas}
with open(os.path.join(workpath, "module.json"), "w", encoding="utf-8") as datei:
    json.dump(sorted(module), datei, indent=1)

zahlen = tuple(int(z) for z in VERSION.split(".")) + (0,)
versionsinfo = VSVersionInfo(
    ffi=FixedFileInfo(filevers=zahlen, prodvers=zahlen),
    kids=[
        StringFileInfo([StringTable("040704B0", [
            StringStruct("CompanyName", "Alexander Unverhau"),
            StringStruct("FileDescription", "Silberkorn"),
            StringStruct("FileVersion", VERSION),
            StringStruct("InternalName", "Silberkorn"),
            StringStruct("LegalCopyright", "Copyright 2026 Alexander Unverhau, MIT License"),
            StringStruct("OriginalFilename", "Silberkorn.exe"),
            StringStruct("ProductName", "Silberkorn"),
            StringStruct("ProductVersion", VERSION),
        ])]),
        VarFileInfo([VarStruct("Translation", [0x0407, 1200])]),
    ],
)

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Silberkorn",
    icon=os.path.join(WURZEL, "silberkorn.ico"),
    version=versionsinfo,
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="Silberkorn", upx=False)
