"""
NVIDIA-Bibliotheken fuer die EXE - beim ersten Start von NVIDIA laden

In der Python-Installation kommen CUDA und cuDNN als pip-Pakete von NVIDIA
(requirements.txt). Die EXE darf sie nicht mitbringen: Die cuDNN-Lizenz erlaubt
die Weitergabe nur fuer einzelne, aeltere Dateien, und NVIDIAs Bedingungen
vertragen sich nicht mit der MIT-Lizenz des Programms. Deshalb laedt die EXE
beim ersten Start - nach Zustimmung zu NVIDIAs Lizenzen - genau diese Pakete
vom Python Package Index, prueft jede Datei gegen ihre SHA-256-Pruefsumme und
legt nur die gebrauchten DLLs flach in einen Ordner:

    nvidia/
      bin/          cudart, nvrtc, nvJitLink, cuBLAS, cuFFT, cuDNN
      include/      die Header der Pakete - CuPy uebersetzt damit seine Kernel
      lizenzen/     die Lizenztexte aus den Paketen
      fertig.json   welche Fassungen vollstaendig ausgepackt sind

CuPy findet den Ordner ueber CUDA_PATH (es erwartet die DLLs in bin), ONNX
Runtime ueber preload_dlls(directory=...). Ausserhalb der EXE tut dieses Modul
nichts.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass

from . import einstellungen

LIZENZEN = (
    ("CUDA Toolkit EULA", "https://docs.nvidia.com/cuda/eula/index.html"),
    ("cuDNN Software License Agreement",
     "https://docs.nvidia.com/deeplearning/cudnn/latest/reference/eula.html"),
)
QUELLE = "https://pypi.org"


@dataclass(frozen=True)
class Paket:
    name: str
    fassung: str
    groesse: int
    sha256: str
    url: str


_PYPI = "https://files.pythonhosted.org/packages/"

# Dieselben Fassungen wie in requirements.txt. ONNX Runtime braucht cudart,
# cuBLAS, cuFFT und cuDNN; CuPy uebersetzt seine Kernel mit NVRTC; cuFFT und
# cuBLAS laden nvJitLink bzw. NVRTC nach. cuRAND, cuSOLVER und cuSPARSE nutzt
# das Programm nicht - sie fehlen hier mit Absicht (rund 600 MB).
PAKETE = (
    Paket("nvidia-cuda-runtime-cu12", "12.8.90", 944318,
          "c0c6027f01505bfed6c3b21ec546f69c687689aad5f1a377554bc6ca4aa993a8",
          _PYPI + "30/a5/a515b7600ad361ea14bfa13fb4d6687abf500adc270f19e89849c0590492/"
          "nvidia_cuda_runtime_cu12-12.8.90-py3-none-win_amd64.whl"),
    Paket("nvidia-cuda-nvrtc-cu12", "12.8.93", 73586838,
          "7a4b6b2904850fe78e0bd179c4b655c404d4bb799ef03ddc60804247099ae909",
          _PYPI + "45/51/52a3d84baa2136cc8df15500ad731d74d3a1114d4c123e043cb608d4a32b/"
          "nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-win_amd64.whl"),
    Paket("nvidia-nvjitlink-cu12", "12.9.86", 35584936,
          "cc6fcec260ca843c10e34c936921a1c426b351753587fdd638e8cff7b16bb9db",
          _PYPI + "dd/7e/2eecb277d8a98184d881fb98a738363fd4f14577a4d2d7f8264266e82623/"
          "nvidia_nvjitlink_cu12-12.9.86-py3-none-win_amd64.whl"),
    Paket("nvidia-cublas-cu12", "12.8.5.5", 567543364,
          "1e272895b82946b4db6f592d9080291fb60f78c9fe253a5c71ba5ebb74864c3e",
          _PYPI + "74/65/d9db5b0754559f6ed279c4a6cf1192dbf581f7d01e5d3d2882f577936049/"
          "nvidia_cublas_cu12-12.8.5.5-py3-none-win_amd64.whl"),
    Paket("nvidia-cufft-cu12", "11.3.3.83", 192216559,
          "7a64a98ef2a7c47f905aaf8931b69a3a43f27c55530c698bb2ed7c75c0b42cb7",
          _PYPI + "7d/ec/ce1629f1e478bb5ccd208986b5f9e0316a78538dd6ab1d0484f012f8e2a1/"
          "nvidia_cufft_cu12-11.3.3.83-py3-none-win_amd64.whl"),
    Paket("nvidia-cudnn-cu12", "9.10.2.21", 692992268,
          "c6288de7d63e6cf62988f0923f96dc339cea362decb1bf5b3141883392a7d65e",
          _PYPI + "3d/90/0bd6e586701b3a890fd38aa71c387dab4883d619d6e5ad912ccbd05bfd67/"
          "nvidia_cudnn_cu12-9.10.2.21-py3-none-win_amd64.whl"),
)
# DLLs aus den Paketen, die niemand laedt
UNGENUTZT = {"nvblas64_12.dll", "cufftw64_11.dll", "nvrtc64_120_0.alt.dll"}
AUSGEPACKT = 2_300_000_000                  # Bytes nach dem Auspacken, gerundet
MARKE = "fertig.json"
FORMAT = "2"                                # 2: mit include/ - aendert sich der Aufbau, neu laden
BLOCK = 1 << 20

_bin_ordner: str | None = None
_dll_verweise: list = []                   # add_dll_directory gilt, solange der Verweis lebt


class LaufzeitFehler(Exception):
    """Download, Pruefsumme oder Auspacken ist gescheitert."""


class Abbruch(Exception):
    """Der Anwender hat das Einrichten abgebrochen."""


def noetig() -> bool:
    """Braucht dieser Programmlauf die nachgeladenen Bibliotheken? Nur die EXE."""
    return einstellungen.ist_eingefroren()


def ordner_kandidaten() -> list[str]:
    """Wie bei den Modellen: neben dem Programm, sonst im Benutzerordner."""
    benutzer = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return [os.path.join(einstellungen.programm_ordner(), "nvidia"),
            os.path.join(benutzer, "Silberkorn", "nvidia")]


def _soll() -> dict[str, str]:
    return {"format": FORMAT, **{p.name: p.fassung for p in PAKETE}}


def vollstaendig(ordner: str) -> bool:
    try:
        with open(os.path.join(ordner, MARKE), encoding="utf-8") as datei:
            return json.load(datei) == _soll()
    except (OSError, ValueError):
        return False


def gefunden() -> str | None:
    """Ordner mit allen Bibliotheken in den richtigen Fassungen - oder None."""
    for ordner in ordner_kandidaten():
        if vollstaendig(ordner):
            return ordner
    return None


def _beschreibbar(ordner: str) -> bool:
    try:
        os.makedirs(ordner, exist_ok=True)
        probe = os.path.join(ordner, ".schreibprobe")
        with open(probe, "w", encoding="ascii") as datei:
            datei.write("x")
        os.remove(probe)
        return True
    except OSError:
        return False


def ziel_ordner() -> str:
    for ordner in ordner_kandidaten():
        if _beschreibbar(ordner):
            return ordner
    return ordner_kandidaten()[-1]


def download_groesse() -> int:
    return sum(p.groesse for p in PAKETE)


def einrichten(fortschritt=None, ordner: str | None = None) -> str:
    """Alle Pakete laden, pruefen und auspacken; Rueckgabe: der Ordner.

    fortschritt(geladen, gesamt) wird nach jedem Block aufgerufen; gibt es
    False zurueck, wird abgebrochen. Die Marke fertig.json entsteht erst ganz
    am Ende - ein abgebrochenes Einrichten gilt beim naechsten Start als
    nicht geschehen und beginnt von vorn.
    """
    ziel = ziel_ordner() if ordner is None else ordner
    _entfernen(os.path.join(ziel, MARKE))
    for unter in ("bin", "include"):
        shutil.rmtree(os.path.join(ziel, unter), ignore_errors=True)
    for unter in ("bin", "include", "lizenzen"):
        os.makedirs(os.path.join(ziel, unter), exist_ok=True)

    groesster = max(p.groesse for p in PAKETE)
    frei = shutil.disk_usage(ziel).free
    if frei < AUSGEPACKT + groesster:
        raise LaufzeitFehler(f"Zu wenig Speicherplatz in {ziel}: "
                             f"{frei / 2**30:.1f} GB frei, gebraucht werden "
                             f"{(AUSGEPACKT + groesster) / 2**30:.1f} GB.")
    gesamt = download_groesse()
    geladen = 0
    for paket in PAKETE:
        teil = os.path.join(ziel, paket.url.rsplit("/", 1)[1] + ".teil")
        try:
            geladen = _laden(paket, teil, geladen, gesamt, fortschritt)
            _auspacken(teil, paket, ziel)
        finally:
            _entfernen(teil)
    with open(os.path.join(ziel, MARKE), "w", encoding="utf-8") as datei:
        json.dump(_soll(), datei, indent=2)
    return ziel


def _laden(paket: Paket, teil: str, geladen: int, gesamt: int, fortschritt) -> int:
    pruef = hashlib.sha256()
    try:
        anfrage = urllib.request.Request(paket.url, headers={"User-Agent": "Silberkorn"})
        with urllib.request.urlopen(anfrage, timeout=30) as antwort, open(teil, "wb") as datei:
            while True:
                block = antwort.read(BLOCK)
                if not block:
                    break
                datei.write(block)
                pruef.update(block)
                geladen += len(block)
                if fortschritt is not None and fortschritt(geladen, gesamt) is False:
                    raise Abbruch()
    except (urllib.error.URLError, OSError, TimeoutError) as fehler:
        raise LaufzeitFehler(f"{paket.name}: {fehler}") from fehler
    if pruef.hexdigest() != paket.sha256:
        raise LaufzeitFehler(f"{paket.name}: Prüfsumme stimmt nicht - die Datei wurde verworfen.")
    return geladen


def _auspacken(wheel: str, paket: Paket, ordner: str):
    """DLLs aus nvidia/*/bin, Header aus nvidia/*/include und die Lizenztexte.

    DLLs und Lizenzen landen unter ihrem Dateinamen, Header unter ihrem Pfad
    unterhalb von include - Pfade mit '..' oder Laufwerk werden uebersprungen,
    damit nichts aus dem Archiv den Zielordner verlassen kann. CuPy braucht die
    Header (cuda_runtime.h, cuda_fp16.h usw.), um seine Kernel zu uebersetzen.
    """
    try:
        with zipfile.ZipFile(wheel) as archiv:
            for eintrag in archiv.infolist():
                teile = eintrag.filename.split("/")
                name = teile[-1]
                if not name or any(t in ("", ".", "..") or ":" in t or "\\" in t
                                   for t in teile):
                    continue
                if (teile[0] == "nvidia" and len(teile) == 4 and teile[2] == "bin"
                        and name.lower().endswith(".dll") and name not in UNGENUTZT):
                    ziel = os.path.join(ordner, "bin", name)
                elif teile[0] == "nvidia" and len(teile) >= 4 and teile[2] == "include":
                    ziel = os.path.join(ordner, "include", *teile[3:])
                    os.makedirs(os.path.dirname(ziel), exist_ok=True)
                elif (teile[0].endswith(".dist-info") and name.lower().startswith("license")):
                    ziel = os.path.join(ordner, "lizenzen", paket.name + ".txt")
                else:
                    continue
                with archiv.open(eintrag) as quelle, open(ziel, "wb") as datei:
                    shutil.copyfileobj(quelle, datei, BLOCK)
    except (zipfile.BadZipFile, OSError) as fehler:
        raise LaufzeitFehler(f"{paket.name}: {fehler}") from fehler


def _entfernen(pfad: str):
    try:
        os.remove(pfad)
    except OSError:
        pass


def aktivieren(ordner: str) -> str:
    """Bibliotheken fuer CuPy und ONNX Runtime auffindbar machen - vor `import cupy`."""
    global _bin_ordner
    bin_ordner = os.path.join(ordner, "bin")
    if _bin_ordner != bin_ordner:
        os.environ["CUDA_PATH"] = ordner
        os.environ["PATH"] = bin_ordner + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            _dll_verweise.append(os.add_dll_directory(bin_ordner))
        _bin_ordner = bin_ordner
    return bin_ordner


def bereitstellen() -> None:
    """In der EXE: vorhandene Bibliotheken aktivieren. Sonst nichts."""
    if noetig() and _bin_ordner is None:
        ordner = gefunden()
        if ordner is not None:
            aktivieren(ordner)


def dll_ordner() -> str | None:
    """Ordner mit den nachgeladenen DLLs - fuer onnxruntime.preload_dlls."""
    return _bin_ordner
