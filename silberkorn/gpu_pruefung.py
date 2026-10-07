"""
Startpruefung: Ist eine geeignete NVIDIA-RTX-Grafikkarte vorhanden?

Die Pruefung laeuft ueber NVML (Paket nvidia-ml-py), bevor CuPy, ONNX Runtime
oder TensorRT geladen werden. Diese Bibliotheken scheitern ohne passende Karte
oder mit zu altem Treiber an fehlenden DLLs - mit einer Meldung, die kein
Anwender versteht, und oft bevor ueberhaupt ein Fenster erscheinen kann.

Die Bewertung selbst (bewerten) ist eine reine Funktion ohne NVML, damit sie
sich ohne Grafikkarte testen laesst.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# CUDA 12.8 ist die erste Fassung mit Unterstuetzung fuer Blackwell (RTX 50).
# Ihr Windows-Treiber traegt die Nummer 570.65; aeltere Treiber koennen die
# Bibliotheken dieser Fassung nicht laden.
MIN_TREIBER = (570, 65)

# Turing (RTX 20) ist die aelteste Generation mit Tensor Cores.
MIN_COMPUTE_CAPABILITY = (7, 5)

GIB = 1024 ** 3

# Untergrenzen der Funktionsstufen in GiB. Grafikkarten melden etwas weniger
# als die Nennkapazitaet (eine 8-GB-Karte etwa 7,6 bis 8 GiB), deshalb liegen
# die Schwellen knapp unter den runden Werten.
STUFEN = (
    ("XL", 22.0),       # 24 GB und mehr
    ("L", 11.0),        # 12 bis 16 GB
    ("M", 5.5),         # 6 bis 8 GB
    ("S", 3.5),         # 4 GB
)
STUFE_OHNE_KI = "-"     # unter 4 GB: nur klassische Filter


# Gruende, aus denen die App nicht startet. Die Texte dazu stehen in
# meldungstext(), damit sie durch die Sprachtabelle laufen.
OK = "ok"
KEIN_TREIBER = "kein_treiber"
KEINE_NVIDIA = "keine_nvidia"
KEINE_RTX = "keine_rtx"
TREIBER_ALT = "treiber_alt"


@dataclass(frozen=True)
class Grafikkarte:
    index: int                              # NVML-Nummer, sortiert nach PCI-Bus
    name: str
    vram_bytes: int
    compute_capability: tuple[int, int]

    @property
    def vram_gib(self) -> float:
        return self.vram_bytes / GIB

    @property
    def ist_rtx(self) -> bool:
        """RTX-Karte mit Tensor Cores?

        Die Compute Capability allein reicht nicht: Die GTX-16-Serie ist
        ebenfalls Turing (7.5), hat aber keine Tensor Cores. Deshalb zaehlt
        zusaetzlich der Name - "RTX" als eigenes Wort, damit auch die
        Profikarten (RTX A2000, RTX 4000 Ada ...) erkannt werden.
        """
        return (re.search(r"\bRTX\b", self.name.upper()) is not None
                and self.compute_capability >= MIN_COMPUTE_CAPABILITY)


@dataclass(frozen=True)
class Befund:
    grund: str
    karte: Grafikkarte | None = None
    treiber: str = ""
    alle_karten: tuple[Grafikkarte, ...] = ()
    fehler: str = ""                        # technische Meldung, nur fuer Fehlerberichte

    @property
    def ok(self) -> bool:
        return self.grund == OK

    @property
    def stufe(self) -> str:
        return stufe_fuer_vram(self.karte.vram_bytes) if self.karte else STUFE_OHNE_KI

    @property
    def fp8(self) -> bool:
        """Ada (RTX 40) und neuer rechnen in TensorRT auch mit FP8."""
        return bool(self.karte) and self.karte.compute_capability >= (8, 9)

    @property
    def fp4(self) -> bool:
        """Blackwell (RTX 50) rechnet zusaetzlich mit FP4."""
        return bool(self.karte) and self.karte.compute_capability >= (12, 0)


def stufe_fuer_vram(vram_bytes: int) -> str:
    gib = vram_bytes / GIB
    for stufe, untergrenze in STUFEN:
        if gib >= untergrenze:
            return stufe
    return STUFE_OHNE_KI


def treiber_tupel(text: str) -> tuple[int, int]:
    """'617.14' -> (617, 14); Unlesbares ergibt (0, 0)."""
    zahlen = re.findall(r"\d+", text or "")
    if not zahlen:
        return (0, 0)
    return (int(zahlen[0]), int(zahlen[1]) if len(zahlen) > 1 else 0)


def bewerten(karten: list[Grafikkarte] | tuple[Grafikkarte, ...], treiber: str) -> Befund:
    """Waehlt die beste RTX-Karte und entscheidet, ob die App starten darf.

    Bei mehreren RTX-Karten gewinnt die mit dem meisten Grafikspeicher - er
    bestimmt den Funktionsumfang staerker als die Generation.
    """
    karten = tuple(karten)
    if not karten:
        return Befund(KEINE_NVIDIA, treiber=treiber)
    rtx = [k for k in karten if k.ist_rtx]
    if not rtx:
        return Befund(KEINE_RTX, karte=karten[0], treiber=treiber, alle_karten=karten)
    beste = max(rtx, key=lambda k: (k.vram_bytes, k.compute_capability))
    if treiber_tupel(treiber) < MIN_TREIBER:
        return Befund(TREIBER_ALT, karte=beste, treiber=treiber, alle_karten=karten)
    return Befund(OK, karte=beste, treiber=treiber, alle_karten=karten)


def _text(wert) -> str:
    return wert.decode("utf-8", "replace") if isinstance(wert, bytes) else str(wert)


def ermitteln() -> Befund:
    """Fragt die Grafikkarten ueber NVML ab und bewertet sie."""
    try:
        import pynvml
    except ImportError as fehler:
        return Befund(KEIN_TREIBER, fehler=f"nvidia-ml-py fehlt: {fehler}")

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as fehler:
        # Ohne NVIDIA-Treiber fehlt nvml.dll; NVML meldet dann
        # "Library Not Found" oder "Driver Not Loaded".
        return Befund(KEIN_TREIBER, fehler=str(fehler))

    try:
        treiber = _text(pynvml.nvmlSystemGetDriverVersion())
        karten = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            geraet = pynvml.nvmlDeviceGetHandleByIndex(index)
            karten.append(Grafikkarte(
                index=index,
                name=_text(pynvml.nvmlDeviceGetName(geraet)),
                vram_bytes=int(pynvml.nvmlDeviceGetMemoryInfo(geraet).total),
                compute_capability=tuple(pynvml.nvmlDeviceGetCudaComputeCapability(geraet)),
            ))
        return bewerten(karten, treiber)
    except pynvml.NVMLError as fehler:
        return Befund(KEIN_TREIBER, fehler=str(fehler))
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


def treiber_anzeige(version: tuple[int, int] = MIN_TREIBER) -> str:
    return f"{version[0]}.{version[1]:02d}"
