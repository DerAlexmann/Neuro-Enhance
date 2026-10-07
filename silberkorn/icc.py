"""
ICC-Farbprofile lesen: RGB-Matrix-Profile in lineares sRGB umrechnen

Die verbreiteten RGB-Arbeitsfarbraeume - sRGB, Adobe RGB, ProPhoto RGB,
Display P3 und die meisten Kameraprofile - sind Matrix-Profile: je Kanal eine
Tonwertkurve (TRC) und eine 3x3-Matrix in den Verbindungsfarbraum XYZ (D50).
Damit laesst sich ein Bild jeder Bittiefe direkt in lineares sRGB umrechnen,
ohne Umweg ueber 8 Bit und ohne Farben ausserhalb von sRGB abzuschneiden.

Profile anderer Bauart (Tabellenprofile, CMYK, Graustufen) erkennt lesen()
nicht; dafuer bleibt LittleCMS ueber Pillow zustaendig.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

TABELLE = 4096                       # Stuetzstellen je Tonwertkurve

# sRGB nach XYZ, an D50 angepasst (Bradford) - so steht es im sRGB-Profil
SRGB_NACH_XYZ_D50 = np.array([
    [0.4360747, 0.3850649, 0.1430804],
    [0.2225045, 0.7168786, 0.0606169],
    [0.0139322, 0.0971045, 0.7141733],
])


def srgb_kurve(groesse: int = TABELLE) -> np.ndarray:
    v = np.linspace(0.0, 1.0, groesse)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


@dataclass(frozen=True)
class Matrixprofil:
    matrix: np.ndarray               # Geraete-RGB (linear) -> XYZ D50
    kurven: tuple[np.ndarray, ...]   # je Kanal: kodierter Wert 0..1 -> linear, TABELLE Werte

    def nach_srgb(self) -> np.ndarray:
        """3x3-Matrix: lineares Geraete-RGB -> lineares sRGB."""
        return np.linalg.inv(SRGB_NACH_XYZ_D50) @ self.matrix

    def ist_srgb(self) -> bool:
        """Entspricht das Profil sRGB so genau, dass die Umrechnung entfallen kann?"""
        if not np.allclose(self.matrix, SRGB_NACH_XYZ_D50, atol=2e-3):
            return False
        soll = srgb_kurve()
        return all(np.allclose(kurve, soll, atol=2e-3) for kurve in self.kurven)

    def ist_linear(self) -> bool:
        gerade = np.linspace(0.0, 1.0, TABELLE)
        return all(np.allclose(kurve, gerade, atol=1e-4) for kurve in self.kurven)


# Lineares sRGB - so liefert LibRaw entwickelte RAW-Bilder
LINEAR_SRGB = Matrixprofil(SRGB_NACH_XYZ_D50,
                           tuple(np.linspace(0.0, 1.0, TABELLE) for _ in range(3)))


def _s15f16(daten: bytes, stelle: int) -> float:
    return struct.unpack_from(">i", daten, stelle)[0] / 65536.0


def _xyz(daten: bytes, stelle: int) -> np.ndarray:
    if daten[stelle:stelle + 4] != b"XYZ ":
        raise ValueError("kein XYZ-Eintrag")
    return np.array([_s15f16(daten, stelle + 8 + 4 * i) for i in range(3)])


def _kurve(daten: bytes, stelle: int) -> np.ndarray:
    """Tonwertkurve ('curv' oder 'para') als Tabelle kodiert -> linear."""
    art = daten[stelle:stelle + 4]
    x = np.linspace(0.0, 1.0, TABELLE)
    if art == b"curv":
        anzahl = struct.unpack_from(">I", daten, stelle + 8)[0]
        if anzahl == 0:
            return x
        if anzahl == 1:
            return x ** (struct.unpack_from(">H", daten, stelle + 12)[0] / 256.0)
        werte = np.frombuffer(daten, ">u2", anzahl, stelle + 12) / 65535.0
        return np.interp(x, np.linspace(0.0, 1.0, anzahl), werte)
    if art == b"para":
        funktion = struct.unpack_from(">H", daten, stelle + 8)[0]
        anzahl = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}[funktion]
        p = [_s15f16(daten, stelle + 12 + 4 * i) for i in range(anzahl)]
        g = p[0]
        if funktion == 0:
            return x ** g
        if funktion == 1:
            a, b = p[1], p[2]
            return np.where(x >= -b / a, np.maximum(a * x + b, 0) ** g, 0.0)
        if funktion == 2:
            a, b, c = p[1], p[2], p[3]
            return np.where(x >= -b / a, np.maximum(a * x + b, 0) ** g + c, c)
        if funktion == 3:
            a, b, c, d = p[1], p[2], p[3], p[4]
            return np.where(x >= d, np.maximum(a * x + b, 0) ** g, c * x)
        a, b, c, d, e, f = p[1:7]
        return np.where(x >= d, np.maximum(a * x + b, 0) ** g + e, c * x + f)
    raise ValueError("unbekannte Kurvenart")


def lesen(daten: bytes | None) -> Matrixprofil | None:
    """Matrix-Profil aus ICC-Bytes; None, wenn es keines ist oder unlesbar."""
    if not daten or len(daten) < 132:
        return None
    try:
        if daten[16:20] != b"RGB " or daten[36:40] != b"acsp":
            return None
        anzahl = struct.unpack_from(">I", daten, 128)[0]
        eintraege = {}
        for i in range(anzahl):
            kennung, stelle, _laenge = struct.unpack_from(">4sII", daten, 132 + 12 * i)
            eintraege[kennung] = stelle
        noetig = (b"rXYZ", b"gXYZ", b"bXYZ", b"rTRC", b"gTRC", b"bTRC")
        if not all(k in eintraege for k in noetig):
            return None
        matrix = np.column_stack([_xyz(daten, eintraege[k]) for k in noetig[:3]])
        kurven = tuple(np.clip(_kurve(daten, eintraege[k]), 0.0, None) for k in noetig[3:])
        return Matrixprofil(matrix, kurven)
    except (struct.error, ValueError, KeyError, IndexError):
        return None


def linearisieren(pixel, profil: Matrixprofil | None):
    """Kodierte Pixel (uint8/uint16, NumPy oder CuPy) -> lineares sRGB als float32.

    profil None heisst sRGB. Farben ausserhalb von sRGB ergeben negative oder
    ueber 1 liegende Werte; sie bleiben bis zur Ausgabe erhalten, wo sie erst
    abgeschnitten werden.
    """
    from . import filter as f
    xp = f.xp_von(pixel)
    hoechstwert = 65535.0 if pixel.dtype == np.uint16 else 255.0
    v = pixel.astype(xp.float32) * xp.float32(1.0 / hoechstwert)
    if profil is None:
        return f.srgb_zu_linear(v)
    if profil.ist_linear():
        linear = v
    else:
        linear = xp.stack([f.tabelle_anwenden(xp.asarray(kurve, dtype=xp.float32), v[..., i])
                           for i, kurve in enumerate(profil.kurven)], axis=-1)
    umrechnung = profil.nach_srgb()
    if np.allclose(umrechnung, np.eye(3), atol=1e-4):
        return linear.astype(xp.float32)
    matrix = xp.asarray(umrechnung.T, dtype=xp.float32)
    return (linear @ matrix).astype(xp.float32)
