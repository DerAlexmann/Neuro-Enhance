"""
LUTs im .cube-Format lesen und anwenden

Ein .cube-File beschreibt eine Farbtabelle: als 3D-LUT ein Gitter von N x N x N
Ausgabefarben, als 1D-LUT je Kanal eine Kurve. Die Eingabe sind sRGB-kodierte
Werte 0..1 - so sind praktisch alle Look-LUTs gebaut. Zwischen den
Gitterpunkten wird tetraedrisch interpoliert: genauer als trilinear, ohne
die leichten Farbverschiebungen an Graukanten.

1D-LUTs werden beim Lesen in eine 3D-LUT umgerechnet; die Anwendung kennt
dann nur noch einen Weg.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import numpy as np

from . import filter as f

MAX_GROESSE = 129
GROESSE_AUS_1D = 65


class LutFehler(Exception):
    """Die Datei ist keine lesbare .cube-LUT."""


@dataclass(frozen=True)
class Lut:
    tabelle: np.ndarray              # (N, N, N, 3) float32, Index [b, g, r] wie im File
    unten: np.ndarray                # DOMAIN_MIN (3,)
    oben: np.ndarray                 # DOMAIN_MAX (3,)
    titel: str

    @property
    def groesse(self) -> int:
        return self.tabelle.shape[0]


def lesen_text(text: str, name: str = "") -> Lut:
    groesse_3d = groesse_1d = None
    titel = name
    unten = np.zeros(3)
    oben = np.ones(3)
    werte = []
    for zeile in text.splitlines():
        zeile = zeile.split("#", 1)[0].strip()
        if not zeile:
            continue
        teile = zeile.split()
        schluessel = teile[0].upper()
        try:
            if schluessel == "TITLE":
                titel = zeile[5:].strip().strip('"') or titel
            elif schluessel == "LUT_3D_SIZE":
                groesse_3d = int(teile[1])
            elif schluessel == "LUT_1D_SIZE":
                groesse_1d = int(teile[1])
            elif schluessel == "DOMAIN_MIN":
                unten = np.array([float(t) for t in teile[1:4]])
            elif schluessel == "DOMAIN_MAX":
                oben = np.array([float(t) for t in teile[1:4]])
            elif schluessel in ("LUT_3D_INPUT_RANGE", "LUT_1D_INPUT_RANGE"):
                unten = np.full(3, float(teile[1]))
                oben = np.full(3, float(teile[2]))
            elif schluessel[0].isalpha():
                continue                      # unbekanntes Schluesselwort
            else:
                werte.append([float(t) for t in teile[:3]])
        except (ValueError, IndexError) as fehler:
            raise LutFehler(f"unlesbare Zeile: {zeile!r}") from fehler

    if (oben <= unten).any():
        raise LutFehler("ungültiger Wertebereich")
    daten = np.asarray(werte, dtype=np.float64)
    if groesse_3d:
        if not 2 <= groesse_3d <= MAX_GROESSE or len(daten) != groesse_3d ** 3:
            raise LutFehler("LUT_3D_SIZE passt nicht zur Zahl der Einträge")
        # Im File laeuft Rot am schnellsten - als Array also [b, g, r]
        tabelle = daten.reshape(groesse_3d, groesse_3d, groesse_3d, 3)
    elif groesse_1d:
        if not 2 <= groesse_1d <= 65536 or len(daten) != groesse_1d:
            raise LutFehler("LUT_1D_SIZE passt nicht zur Zahl der Einträge")
        tabelle = _aus_1d(daten)
    else:
        raise LutFehler("weder LUT_3D_SIZE noch LUT_1D_SIZE")
    return Lut(tabelle.astype(np.float32), unten.astype(np.float32),
               oben.astype(np.float32), titel)


def _aus_1d(kurven: np.ndarray) -> np.ndarray:
    """1D-LUT (je Kanal eine Kurve) als 3D-Gitter."""
    n = GROESSE_AUS_1D
    stellen = np.linspace(0.0, 1.0, n)
    x = np.linspace(0.0, 1.0, len(kurven))
    kanal = [np.interp(stellen, x, kurven[:, i]) for i in range(3)]
    b, g, r = np.meshgrid(stellen, stellen, stellen, indexing="ij")
    tabelle = np.empty((n, n, n, 3))
    tabelle[..., 0] = np.interp(r, stellen, kanal[0])
    tabelle[..., 1] = np.interp(g, stellen, kanal[1])
    tabelle[..., 2] = np.interp(b, stellen, kanal[2])
    return tabelle


@functools.lru_cache(maxsize=8)
def _laden(pfad: str, _zeitstempel: float) -> Lut:
    try:
        with open(pfad, encoding="utf-8", errors="replace") as datei:
            return lesen_text(datei.read(), os.path.splitext(os.path.basename(pfad))[0])
    except OSError as fehler:
        raise LutFehler(str(fehler)) from fehler


def laden(pfad: str) -> Lut:
    """LUT aus einer Datei - zwischengespeichert, bis sich die Datei aendert."""
    try:
        zeitstempel = os.path.getmtime(pfad)
    except OSError as fehler:
        raise LutFehler(str(fehler)) from fehler
    return _laden(pfad, zeitstempel)


def anwenden(v, lut: Lut):
    """Tetraedrische Interpolation; v sind sRGB-kodierte Werte (..., 3)."""
    xp = f.xp_von(v)
    n = lut.groesse
    tabelle = xp.asarray(lut.tabelle)
    unten = xp.asarray(lut.unten)
    spanne = xp.asarray(lut.oben - lut.unten)
    pos = xp.clip((v - unten) / spanne, 0, 1) * (n - 1)
    basis = xp.minimum(xp.floor(pos).astype(xp.int32), n - 2)
    anteil = (pos - basis).astype(xp.float32)
    r0, g0, b0 = basis[..., 0], basis[..., 1], basis[..., 2]
    fr, fg, fb = (anteil[..., i:i + 1] for i in range(3))

    def ecke(dr, dg, db):
        return tabelle[b0 + db, g0 + dg, r0 + dr]

    c000, c111 = ecke(0, 0, 0), ecke(1, 1, 1)
    c100, c010, c001 = ecke(1, 0, 0), ecke(0, 1, 0), ecke(0, 0, 1)
    c110, c101, c011 = ecke(1, 1, 0), ecke(1, 0, 1), ecke(0, 1, 1)
    # Sechs Tetraeder, ausgewaehlt nach der Reihenfolge der Anteile
    rg, gb, rb = fr > fg, fg > fb, fr > fb
    return xp.where(rg, xp.where(gb, c000 + fr * (c100 - c000) + fg * (c110 - c100)
                                 + fb * (c111 - c110),
                                 xp.where(rb, c000 + fr * (c100 - c000) + fb * (c101 - c100)
                                          + fg * (c111 - c101),
                                          c000 + fb * (c001 - c000) + fr * (c101 - c001)
                                          + fg * (c111 - c101))),
                    xp.where(~gb, c000 + fb * (c001 - c000) + fg * (c011 - c001)
                             + fr * (c111 - c011),
                             xp.where(~rb, c000 + fg * (c010 - c000) + fb * (c011 - c010)
                                      + fr * (c111 - c011),
                                      c000 + fg * (c010 - c000) + fr * (c110 - c010)
                                      + fb * (c111 - c110)))).astype(xp.float32)
