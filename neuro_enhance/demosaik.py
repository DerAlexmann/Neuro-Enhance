"""
RAW-Entwicklung auf der Grafikkarte: Bayer-Mosaik -> lineares sRGB

LibRaw (ueber rawpy) liest und entpackt die Datei und liefert Schwarzwerte,
Weisspunkt, Weissabgleich der Kamera und die Farbmatrix. Alles Weitere
rechnet die GPU in einem Durchlauf je Pixel:

  1. Schwarzwert abziehen, auf 0..1 bringen, Weissabgleich, bei 1 abschneiden
  2. Demosaicing - als Standard RCD (rcd.py), das an feinen Mustern die
     wenigsten Farbsaeume hinterlaesst; wahlweise das schnellere Verfahren von
     Malvar, He und Cutler (2004), das die fehlenden Farben aus den Nachbarn
     interpoliert und mit der Steigung des eigenen Kanals korrigiert
  3. Kamera-RGB -> lineares sRGB mit der Farbmatrix aus LibRaw

Die Konventionen folgen LibRaw: Ein ohne automatische Aufhellung entwickeltes
Bild entspricht dem von postprocess(), nur ohne Umweg ueber den Prozessor.

Nur das klassische 2x2-Bayer-Mosaik mit Rot, Gruen und Blau rechnet die GPU.
Fuer X-Trans, Foveon, bereits entwickelte DNG und andere Sensoren liefert
aus_rawpy() None, und LibRaw entwickelt wie bisher.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import filter as f
from . import rcd

VERFAHREN = ("rcd", "malvar")


@dataclass
class RawMosaik:
    daten: np.ndarray                # (H, W) uint16, sichtbarer Bereich
    muster: np.ndarray               # (2, 2) Kanal 0/1/2 je Position im Mosaik
    schwarz: np.ndarray              # (2, 2) Schwarzwert je Position
    weiss: float
    weissabgleich: np.ndarray        # (3,) Faktoren R, G, B, kleinster = 1
    matrix: np.ndarray               # (3, 3) Kamera-RGB -> lineares sRGB
    drehung: int                     # LibRaw-flip: 0, 3 (180), 5 (90 links), 6 (90 rechts)

    @property
    def form(self) -> tuple[int, int]:
        hoehe, breite = self.daten.shape
        return (breite, hoehe) if self.drehung in (5, 6) else (hoehe, breite)


# Lineares sRGB -> XYZ (D65), mit den Zahlen aus dcraw/LibRaw
SRGB_NACH_XYZ = np.array([[0.412453, 0.357580, 0.180423],
                          [0.212671, 0.715160, 0.072169],
                          [0.019334, 0.119193, 0.950227]])


def farbmatrix_aus_xyz(kamera_aus_xyz: np.ndarray) -> np.ndarray | None:
    """Farbmatrix Kamera-RGB -> lineares sRGB aus der Matrix XYZ -> Kamera.

    Wie cam_xyz_coeff() in dcraw/LibRaw: Kamera aus sRGB bilden, jede Zeile so
    normieren, dass Weiss (1, 1, 1) wieder Weiss ergibt - den Rest erledigt der
    Weissabgleich - und umkehren.
    """
    if not np.isfinite(kamera_aus_xyz).all() or np.allclose(kamera_aus_xyz, 0):
        return None
    kamera_aus_srgb = kamera_aus_xyz @ SRGB_NACH_XYZ
    zeilen = kamera_aus_srgb.sum(axis=1, keepdims=True)
    if np.any(np.abs(zeilen) < 1e-9):
        return None
    try:
        return np.linalg.inv(kamera_aus_srgb / zeilen)
    except np.linalg.LinAlgError:
        return None


def aus_rawpy(roh) -> RawMosaik | None:
    """Mosaik und Farbdaten aus einer geoeffneten rawpy-Datei - oder None."""
    import rawpy
    if roh.raw_type != rawpy.RawType.Flat or roh.raw_pattern is None:
        return None
    if roh.raw_pattern.shape != (2, 2) or roh.num_colors != 3:
        return None
    beschreibung = roh.color_desc.decode("ascii", "replace")
    # Farbindex der Datei (0..3) -> Kanal R, G, B; die vierte Farbe ist das zweite Gruen
    zuordnung = {"R": 0, "G": 1, "B": 2}
    if sorted(set(beschreibung)) != ["B", "G", "R"]:
        return None
    farben = np.array(roh.raw_colors_visible[:2, :2])
    muster = np.vectorize(lambda i: zuordnung[beschreibung[i]])(farben)
    if sorted(muster.ravel().tolist()) != [0, 1, 1, 2]:
        return None

    schwarz_je_farbe = np.asarray(roh.black_level_per_channel, dtype=np.float64)
    schwarz = schwarz_je_farbe[farben]
    weissabgleich = np.asarray(roh.camera_whitebalance[:3], dtype=np.float64)
    if not (weissabgleich > 0).all():
        weissabgleich = np.asarray(roh.daylight_whitebalance[:3], dtype=np.float64)
    if not (weissabgleich > 0).all():
        weissabgleich = np.ones(3)
    matrix = np.asarray(roh.color_matrix, dtype=np.float64)[:, :3]
    if not np.isfinite(matrix).all() or np.allclose(matrix, 0):
        # Fuer die meisten Kameras (Nikon, Canon ...) fuellt LibRaw color_matrix
        # erst intern beim Entwickeln; rawpy zeigt dann Nullen. Die Rohmatrix
        # XYZ -> Kamera aus der Kameratabelle ist aber da.
        matrix = farbmatrix_aus_xyz(np.asarray(roh.rgb_xyz_matrix, dtype=np.float64)[:3])
        if matrix is None:
            return None
    return RawMosaik(
        # Eine echte Kopie: raw_image_visible zeigt in den Speicher von LibRaw,
        # der mit dem Schliessen der Datei freigegeben wird.
        daten=np.array(roh.raw_image_visible, dtype=np.uint16, copy=True, order="C"),
        muster=muster.astype(np.int32),
        schwarz=schwarz,
        weiss=float(roh.white_level),
        weissabgleich=weissabgleich / weissabgleich.min(),
        matrix=matrix,
        drehung=int(roh.sizes.flip),
    )


# --------------------------------------------------------------------------
# Referenz mit NumPy/CuPy-Arrays - beschreibt, was der CUDA-Kernel rechnet
# --------------------------------------------------------------------------

def vorbereiten(daten, mosaik: RawMosaik):
    """Schwarzwert, Skalierung, Weissabgleich und Abschneiden - je Mosaikposition."""
    xp = f.xp_von(daten)
    hoehe, breite = daten.shape
    ergebnis = xp.empty((hoehe, breite), dtype=xp.float32)
    for y in range(2):
        for x in range(2):
            kanal = int(mosaik.muster[y, x])
            schwarz = mosaik.schwarz[y, x]
            faktor = mosaik.weissabgleich[kanal] / (mosaik.weiss - schwarz)
            teil = (daten[y::2, x::2].astype(xp.float32) - xp.float32(schwarz)) \
                * xp.float32(faktor)
            ergebnis[y::2, x::2] = xp.clip(teil, 0, 1)
    return ergebnis


# Die vier Kerne von Malvar, He und Cutler (alle durch 8 zu teilen), als
# (dy, dx, Gewicht). Sie werden auf die vorbereiteten Mosaikwerte angewendet.
G_AN_RB = [(0, 0, 4), (-1, 0, 2), (1, 0, 2), (0, -1, 2), (0, 1, 2),
           (-2, 0, -1), (2, 0, -1), (0, -2, -1), (0, 2, -1)]
# R (bzw. B) an Gruen, wenn die gesuchte Farbe links und rechts liegt
WAAGRECHT = [(0, 0, 5), (0, -1, 4), (0, 1, 4), (-1, -1, -1), (-1, 1, -1), (1, -1, -1),
             (1, 1, -1), (0, -2, -1), (0, 2, -1), (-2, 0, 0.5), (2, 0, 0.5)]
# ... wenn sie oben und unten liegt
SENKRECHT = [(dx, dy, w) for dy, dx, w in WAAGRECHT]
# R an B bzw. B an R
DIAGONAL = [(0, 0, 6), (-1, -1, 2), (-1, 1, 2), (1, -1, 2), (1, 1, 2),
            (-2, 0, -1.5), (2, 0, -1.5), (0, -2, -1.5), (0, 2, -1.5)]


def _falten(gepolstert, kern, hoehe, breite):
    summe = None
    for dy, dx, gewicht in kern:
        teil = gepolstert[2 + dy:2 + dy + hoehe, 2 + dx:2 + dx + breite] * (gewicht / 8)
        summe = teil if summe is None else summe + teil
    return summe


def malvar(m, muster):
    """Demosaicing auf dem vorbereiteten Mosaik m (H, W) -> (H, W, 3)."""
    xp = f.xp_von(m)
    hoehe, breite = m.shape
    # "reflect" spiegelt ohne Randpixel und haelt so das Mosaikmuster am Rand ein
    gepolstert = xp.pad(m, 2, mode="reflect")
    g_rb = _falten(gepolstert, G_AN_RB, hoehe, breite)
    waag = _falten(gepolstert, WAAGRECHT, hoehe, breite)
    senk = _falten(gepolstert, SENKRECHT, hoehe, breite)
    diag = _falten(gepolstert, DIAGONAL, hoehe, breite)

    kanal = xp.asarray(muster)[xp.arange(hoehe)[:, None] % 2, xp.arange(breite)[None, :] % 2]
    # Liegt Rot in dieser Zeile? (bei Gruen-Pixeln entscheidet das waagrecht/senkrecht)
    rot_in_zeile = xp.asarray([(muster[y] == 0).any() for y in range(2)])[
        xp.arange(hoehe)[:, None] % 2] & xp.ones((1, breite), dtype=bool)

    rgb = xp.empty((hoehe, breite, 3), dtype=xp.float32)
    ist_r, ist_g, ist_b = kanal == 0, kanal == 1, kanal == 2
    rgb[..., 0] = xp.where(ist_r, m, xp.where(ist_b, diag, xp.where(rot_in_zeile, waag, senk)))
    rgb[..., 1] = xp.where(ist_g, m, g_rb)
    rgb[..., 2] = xp.where(ist_b, m, xp.where(ist_r, diag, xp.where(rot_in_zeile, senk, waag)))
    return xp.maximum(rgb, 0)


def drehen(bild, drehung: int):
    xp = f.xp_von(bild)
    if drehung == 3:
        return xp.ascontiguousarray(bild[::-1, ::-1])
    if drehung == 5:
        return xp.ascontiguousarray(xp.rot90(bild, 1))
    if drehung == 6:
        return xp.ascontiguousarray(xp.rot90(bild, 3))
    return bild


def entwickeln_referenz(daten, mosaik: RawMosaik, verfahren: str = "rcd"):
    xp = f.xp_von(daten)
    vorbereitet = vorbereiten(daten, mosaik)
    if verfahren == "rcd":
        rgb = rcd.rcd(vorbereitet, mosaik.muster)
    else:
        rgb = malvar(vorbereitet, mosaik.muster)
    linear = rgb @ xp.asarray(mosaik.matrix.T, dtype=xp.float32)
    return drehen(linear.astype(xp.float32), mosaik.drehung)


# --------------------------------------------------------------------------
# CUDA-Kernel: alles in einem Durchlauf je Pixel
# --------------------------------------------------------------------------

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        from .cuda import cupy as cp
        _KERNEL = cp.ElementwiseKernel(
            "raw uint16 roh, int32 hoehe, int32 breite, raw int32 muster, raw float32 schwarz, "
            "raw float32 faktor, raw float32 matrix",
            "raw float32 ziel",
            r"""
            int y = i / breite, x = i % breite;
            #define WERT(dy, dx) \
                wert(&roh[0], hoehe, breite, &muster[0], &schwarz[0], &faktor[0], \
                     y + (dy), x + (dx))
            float m = WERT(0, 0);
            int kanal = muster[(y & 1) * 2 + (x & 1)];
            bool rot_in_zeile = muster[(y & 1) * 2] == 0 || muster[(y & 1) * 2 + 1] == 0;
            float g_rb = (4.0f * m + 2.0f * (WERT(-1, 0) + WERT(1, 0) + WERT(0, -1) + WERT(0, 1))
                          - (WERT(-2, 0) + WERT(2, 0) + WERT(0, -2) + WERT(0, 2))) / 8.0f;
            float waag = (5.0f * m + 4.0f * (WERT(0, -1) + WERT(0, 1))
                          - (WERT(-1, -1) + WERT(-1, 1) + WERT(1, -1) + WERT(1, 1))
                          - (WERT(0, -2) + WERT(0, 2)) + 0.5f * (WERT(-2, 0) + WERT(2, 0))) / 8.0f;
            float senk = (5.0f * m + 4.0f * (WERT(-1, 0) + WERT(1, 0))
                          - (WERT(-1, -1) + WERT(1, -1) + WERT(-1, 1) + WERT(1, 1))
                          - (WERT(-2, 0) + WERT(2, 0)) + 0.5f * (WERT(0, -2) + WERT(0, 2))) / 8.0f;
            float diag = (6.0f * m + 2.0f * (WERT(-1, -1) + WERT(-1, 1) + WERT(1, -1) + WERT(1, 1))
                          - 1.5f * (WERT(-2, 0) + WERT(2, 0) + WERT(0, -2) + WERT(0, 2))) / 8.0f;
            float r, g, b;
            if (kanal == 0) { r = m; g = g_rb; b = diag; }
            else if (kanal == 2) { r = diag; g = g_rb; b = m; }
            else if (rot_in_zeile) { r = waag; g = m; b = senk; }
            else { r = senk; g = m; b = waag; }
            r = fmaxf(r, 0.0f); g = fmaxf(g, 0.0f); b = fmaxf(b, 0.0f);
            ziel[3 * i] = matrix[0] * r + matrix[1] * g + matrix[2] * b;
            ziel[3 * i + 1] = matrix[3] * r + matrix[4] * g + matrix[5] * b;
            ziel[3 * i + 2] = matrix[6] * r + matrix[7] * g + matrix[8] * b;
            """,
            "neuro_enhance_demosaik",
            preamble=r"""
            __device__ __forceinline__ float wert(const unsigned short* roh, int hoehe, int breite,
                                                  const int* muster, const float* schwarz,
                                                  const float* faktor, int y, int x) {
                y = y < 0 ? -y : (y >= hoehe ? 2 * (hoehe - 1) - y : y);
                x = x < 0 ? -x : (x >= breite ? 2 * (breite - 1) - x : x);
                int stelle = (y & 1) * 2 + (x & 1);
                float v = ((float)roh[y * breite + x] - schwarz[stelle]) * faktor[stelle];
                return fminf(fmaxf(v, 0.0f), 1.0f);
            }
            """)
    return _KERNEL


def entwickeln(daten, mosaik: RawMosaik, verfahren: str = "rcd"):
    """Mosaik (NumPy oder CuPy, uint16) -> lineares sRGB float32 (H, W, 3), gedreht."""
    xp = f.xp_von(daten)
    if xp is np:
        return entwickeln_referenz(daten, mosaik, verfahren)
    if verfahren == "rcd":
        linear = rcd.rcd_gpu(vorbereiten(daten, mosaik), mosaik.muster, mosaik.matrix)
        return drehen(linear, mosaik.drehung)
    hoehe, breite = daten.shape
    faktor = np.array([[mosaik.weissabgleich[mosaik.muster[y, x]]
                        / (mosaik.weiss - mosaik.schwarz[y, x]) for x in range(2)]
                       for y in range(2)])
    ziel = xp.empty((hoehe, breite, 3), dtype=xp.float32)
    _kernel()(xp.ascontiguousarray(daten).ravel(), xp.int32(hoehe), xp.int32(breite),
              xp.asarray(mosaik.muster.ravel(), dtype=xp.int32),
              xp.asarray(mosaik.schwarz.ravel(), dtype=xp.float32),
              xp.asarray(faktor.ravel(), dtype=xp.float32),
              xp.asarray(mosaik.matrix.ravel(), dtype=xp.float32),
              ziel, size=hoehe * breite)
    return drehen(ziel, mosaik.drehung)
