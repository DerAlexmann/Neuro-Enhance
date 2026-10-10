"""
RAW-Entwicklung auf der Grafikkarte: Bayer-Mosaik -> lineares sRGB

LibRaw (ueber rawpy) liest und entpackt die Datei und liefert Schwarzwerte,
Weisspunkt, Weissabgleich der Kamera und die Farbmatrix. Alles Weitere
rechnet die GPU in einem Durchlauf je Pixel:

  1. Schwarzwert abziehen, auf 0..1 bringen, Weissabgleich, bei 1 abschneiden
  2. Demosaicing nach Malvar, He und Cutler (2004): Die fehlenden Farben
     werden aus den Nachbarn interpoliert und mit der Steigung des eigenen
     Kanals korrigiert - vier feste 5x5-Kerne aus dem Paper. Wer es noch
     genauer will, laesst LibRaw entwickeln (bilddatei.laden mit
     raw_qualitaet="beste", Verfahren DHT, auf dem Prozessor).
  3. Kamera-RGB -> lineares sRGB mit der Farbmatrix aus LibRaw

Fujis X-Trans-Sensoren haben ein 6x6-Muster, in dem jede Zeile und Spalte alle
drei Farben enthaelt. Dafuer rechnet die GPU zwei Durchlaeufe:

  1. Gruen: In jeder der vier Richtungen (waagrecht, senkrecht, zwei
     Diagonalen) liegt hoechstens zwei Pixel entfernt ein gruener Nachbar auf
     jeder Seite. Jede Richtung liefert eine Schaetzung; gewichtet wird mit dem
     Kehrwert des Gefaelles im Quadrat - so wird entlang von Kanten
     interpoliert, nicht ueber sie hinweg.
  2. Rot und Blau ueber Farbdifferenzen: Rot minus Gruen der roten Pixel im
     5x5-Fenster, gewichtet nach Abstand und nach der Aehnlichkeit des Gruens.

Am Rand fehlen Nachbarn; dort springt der Zugriff um ganze Musterperioden
nach innen, damit jeder gelesene Pixel die erwartete Farbe hat.

Die Konventionen folgen LibRaw: Ein ohne automatische Aufhellung entwickeltes
Bild entspricht dem von postprocess(), nur ohne Umweg ueber den Prozessor.

Bayer- und X-Trans-Mosaike mit Rot, Gruen und Blau rechnet die GPU. Fuer
Foveon, bereits entwickelte DNG und andere Sensoren liefert aus_rawpy() None,
und LibRaw entwickelt wie bisher.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import filter as f


@dataclass
class RawMosaik:
    daten: np.ndarray                # (H, W) uint16, sichtbarer Bereich
    muster: np.ndarray               # (2, 2) Bayer oder (6, 6) X-Trans: Kanal 0/1/2
    schwarz: np.ndarray              # wie muster: Schwarzwert je Position
    weiss: float
    weissabgleich: np.ndarray        # (3,) Faktoren R, G, B, kleinster = 1
    matrix: np.ndarray               # (3, 3) Kamera-RGB -> lineares sRGB
    drehung: int                     # LibRaw-flip: 0, 3 (180), 5 (90 links), 6 (90 rechts)

    @property
    def form(self) -> tuple[int, int]:
        hoehe, breite = self.daten.shape
        return (breite, hoehe) if self.drehung in (5, 6) else (hoehe, breite)

    @property
    def periode(self) -> int:
        return int(self.muster.shape[0])


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
    form = roh.raw_pattern.shape
    if form not in ((2, 2), (6, 6)) or roh.num_colors != 3:
        return None
    beschreibung = roh.color_desc.decode("ascii", "replace")
    # Farbindex der Datei (0..3) -> Kanal R, G, B; die vierte Farbe ist das zweite Gruen
    zuordnung = {"R": 0, "G": 1, "B": 2}
    if sorted(set(beschreibung)) != ["B", "G", "R"]:
        return None
    periode = form[0]
    farben = np.array(roh.raw_colors_visible[:periode, :periode])
    if farben.shape != form or farben.max() >= len(beschreibung):
        return None
    muster = np.vectorize(lambda i: zuordnung[beschreibung[i]])(farben)
    anzahl = np.bincount(muster.ravel(), minlength=3).tolist()
    if anzahl != ([1, 2, 1] if periode == 2 else [8, 20, 8]):
        return None
    if periode == 6 and not _xtrans_tauglich(muster):
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
    p = mosaik.periode
    ergebnis = xp.empty((hoehe, breite), dtype=xp.float32)
    for y in range(p):
        for x in range(p):
            schwarz = mosaik.schwarz[y, x]
            teil = (daten[y::p, x::p].astype(xp.float32) - xp.float32(schwarz)) \
                * xp.float32(_faktoren(mosaik)[y, x])
            ergebnis[y::p, x::p] = xp.clip(teil, 0, 1)
    return ergebnis


def _faktoren(mosaik: RawMosaik) -> np.ndarray:
    """Weissabgleich / (Weiss - Schwarz) je Mosaikposition."""
    return mosaik.weissabgleich[mosaik.muster] / (mosaik.weiss - mosaik.schwarz)


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


# --------------------------------------------------------------------------
# X-Trans (6x6)
# --------------------------------------------------------------------------

XT_RICHTUNGEN = ((0, 1), (1, 0), (1, 1), (1, -1))
XT_SUCHE = 3                 # gruene Nachbarn hoechstens so weit entfernt
XT_STEIGUNG = 1e-3           # gegen Teilen durch null bei glatten Flaechen
XT_FENSTER = 2               # Rot/Blau aus dem (2*2+1)^2-Fenster
XT_GRUEN_SIGMA = 0.02        # Gruenunterschied, bei dem das Gewicht halbiert ist
XT_RAND = 12                 # Polster: zwei Musterperioden


def _xtrans_abstaende(muster, y, x, dy, dx):
    """Abstand zum naechsten gruenen Pixel rueckwaerts und vorwaerts - oder None."""
    def suche(vorzeichen):
        for k in range(1, XT_SUCHE + 1):
            if muster[(y + vorzeichen * k * dy) % 6, (x + vorzeichen * k * dx) % 6] == 1:
                return k
        return None
    return suche(-1), suche(1)


def _xtrans_tauglich(muster) -> bool:
    """Hat jeder rote und blaue Pixel in jeder Richtung Gruen und im Fenster beide Farben?"""
    for y in range(6):
        for x in range(6):
            if muster[y, x] != 1 and any(
                    None in _xtrans_abstaende(muster, y, x, dy, dx) for dy, dx in XT_RICHTUNGEN):
                return False
            fenster = {int(muster[(y + dy) % 6, (x + dx) % 6])
                       for dy in range(-XT_FENSTER, XT_FENSTER + 1)
                       for dx in range(-XT_FENSTER, XT_FENSTER + 1)}
            if fenster != {0, 1, 2}:
                return False
    return True


def _rand_index(xp, n: int):
    """Indizes -XT_RAND .. n+XT_RAND-1, ausserhalb um ganze Perioden nach innen."""
    i = np.arange(-XT_RAND, n + XT_RAND)
    i = np.where(i < 0, i % 6, np.where(i >= n, i - 6 * ((i - n) // 6 + 1), i))
    return xp.asarray(i)


def xtrans(m, muster):
    """Demosaicing auf dem vorbereiteten X-Trans-Mosaik m (H, W) -> (H, W, 3).

    Gerechnet wird je Musterposition (36 Teilgitter), so sind Nachbarfarben und
    Abstaende fuer alle Pixel eines Teilgitters gleich."""
    xp = f.xp_von(m)
    hoehe, breite = m.shape
    zeilen, spalten = _rand_index(xp, hoehe), _rand_index(xp, breite)

    def polster(ebene):
        return ebene[zeilen[:, None], spalten[None, :]]

    def nachbar(gepolstert, y0, x0, dy, dx):
        n_y, n_x = len(range(y0, hoehe, 6)), len(range(x0, breite, 6))
        y, x = XT_RAND + y0 + dy, XT_RAND + x0 + dx
        return gepolstert[y:y + 6 * (n_y - 1) + 1:6, x:x + 6 * (n_x - 1) + 1:6]

    mp = polster(m)
    gruen = m.copy()
    for y0 in range(6):
        for x0 in range(6):
            if muster[y0, x0] == 1:
                continue
            summe = gewicht = 0
            for dy, dx in XT_RICHTUNGEN:
                a, b = _xtrans_abstaende(muster, y0, x0, dy, dx)
                ga = nachbar(mp, y0, x0, -a * dy, -a * dx)
                gb = nachbar(mp, y0, x0, b * dy, b * dx)
                steigung = xp.abs(ga - gb) * xp.float32(math.hypot(dy, dx) / (a + b))
                w = 1 / (XT_STEIGUNG + steigung) ** 2
                summe = summe + w * (ga * b + gb * a) / (a + b)
                gewicht = gewicht + w
            gruen[y0::6, x0::6] = summe / gewicht

    gp = polster(gruen)
    rgb = xp.empty((hoehe, breite, 3), dtype=xp.float32)
    rgb[..., 1] = gruen
    for kanal in (0, 2):
        for y0 in range(6):
            for x0 in range(6):
                if muster[y0, x0] == kanal:
                    rgb[y0::6, x0::6, kanal] = m[y0::6, x0::6]
                    continue
                g0 = gruen[y0::6, x0::6]
                summe = gewicht = 0
                for dy in range(-XT_FENSTER, XT_FENSTER + 1):
                    for dx in range(-XT_FENSTER, XT_FENSTER + 1):
                        if muster[(y0 + dy) % 6, (x0 + dx) % 6] != kanal:
                            continue
                        gq = nachbar(gp, y0, x0, dy, dx)
                        w = xp.float32(1 / (dy * dy + dx * dx)) \
                            / (1 + ((gq - g0) / xp.float32(XT_GRUEN_SIGMA)) ** 2)
                        summe = summe + w * (nachbar(mp, y0, x0, dy, dx) - gq)
                        gewicht = gewicht + w
                rgb[y0::6, x0::6, kanal] = g0 + summe / gewicht
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


def entwickeln_referenz(daten, mosaik: RawMosaik):
    xp = f.xp_von(daten)
    verfahren = xtrans if mosaik.periode == 6 else malvar
    rgb = verfahren(vorbereiten(daten, mosaik), mosaik.muster)
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
            "silberkorn_demosaik",
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


_XT_PRAEAMBEL = r"""
__device__ __forceinline__ int rand6(int i, int n) {
    if (i < 0) return i + 6 * ((5 - i) / 6);
    if (i >= n) return i - 6 * ((i - n) / 6 + 1);
    return i;
}
__device__ __forceinline__ int stelle6(int y, int x) {
    return (((y % 6) + 6) % 6) * 6 + ((x % 6) + 6) % 6;
}
__device__ __forceinline__ float lesen6(const float* ebene, int hoehe, int breite, int y, int x) {
    return ebene[rand6(y, hoehe) * breite + rand6(x, breite)];
}
"""

_XT_KERNEL = None


def _xt_kernel():
    """Zwei Durchlaeufe: vorbereiten und Gruen, dann Rot/Blau und Farbmatrix."""
    global _XT_KERNEL
    if _XT_KERNEL is None:
        from .cuda import cupy as cp
        gruen = cp.ElementwiseKernel(
            "raw uint16 roh, int32 hoehe, int32 breite, raw int32 muster, raw float32 schwarz, "
            "raw float32 faktor",
            "raw float32 m, raw float32 gruen",
            r"""
            int y = i / breite, x = i % breite;
            #define WERT(yy, xx) wert6(&roh[0], hoehe, breite, &schwarz[0], &faktor[0], yy, xx)
            float mw = WERT(y, x);
            m[i] = mw;
            if (muster[stelle6(y, x)] == 1) {
                gruen[i] = mw;
            } else {
                const int rdy[4] = {0, 1, 1, 1}, rdx[4] = {1, 0, 1, -1};
                float summe = 0.0f, gewicht = 0.0f;
                for (int r = 0; r < 4; r++) {
                    int dy = rdy[r], dx = rdx[r], a = 0, b = 0;
                    for (int k = SUCHE; k >= 1; k--) {
                        if (muster[stelle6(y - k * dy, x - k * dx)] == 1) a = k;
                        if (muster[stelle6(y + k * dy, x + k * dx)] == 1) b = k;
                    }
                    float ga = WERT(y - a * dy, x - a * dx), gb = WERT(y + b * dy, x + b * dx);
                    float steigung = fabsf(ga - gb) * ((dx != 0 && dy != 0 ? 1.41421356f : 1.0f)
                                                       / (float)(a + b));
                    float w = 1.0f / ((STEIGUNG + steigung) * (STEIGUNG + steigung));
                    summe += w * (ga * b + gb * a) / (float)(a + b);
                    gewicht += w;
                }
                gruen[i] = summe / gewicht;
            }
            """,
            "silberkorn_xtrans_gruen",
            preamble=_XT_PRAEAMBEL + f"#define SUCHE {XT_SUCHE}\n#define STEIGUNG {XT_STEIGUNG}f\n"
            + r"""
            __device__ __forceinline__ float wert6(const unsigned short* roh, int hoehe, int breite,
                                                   const float* schwarz, const float* faktor,
                                                   int y, int x) {
                int s = stelle6(y, x);
                float v = ((float)roh[rand6(y, hoehe) * breite + rand6(x, breite)] - schwarz[s])
                          * faktor[s];
                return fminf(fmaxf(v, 0.0f), 1.0f);
            }
            """)
        farben = cp.ElementwiseKernel(
            "raw float32 m, raw float32 gruen, int32 hoehe, int32 breite, raw int32 muster, "
            "raw float32 matrix",
            "raw float32 ziel",
            r"""
            int y = i / breite, x = i % breite;
            int kanal = muster[stelle6(y, x)];
            float g0 = gruen[i], mw = m[i];
            float rgb[3];
            rgb[1] = g0;
            for (int c = 0; c <= 2; c += 2) {
                if (kanal == c) { rgb[c] = mw; continue; }
                float summe = 0.0f, gewicht = 0.0f;
                for (int dy = -FENSTER; dy <= FENSTER; dy++) {
                    for (int dx = -FENSTER; dx <= FENSTER; dx++) {
                        if (muster[stelle6(y + dy, x + dx)] != c) continue;
                        float gq = lesen6(&gruen[0], hoehe, breite, y + dy, x + dx);
                        float u = (gq - g0) / SIGMA;
                        float w = (1.0f / (float)(dy * dy + dx * dx)) / (1.0f + u * u);
                        summe += w * (lesen6(&m[0], hoehe, breite, y + dy, x + dx) - gq);
                        gewicht += w;
                    }
                }
                rgb[c] = g0 + summe / gewicht;
            }
            float r = fmaxf(rgb[0], 0.0f), g = fmaxf(rgb[1], 0.0f), b = fmaxf(rgb[2], 0.0f);
            ziel[3 * i] = matrix[0] * r + matrix[1] * g + matrix[2] * b;
            ziel[3 * i + 1] = matrix[3] * r + matrix[4] * g + matrix[5] * b;
            ziel[3 * i + 2] = matrix[6] * r + matrix[7] * g + matrix[8] * b;
            """,
            "silberkorn_xtrans_farben",
            preamble=_XT_PRAEAMBEL
            + f"#define FENSTER {XT_FENSTER}\n#define SIGMA {XT_GRUEN_SIGMA}f\n")
        _XT_KERNEL = (gruen, farben)
    return _XT_KERNEL


def entwickeln(daten, mosaik: RawMosaik):
    """Mosaik (NumPy oder CuPy, uint16) -> lineares sRGB float32 (H, W, 3), gedreht."""
    xp = f.xp_von(daten)
    if xp is np:
        return entwickeln_referenz(daten, mosaik)
    hoehe, breite = daten.shape
    roh = xp.ascontiguousarray(daten).ravel()
    muster = xp.asarray(mosaik.muster.ravel(), dtype=xp.int32)
    schwarz = xp.asarray(mosaik.schwarz.ravel(), dtype=xp.float32)
    faktor = xp.asarray(_faktoren(mosaik).ravel(), dtype=xp.float32)
    matrix = xp.asarray(mosaik.matrix.ravel(), dtype=xp.float32)
    ziel = xp.empty((hoehe, breite, 3), dtype=xp.float32)
    if mosaik.periode == 6:
        gruen_kernel, farben_kernel = _xt_kernel()
        m = xp.empty((hoehe, breite), dtype=xp.float32)
        gruen = xp.empty((hoehe, breite), dtype=xp.float32)
        gruen_kernel(roh, xp.int32(hoehe), xp.int32(breite), muster, schwarz, faktor,
                     m, gruen, size=hoehe * breite)
        farben_kernel(m, gruen, xp.int32(hoehe), xp.int32(breite), muster, matrix,
                      ziel, size=hoehe * breite)
        del m, gruen
    else:
        _kernel()(roh, xp.int32(hoehe), xp.int32(breite), muster, schwarz, faktor, matrix,
                  ziel, size=hoehe * breite)
    return drehen(ziel, mosaik.drehung)
