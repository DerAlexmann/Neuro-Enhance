"""
Geometrie: Zuschnitt, Begradigen, Perspektive, Drehen, Spiegeln und Objektivkorrektur

Alles ist eine einzige Abbildung, rueckwaerts gerechnet: Fuer jeden Pixel des
Ergebnisses wird bestimmt, woher er im Original stammt, und dort bikubisch
(Catmull-Rom) gelesen. So wird jeder Pixel genau einmal neu abgetastet, gleich
wie viele Korrekturen zusammenkommen.

Der Weg eines Ergebnispixels zurueck ins Original:

  Zuschnitt   -> Rahmen (das gedrehte Bild, auf 0..1 normiert)
  Vergroessern um s >= 1, damit keine leeren Ecken entstehen
  Begradigen  -> Drehung um den Winkel
  Perspektive -> projektive Entzerrung, senkrecht und waagrecht
  Spiegeln, Drehen um 90 Grad -> Lage im Original
  Verzeichnung -> radial um die Bildmitte; je Farbkanal etwas anders skaliert,
                  das gleicht Farbsaeume (chromatische Aberration) aus

Die Vignette wird beim Lesen ausgeglichen: ein Faktor, der zum Rand hin waechst.

Alle Groessen sind auf das Bild bezogen, nicht auf Pixel - Vorschau und Export
zeigen deshalb denselben Ausschnitt.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import filter as f

VOLLER_ZUSCHNITT = (0.0, 0.0, 1.0, 1.0)
PERSPEKTIVE = 0.3            # Regler 100 kippt die Ebene um 30 % der halben Kante
VERZEICHNUNG = 0.2           # Regler 100: Radius am Bildrand um 20 % verschoben
VIGNETTE = 0.8               # Regler 100: Ecken 80 % heller
FARBSAEUME = 0.003           # Regler 100: Kanal um 0,3 % radial skaliert


@dataclass(frozen=True)
class Geometrie:
    drehung90: int = 0                   # Vierteldrehungen im Uhrzeigersinn
    spiegeln: bool = False
    begradigen: float = 0.0              # Grad
    zuschnitt: tuple[float, float, float, float] = VOLLER_ZUSCHNITT   # x0, y0, x1, y1
    perspektive_v: float = 0.0
    perspektive_h: float = 0.0
    verzeichnung: float = 0.0
    vignette: float = 0.0
    ca_rot: float = 0.0
    ca_blau: float = 0.0

    def ist_neutral(self) -> bool:
        return self == Geometrie(zuschnitt=VOLLER_ZUSCHNITT)

    def ohne_zuschnitt(self) -> Geometrie:
        from dataclasses import replace
        return replace(self, zuschnitt=VOLLER_ZUSCHNITT)


def aus(werte) -> Geometrie:
    return Geometrie(werte.drehung90 % 4, bool(werte.spiegeln), werte.begradigen,
                     tuple(werte.zuschnitt), werte.perspektive_v, werte.perspektive_h,
                     werte.verzeichnung, werte.vignette, werte.ca_rot, werte.ca_blau)


def rahmen(form_quelle, g: Geometrie) -> tuple[int, int]:
    """Hoehe und Breite des gedrehten, noch nicht zugeschnittenen Bildes."""
    hoehe, breite = form_quelle[:2]
    return (breite, hoehe) if g.drehung90 % 2 else (hoehe, breite)


def ausgabe_form(form_quelle, g: Geometrie) -> tuple[int, int]:
    hf, wf = rahmen(form_quelle, g)
    x0, y0, x1, y1 = g.zuschnitt
    return max(1, round((y1 - y0) * hf)), max(1, round((x1 - x0) * wf))


@dataclass(frozen=True)
class _Parameter:
    """Alles, was die Abbildung fuer ein Bild braucht - fuer Referenz und Kernel gleich."""
    hs: int
    ws: int
    hf: float
    wf: float
    hout: int
    wout: int
    x0: float
    y0: float
    sx: float                # Breite des Zuschnitts im Rahmen
    sy: float
    zoom: float
    cos_t: float
    sin_t: float
    a: float                 # Perspektive senkrecht
    b: float                 # Perspektive waagrecht
    drehung90: int
    spiegeln: int
    k: float                 # Verzeichnung
    halbdiagonale2: float
    vignette: float
    skala: tuple[float, float, float]   # je Kanal, fuer die Farbsaeume


def _parameter(form_quelle, g: Geometrie, zoom: float, form_aus=None) -> _Parameter:
    hs, ws = form_quelle[:2]
    hf, wf = rahmen(form_quelle, g)
    hout, wout = form_aus or ausgabe_form(form_quelle, g)
    x0, y0, x1, y1 = g.zuschnitt
    winkel = math.radians(g.begradigen)
    return _Parameter(
        hs, ws, float(hf), float(wf), hout, wout, x0, y0, x1 - x0, y1 - y0, zoom,
        math.cos(winkel), math.sin(winkel), g.perspektive_v / 100 * PERSPEKTIVE,
        g.perspektive_h / 100 * PERSPEKTIVE, g.drehung90 % 4, int(g.spiegeln),
        -g.verzeichnung / 100 * VERZEICHNUNG, (hs * hs + ws * ws) / 4.0,
        g.vignette / 100 * VIGNETTE,
        (1 + g.ca_rot / 100 * FARBSAEUME, 1.0, 1 + g.ca_blau / 100 * FARBSAEUME))


def _rueckwaerts(xp, p: _Parameter, u, v):
    """Ergebnis (u, v in 0..1) -> Ort im Original, zentriert, vor der Verzeichnung."""
    px = p.x0 + u * p.sx
    py = p.y0 + v * p.sy
    qx = (px - 0.5) * p.wf / p.zoom
    qy = (py - 0.5) * p.hf / p.zoom
    rx = p.cos_t * qx + p.sin_t * qy
    ry = -p.sin_t * qx + p.cos_t * qy
    w = 1 + p.a * (ry / (p.hf / 2)) + p.b * (rx / (p.wf / 2))
    tx, ty = rx / w, ry / w
    if p.spiegeln:
        tx = -tx
    if p.drehung90 == 1:
        return ty, -tx
    if p.drehung90 == 2:
        return -tx, -ty
    if p.drehung90 == 3:
        return -ty, tx
    return tx, ty


def _verzeichnet(p: _Parameter, cx, cy):
    faktor = 1 + p.k * (cx * cx + cy * cy) / p.halbdiagonale2
    return cx * faktor, cy * faktor


def automatischer_zoom(form_quelle, g: Geometrie) -> float:
    """Kleinste Vergroesserung, bei der der ganze Rahmen im Original liegt.

    Geprueft werden Punkte entlang des Rahmens (ohne Zuschnitt); die Suche
    halbiert das Intervall, bis die Vergroesserung auf 0,01 % genau ist.
    """
    voll = g.ohne_zuschnitt()
    if g.begradigen == 0 and g.perspektive_v == 0 and g.perspektive_h == 0 \
            and g.verzeichnung == 0:
        return 1.0
    t = np.linspace(0.0, 1.0, 65)
    u = np.concatenate([t, t, np.zeros_like(t), np.ones_like(t)])
    v = np.concatenate([np.zeros_like(t), np.ones_like(t), t, t])

    def passt(zoom):
        p = _parameter(form_quelle, voll, zoom, (2, 2))
        cx, cy = _verzeichnet(p, *_rueckwaerts(np, p, u, v))
        skala = max(p.skala)
        return bool((np.abs(cx * skala) <= p.ws / 2 - 0.5).all()
                    and (np.abs(cy * skala) <= p.hs / 2 - 0.5).all())

    unten, oben = 1.0, 1.0
    while not passt(oben) and oben < 16:
        unten, oben = oben, oben * 1.5
    if unten == oben:
        return 1.0
    for _ in range(40):
        mitte = (unten + oben) / 2
        unten, oben = (unten, mitte) if passt(mitte) else (mitte, oben)
    return oben


def zur_quelle(form_quelle, g: Geometrie, x: float, y: float) -> tuple[float, float]:
    """Punkt (x, y) im fertigen Bild, in Pixeln - wo liegt er im Original?"""
    p = _parameter(form_quelle, g, automatischer_zoom(form_quelle, g))
    cx, cy = _verzeichnet(p, *_rueckwaerts(np, p, x / p.wout, y / p.hout))
    return float(cx + p.ws / 2), float(cy + p.hs / 2)


def von_quelle(form_quelle, g: Geometrie, punkte) -> np.ndarray:
    """Punkte im Original (n, 2), in Pixeln - wo liegen sie im fertigen Bild?

    Die Abbildung ist nur rueckwaerts geschlossen formuliert; vorwaerts loest das
    Newton-Verfahren sie fuer alle Punkte zugleich. Sie ist fast affin, nach
    wenigen Schritten stimmt es auf Bruchteile eines Pixels."""
    punkte = np.asarray(punkte, dtype=np.float64).reshape(-1, 2)
    p = _parameter(form_quelle, g, automatischer_zoom(form_quelle, g))
    sx, sy = punkte[:, 0], punkte[:, 1]

    def rueck(u, v):
        cx, cy = _verzeichnet(p, *_rueckwaerts(np, p, u, v))
        return cx + p.ws / 2, cy + p.hs / 2

    u = np.full(len(punkte), 0.5)
    v = np.full(len(punkte), 0.5)
    eps = 1e-5
    for _ in range(12):
        qx, qy = rueck(u, v)
        ax, ay = rueck(u + eps, v)
        bx, by = rueck(u, v + eps)
        j11, j12 = (ax - qx) / eps, (bx - qx) / eps
        j21, j22 = (ay - qy) / eps, (by - qy) / eps
        det = j11 * j22 - j12 * j21
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        fx, fy = qx - sx, qy - sy
        du, dv = (j22 * fx - j12 * fy) / det, (j11 * fy - j21 * fx) / det
        u, v = u - du, v - dv
        if max(np.abs(du).max(initial=0), np.abs(dv).max(initial=0)) < 1e-9:
            break
    return np.stack([u * p.wout, v * p.hout], axis=1)


def _catmull_rom(xp, t):
    """Gewichte der vier Nachbarn fuer den Anteil t (Catmull-Rom, a = -0.5)."""
    t2, t3 = t * t, t * t * t
    return ((-0.5 * t3 + t2 - 0.5 * t), (1.5 * t3 - 2.5 * t2 + 1),
            (-1.5 * t3 + 2 * t2 + 0.5 * t), (0.5 * t3 - 0.5 * t2))


def _bikubisch(xp, kanal, sx, sy):
    """Kanal (H, W) an den Stellen (sx, sy) lesen, Rand fortgesetzt."""
    hoehe, breite = kanal.shape
    x0, y0 = xp.floor(sx), xp.floor(sy)
    gx, gy = _catmull_rom(xp, sx - x0), _catmull_rom(xp, sy - y0)
    x0, y0 = x0.astype(xp.int32), y0.astype(xp.int32)
    ergebnis = 0
    for j in range(4):
        zeile = xp.clip(y0 + j - 1, 0, hoehe - 1)
        teil = 0
        for i in range(4):
            spalte = xp.clip(x0 + i - 1, 0, breite - 1)
            teil = teil + gx[i] * kanal[zeile, spalte]
        ergebnis = ergebnis + gy[j] * teil
    return ergebnis


def anwenden(rgb, g: Geometrie, form_aus=None):
    """Geometrie auf ein lineares Bild (H, W, 3) - NumPy, CuPy oder GPU-Kernel."""
    if g.ist_neutral():
        return rgb
    xp = f.xp_von(rgb)
    p = _parameter(rgb.shape, g, automatischer_zoom(rgb.shape, g), form_aus)
    if xp is not np:
        return _anwenden_gpu(rgb, p)
    v, u = np.meshgrid((np.arange(p.hout) + 0.5) / p.hout, (np.arange(p.wout) + 0.5) / p.wout,
                       indexing="ij")
    cx, cy = _verzeichnet(p, *_rueckwaerts(np, p, u, v))
    vignette = 1 + p.vignette * (cx * cx + cy * cy) / p.halbdiagonale2
    kanaele = []
    for nummer in range(3):
        sx = cx * p.skala[nummer] + p.ws / 2 - 0.5
        sy = cy * p.skala[nummer] + p.hs / 2 - 0.5
        kanaele.append(_bikubisch(np, rgb[..., nummer], sx, sy) * vignette)
    return np.stack(kanaele, axis=-1).astype(np.float32)


# --------------------------------------------------------------------------
# CUDA
# --------------------------------------------------------------------------

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        from .cuda import cupy as cp
        _KERNEL = cp.ElementwiseKernel(
            "raw float32 quelle, raw float32 p, int32 hs, int32 ws, int32 hout, int32 wout, "
            "int32 drehung90, int32 spiegeln",
            "raw float32 ziel",
            r"""
            // p: hf wf x0 y0 sx sy zoom cos sin a b k halbdiag2 vignette skala_r skala_g skala_b
            float hf = p[0], wf = p[1];
            int i_y = i / wout, i_x = i % wout;
            float u = ((float)i_x + 0.5f) / (float)wout, v = ((float)i_y + 0.5f) / (float)hout;
            float px = p[2] + u * p[4], py = p[3] + v * p[5];
            float qx = (px - 0.5f) * wf / p[6], qy = (py - 0.5f) * hf / p[6];
            float rx = p[7] * qx + p[8] * qy, ry = -p[8] * qx + p[7] * qy;
            float w = 1.0f + p[9] * (ry / (hf / 2.0f)) + p[10] * (rx / (wf / 2.0f));
            float tx = rx / w, ty = ry / w;
            if (spiegeln) tx = -tx;
            float cx, cy;
            if (drehung90 == 1) { cx = ty; cy = -tx; }
            else if (drehung90 == 2) { cx = -tx; cy = -ty; }
            else if (drehung90 == 3) { cx = -ty; cy = tx; }
            else { cx = tx; cy = ty; }
            float faktor = 1.0f + p[11] * (cx * cx + cy * cy) / p[12];
            cx *= faktor; cy *= faktor;
            float vignette = 1.0f + p[13] * (cx * cx + cy * cy) / p[12];
            for (int k = 0; k < 3; k++) {
                float sx = cx * p[14 + k] + (float)ws / 2.0f - 0.5f;
                float sy = cy * p[14 + k] + (float)hs / 2.0f - 0.5f;
                float x0 = floorf(sx), y0 = floorf(sy);
                float gx[4], gy[4];
                gewichte(sx - x0, gx);
                gewichte(sy - y0, gy);
                float summe = 0.0f;
                for (int j = 0; j < 4; j++) {
                    int zeile = min(max((int)y0 + j - 1, 0), hs - 1);
                    float teil = 0.0f;
                    for (int m = 0; m < 4; m++) {
                        int spalte = min(max((int)x0 + m - 1, 0), ws - 1);
                        teil += gx[m] * quelle[(zeile * ws + spalte) * 3 + k];
                    }
                    summe += gy[j] * teil;
                }
                ziel[3 * i + k] = summe * vignette;
            }
            """,
            "silberkorn_geometrie",
            preamble=r"""
            __device__ __forceinline__ void gewichte(float t, float* g) {
                float t2 = t * t, t3 = t2 * t;
                g[0] = -0.5f * t3 + t2 - 0.5f * t;
                g[1] = 1.5f * t3 - 2.5f * t2 + 1.0f;
                g[2] = -1.5f * t3 + 2.0f * t2 + 0.5f * t;
                g[3] = 0.5f * t3 - 0.5f * t2;
            }
            """)
    return _KERNEL


def _anwenden_gpu(rgb, p: _Parameter):
    from .cuda import cupy as cp
    werte = np.array([p.hf, p.wf, p.x0, p.y0, p.sx, p.sy, p.zoom, p.cos_t, p.sin_t, p.a, p.b,
                      p.k, p.halbdiagonale2, p.vignette, *p.skala], dtype=np.float32)
    ziel = cp.empty((p.hout, p.wout, 3), dtype=cp.float32)
    _kernel()(cp.ascontiguousarray(rgb, dtype=cp.float32), cp.asarray(werte), cp.int32(p.hs),
              cp.int32(p.ws), cp.int32(p.hout), cp.int32(p.wout), cp.int32(p.drehung90),
              cp.int32(p.spiegeln), ziel, size=p.hout * p.wout)
    return ziel
