"""
Klassische Filter als zusammengefasste CUDA-Kernel

filter.py beschreibt jeden Schritt mit Array-Operationen. Auf der Grafikkarte
waere das ein eigener Kernelaufruf mit eigenem Zwischenspeicher je Operation -
bei einer 10-Megapixel-Vorschau rund 90 ms. Hier sind dieselben Formeln zu drei
Durchlaeufen zusammengefasst:

  1. punkt:     Weissabgleich, Belichtung, Tonwerte, Farbe - je Pixel
  2. helligkeit: wahrnehmungsgerechte Helligkeit fuer das Schaerfen
  3. ausgabe:   Schaerfen und Umwandlung nach sRGB-uint8

dazwischen die Gauss-Unschaerfe als zwei eindimensionale Faltungen.

Die Formeln muessen mit filter.py uebereinstimmen; tests/test_filter.py
vergleicht beide Wege Pixel fuer Pixel.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import math

from cupyx.scipy import ndimage

from . import filter as f
from .cuda import cupy as cp

_GEMEINSAM = r"""
#define EPS 1e-6f
__device__ __forceinline__ float luma(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}
__device__ __forceinline__ float kodieren(float x) {      // linear -> sRGB
    x = fmaxf(x, 0.0f);
    return x <= 0.0031308f ? x * 12.92f : 1.055f * powf(x, 1.0f / 2.4f) - 0.055f;
}
__device__ __forceinline__ float dekodieren(float v) {    // sRGB -> linear
    return v <= 0.04045f ? v / 12.92f : powf((v + 0.055f) / 1.055f, 2.4f);
}
__device__ __forceinline__ float glatt(float k0, float k1, float v) {
    float t = fminf(fmaxf((v - k0) / (k1 - k0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
"""

_punkt = cp.ElementwiseKernel(
    "raw float32 quelle, float32 gr, float32 gg, float32 gb, int32 ton, float32 kontrast, "
    "float32 lichter, float32 tiefen, int32 farbig, float32 saettigung, float32 dynamik",
    "raw float32 ziel",
    r"""
    float r = quelle[3 * i] * gr, g = quelle[3 * i + 1] * gg, b = quelle[3 * i + 2] * gb;
    if (ton) {
        float y = luma(r, g, b);
        float v = kodieren(y);
        float innen = fminf(fmaxf(v, 0.0f), 1.0f);
        float ueber = v - innen;
        if (kontrast > 0.0f) {
            innen = innen + kontrast * (glatt(0.0f, 1.0f, innen) - innen);
        } else if (kontrast < 0.0f) {
            innen = innen + (-kontrast) * ((0.5f + (innen - 0.5f) * 0.5f) - innen);
        }
        if (lichter != 0.0f) {
            innen = innen + lichter * 0.6f * glatt(0.5f, 1.0f, innen) * (innen - 0.5f);
        }
        if (tiefen != 0.0f) {
            innen = innen + tiefen * 0.6f * (1.0f - glatt(0.0f, 0.5f, innen)) * (0.5f - innen);
        }
        float y_neu = dekodieren(fminf(fmaxf(innen, 0.0f), 1.0f) + ueber);
        if (y <= EPS) {
            r = g = b = y_neu;
        } else {
            float q = y_neu / y;
            r *= q; g *= q; b *= q;
        }
    }
    if (farbig) {
        float y = luma(r, g, b);
        float faktor = 1.0f + saettigung;
        if (dynamik != 0.0f) {
            float hoechster = fmaxf(r, fmaxf(g, b));
            float niedrigster = fminf(r, fminf(g, b));
            float vorhanden = (hoechster - niedrigster) / fmaxf(hoechster, EPS);
            faktor = faktor * (1.0f + dynamik * (1.0f - vorhanden));
        }
        faktor = fmaxf(faktor, 0.0f);
        r = y + (r - y) * faktor; g = y + (g - y) * faktor; b = y + (b - y) * faktor;
    }
    ziel[3 * i] = r; ziel[3 * i + 1] = g; ziel[3 * i + 2] = b;
    """,
    "neuro_enhance_punkt", preamble=_GEMEINSAM)

_helligkeit = cp.ElementwiseKernel(
    "raw float32 quelle", "raw float32 v",
    "v[i] = kodieren(luma(quelle[3 * i], quelle[3 * i + 1], quelle[3 * i + 2]));",
    "neuro_enhance_helligkeit", preamble=_GEMEINSAM)

_ausgabe = cp.ElementwiseKernel(
    "raw float32 quelle, raw float32 v, raw float32 weich, float32 staerke",
    "raw uint8 ziel",
    r"""
    float r = quelle[3 * i], g = quelle[3 * i + 1], b = quelle[3 * i + 2];
    if (staerke > 0.0f) {
        float y = luma(r, g, b);
        float v_neu = v[i] + staerke * (v[i] - weich[i]);
        float q = dekodieren(fmaxf(v_neu, 0.0f)) / fmaxf(y, EPS);
        r *= q; g *= q; b *= q;
    }
    ziel[3 * i] = (unsigned char)(kodieren(fminf(fmaxf(r, 0.0f), 1.0f)) * 255.0f + 0.5f);
    ziel[3 * i + 1] = (unsigned char)(kodieren(fminf(fmaxf(g, 0.0f), 1.0f)) * 255.0f + 0.5f);
    ziel[3 * i + 2] = (unsigned char)(kodieren(fminf(fmaxf(b, 0.0f), 1.0f)) * 255.0f + 0.5f);
    """,
    "neuro_enhance_ausgabe", preamble=_GEMEINSAM)


def _verstaerkungen(werte: f.Einstellungen) -> list[float]:
    """Weissabgleich und Belichtung als ein Faktor je Kanal - wie in filter.py."""
    k = werte.temperatur / 100 * 0.35
    m = werte.toenung / 100 * 0.25
    faktoren = [math.exp(k), math.exp(-m), math.exp(-k)]
    norm = sum(a * w for a, w in zip(faktoren, f.LUMA, strict=True))
    belichtung = 2.0 ** werte.belichtung
    return [a / norm * belichtung for a in faktoren]


def anwenden_8bit(rgb_linear, werte: f.Einstellungen, massstab: float = 1.0):
    """Ganze Filterkette auf der Grafikkarte, Ergebnis als sRGB-uint8 (H, W, 3)."""
    hoehe, breite = rgb_linear.shape[:2]
    pixel = hoehe * breite
    quelle = cp.ascontiguousarray(rgb_linear, dtype=cp.float32)

    gr, gg, gb = _verstaerkungen(werte)
    ton = int(werte.kontrast != 0 or werte.lichter != 0 or werte.tiefen != 0)
    farbig = int(werte.dynamik != 0 or werte.saettigung != 0)
    zwischen = cp.empty_like(quelle)
    _punkt(quelle, cp.float32(gr), cp.float32(gg), cp.float32(gb), cp.int32(ton),
           cp.float32(werte.kontrast / 100), cp.float32(werte.lichter / 100),
           cp.float32(werte.tiefen / 100), cp.int32(farbig),
           cp.float32(werte.saettigung / 100), cp.float32(werte.dynamik / 100),
           zwischen, size=pixel)

    staerke = werte.schaerfe / 100
    radius = werte.schaerfe_radius * massstab
    if staerke > 0 and radius > 0:
        v = cp.empty((hoehe, breite), dtype=cp.float32)
        _helligkeit(zwischen, v, size=pixel)
        kern = cp.asarray(f.gauss_kern(radius))
        # "mirror" spiegelt ohne den Randpixel zu wiederholen - wie np.pad(mode="reflect")
        weich = ndimage.correlate1d(v, kern, axis=0, mode="mirror")
        weich = ndimage.correlate1d(weich, kern, axis=1, mode="mirror")
    else:
        staerke = 0.0
        v = weich = zwischen                  # werden bei Staerke 0 nicht gelesen

    ziel = cp.empty((hoehe, breite, 3), dtype=cp.uint8)
    _ausgabe(zwischen, v, weich, cp.float32(staerke), ziel, size=pixel)
    return ziel
