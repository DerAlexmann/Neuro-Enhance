"""
Klassische Filter als zusammengefasste CUDA-Kernel

filter.py beschreibt jeden Schritt mit Array-Operationen. Auf der Grafikkarte
waere das ein eigener Kernelaufruf mit eigenem Zwischenspeicher je Operation -
bei einer 10-Megapixel-Vorschau rund 90 ms. Hier sind dieselben Formeln zu
wenigen Durchlaeufen zusammengefasst:

  licht:          Weissabgleich, Belichtung, Tonwerte
  klarheit:       lokaler Kontrast aus Helligkeit und ihrer weichen Fassung
  kurven_farbe:   Gradationskurven, Dynamik, Saettigung
  ausgabe:        Schaerfen und Umwandlung nach sRGB-uint8

Was auf grossen Nachbarschaften rechnet - Dunst schaetzen, Klarheit und
Schaerfen weichzeichnen -, uebernimmt filter.py unveraendert; diese Schritte
laufen auf verkleinerten Kopien und kosten wenig.

Die Formeln muessen mit filter.py uebereinstimmen; tests/test_filter.py
vergleicht beide Wege Pixel fuer Pixel.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import math

import numpy as np
from cupyx.scipy import ndimage

from . import filter as f
from . import kurven
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
// Bilinear aus einem kleinen Bild (h0 x b0) an der Stelle des grossen Pixels
// (y, x) lesen - dieselbe Rechnung wie filter.vergroessern().
__device__ __forceinline__ float bilinear(const float* klein, int h0, int b0, float sy,
                                          float sx, int y, int x) {
    float fy = fminf(fmaxf(((float)y + 0.5f) * sy - 0.5f, 0.0f), (float)(h0 - 1));
    float fx = fminf(fmaxf(((float)x + 0.5f) * sx - 0.5f, 0.0f), (float)(b0 - 1));
    int y0 = (int)floorf(fy), x0 = (int)floorf(fx);
    int y1 = min(y0 + 1, h0 - 1), x1 = min(x0 + 1, b0 - 1);
    float ay = fy - (float)y0, ax = fx - (float)x0;
    float oben = klein[y0 * b0 + x0] * (1.0f - ax) + klein[y0 * b0 + x1] * ax;
    float unten = klein[y1 * b0 + x0] * (1.0f - ax) + klein[y1 * b0 + x1] * ax;
    return oben * (1.0f - ay) + unten * ay;
}
__device__ __forceinline__ float nachschlagen(const float* tabelle, int n, float v) {
    float pos = fminf(fmaxf(v, 0.0f), 1.0f) * (float)(n - 1);
    int i0 = min((int)floorf(pos), n - 2);
    float anteil = pos - (float)i0;
    return tabelle[i0] * (1.0f - anteil) + tabelle[i0 + 1] * anteil;
}
"""

_licht = cp.ElementwiseKernel(
    "raw float32 quelle, float32 gr, float32 gg, float32 gb, int32 ton, float32 kontrast, "
    "float32 lichter, float32 tiefen",
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
    ziel[3 * i] = r; ziel[3 * i + 1] = g; ziel[3 * i + 2] = b;
    """,
    "neuro_enhance_licht", preamble=_GEMEINSAM)

_helligkeit = cp.ElementwiseKernel(
    "raw float32 quelle", "raw float32 v",
    "v[i] = kodieren(luma(quelle[3 * i], quelle[3 * i + 1], quelle[3 * i + 2]));",
    "neuro_enhance_helligkeit", preamble=_GEMEINSAM)

_dunst = cp.ElementwiseKernel(
    "raw float32 quelle, raw float32 durchlass, int32 h0, int32 b0, float32 sy, float32 sx, "
    "int32 breite, float32 lr, float32 lg, float32 lb, float32 staerke, float32 minimum",
    "raw float32 ziel",
    r"""
    float r = quelle[3 * i], g = quelle[3 * i + 1], b = quelle[3 * i + 2];
    if (staerke < 0.0f) {
        float anteil = -staerke * 0.6f;
        r = r * (1.0f - anteil) + lr * anteil;
        g = g * (1.0f - anteil) + lg * anteil;
        b = b * (1.0f - anteil) + lb * anteil;
    } else {
        float t = bilinear(&durchlass[0], h0, b0, sy, sx, i / breite, i % breite);
        float d = fmaxf(t, minimum);
        r = fmaxf((r - lr) / d + lr, 0.0f);
        g = fmaxf((g - lg) / d + lg, 0.0f);
        b = fmaxf((b - lb) / d + lb, 0.0f);
    }
    ziel[3 * i] = r; ziel[3 * i + 1] = g; ziel[3 * i + 2] = b;
    """,
    "neuro_enhance_dunst", preamble=_GEMEINSAM)

_klarheit = cp.ElementwiseKernel(
    "raw float32 quelle, raw float32 v, raw float32 weich, int32 h0, int32 b0, float32 sy, "
    "float32 sx, int32 breite, float32 staerke",
    "raw float32 ziel",
    r"""
    float r = quelle[3 * i], g = quelle[3 * i + 1], b = quelle[3 * i + 2];
    float w = bilinear(&weich[0], h0, b0, sy, sx, i / breite, i % breite);
    float gewicht = fminf(fmaxf(4.0f * v[i] * (1.0f - v[i]), 0.0f), 1.0f);
    float v_neu = v[i] + staerke * gewicht * (v[i] - w);
    float q = dekodieren(fmaxf(v_neu, 0.0f)) / fmaxf(luma(r, g, b), EPS);
    ziel[3 * i] = r * q; ziel[3 * i + 1] = g * q; ziel[3 * i + 2] = b * q;
    """,
    "neuro_enhance_klarheit", preamble=_GEMEINSAM)

_kurven_farbe = cp.ElementwiseKernel(
    "raw float32 quelle, raw float32 tabellen, int32 n, int32 hell, int32 rot, int32 gruen, "
    "int32 blau, int32 farbig, float32 saettigung, float32 dynamik",
    "raw float32 ziel",
    r"""
    float r = quelle[3 * i], g = quelle[3 * i + 1], b = quelle[3 * i + 2];
    if (hell) {
        float y = luma(r, g, b);
        float v = kodieren(y);
        float innen = fminf(fmaxf(v, 0.0f), 1.0f);
        float y_neu = dekodieren(nachschlagen(&tabellen[0], n, innen) + (v - innen));
        if (y <= EPS) {
            r = g = b = y_neu;
        } else {
            float q = y_neu / y;
            r *= q; g *= q; b *= q;
        }
    }
    if (rot) {
        float v = kodieren(r), innen = fminf(fmaxf(v, 0.0f), 1.0f);
        r = dekodieren(nachschlagen(&tabellen[n], n, innen) + (v - innen));
    }
    if (gruen) {
        float v = kodieren(g), innen = fminf(fmaxf(v, 0.0f), 1.0f);
        g = dekodieren(nachschlagen(&tabellen[2 * n], n, innen) + (v - innen));
    }
    if (blau) {
        float v = kodieren(b), innen = fminf(fmaxf(v, 0.0f), 1.0f);
        b = dekodieren(nachschlagen(&tabellen[3 * n], n, innen) + (v - innen));
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
    "neuro_enhance_kurven_farbe", preamble=_GEMEINSAM)

_AUSGABE = r"""
    float r = quelle[3 * i], g = quelle[3 * i + 1], b = quelle[3 * i + 2];
    if (staerke > 0.0f) {
        float y = luma(r, g, b);
        float v_neu = v[i] + staerke * (v[i] - weich[i]);
        float q = dekodieren(fmaxf(v_neu, 0.0f)) / fmaxf(y, EPS);
        r *= q; g *= q; b *= q;
    }
    ziel[3 * i] = (TYP)(kodieren(fminf(fmaxf(r, 0.0f), 1.0f)) * HOECHST + 0.5f);
    ziel[3 * i + 1] = (TYP)(kodieren(fminf(fmaxf(g, 0.0f), 1.0f)) * HOECHST + 0.5f);
    ziel[3 * i + 2] = (TYP)(kodieren(fminf(fmaxf(b, 0.0f), 1.0f)) * HOECHST + 0.5f);
"""

# Dieselbe Ausgabe fuer 8 und 16 Bit - nur Zieltyp und Hoechstwert unterscheiden sich
_ausgabe = {
    bits: cp.ElementwiseKernel(
        "raw float32 quelle, raw float32 v, raw float32 weich, float32 staerke",
        f"raw {typ} ziel",
        _AUSGABE.replace("TYP", ctyp).replace("HOECHST", hoechst),
        f"neuro_enhance_ausgabe{bits}", preamble=_GEMEINSAM)
    for bits, typ, ctyp, hoechst in ((8, "uint8", "unsigned char", "255.0f"),
                                     (16, "uint16", "unsigned short", "65535.0f"))
}


def _verstaerkungen(werte: f.Einstellungen) -> list[float]:
    """Weissabgleich und Belichtung als ein Faktor je Kanal - wie in filter.py."""
    k = werte.temperatur / 100 * 0.35
    m = werte.toenung / 100 * 0.25
    faktoren = [math.exp(k), math.exp(-m), math.exp(-k)]
    norm = sum(a * w for a, w in zip(faktoren, f.LUMA, strict=True))
    belichtung = 2.0 ** werte.belichtung
    return [a / norm * belichtung for a in faktoren]


def _helligkeit_von(rgb, pixel):
    v = cp.empty(rgb.shape[:2], dtype=cp.float32)
    _helligkeit(rgb, v, size=pixel)
    return v


def _skalierung(klein, hoehe, breite):
    """Kantenmasse und Schrittweiten fuer bilinear() - wie in filter.vergroessern()."""
    h0, b0 = klein.shape[:2]
    return (cp.int32(h0), cp.int32(b0), cp.float32(np.float32(h0 / hoehe)),
            cp.float32(np.float32(b0 / breite)), cp.int32(breite))


def _dunst_schaetzung(bild, werte: f.Einstellungen, speicher: dict | None):
    """Lichtfarbe und Durchlasskarte - aus dem Zwischenspeicher, wenn moeglich.

    Die Schaetzung haengt nur vom Bild vor dem Dunstschritt ab, also von
    Weissabgleich und Belichtung - nicht von der Dunststaerke selbst. Beim
    Ziehen am Dunstregler muss sie deshalb nur einmal gerechnet werden.
    """
    schluessel = ("dunst", bild.shape, werte.temperatur, werte.toenung, werte.belichtung)
    if speicher is not None and schluessel in speicher:
        return speicher[schluessel]
    ergebnis = f.dunst_schaetzen(bild)
    if speicher is not None:
        for alt in [k for k in speicher if k[0] == "dunst" and k[1] == bild.shape]:
            del speicher[alt]
        speicher[schluessel] = ergebnis
    return ergebnis


def anwenden_ausgabe(rgb_linear, werte: f.Einstellungen, massstab: float = 1.0,
                     speicher: dict | None = None, bits: int = 8):
    """Ganze Filterkette auf der Grafikkarte, Ergebnis als sRGB mit 8 oder 16 Bit (H, W, 3)."""
    hoehe, breite = rgb_linear.shape[:2]
    pixel = hoehe * breite
    bild = cp.ascontiguousarray(rgb_linear, dtype=cp.float32)

    gr, gg, gb = _verstaerkungen(werte)
    ton = int(werte.kontrast != 0 or werte.lichter != 0 or werte.tiefen != 0)
    tonwerte = (cp.float32(werte.kontrast / 100), cp.float32(werte.lichter / 100),
                cp.float32(werte.tiefen / 100))

    # Weissabgleich, Belichtung - und, wenn kein Dunst zu entfernen ist, gleich
    # die Tonwerte im selben Durchlauf. Sonst gehoert der Dunst dazwischen.
    zwischen = cp.empty_like(bild)
    erst_dunst = werte.dunst != 0
    _licht(bild, cp.float32(gr), cp.float32(gg), cp.float32(gb),
           cp.int32(0 if erst_dunst else ton), *tonwerte, zwischen, size=pixel)
    if erst_dunst:
        licht, durchlass = _dunst_schaetzung(zwischen, werte, speicher)
        staerke = werte.dunst / 100
        # Die Staerke mischt wie in filter.dunst() schon auf der kleinen Karte.
        durchlass = cp.ascontiguousarray(1 - cp.float32(staerke) * (1 - durchlass))
        lr, lg, lb = (cp.float32(wert) for wert in cp.asnumpy(licht))
        entdunstet = cp.empty_like(zwischen)
        _dunst(zwischen, durchlass, *_skalierung(durchlass, hoehe, breite), lr, lg, lb,
               cp.float32(staerke), cp.float32(f.DUNST_MIN_DURCHLASS), entdunstet, size=pixel)
        if ton:
            _licht(entdunstet, cp.float32(1), cp.float32(1), cp.float32(1), cp.int32(1),
                   *tonwerte, zwischen, size=pixel)
        else:
            zwischen = entdunstet

    if werte.klarheit != 0:
        v = _helligkeit_von(zwischen, pixel)
        weich = cp.ascontiguousarray(f.grob_weich_klein(v, f.klarheit_sigma(zwischen.shape)))
        _klarheit(zwischen, v, weich, *_skalierung(weich, hoehe, breite),
                  cp.float32(werte.klarheit / 100), zwischen, size=pixel)

    tabellen = f.kurven_tabellen(werte)
    farbig = int(werte.dynamik != 0 or werte.saettigung != 0)
    if tabellen or farbig:
        alle = np.concatenate([tabellen.get(name, kurven.tabelle(kurven.IDENTITAET))
                               for name in f.KURVEN])
        _kurven_farbe(zwischen, cp.asarray(alle), cp.int32(kurven.TABELLE),
                      *(cp.int32(name in tabellen) for name in f.KURVEN), cp.int32(farbig),
                      cp.float32(werte.saettigung / 100), cp.float32(werte.dynamik / 100),
                      zwischen, size=pixel)

    staerke = werte.schaerfe / 100
    radius = werte.schaerfe_radius * massstab
    if staerke > 0 and radius > 0:
        v = _helligkeit_von(zwischen, pixel)
        kern = cp.asarray(f.gauss_kern(radius))
        # "mirror" spiegelt ohne den Randpixel zu wiederholen - wie np.pad(mode="reflect")
        weich = ndimage.correlate1d(v, kern, axis=0, mode="mirror")
        weich = ndimage.correlate1d(weich, kern, axis=1, mode="mirror")
    else:
        staerke = 0.0
        v = weich = zwischen                  # werden bei Staerke 0 nicht gelesen

    ziel = cp.empty((hoehe, breite, 3), dtype=cp.uint16 if bits == 16 else cp.uint8)
    _ausgabe[bits](zwischen, v, weich, cp.float32(staerke), ziel, size=pixel)
    return ziel
