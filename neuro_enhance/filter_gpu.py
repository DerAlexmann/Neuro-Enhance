"""
Klassische Filter als zusammengefasste CUDA-Kernel

filter.py beschreibt jeden Schritt mit Array-Operationen. Auf der Grafikkarte
waere das ein eigener Kernelaufruf mit eigenem Zwischenspeicher je Operation -
bei einer 10-Megapixel-Vorschau rund 90 ms. Hier sind dieselben Formeln zu
wenigen Durchlaeufen zusammengefasst:

  licht:          Weissabgleich, Belichtung, Tonwerte
  nlm, ...:       Entrauschen - Ergebnis je Bild und Einstellung zwischengespeichert
  klarheit:       lokaler Kontrast aus Helligkeit und ihrer weichen Fassung
  kurven_farbe:   Gradationskurven, HSL je Farbbereich, Dynamik, Saettigung, LUT
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
from . import geometrie, kurven
from .cuda import cupy as cp


def _c_feld(name, werte):
    """Konstantes float-Feld fuer den CUDA-Code - mit denselben float32-Werten wie filter.py."""
    zahlen = ", ".join(f"{float(np.float32(w))!r}f" for w in np.ravel(werte))
    return f"__device__ const float {name}[{np.size(werte)}] = {{{zahlen}}};\n"


_GEMEINSAM = (
    _c_feld("OKLAB_M1", f.OKLAB_M1) + _c_feld("OKLAB_M2", f.OKLAB_M2)
    + _c_feld("OKLAB_M1_INV", f.OKLAB_M1_INV) + _c_feld("OKLAB_M2_INV", f.OKLAB_M2_INV)
    + _c_feld("BEREICH_MITTE", f.FARBBEREICH_MITTE)
    + f"#define BEREICHE {len(f.FARBBEREICHE)}\n"
    + f"#define HSL_FARBTON_GRAD {f.HSL_FARBTON_GRAD}f\n"
    + f"#define HSL_LUMINANZ {f.HSL_LUMINANZ}f\n"
    + f"#define HSL_CHROMA_VOLL {f.HSL_CHROMA_VOLL}f\n"
) + r"""
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
__device__ __forceinline__ void matrix3(const float* m, float& a, float& b, float& c) {
    float x = m[0] * a + m[1] * b + m[2] * c;
    float y = m[3] * a + m[4] * b + m[5] * c;
    float z = m[6] * a + m[7] * b + m[8] * c;
    a = x; b = y; c = z;
}
// HSL je Farbbereich in OkLCh - dieselbe Rechnung wie filter.hsl()
__device__ void hsl_verschieben(float& r, float& g, float& b, const float* werte) {
    float l = r, a = g, bb = b;
    matrix3(OKLAB_M1, l, a, bb);
    l = cbrtf(l); a = cbrtf(a); bb = cbrtf(bb);
    matrix3(OKLAB_M2, l, a, bb);
    float chroma = sqrtf(a * a + bb * bb);
    float winkel = atan2f(bb, a) * (180.0f / 3.14159265358979f);
    winkel = winkel - 360.0f * floorf(winkel / 360.0f);
    float d_ton = 0.0f, d_satt = 0.0f, d_lum = 0.0f;
    for (int k = 0; k < BEREICHE; k++) {
        float mitte = BEREICH_MITTE[k];
        float vorher = k == 0 ? BEREICH_MITTE[BEREICHE - 1] - 360.0f : BEREICH_MITTE[k - 1];
        float nachher = k == BEREICHE - 1 ? BEREICH_MITTE[0] + 360.0f : BEREICH_MITTE[k + 1];
        float t = winkel - mitte + 180.0f;
        float abstand = t - 360.0f * floorf(t / 360.0f) - 180.0f;
        float gewicht = abstand < 0.0f
            ? 0.5f * (1.0f + cosf(3.14159265358979f
                                  * fminf(fmaxf(-abstand / (mitte - vorher), 0.0f), 1.0f)))
            : 0.5f * (1.0f + cosf(3.14159265358979f
                                  * fminf(fmaxf(abstand / (nachher - mitte), 0.0f), 1.0f)));
        d_ton += gewicht * werte[k];
        d_satt += gewicht * werte[BEREICHE + k];
        d_lum += gewicht * werte[2 * BEREICHE + k];
    }
    float winkel_neu = (winkel + d_ton * (HSL_FARBTON_GRAD / 100.0f))
                       * (3.14159265358979f / 180.0f);
    float chroma_neu = chroma * fmaxf(1.0f + d_satt * (1.0f / 100.0f), 0.0f);
    float buntheit = fminf(fmaxf(chroma / HSL_CHROMA_VOLL, 0.0f), 1.0f);
    l = l * (1.0f + d_lum * (HSL_LUMINANZ / 100.0f) * buntheit);
    a = chroma_neu * cosf(winkel_neu);
    bb = chroma_neu * sinf(winkel_neu);
    matrix3(OKLAB_M2_INV, l, a, bb);
    l = l * l * l; a = a * a * a; bb = bb * bb * bb;
    matrix3(OKLAB_M1_INV, l, a, bb);
    r = l; g = a; b = bb;
}
// Tetraedrische Interpolation in einer 3D-LUT [b][g][r] - wie lut.anwenden()
__device__ void lut_anwenden(float& r, float& g, float& b, const float* tabelle, int n,
                             const float* bereich, float staerke) {
    float v[3] = {kodieren(fminf(fmaxf(r, 0.0f), 1.0f)), kodieren(fminf(fmaxf(g, 0.0f), 1.0f)),
                  kodieren(fminf(fmaxf(b, 0.0f), 1.0f))};
    int basis[3];
    float anteil[3];
    for (int k = 0; k < 3; k++) {
        float pos = fminf(fmaxf((v[k] - bereich[k]) / bereich[3 + k], 0.0f), 1.0f) * (n - 1);
        basis[k] = min((int)floorf(pos), n - 2);
        anteil[k] = pos - (float)basis[k];
    }
    float fr = anteil[0], fg = anteil[1], fb = anteil[2];
    #define ECKE(dr, dg, db, k) tabelle[(((basis[2] + (db)) * n + basis[1] + (dg)) * n \
                                         + basis[0] + (dr)) * 3 + (k)]
    for (int k = 0; k < 3; k++) {
        float c000 = ECKE(0, 0, 0, k), c111 = ECKE(1, 1, 1, k);
        float c100 = ECKE(1, 0, 0, k), c010 = ECKE(0, 1, 0, k), c001 = ECKE(0, 0, 1, k);
        float c110 = ECKE(1, 1, 0, k), c101 = ECKE(1, 0, 1, k), c011 = ECKE(0, 1, 1, k);
        float aus;
        if (fr > fg) {
            if (fg > fb) aus = c000 + fr * (c100 - c000) + fg * (c110 - c100) + fb * (c111 - c110);
            else if (fr > fb) aus = c000 + fr * (c100 - c000) + fb * (c101 - c100)
                                    + fg * (c111 - c101);
            else aus = c000 + fb * (c001 - c000) + fr * (c101 - c001) + fg * (c111 - c101);
        } else {
            if (!(fg > fb)) aus = c000 + fb * (c001 - c000) + fg * (c011 - c001)
                                  + fr * (c111 - c011);
            else if (!(fr > fb)) aus = c000 + fg * (c010 - c000) + fb * (c011 - c010)
                                       + fr * (c111 - c011);
            else aus = c000 + fg * (c010 - c000) + fr * (c110 - c010) + fb * (c111 - c110);
        }
        v[k] = dekodieren(fminf(fmaxf(v[k] + staerke * (aus - v[k]), 0.0f), 1.0f));
    }
    #undef ECKE
    r = v[0]; g = v[1]; b = v[2];
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
    "int32 blau, int32 farbig, float32 saettigung, float32 dynamik, int32 hsl_an, "
    "raw float32 hsl, int32 lut_an, raw float32 lut, int32 lut_n, raw float32 lut_bereich, "
    "float32 lut_staerke",
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
    if (hsl_an) {
        hsl_verschieben(r, g, b, &hsl[0]);
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
    if (lut_an) {
        lut_anwenden(r, g, b, &lut[0], lut_n, &lut_bereich[0], lut_staerke);
    }
    ziel[3 * i] = r; ziel[3 * i + 1] = g; ziel[3 * i + 2] = b;
    """,
    "neuro_enhance_kurven_farbe", preamble=_GEMEINSAM)

# --------------------------------------------------------------------------
# Entrauschen
# --------------------------------------------------------------------------

_nlm = cp.ElementwiseKernel(
    "raw float32 v, int32 h, int32 w, float32 h2", "raw float32 ziel",
    f"#define SUCHE {f.NLM_SUCHE}\n#define FLECK {f.NLM_FLECK}\n" + r"""
    int y = i / w, x = i % w;
    // gespiegelt ohne Randpixel - wie np.pad(mode="reflect") in filter.nlm()
    #define M(yy, xx) v[spiegel(yy, h) * w + spiegel(xx, w)]
    const float flaeche = (float)((2 * FLECK + 1) * (2 * FLECK + 1));
    float summe = 0.0f, gewichte = 0.0f;
    for (int dy = -SUCHE; dy <= SUCHE; dy++) {
        for (int dx = -SUCHE; dx <= SUCHE; dx++) {
            float abstand = 0.0f;
            for (int fy = -FLECK; fy <= FLECK; fy++) {
                for (int fx = -FLECK; fx <= FLECK; fx++) {
                    float u = M(y + fy, x + fx) - M(y + dy + fy, x + dx + fx);
                    abstand = abstand + u * u;
                }
            }
            float gewicht = expf(-(abstand / flaeche) / h2);
            summe = summe + gewicht * M(y + dy, x + dx);
            gewichte = gewichte + gewicht;
        }
    }
    ziel[i] = summe / gewichte;
    """,
    "neuro_enhance_nlm", preamble=_GEMEINSAM + r"""
    __device__ __forceinline__ int spiegel(int a, int n) {
        return a < 0 ? -a : (a >= n ? 2 * (n - 1) - a : a);
    }
    """)

_helligkeit_setzen = cp.ElementwiseKernel(
    "raw float32 quelle, raw float32 v_neu", "raw float32 ziel",
    r"""
    float r = quelle[3 * i], g = quelle[3 * i + 1], b = quelle[3 * i + 2];
    float q = dekodieren(fmaxf(v_neu[i], 0.0f)) / fmaxf(luma(r, g, b), EPS);
    ziel[3 * i] = r * q; ziel[3 * i + 1] = g * q; ziel[3 * i + 2] = b * q;
    """,
    "neuro_enhance_helligkeit_setzen", preamble=_GEMEINSAM)

_farbrauschen = cp.ElementwiseKernel(
    "raw float32 quelle, raw float32 v, raw float32 a, raw float32 b, int32 h0, int32 b0, "
    "float32 sy, float32 sx, int32 breite, float32 staerke",
    "raw float32 ziel",
    r"""
    int y = i / breite, x = i % breite;
    float hell = v[i];
    float kodiert[3] = {kodieren(quelle[3 * i]), kodieren(quelle[3 * i + 1]),
                        kodieren(quelle[3 * i + 2])};
    int ebene = h0 * b0;
    for (int k = 0; k < 3; k++) {
        float chroma = kodiert[k] - hell;
        // a liegt als 9 Ebenen [k * 3 + j], b als 3 Ebenen vor
        float glatt = bilinear(&b[k * ebene], h0, b0, sy, sx, y, x);
        for (int j = 0; j < 3; j++) {
            glatt += bilinear(&a[(k * 3 + j) * ebene], h0, b0, sy, sx, y, x) * kodiert[j];
        }
        chroma = chroma + staerke * (glatt - chroma);
        ziel[3 * i + k] = dekodieren(fmaxf(hell + chroma, 0.0f));
    }
    """,
    "neuro_enhance_farbrauschen", preamble=_GEMEINSAM)

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


class _GpuLut:
    def __init__(self, tabelle):
        self.gpu = cp.asarray(tabelle.tabelle.ravel())
        self.groesse = tabelle.groesse
        self.unten, self.oben = tabelle.unten, tabelle.oben


def _lut_auf_gpu(pfad: str, speicher: dict | None):
    """LUT laden und hochladen - einmal je Datei und Sitzung."""
    from . import lut
    tabelle = lut.laden(pfad)
    schluessel = ("lut", pfad, id(tabelle))
    if speicher is not None and schluessel in speicher:
        return speicher[schluessel]
    ergebnis = _GpuLut(tabelle)
    if speicher is not None:
        for alt in [k for k in speicher if k[0] == "lut"]:
            del speicher[alt]
        speicher[schluessel] = ergebnis
    return ergebnis


def _helligkeit_von(rgb, pixel):
    v = cp.empty(rgb.shape[:2], dtype=cp.float32)
    _helligkeit(rgb, v, size=pixel)
    return v


def _skalierung(klein, hoehe, breite):
    """Kantenmasse und Schrittweiten fuer bilinear() - wie in filter.vergroessern()."""
    h0, b0 = klein.shape[:2]
    return (cp.int32(h0), cp.int32(b0), cp.float32(np.float32(h0 / hoehe)),
            cp.float32(np.float32(b0 / breite)), cp.int32(breite))


def _entrauscht(bild, werte: f.Einstellungen, speicher: dict | None):
    """Entrauschtes Bild - aus dem Zwischenspeicher, wenn moeglich.

    Non-Local Means ist der teuerste Schritt der Kette. Sein Ergebnis haengt
    nur vom Bild vor dem Entrauschen und den beiden Rauschreglern ab; alle
    spaeteren Regler koennen es wiederverwenden. Herausgegeben wird eine
    Kopie, denn die folgenden Schritte schreiben in ihr Bild hinein.
    """
    schluessel = ("rauschen", bild.shape, werte.temperatur, werte.toenung, werte.belichtung,
                  werte.rauschen_luminanz, werte.rauschen_farbe)
    if speicher is not None and schluessel in speicher:
        return speicher[schluessel].copy()
    hoehe, breite = bild.shape[:2]
    pixel = hoehe * breite
    rgb = bild
    v = _helligkeit_von(rgb, pixel)
    if werte.rauschen_luminanz > 0:
        v_neu = cp.empty_like(v)
        h2 = (werte.rauschen_luminanz / 100 * f.NLM_H) ** 2
        _nlm(v, cp.int32(hoehe), cp.int32(breite), cp.float32(h2), v_neu, size=pixel)
        neu = cp.empty_like(rgb)
        _helligkeit_setzen(rgb, v_neu, neu, size=pixel)
        rgb, v = neu, v_neu
    if werte.rauschen_farbe > 0:
        kodiert = f.linear_zu_srgb(rgb)
        chroma = kodiert - v[..., None]
        a, b = f.farbrauschen_koeffizienten(kodiert, chroma, werte.rauschen_farbe, rgb.shape)
        del chroma, kodiert
        neu = cp.empty_like(rgb)
        ebenen_a = cp.ascontiguousarray(cp.moveaxis(a.reshape(*a.shape[:2], 9), -1, 0))
        _farbrauschen(rgb, v, ebenen_a, cp.ascontiguousarray(cp.moveaxis(b, -1, 0)),
                      *_skalierung(a, hoehe, breite),
                      cp.float32(min(1.0, werte.rauschen_farbe / 50)), neu, size=pixel)
        rgb = neu
    if speicher is not None:
        for alt in [k for k in speicher if k[0] == "rauschen" and k[1] == bild.shape]:
            del speicher[alt]
        speicher[schluessel] = rgb
        return rgb.copy()
    return rgb


def _dunst_schaetzung(bild, werte: f.Einstellungen, speicher: dict | None):
    """Lichtfarbe und Durchlasskarte - aus dem Zwischenspeicher, wenn moeglich.

    Die Schaetzung haengt nur vom Bild vor dem Dunstschritt ab, also von
    Weissabgleich und Belichtung - nicht von der Dunststaerke selbst. Beim
    Ziehen am Dunstregler muss sie deshalb nur einmal gerechnet werden.
    """
    schluessel = ("dunst", bild.shape, werte.temperatur, werte.toenung, werte.belichtung,
                  werte.rauschen_luminanz, werte.rauschen_farbe)
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

    # Weissabgleich, Belichtung - und, wenn weder Rauschen noch Dunst zu entfernen
    # sind, gleich die Tonwerte im selben Durchlauf. Sonst gehoeren diese dazwischen.
    zwischen = cp.empty_like(bild)
    erst_dunst = werte.dunst != 0
    geo = geometrie.aus(werte)
    vorstufe = erst_dunst or werte.rauschen_aktiv() or not geo.ist_neutral()
    _licht(bild, cp.float32(gr), cp.float32(gg), cp.float32(gb),
           cp.int32(0 if vorstufe else ton), *tonwerte, zwischen, size=pixel)
    if werte.rauschen_aktiv():
        zwischen = _entrauscht(zwischen, werte, speicher)
    if erst_dunst:
        licht, durchlass = _dunst_schaetzung(zwischen, werte, speicher)
        staerke = werte.dunst / 100
        # Die Staerke mischt wie in filter.dunst() schon auf der kleinen Karte.
        durchlass = cp.ascontiguousarray(1 - cp.float32(staerke) * (1 - durchlass))
        lr, lg, lb = (cp.float32(wert) for wert in cp.asnumpy(licht))
        entdunstet = cp.empty_like(zwischen)
        _dunst(zwischen, durchlass, *_skalierung(durchlass, hoehe, breite), lr, lg, lb,
               cp.float32(staerke), cp.float32(f.DUNST_MIN_DURCHLASS), entdunstet, size=pixel)
        zwischen = entdunstet
    if not geo.ist_neutral():
        # Ab hier hat das Bild die Groesse des Zuschnitts
        zwischen = geometrie.anwenden(zwischen, geo)
        hoehe, breite = zwischen.shape[:2]
        pixel = hoehe * breite
    if vorstufe and ton:
        _licht(zwischen, cp.float32(1), cp.float32(1), cp.float32(1), cp.int32(1),
               *tonwerte, zwischen, size=pixel)

    if werte.klarheit != 0:
        v = _helligkeit_von(zwischen, pixel)
        weich = cp.ascontiguousarray(f.grob_weich_klein(v, f.klarheit_sigma(zwischen.shape)))
        _klarheit(zwischen, v, weich, *_skalierung(weich, hoehe, breite),
                  cp.float32(werte.klarheit / 100), zwischen, size=pixel)

    tabellen = f.kurven_tabellen(werte)
    farbig = int(werte.dynamik != 0 or werte.saettigung != 0)
    if tabellen or farbig or werte.hsl_aktiv() or werte.lut_aktiv():
        alle = np.concatenate([tabellen.get(name, kurven.tabelle(kurven.IDENTITAET))
                               for name in f.KURVEN])
        hsl = np.concatenate([werte.hsl_farbton, werte.hsl_saettigung, werte.hsl_luminanz])
        if werte.lut_aktiv():
            tabelle = _lut_auf_gpu(werte.lut, speicher)
            bereich = np.concatenate([tabelle.unten, tabelle.oben - tabelle.unten])
            lut_daten, lut_n = tabelle.gpu, tabelle.groesse
        else:
            lut_daten, lut_n, bereich = cp.zeros(3, dtype=cp.float32), 2, np.ones(6)
        _kurven_farbe(zwischen, cp.asarray(alle), cp.int32(kurven.TABELLE),
                      *(cp.int32(name in tabellen) for name in f.KURVEN), cp.int32(farbig),
                      cp.float32(werte.saettigung / 100), cp.float32(werte.dynamik / 100),
                      cp.int32(werte.hsl_aktiv()), cp.asarray(hsl, dtype=cp.float32),
                      cp.int32(werte.lut_aktiv()), lut_daten, cp.int32(lut_n),
                      cp.asarray(bereich, dtype=cp.float32),
                      cp.float32(werte.lut_staerke / 100), zwischen, size=pixel)

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
