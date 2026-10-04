"""
Demosaicing nach RCD (Ratio Corrected Demosaicing, Luis Sanz Rodriguez)

RCD entscheidet je Pixel anhand der Bildstruktur, ob waagrecht, senkrecht oder
diagonal interpoliert wird, und rechnet ueber Farbverhaeltnisse statt nur
ueber Differenzen. An feinen Mustern entstehen dadurch deutlich weniger
Farbsaeume als bei Malvar-He-Cutler. Die Schritte folgen der Fassung in
RawTherapee und darktable:

  1. Richtung waagrecht/senkrecht aus quadrierten Hochpaessen (VH)
  2. Tiefpass aus Rot, Gruen und Blau an den Rot- und Blaupixeln
  3. Gruen an Rot und Blau, ueber Verhaeltnisse zum Tiefpass
  4. Rot an Blau und Blau an Rot - diagonal (PQ), ueber Farbdifferenzen zu Gruen
  5. Rot und Blau an Gruen

Jeder Schritt braucht die Ergebnisse seiner Nachbarn aus dem vorigen; auf der
GPU sind es deshalb vier Durchlaeufe. Hoch- und Tiefpaesse rechnet die GPU
dort neu, wo sie gebraucht werden, statt sie zwischenzuspeichern - Rechnen
ist billiger als Grafikspeicher, und bei grossen RAWs zaehlt jedes Bild an
Zwischenspeicher. Damit auch am Rand genug Nachbarn da
sind, wird das Mosaik vorher gespiegelt um RAND Pixel erweitert - RAND ist
gerade, so bleibt das Bayer-Muster erhalten.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import numpy as np

from . import filter as f

RAND = 12                       # reicht fuer die Reichweite aller fuenf Schritte
EPS = 1e-5
EPS_QUADRAT = 1e-10


# --------------------------------------------------------------------------
# Referenz mit NumPy/CuPy-Arrays
# --------------------------------------------------------------------------

def _verschoben(a, dy: int, dx: int):
    """Wert an (y + dy, x + dx) fuer jeden Pixel. Am Rand umlaufend - der Rand
    wird am Ende abgeschnitten und hat keinen Einfluss auf das Innere."""
    xp = f.xp_von(a)
    return xp.roll(a, (-dy, -dx), axis=(0, 1))


def rcd(m, muster):
    """Demosaicing auf dem vorbereiteten Mosaik m (H, W, Werte 0..1) -> (H, W, 3)."""
    xp = f.xp_von(m)
    hoehe, breite = m.shape
    c = xp.pad(m, RAND, mode="reflect").astype(xp.float32)
    hp, bp = c.shape

    def s(a, dy, dx):
        return _verschoben(a, dy, dx)

    kanal = xp.asarray(muster)[xp.arange(hp)[:, None] % 2, xp.arange(bp)[None, :] % 2]
    ist_g = kanal == 1
    ist_r = kanal == 0
    ist_b = kanal == 2

    # Schritt 1: Richtung waagrecht / senkrecht
    vb = ((s(c, -3, 0) - s(c, -1, 0) - s(c, 1, 0) + s(c, 3, 0))
          - 3 * (s(c, -2, 0) + s(c, 2, 0)) + 6 * c) ** 2
    hb = ((s(c, 0, -3) - s(c, 0, -1) - s(c, 0, 1) + s(c, 0, 3))
          - 3 * (s(c, 0, -2) + s(c, 0, 2)) + 6 * c) ** 2
    v_stat = xp.maximum(EPS_QUADRAT, s(vb, -1, 0) + vb + s(vb, 1, 0))
    h_stat = xp.maximum(EPS_QUADRAT, s(hb, 0, -1) + hb + s(hb, 0, 1))
    vh = v_stat / (v_stat + h_stat)

    # Schritt 2: Tiefpass
    lpf = (0.25 * c + 0.125 * (s(c, -1, 0) + s(c, 1, 0) + s(c, 0, -1) + s(c, 0, 1))
           + 0.0625 * (s(c, -1, -1) + s(c, -1, 1) + s(c, 1, -1) + s(c, 1, 1)))

    # Schritt 3: Gruen an Rot und Blau
    n_grad = (EPS + abs(s(c, -1, 0) - s(c, 1, 0)) + abs(c - s(c, -2, 0))
              + abs(s(c, -1, 0) - s(c, -3, 0)) + abs(s(c, -2, 0) - s(c, -4, 0)))
    s_grad = (EPS + abs(s(c, 1, 0) - s(c, -1, 0)) + abs(c - s(c, 2, 0))
              + abs(s(c, 1, 0) - s(c, 3, 0)) + abs(s(c, 2, 0) - s(c, 4, 0)))
    w_grad = (EPS + abs(s(c, 0, -1) - s(c, 0, 1)) + abs(c - s(c, 0, -2))
              + abs(s(c, 0, -1) - s(c, 0, -3)) + abs(s(c, 0, -2) - s(c, 0, -4)))
    e_grad = (EPS + abs(s(c, 0, 1) - s(c, 0, -1)) + abs(c - s(c, 0, 2))
              + abs(s(c, 0, 1) - s(c, 0, 3)) + abs(s(c, 0, 2) - s(c, 0, 4)))

    def schaetzung(nachbar, dy, dx):
        anderes = s(lpf, dy, dx)
        return nachbar * (1 + (lpf - anderes) / (EPS + lpf + anderes))

    n_est = schaetzung(s(c, -1, 0), -2, 0)
    s_est = schaetzung(s(c, 1, 0), 2, 0)
    w_est = schaetzung(s(c, 0, -1), 0, -2)
    e_est = schaetzung(s(c, 0, 1), 0, 2)
    v_est = (s_grad * n_est + n_grad * s_est) / (n_grad + s_grad)
    h_est = (w_grad * e_est + e_grad * w_est) / (e_grad + w_grad)

    def verfeinert(richtung, a, b, c_, d):
        mitte = richtung
        umgebung = 0.25 * (a + b + c_ + d)
        return xp.where(abs(0.5 - mitte) < abs(0.5 - umgebung), umgebung, mitte)

    vh_disc = verfeinert(vh, s(vh, -1, -1), s(vh, -1, 1), s(vh, 1, -1), s(vh, 1, 1))
    g = xp.where(ist_g, c, xp.clip(vh_disc * (h_est - v_est) + v_est, 0, 1))

    # Schritt 4: Rot an Blau und Blau an Rot, diagonal
    pb = ((s(c, -3, -3) - s(c, -1, -1) - s(c, 1, 1) + s(c, 3, 3))
          - 3 * (s(c, -2, -2) + s(c, 2, 2)) + 6 * c) ** 2
    qb = ((s(c, -3, 3) - s(c, -1, 1) - s(c, 1, -1) + s(c, 3, -3))
          - 3 * (s(c, -2, 2) + s(c, 2, -2)) + 6 * c) ** 2
    p_stat = xp.maximum(EPS_QUADRAT, s(pb, -1, -1) + pb + s(pb, 1, 1))
    q_stat = xp.maximum(EPS_QUADRAT, s(qb, -1, 1) + qb + s(qb, 1, -1))
    pq = p_stat / (p_stat + q_stat)
    pq_disc = verfeinert(pq, s(pq, -1, -1), s(pq, -1, 1), s(pq, 1, -1), s(pq, 1, 1))

    # An Rot- und Blaupixeln liegt die gesuchte Farbe diagonal - als Rohwert in c
    nw_grad = (EPS + abs(s(c, -1, -1) - s(c, 1, 1)) + abs(s(c, -1, -1) - s(c, -3, -3))
               + abs(g - s(g, -2, -2)))
    ne_grad = (EPS + abs(s(c, -1, 1) - s(c, 1, -1)) + abs(s(c, -1, 1) - s(c, -3, 3))
               + abs(g - s(g, -2, 2)))
    sw_grad = (EPS + abs(s(c, 1, -1) - s(c, -1, 1)) + abs(s(c, 1, -1) - s(c, 3, -3))
               + abs(g - s(g, 2, -2)))
    se_grad = (EPS + abs(s(c, 1, 1) - s(c, -1, -1)) + abs(s(c, 1, 1) - s(c, 3, 3))
               + abs(g - s(g, 2, 2)))
    nw_est = s(c, -1, -1) - s(g, -1, -1)
    ne_est = s(c, -1, 1) - s(g, -1, 1)
    sw_est = s(c, 1, -1) - s(g, 1, -1)
    se_est = s(c, 1, 1) - s(g, 1, 1)
    p_est = (nw_grad * se_est + se_grad * nw_est) / (nw_grad + se_grad)
    q_est = (ne_grad * sw_est + sw_grad * ne_est) / (ne_grad + sw_grad)
    diagonal = xp.clip(g + pq_disc * (q_est - p_est) + p_est, 0, 1)
    r = xp.where(ist_r, c, xp.where(ist_b, diagonal, 0))
    b = xp.where(ist_b, c, xp.where(ist_r, diagonal, 0))

    # Schritt 5: Rot und Blau an Gruen
    n1 = EPS + abs(g - s(g, -2, 0))
    s1 = EPS + abs(g - s(g, 2, 0))
    w1 = EPS + abs(g - s(g, 0, -2))
    e1 = EPS + abs(g - s(g, 0, 2))

    def an_gruen(farbe):
        sn = abs(s(farbe, -1, 0) - s(farbe, 1, 0))
        ew = abs(s(farbe, 0, -1) - s(farbe, 0, 1))
        n_g = n1 + sn + abs(s(farbe, -1, 0) - s(farbe, -3, 0))
        s_g = s1 + sn + abs(s(farbe, 1, 0) - s(farbe, 3, 0))
        w_g = w1 + ew + abs(s(farbe, 0, -1) - s(farbe, 0, -3))
        e_g = e1 + ew + abs(s(farbe, 0, 1) - s(farbe, 0, 3))
        n_e = s(farbe, -1, 0) - s(g, -1, 0)
        s_e = s(farbe, 1, 0) - s(g, 1, 0)
        w_e = s(farbe, 0, -1) - s(g, 0, -1)
        e_e = s(farbe, 0, 1) - s(g, 0, 1)
        v_e = (n_g * s_e + s_g * n_e) / (n_g + s_g)
        h_e = (e_g * w_e + w_g * e_e) / (e_g + w_g)
        return xp.where(ist_g, xp.clip(g + vh_disc * (h_e - v_e) + v_e, 0, 1), farbe)

    r, b = an_gruen(r), an_gruen(b)
    innen = (slice(RAND, RAND + hoehe), slice(RAND, RAND + breite))
    return xp.stack([r[innen], g[innen], b[innen]], axis=-1).astype(xp.float32)


# --------------------------------------------------------------------------
# CUDA: fuenf Durchlaeufe auf dem erweiterten Mosaik
# --------------------------------------------------------------------------

_GEMEINSAM = r"""
#define EPS 1e-5f
#define EPS_QUADRAT 1e-10f
// Wert an (y + dy, x + dx); ausserhalb wird an den Rand geklemmt. Das betrifft
// nur den erweiterten Rand, der am Ende wegfaellt.
__device__ __forceinline__ float lesen(const float* a, int hoehe, int breite, int y, int x) {
    y = min(max(y, 0), hoehe - 1);
    x = min(max(x, 0), breite - 1);
    return a[y * breite + x];
}
__device__ __forceinline__ float sqr(float v) { return v * v; }
// Quadrierter Hochpass entlang der Richtung (sy, sx) am Punkt (y, x)
__device__ __forceinline__ float hochpass(const float* c, int h, int w, int y, int x,
                                          int sy, int sx) {
    #define P(k) lesen(c, h, w, y + (k) * sy, x + (k) * sx)
    float v = (P(-3) - P(-1) - P(1) + P(3)) - 3.0f * (P(-2) + P(2)) + 6.0f * P(0);
    #undef P
    return v * v;
}
__device__ __forceinline__ float tiefpass(const float* c, int h, int w, int y, int x) {
    #define Q(dy, dx) lesen(c, h, w, y + (dy), x + (dx))
    return 0.25f * Q(0, 0) + 0.125f * (Q(-1, 0) + Q(1, 0) + Q(0, -1) + Q(0, 1))
           + 0.0625f * (Q(-1, -1) + Q(-1, 1) + Q(1, -1) + Q(1, 1));
    #undef Q
}
__device__ __forceinline__ float verfeinert(const float* r, int h, int w, int y, int x) {
    float mitte = lesen(r, h, w, y, x);
    float umgebung = 0.25f * (lesen(r, h, w, y - 1, x - 1) + lesen(r, h, w, y - 1, x + 1)
                              + lesen(r, h, w, y + 1, x - 1) + lesen(r, h, w, y + 1, x + 1));
    return fabsf(0.5f - mitte) < fabsf(0.5f - umgebung) ? umgebung : mitte;
}
"""

_KERNELS = None


def _kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    from .cuda import cupy as cp

    def kernel(parameter, ziele, rumpf, name):
        return cp.ElementwiseKernel(
            "raw float32 c, int32 h, int32 w, raw int32 muster" + parameter, ziele,
            # Der Rumpf steht bei CuPy in einer Schleife ueber die Pixel: continue
            # beendet den Pixel, ein return wuerde weitere Pixel des Threads auslassen.
            "int y = i / w, x = i % w;\n"
            "#define C(dy, dx) lesen(&c[0], h, w, y + (dy), x + (dx))\n" + rumpf,
            name, preamble=_GEMEINSAM)

    richtungen = kernel("", "raw float32 vh, raw float32 pq", r"""
        #define HP(dy, dx, sy, sx) hochpass(&c[0], h, w, y + (dy), x + (dx), sy, sx)
        float v = fmaxf(EPS_QUADRAT, HP(-1, 0, 1, 0) + HP(0, 0, 1, 0) + HP(1, 0, 1, 0));
        float hh = fmaxf(EPS_QUADRAT, HP(0, -1, 0, 1) + HP(0, 0, 0, 1) + HP(0, 1, 0, 1));
        vh[i] = v / (v + hh);
        float p = fmaxf(EPS_QUADRAT, HP(-1, -1, 1, 1) + HP(0, 0, 1, 1) + HP(1, 1, 1, 1));
        float q = fmaxf(EPS_QUADRAT, HP(-1, 1, 1, -1) + HP(0, 0, 1, -1) + HP(1, -1, 1, -1));
        pq[i] = p / (p + q);
        """, "neuro_enhance_rcd_richtungen")

    gruen = kernel(", raw float32 vh", "raw float32 g", r"""
        #define L(dy, dx) tiefpass(&c[0], h, w, y + (dy), x + (dx))
        float c0 = C(0, 0);
        if (muster[(y & 1) * 2 + (x & 1)] == 1) { g[i] = c0; continue; }
        float n_grad = EPS + fabsf(C(-1, 0) - C(1, 0)) + fabsf(c0 - C(-2, 0))
                       + fabsf(C(-1, 0) - C(-3, 0)) + fabsf(C(-2, 0) - C(-4, 0));
        float s_grad = EPS + fabsf(C(1, 0) - C(-1, 0)) + fabsf(c0 - C(2, 0))
                       + fabsf(C(1, 0) - C(3, 0)) + fabsf(C(2, 0) - C(4, 0));
        float w_grad = EPS + fabsf(C(0, -1) - C(0, 1)) + fabsf(c0 - C(0, -2))
                       + fabsf(C(0, -1) - C(0, -3)) + fabsf(C(0, -2) - C(0, -4));
        float e_grad = EPS + fabsf(C(0, 1) - C(0, -1)) + fabsf(c0 - C(0, 2))
                       + fabsf(C(0, 1) - C(0, 3)) + fabsf(C(0, 2) - C(0, 4));
        float l0 = L(0, 0), ln = L(-2, 0), ls = L(2, 0), lw = L(0, -2), le = L(0, 2);
        float n_est = C(-1, 0) * (1.0f + (l0 - ln) / (EPS + l0 + ln));
        float s_est = C(1, 0) * (1.0f + (l0 - ls) / (EPS + l0 + ls));
        float w_est = C(0, -1) * (1.0f + (l0 - lw) / (EPS + l0 + lw));
        float e_est = C(0, 1) * (1.0f + (l0 - le) / (EPS + l0 + le));
        float v_est = (s_grad * n_est + n_grad * s_est) / (n_grad + s_grad);
        float h_est = (w_grad * e_est + e_grad * w_est) / (e_grad + w_grad);
        float disc = verfeinert(&vh[0], h, w, y, x);
        g[i] = fminf(fmaxf(disc * (h_est - v_est) + v_est, 0.0f), 1.0f);
        """, "neuro_enhance_rcd_gruen")

    diagonal = kernel(", raw float32 g, raw float32 pq", "raw float32 r, raw float32 b", r"""
        #define G(dy, dx) lesen(&g[0], h, w, y + (dy), x + (dx))
        int kanal = muster[(y & 1) * 2 + (x & 1)];
        float c0 = C(0, 0);
        if (kanal == 1) { r[i] = 0.0f; b[i] = 0.0f; continue; }
        float g0 = G(0, 0);
        float nw_grad = EPS + fabsf(C(-1, -1) - C(1, 1)) + fabsf(C(-1, -1) - C(-3, -3))
                        + fabsf(g0 - G(-2, -2));
        float ne_grad = EPS + fabsf(C(-1, 1) - C(1, -1)) + fabsf(C(-1, 1) - C(-3, 3))
                        + fabsf(g0 - G(-2, 2));
        float sw_grad = EPS + fabsf(C(1, -1) - C(-1, 1)) + fabsf(C(1, -1) - C(3, -3))
                        + fabsf(g0 - G(2, -2));
        float se_grad = EPS + fabsf(C(1, 1) - C(-1, -1)) + fabsf(C(1, 1) - C(3, 3))
                        + fabsf(g0 - G(2, 2));
        float nw_est = C(-1, -1) - G(-1, -1);
        float ne_est = C(-1, 1) - G(-1, 1);
        float sw_est = C(1, -1) - G(1, -1);
        float se_est = C(1, 1) - G(1, 1);
        float p_est = (nw_grad * se_est + se_grad * nw_est) / (nw_grad + se_grad);
        float q_est = (ne_grad * sw_est + sw_grad * ne_est) / (ne_grad + sw_grad);
        float disc = verfeinert(&pq[0], h, w, y, x);
        float wert = fminf(fmaxf(g0 + disc * (q_est - p_est) + p_est, 0.0f), 1.0f);
        if (kanal == 0) { r[i] = c0; b[i] = wert; } else { r[i] = wert; b[i] = c0; }
        """, "neuro_enhance_rcd_diagonal")

    ausgabe = kernel(", raw float32 g, raw float32 r, raw float32 b, raw float32 vh, "
                     "int32 rand, int32 breite_aus, raw float32 matrix", "raw float32 ziel", r"""
        // i zaehlt hier die Pixel des Ergebnisses; y, x werden auf das erweiterte Mosaik umgelegt
        y = i / breite_aus + rand;
        x = i % breite_aus + rand;
        #define G(dy, dx) lesen(&g[0], h, w, y + (dy), x + (dx))
        #define R(dy, dx) lesen(&r[0], h, w, y + (dy), x + (dx))
        #define B(dy, dx) lesen(&b[0], h, w, y + (dy), x + (dx))
        float g0 = G(0, 0), rr = R(0, 0), bb = B(0, 0);
        if (muster[(y & 1) * 2 + (x & 1)] == 1) {
            float disc = verfeinert(&vh[0], h, w, y, x);
            float n1 = EPS + fabsf(g0 - G(-2, 0)), s1 = EPS + fabsf(g0 - G(2, 0));
            float w1 = EPS + fabsf(g0 - G(0, -2)), e1 = EPS + fabsf(g0 - G(0, 2));
            #define AN_GRUEN(F, ERGEBNIS) { \
                float sn = fabsf(F(-1, 0) - F(1, 0)), ew = fabsf(F(0, -1) - F(0, 1)); \
                float n_g = n1 + sn + fabsf(F(-1, 0) - F(-3, 0)); \
                float s_g = s1 + sn + fabsf(F(1, 0) - F(3, 0)); \
                float w_g = w1 + ew + fabsf(F(0, -1) - F(0, -3)); \
                float e_g = e1 + ew + fabsf(F(0, 1) - F(0, 3)); \
                float n_e = F(-1, 0) - G(-1, 0), s_e = F(1, 0) - G(1, 0); \
                float w_e = F(0, -1) - G(0, -1), e_e = F(0, 1) - G(0, 1); \
                float v_e = (n_g * s_e + s_g * n_e) / (n_g + s_g); \
                float h_e = (e_g * w_e + w_g * e_e) / (e_g + w_g); \
                ERGEBNIS = fminf(fmaxf(g0 + disc * (h_e - v_e) + v_e, 0.0f), 1.0f); }
            AN_GRUEN(R, rr)
            AN_GRUEN(B, bb)
        }
        ziel[3 * i] = matrix[0] * rr + matrix[1] * g0 + matrix[2] * bb;
        ziel[3 * i + 1] = matrix[3] * rr + matrix[4] * g0 + matrix[5] * bb;
        ziel[3 * i + 2] = matrix[6] * rr + matrix[7] * g0 + matrix[8] * bb;
        """, "neuro_enhance_rcd_ausgabe")

    _KERNELS = (richtungen, gruen, diagonal, ausgabe)
    return _KERNELS


def rcd_gpu(m, muster, matrix):
    """RCD und Farbmatrix auf der GPU: vorbereitetes Mosaik (CuPy) -> lineares sRGB."""
    from .cuda import cupy as cp
    richtungen, gruen, diagonal, ausgabe = _kernels()
    hoehe, breite = m.shape
    c = cp.ascontiguousarray(cp.pad(m, RAND, mode="reflect"), dtype=cp.float32)
    del m                               # jeder Puffer so frueh frei wie moeglich -
    hp, bp = c.shape                    # bei 45 Megapixeln sind es je 180 MB
    n = hp * bp
    grund = (c, cp.int32(hp), cp.int32(bp), cp.asarray(np.ravel(muster), dtype=cp.int32))

    vh, pq = cp.empty_like(c), cp.empty_like(c)
    richtungen(*grund, vh, pq, size=n)
    g = cp.empty_like(c)
    gruen(*grund, vh, g, size=n)
    r, b = cp.empty_like(c), cp.empty_like(c)
    diagonal(*grund, g, pq, r, b, size=n)
    del pq
    ziel = cp.empty((hoehe, breite, 3), dtype=cp.float32)
    ausgabe(*grund, g, r, b, vh, cp.int32(RAND), cp.int32(breite),
            cp.asarray(np.ravel(matrix), dtype=cp.float32), ziel, size=hoehe * breite)
    return ziel
