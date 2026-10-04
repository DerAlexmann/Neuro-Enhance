"""
Klassische Filter

Alle Funktionen arbeiten mit NumPy- oder CuPy-Arrays gleichermassen: xp_von()
waehlt das passende Modul nach dem Array. Im Programm liegen die Bilder als
CuPy-Arrays auf der Grafikkarte; die Tests rechnen dieselben Formeln mit
NumPy, ganz ohne Grafikkarte.

Gerechnet wird in linearem Licht mit float32 (Bildwerte 0..1, sRGB-Primaer-
farben). Erst fuer Anzeige und Export wird zurueck nach sRGB gewandelt.

Die Reihenfolge der Schritte ist fest, wie in der Bildentwicklung ueblich:
Weissabgleich -> Belichtung -> Tonwerte (Kontrast, Lichter, Tiefen)
-> Farbe (Dynamik, Saettigung) -> Schaerfen.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

from .cuda import cupy as _cupy

# Gewichte fuer die Luminanz in linearem sRGB (Rec. 709)
LUMA = (0.2126, 0.7152, 0.0722)

EPS = 1e-6


def xp_von(array):
    """numpy oder cupy - je nachdem, wo das Array liegt."""
    if _cupy is not None and isinstance(array, _cupy.ndarray):
        return _cupy
    return np


# --------------------------------------------------------------------------
# Regler
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Regler:
    """Ein Regler: Wertebereich, Vorgabe und wie der Wert angezeigt wird."""
    name: str
    gruppe: str
    minimum: float
    maximum: float
    vorgabe: float = 0.0
    schritt: float = 1.0             # kleinste Aenderung am Schieberegler
    nachkomma: int = 0


REGLER = (
    Regler("temperatur", "weissabgleich", -100, 100),
    Regler("toenung", "weissabgleich", -100, 100),
    Regler("belichtung", "licht", -5.0, 5.0, schritt=0.05, nachkomma=2),
    Regler("kontrast", "licht", -100, 100),
    Regler("lichter", "licht", -100, 100),
    Regler("tiefen", "licht", -100, 100),
    Regler("dynamik", "farbe", -100, 100),
    Regler("saettigung", "farbe", -100, 100),
    Regler("schaerfe", "details", 0, 150),
    Regler("schaerfe_radius", "details", 0.5, 3.0, vorgabe=1.0, schritt=0.1, nachkomma=1),
)
REGLER_NACH_NAME = {r.name: r for r in REGLER}


@dataclass
class Einstellungen:
    """Die Werte aller Regler. Die Vorgaben lassen das Bild unveraendert."""
    temperatur: float = 0.0
    toenung: float = 0.0
    belichtung: float = 0.0
    kontrast: float = 0.0
    lichter: float = 0.0
    tiefen: float = 0.0
    dynamik: float = 0.0
    saettigung: float = 0.0
    schaerfe: float = 0.0
    schaerfe_radius: float = 1.0

    def ist_neutral(self) -> bool:
        return all(getattr(self, f.name) == REGLER_NACH_NAME[f.name].vorgabe
                   for f in fields(self) if f.name != "schaerfe_radius")


# --------------------------------------------------------------------------
# Farbraum
# --------------------------------------------------------------------------

def srgb_zu_linear(v):
    xp = xp_von(v)
    v = xp.asarray(v, dtype=xp.float32)
    return xp.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(xp.float32)


def linear_zu_srgb(x):
    xp = xp_von(x)
    x = xp.maximum(x, 0)
    return xp.where(x <= 0.0031308, x * 12.92,
                    1.055 * x ** (1 / 2.4) - 0.055).astype(xp.float32)


def luminanz(rgb):
    return rgb[..., 0] * LUMA[0] + rgb[..., 1] * LUMA[1] + rgb[..., 2] * LUMA[2]


def nach_8bit(rgb_linear):
    """Lineares float-Bild -> sRGB uint8 fuer Anzeige und Export."""
    xp = xp_von(rgb_linear)
    v = linear_zu_srgb(xp.clip(rgb_linear, 0, 1))
    return (v * 255 + 0.5).astype(xp.uint8)


def von_8bit(rgb_uint8):
    xp = xp_von(rgb_uint8)
    return srgb_zu_linear(rgb_uint8.astype(xp.float32) / 255)


# --------------------------------------------------------------------------
# Die einzelnen Schritte
# --------------------------------------------------------------------------

def weissabgleich(rgb, temperatur, toenung):
    """Temperatur verschiebt zwischen Blau und Gelb, Toenung zwischen Gruen und Magenta.

    Die Faktoren werden so normiert, dass die Helligkeit erhalten bleibt -
    sonst wuerde ein waermeres Bild zugleich dunkler oder heller.
    """
    if temperatur == 0 and toenung == 0:
        return rgb
    xp = xp_von(rgb)
    k = temperatur / 100 * 0.35
    m = toenung / 100 * 0.25
    faktoren = [math.exp(k), math.exp(-m), math.exp(-k)]
    norm = sum(f * w for f, w in zip(faktoren, LUMA, strict=True))
    return rgb * xp.asarray([f / norm for f in faktoren], dtype=xp.float32)


def belichtung(rgb, ev):
    """Belichtung in Lichtwerten: +1 verdoppelt das Licht."""
    return rgb if ev == 0 else rgb * xp_von(rgb).float32(2.0 ** ev)


def _glatt(xp, kante0, kante1, v):
    t = xp.clip((v - kante0) / (kante1 - kante0), 0, 1)
    return t * t * (3 - 2 * t)


def tonwerte(rgb, kontrast, lichter, tiefen):
    """Kontrast, Lichter und Tiefen ueber eine Tonkurve auf der Helligkeit.

    Die Kurve wirkt auf die wahrnehmungsgerechte Helligkeit (sRGB-kodierte
    Luminanz), die Farbkanaele werden anschliessend im selben Verhaeltnis
    skaliert. So bleiben Farbton und Saettigung erhalten - eine Kurve je
    Kanal wuerde die Farben verschieben.
    """
    if kontrast == 0 and lichter == 0 and tiefen == 0:
        return rgb
    xp = xp_von(rgb)
    y = luminanz(rgb)
    v = linear_zu_srgb(y)
    innen = xp.clip(v, 0, 1)
    ueber = v - innen                       # Werte ueber 1 bleiben unberuehrt

    c = kontrast / 100
    if c > 0:
        innen = innen + c * (_glatt(xp, 0.0, 1.0, innen) - innen)
    elif c < 0:
        innen = innen + (-c) * ((0.5 + (innen - 0.5) * 0.5) - innen)

    if lichter:
        gewicht = _glatt(xp, 0.5, 1.0, innen)
        innen = innen + (lichter / 100) * 0.6 * gewicht * (innen - 0.5)
    if tiefen:
        gewicht = 1 - _glatt(xp, 0.0, 0.5, innen)
        innen = innen + (tiefen / 100) * 0.6 * gewicht * (0.5 - innen)

    y_neu = srgb_zu_linear(xp.clip(innen, 0, 1) + ueber)
    verhaeltnis = y_neu / xp.maximum(y, EPS)
    # Reines Schwarz hat kein Verhaeltnis - dort wird das neue Grau gesetzt.
    schwarz = (y <= EPS)[..., None]
    return xp.where(schwarz, y_neu[..., None], rgb * verhaeltnis[..., None]).astype(xp.float32)


def farbe(rgb, dynamik, saettigung):
    """Saettigung wirkt gleichmaessig, Dynamik vor allem auf blasse Farben."""
    if dynamik == 0 and saettigung == 0:
        return rgb
    xp = xp_von(rgb)
    y = luminanz(rgb)[..., None]
    faktor = 1 + saettigung / 100
    if dynamik:
        hoechster = rgb.max(axis=-1, keepdims=True)
        niedrigster = rgb.min(axis=-1, keepdims=True)
        vorhanden = (hoechster - niedrigster) / xp.maximum(hoechster, EPS)
        faktor = faktor * (1 + dynamik / 100 * (1 - vorhanden))
    faktor = xp.maximum(faktor, 0)
    return (y + (rgb - y) * faktor).astype(xp.float32)


def gauss_kern(sigma):
    radius = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kern = np.exp(-(x * x) / (2 * sigma * sigma))
    return (kern / kern.sum()).astype(np.float32)


def gauss(bild, sigma):
    """Separierbare Gauss-Unschaerfe auf einem 2D-Array, Rand gespiegelt."""
    if sigma <= 0:
        return bild
    xp = xp_von(bild)
    kern = gauss_kern(sigma)
    r = len(kern) // 2
    ergebnis = bild
    for achse in (0, 1):
        breite = [(0, 0), (0, 0)]
        breite[achse] = (r, r)
        gepolstert = xp.pad(ergebnis, breite, mode="reflect")
        summe = xp.zeros_like(ergebnis)
        laenge = ergebnis.shape[achse]
        for i, gewicht in enumerate(kern):
            ausschnitt = [slice(None), slice(None)]
            ausschnitt[achse] = slice(i, i + laenge)
            summe += gepolstert[tuple(ausschnitt)] * xp.float32(gewicht)
        ergebnis = summe
    return ergebnis


def schaerfen(rgb, staerke, radius):
    """Unscharf maskieren auf der Helligkeit.

    Geschaerft wird nur die wahrnehmungsgerechte Helligkeit; die Farben folgen
    ueber das Verhaeltnis. Das vermeidet farbige Saeume an Kanten.
    """
    if staerke <= 0 or radius <= 0:
        return rgb
    xp = xp_von(rgb)
    y = luminanz(rgb)
    v = linear_zu_srgb(y)
    v_neu = v + (staerke / 100) * (v - gauss(v, radius))
    y_neu = srgb_zu_linear(xp.clip(v_neu, 0, None))
    verhaeltnis = y_neu / xp.maximum(y, EPS)
    return (rgb * verhaeltnis[..., None]).astype(xp.float32)


# --------------------------------------------------------------------------
# Die ganze Kette
# --------------------------------------------------------------------------

def anwenden(rgb_linear, werte: Einstellungen, massstab: float = 1.0):
    """Wendet alle Regler in fester Reihenfolge an und gibt ein neues Bild zurueck.

    massstab ist das Verhaeltnis der bearbeiteten Groesse zur Originalgroesse.
    Die Vorschau ist verkleinert; ohne diesen Faktor waere ein Schaerferadius
    von 1 Pixel in der Vorschau viel groeber als im exportierten Bild.
    """
    bild = weissabgleich(rgb_linear, werte.temperatur, werte.toenung)
    bild = belichtung(bild, werte.belichtung)
    bild = tonwerte(bild, werte.kontrast, werte.lichter, werte.tiefen)
    bild = farbe(bild, werte.dynamik, werte.saettigung)
    return schaerfen(bild, werte.schaerfe, werte.schaerfe_radius * massstab)


def anwenden_8bit(rgb_linear, werte: Einstellungen, massstab: float = 1.0):
    """Ganze Kette bis zum sRGB-uint8 fuer Anzeige und Export.

    Liegt das Bild auf der Grafikkarte, rechnen die zusammengefassten Kernel
    aus filter_gpu.py - mit denselben Formeln wie oben, aber in drei statt
    rund vierzig Durchlaeufen.
    """
    if xp_von(rgb_linear) is not np:
        from . import filter_gpu
        return filter_gpu.anwenden_8bit(rgb_linear, werte, massstab)
    return nach_8bit(anwenden(rgb_linear, werte, massstab))
