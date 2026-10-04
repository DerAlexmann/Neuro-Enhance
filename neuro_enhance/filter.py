"""
Klassische Filter

Alle Funktionen arbeiten mit NumPy- oder CuPy-Arrays gleichermassen: xp_von()
waehlt das passende Modul nach dem Array. Im Programm liegen die Bilder als
CuPy-Arrays auf der Grafikkarte; die Tests rechnen dieselben Formeln mit
NumPy, ganz ohne Grafikkarte.

Gerechnet wird in linearem Licht mit float32 (Bildwerte 0..1, sRGB-Primaer-
farben). Erst fuer Anzeige und Export wird zurueck nach sRGB gewandelt.

Die Reihenfolge der Schritte ist fest, wie in der Bildentwicklung ueblich:
Weissabgleich -> Belichtung -> Entrauschen -> Dunst entfernen -> Tonwerte
(Kontrast, Lichter, Tiefen) -> Klarheit -> Gradationskurven -> HSL je
Farbbereich -> Farbe (Dynamik, Saettigung) -> LUT -> Schaerfen.

Dunst entfernen und Klarheit arbeiten mit grossen Nachbarschaften. Sie
rechnen deshalb auf einer verkleinerten Kopie und vergroessern das Ergebnis
wieder - so wirken sie in Vorschau und Export gleich, unabhaengig von der
Aufloesung.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import numpy as np

from . import kurven
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
    Regler("klarheit", "praesenz", -100, 100),
    Regler("dunst", "praesenz", -100, 100),
    Regler("dynamik", "farbe", -100, 100),
    Regler("saettigung", "farbe", -100, 100),
    Regler("rauschen_luminanz", "rauschen", 0, 100),
    Regler("rauschen_farbe", "rauschen", 0, 100),
    Regler("schaerfe", "details", 0, 150),
    Regler("schaerfe_radius", "details", 0.5, 3.0, vorgabe=1.0, schritt=0.1, nachkomma=1),
    Regler("lut_staerke", "lut", 0, 100, vorgabe=100),
)

# HSL je Farbbereich: acht Bereiche mit ihrer Mitte als Farbwinkel in OkLCh
FARBBEREICHE = ("rot", "orange", "gelb", "gruen", "aqua", "blau", "lila", "magenta")
FARBBEREICH_MITTE = (29.0, 55.0, 105.0, 142.0, 195.0, 264.0, 300.0, 330.0)
HSL = ("hsl_farbton", "hsl_saettigung", "hsl_luminanz")
HSL_NEUTRAL = (0.0,) * len(FARBBEREICHE)
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
    klarheit: float = 0.0
    dunst: float = 0.0
    dynamik: float = 0.0
    saettigung: float = 0.0
    schaerfe: float = 0.0
    schaerfe_radius: float = 1.0
    # Gradationskurven: eine fuer die Helligkeit, je eine fuer Rot, Gruen, Blau
    kurve_hell: kurven.Punkte = field(default=kurven.IDENTITAET)
    kurve_rot: kurven.Punkte = field(default=kurven.IDENTITAET)
    kurve_gruen: kurven.Punkte = field(default=kurven.IDENTITAET)
    kurve_blau: kurven.Punkte = field(default=kurven.IDENTITAET)
    # HSL: je ein Wert -100..100 fuer die acht Farbbereiche
    hsl_farbton: tuple[float, ...] = HSL_NEUTRAL
    hsl_saettigung: tuple[float, ...] = HSL_NEUTRAL
    hsl_luminanz: tuple[float, ...] = HSL_NEUTRAL
    # Pfad einer .cube-Datei; leer heisst keine LUT
    lut: str = ""
    rauschen_luminanz: float = 0.0
    rauschen_farbe: float = 0.0
    lut_staerke: float = 100.0

    def ist_neutral(self) -> bool:
        for feld in fields(self):
            wert = getattr(self, feld.name)
            if feld.name in KURVEN:
                if not kurven.ist_identitaet(wert):
                    return False
            elif feld.name in HSL:
                if any(wert):
                    return False
            elif feld.name == "lut":
                if wert:
                    return False
            elif feld.name not in ("schaerfe_radius", "lut_staerke") \
                    and wert != REGLER_NACH_NAME[feld.name].vorgabe:
                return False
        return True

    def hsl_aktiv(self) -> bool:
        return any(any(getattr(self, name)) for name in HSL)

    def lut_aktiv(self) -> bool:
        return bool(self.lut) and self.lut_staerke > 0

    def rauschen_aktiv(self) -> bool:
        return self.rauschen_luminanz > 0 or self.rauschen_farbe > 0


KURVEN = ("kurve_hell", "kurve_rot", "kurve_gruen", "kurve_blau")


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


def nach_16bit(rgb_linear):
    """Lineares float-Bild -> sRGB uint16 fuer den Export mit 16 Bit."""
    xp = xp_von(rgb_linear)
    v = linear_zu_srgb(xp.clip(rgb_linear, 0, 1))
    return (v * 65535 + 0.5).astype(xp.uint16)


def verkleinern_auf(bild, laengste_kante: int):
    """Vorschau: um einen ganzzahligen Faktor verkleinern, bis die laengste Kante passt.

    Gemittelt wird in linearem Licht - so bleiben feine helle Strukturen so hell,
    wie sie wirken. Rueckgabe: verkleinertes Bild und Massstab zum Original.
    """
    faktor = max(1, math.ceil(max(bild.shape[:2]) / laengste_kante))
    if faktor == 1:
        return bild, 1.0
    klein = verkleinern_box(bild, faktor).astype(xp_von(bild).float32)
    return klein, 1.0 / faktor


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
    if xp is not np:
        # gleiche Rechnung, ein Kernel je Achse; "mirror" = np.pad(mode="reflect")
        from cupyx.scipy import ndimage
        gewichte = xp.asarray(kern)
        ergebnis = ndimage.correlate1d(bild, gewichte, axis=0, mode="mirror")
        return ndimage.correlate1d(ergebnis, gewichte, axis=1, mode="mirror")
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
# Werkzeuge fuer grosse Nachbarschaften
# --------------------------------------------------------------------------

def verkleinern_box(bild, faktor: int):
    """Mittelwert ueber faktor x faktor Pixel; der Rest am Rand faellt weg."""
    if faktor <= 1:
        return bild
    hoehe = bild.shape[0] // faktor * faktor
    breite = bild.shape[1] // faktor * faktor
    zuschnitt = bild[:hoehe, :breite]
    form = (hoehe // faktor, faktor, breite // faktor, faktor) + bild.shape[2:]
    return zuschnitt.reshape(form).mean(axis=(1, 3))


def vergroessern(bild, hoehe: int, breite: int):
    """Bilinear auf (hoehe, breite) vergroessern, Pixelmitten aufeinander ausgerichtet."""
    xp = xp_von(bild)
    h0, b0 = bild.shape[:2]
    if (h0, b0) == (hoehe, breite):
        return bild
    y = (xp.arange(hoehe, dtype=xp.float32) + 0.5) * xp.float32(h0 / hoehe) - 0.5
    x = (xp.arange(breite, dtype=xp.float32) + 0.5) * xp.float32(b0 / breite) - 0.5
    y = xp.clip(y, 0, h0 - 1)
    x = xp.clip(x, 0, b0 - 1)
    y0 = xp.floor(y).astype(xp.int32)
    x0 = xp.floor(x).astype(xp.int32)
    y1 = xp.minimum(y0 + 1, h0 - 1)
    x1 = xp.minimum(x0 + 1, b0 - 1)
    fy = (y - y0)[:, None]
    fx = (x - x0)[None, :]
    for _ in range(bild.ndim - 2):
        fy, fx = fy[..., None], fx[..., None]
    zeile0, zeile1 = bild[y0], bild[y1]
    oben = zeile0[:, x0] * (1 - fx) + zeile0[:, x1] * fx
    unten = zeile1[:, x0] * (1 - fx) + zeile1[:, x1] * fx
    return (oben * (1 - fy) + unten * fy).astype(xp.float32)


def box_mittel(bild, radius: int):
    """Mittelwert ueber ein (2r+1)-Quadrat per Summentabelle; am Rand ueber das Vorhandene."""
    xp = xp_von(bild)
    hoehe, breite = bild.shape[:2]
    if xp is not np:
        # Gleiche Rechnung in zwei Kastenfiltern: ausserhalb zaehlen Nullen, und die
        # Teilung durch den gefilterten Einser-Rahmen ergibt das Mittel ueber das
        # Vorhandene - wie die Summentabellen unten.
        from cupyx.scipy import ndimage
        groesse = (2 * radius + 1, 2 * radius + 1) + (1,) * (bild.ndim - 2)
        summe = ndimage.uniform_filter(bild.astype(xp.float32), groesse, mode="constant")
        anzahl = ndimage.uniform_filter(xp.ones((hoehe, breite), dtype=xp.float32),
                                        groesse[:2], mode="constant")
        if bild.ndim > 2:
            anzahl = anzahl.reshape(anzahl.shape + (1,) * (bild.ndim - 2))
        return (summe / anzahl).astype(xp.float32)

    def laengs(a, achse, laenge):
        summe = xp.cumsum(a, axis=achse, dtype=xp.float64)
        nullen = xp.zeros_like(xp.take(summe, xp.asarray([0]), axis=achse))
        summe = xp.concatenate([nullen, summe], axis=achse)
        i = xp.arange(laenge)
        oben = xp.minimum(i + radius + 1, laenge)
        unten = xp.maximum(i - radius, 0)
        return xp.take(summe, oben, axis=achse) - xp.take(summe, unten, axis=achse)

    summe = laengs(laengs(bild, 0, hoehe), 1, breite)
    eins = xp.ones((hoehe, breite), dtype=xp.float64)
    anzahl = laengs(laengs(eins, 0, hoehe), 1, breite)
    if bild.ndim == 3:
        anzahl = anzahl[..., None]
    return (summe / anzahl).astype(xp.float32)


def min_filter(bild, radius: int):
    """Minimum ueber ein (2r+1)-Quadrat, separierbar, Randpixel wiederholt."""
    xp = xp_von(bild)
    if xp is not np:
        # gleiches Ergebnis in einem Kernel; "nearest" wiederholt den Randpixel
        from cupyx.scipy import ndimage
        return ndimage.minimum_filter(bild, size=2 * radius + 1, mode="nearest")
    ergebnis = bild
    for achse in (0, 1):
        breite = [(0, 0), (0, 0)]
        breite[achse] = (radius, radius)
        gepolstert = xp.pad(ergebnis, breite, mode="edge")
        laenge = ergebnis.shape[achse]
        minimum = None
        for i in range(2 * radius + 1):
            ausschnitt = [slice(None), slice(None)]
            ausschnitt[achse] = slice(i, i + laenge)
            teil = gepolstert[tuple(ausschnitt)]
            minimum = teil if minimum is None else xp.minimum(minimum, teil)
        ergebnis = minimum
    return ergebnis


def gefuehrter_filter(fuehrung, eingabe, radius: int, eps: float):
    """Kantenerhaltende Glaettung nach He, Sun und Tang (Guided Filter)."""
    mittel_f = box_mittel(fuehrung, radius)
    mittel_e = box_mittel(eingabe, radius)
    kovarianz = box_mittel(fuehrung * eingabe, radius) - mittel_f * mittel_e
    varianz = box_mittel(fuehrung * fuehrung, radius) - mittel_f * mittel_f
    a = kovarianz / (varianz + eps)
    b = mittel_e - a * mittel_f
    return box_mittel(a, radius) * fuehrung + box_mittel(b, radius)


def grob_weich_klein(bild, sigma: float):
    """Gauss mit grossem Radius auf einer verkleinerten Kopie - noch nicht vergroessert."""
    faktor = max(1, int(sigma // 4))
    return gauss(verkleinern_box(bild, faktor), sigma / faktor)


def grob_weichzeichnen(bild, sigma: float):
    """Gauss mit grossem Radius: verkleinern, weichzeichnen, wieder vergroessern."""
    return vergroessern(grob_weich_klein(bild, sigma), bild.shape[0], bild.shape[1])


# --------------------------------------------------------------------------
# Dunst, Klarheit, Kurven
# --------------------------------------------------------------------------

DUNST_KANTE = 512          # Schaetzung von Dunst und Lichtfarbe auf dieser Groesse
DUNST_MIN_DURCHLASS = 0.1  # schuetzt dichten Dunst vor Rauschen und Ausreissern
KLARHEIT_SIGMA = 0.008     # Radius der Klarheit relativ zur laengsten Kante


def dunst_schaetzen(rgb):
    """Lichtfarbe des Dunstes und Durchlasskarte - nach dem Dark Channel Prior.

    In dunstfreien Bildern hat fast jedes Fleckchen einen Kanal nahe Schwarz.
    Wo das nicht so ist, liegt Dunst darueber; je heller der dunkelste Kanal,
    desto dichter. Die Lichtfarbe des Dunstes stammt aus den dunstigsten 0,1 %.
    Gerechnet wird auf einer Kopie mit etwa 512 Pixeln Kantenlaenge.
    """
    xp = xp_von(rgb)
    faktor = max(1, max(rgb.shape[:2]) // DUNST_KANTE)
    klein = verkleinern_box(rgb, faktor)
    radius = max(1, round(7 * max(klein.shape[:2]) / DUNST_KANTE))

    dunkel = min_filter(klein.min(axis=-1), radius)
    flach = dunkel.reshape(-1)
    anzahl = max(1, flach.size // 1000)
    oben = xp.argsort(flach)[-anzahl:]
    licht = xp.maximum(klein.reshape(-1, 3)[oben].mean(axis=0), 0.05)

    normiert = min_filter((klein / licht).min(axis=-1), radius)
    durchlass = 1 - 0.95 * normiert
    durchlass = gefuehrter_filter(luminanz(klein), durchlass, radius * 4, 1e-3)
    return licht.astype(xp.float32), xp.clip(durchlass, 0, 1).astype(xp.float32)


def dunst(rgb, staerke):
    """Positiv: Dunst entfernen. Negativ: Dunst hinzufuegen."""
    if staerke == 0:
        return rgb
    xp = xp_von(rgb)
    licht, durchlass = dunst_schaetzen(rgb)
    if staerke < 0:
        anteil = xp.float32(-staerke / 100 * 0.6)
        return (rgb * (1 - anteil) + licht * anteil).astype(xp.float32)
    # Die Staerke mischt zwischen keinem und dem vollen Entfernen des Dunstes.
    durchlass = 1 - xp.float32(staerke / 100) * (1 - durchlass)
    durchlass = vergroessern(durchlass, rgb.shape[0], rgb.shape[1])
    durchlass = xp.maximum(durchlass, DUNST_MIN_DURCHLASS)[..., None]
    return xp.maximum((rgb - licht) / durchlass + licht, 0).astype(xp.float32)


def klarheit_sigma(form) -> float:
    return KLARHEIT_SIGMA * max(form[:2])


def klarheit(rgb, staerke):
    """Lokaler Kontrast in den Mitteltoenen - unscharf maskieren mit grossem Radius.

    Die Gewichtung 4 v (1 - v) laesst Tiefen und Lichter aus, sonst entstuenden
    dort Saeume und abgesoffene Schatten.
    """
    if staerke == 0:
        return rgb
    xp = xp_von(rgb)
    y = luminanz(rgb)
    v = linear_zu_srgb(y)
    weich = grob_weichzeichnen(v, klarheit_sigma(rgb.shape))
    gewicht = xp.clip(4 * v * (1 - v), 0, 1)
    v_neu = v + xp.float32(staerke / 100) * gewicht * (v - weich)
    y_neu = srgb_zu_linear(xp.clip(v_neu, 0, None))
    return (rgb * (y_neu / xp.maximum(y, EPS))[..., None]).astype(xp.float32)


def tabelle_anwenden(tabelle, v):
    """Wert 0..1 ueber die Tabelle abbilden, linear zwischen den Stuetzstellen."""
    xp = xp_von(v)
    n = tabelle.shape[0]
    pos = xp.clip(v, 0, 1) * (n - 1)
    i0 = xp.minimum(xp.floor(pos).astype(xp.int32), n - 2)
    anteil = pos - i0
    return tabelle[i0] * (1 - anteil) + tabelle[i0 + 1] * anteil


def kurven_tabellen(werte: Einstellungen) -> dict[str, np.ndarray]:
    """Nur die Kurven, die etwas veraendern, als Tabellen."""
    return {name: kurven.tabelle(getattr(werte, name))
            for name in KURVEN if not kurven.ist_identitaet(getattr(werte, name))}


def gradation(rgb, werte: Einstellungen):
    """Helligkeitskurve farbtreu, danach die Kanalkurven auf R, G, B einzeln.

    Wie bei den Tonwerten wirkt die Helligkeitskurve auf die sRGB-kodierte
    Luminanz und skaliert die Kanaele im selben Verhaeltnis. Die Kanalkurven
    duerfen die Farbe dagegen gezielt verschieben - dafuer sind sie da.
    """
    tabellen = kurven_tabellen(werte)
    if not tabellen:
        return rgb
    xp = xp_von(rgb)
    if "kurve_hell" in tabellen:
        y = luminanz(rgb)
        v = linear_zu_srgb(y)
        innen = xp.clip(v, 0, 1)
        neu = tabelle_anwenden(xp.asarray(tabellen["kurve_hell"]), innen) + (v - innen)
        y_neu = srgb_zu_linear(neu)
        schwarz = (y <= EPS)[..., None]
        rgb = xp.where(schwarz, y_neu[..., None],
                       rgb * (y_neu / xp.maximum(y, EPS))[..., None]).astype(xp.float32)
    kanaele = []
    for nummer, name in enumerate(("kurve_rot", "kurve_gruen", "kurve_blau")):
        kanal = rgb[..., nummer]
        if name in tabellen:
            v = linear_zu_srgb(kanal)
            innen = xp.clip(v, 0, 1)
            kanal = srgb_zu_linear(tabelle_anwenden(xp.asarray(tabellen[name]), innen)
                                   + (v - innen))
        kanaele.append(kanal)
    return xp.stack(kanaele, axis=-1).astype(xp.float32)


# --------------------------------------------------------------------------
# HSL je Farbbereich - in OkLab / OkLCh
# --------------------------------------------------------------------------

# Lineares sRGB -> LMS -> OkLab (Bjoern Ottosson, 2020)
OKLAB_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                     [0.2119034982, 0.6806995451, 0.1073969566],
                     [0.0883024619, 0.2817188376, 0.6299787005]])
OKLAB_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                     [1.9779984951, -2.4285922050, 0.4505937099],
                     [0.0259040371, 0.7827717662, -0.8086757660]])
OKLAB_M1_INV = np.linalg.inv(OKLAB_M1)
OKLAB_M2_INV = np.linalg.inv(OKLAB_M2)

HSL_FARBTON_GRAD = 30.0      # Farbton +-100 verschiebt um bis zu 30 Grad
HSL_LUMINANZ = 0.3           # Luminanz +-100 aendert die OkLab-Helligkeit um bis zu 30 %
HSL_CHROMA_VOLL = 0.08       # ab dieser Buntheit wirkt die Luminanz voll; Grau bleibt Grau


def _matrix(rgb, m):
    xp = xp_von(rgb)
    return rgb @ xp.asarray(m.T, dtype=xp.float32)


def nach_oklab(rgb):
    xp = xp_von(rgb)
    return _matrix(xp.cbrt(_matrix(rgb, OKLAB_M1)), OKLAB_M2)


def von_oklab(lab):
    return _matrix(_matrix(lab, OKLAB_M2_INV) ** 3, OKLAB_M1_INV)


def bereichsgewichte(farbwinkel):
    """Gewicht jedes Farbbereichs je Pixel (..., 8); die Summe ist immer 1.

    Zwischen zwei benachbarten Bereichsmitten wird mit einer Kosinuskurve
    uebergeblendet - so gibt es keine Stufen, wenn ein Regler bewegt wird.
    """
    xp = xp_von(farbwinkel)
    mitten = list(FARBBEREICH_MITTE)
    gewichte = []
    for i, mitte in enumerate(mitten):
        vorher = mitten[i - 1] - (360.0 if i == 0 else 0.0)
        nachher = mitten[(i + 1) % len(mitten)] + (360.0 if i == len(mitten) - 1 else 0.0)
        # Abstand auf dem Farbkreis, vorzeichenbehaftet in (-180, 180]
        abstand = (farbwinkel - mitte + 180.0) % 360.0 - 180.0
        steigend = 0.5 * (1 + xp.cos(np.pi * xp.clip(-abstand / (mitte - vorher), 0, 1)))
        fallend = 0.5 * (1 + xp.cos(np.pi * xp.clip(abstand / (nachher - mitte), 0, 1)))
        gewichte.append(xp.where(abstand < 0, steigend, fallend))
    return xp.stack(gewichte, axis=-1).astype(xp.float32)


def hsl(rgb, werte: Einstellungen):
    """Farbton, Saettigung und Luminanz je Farbbereich verschieben."""
    if not werte.hsl_aktiv():
        return rgb
    xp = xp_von(rgb)
    lab = nach_oklab(rgb)
    hell, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = xp.sqrt(a * a + b * b)
    winkel = xp.degrees(xp.arctan2(b, a)) % 360.0
    gewicht = bereichsgewichte(winkel)

    def summe(werte_tupel, faktor):
        return (gewicht * xp.asarray(werte_tupel, dtype=xp.float32)).sum(axis=-1) \
            * xp.float32(faktor / 100)

    winkel_neu = xp.radians(winkel + summe(werte.hsl_farbton, HSL_FARBTON_GRAD))
    chroma_neu = chroma * xp.maximum(1 + summe(werte.hsl_saettigung, 1.0), 0)
    buntheit = xp.clip(chroma / HSL_CHROMA_VOLL, 0, 1)
    hell_neu = hell * (1 + summe(werte.hsl_luminanz, HSL_LUMINANZ) * buntheit)
    lab_neu = xp.stack([hell_neu, chroma_neu * xp.cos(winkel_neu),
                        chroma_neu * xp.sin(winkel_neu)], axis=-1)
    return von_oklab(lab_neu).astype(xp.float32)


# --------------------------------------------------------------------------
# LUT
# --------------------------------------------------------------------------

def lut_anwenden(rgb, werte: Einstellungen):
    """LUT auf die sRGB-kodierten Werte, mit der Staerke zum Original gemischt."""
    if not werte.lut_aktiv():
        return rgb
    from . import lut
    xp = xp_von(rgb)
    tabelle = lut.laden(werte.lut)
    v = linear_zu_srgb(xp.clip(rgb, 0, 1))
    neu = lut.anwenden(v, tabelle)
    staerke = xp.float32(werte.lut_staerke / 100)
    return srgb_zu_linear(xp.clip(v + staerke * (neu - v), 0, 1))


# --------------------------------------------------------------------------
# Entrauschen
# --------------------------------------------------------------------------

NLM_SUCHE = 3                # Suchfenster 7 x 7
NLM_FLECK = 1                # Vergleichsflecken 3 x 3
NLM_H = 0.06                 # Filterstaerke bei Regler 100, in sRGB-kodierter Helligkeit
FARBRAUSCHEN_RADIUS = 0.004  # relativ zur laengsten Kante


def _gespiegelt(bild, rand):
    return xp_von(bild).pad(bild, rand, mode="reflect")


def nlm(v, staerke):
    """Non-Local Means auf einem 2D-Bild: Mittel ueber aehnliche Flecken der Umgebung."""
    xp = xp_von(v)
    h2 = xp.float32((staerke / 100 * NLM_H) ** 2)
    rand = NLM_SUCHE + NLM_FLECK
    hoehe, breite = v.shape
    gepolstert = _gespiegelt(v, rand)
    flaeche = (2 * NLM_FLECK + 1) ** 2

    def ausschnitt(dy, dx):
        return gepolstert[rand + dy:rand + dy + hoehe, rand + dx:rand + dx + breite]

    summe = xp.zeros_like(v)
    gewichte = xp.zeros_like(v)
    for dy in range(-NLM_SUCHE, NLM_SUCHE + 1):
        for dx in range(-NLM_SUCHE, NLM_SUCHE + 1):
            abstand = xp.zeros_like(v)
            for fy in range(-NLM_FLECK, NLM_FLECK + 1):
                for fx in range(-NLM_FLECK, NLM_FLECK + 1):
                    unterschied = ausschnitt(fy, fx) - ausschnitt(dy + fy, dx + fx)
                    abstand = abstand + unterschied * unterschied
            gewicht = xp.exp(-(abstand / flaeche) / h2)
            summe = summe + gewicht * ausschnitt(dy, dx)
            gewichte = gewichte + gewicht
    return (summe / gewichte).astype(xp.float32)


def _inverse_3x3(m):
    """Inverse vieler symmetrischer 3x3-Matrizen (..., 3, 3) ueber die Adjunkte."""
    xp = xp_von(m)
    a, b, c = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    e, f_, i = m[..., 1, 1], m[..., 1, 2], m[..., 2, 2]
    k00, k01, k02 = e * i - f_ * f_, c * f_ - b * i, b * f_ - c * e
    k11, k12, k22 = a * i - c * c, b * c - a * f_, a * e - b * b
    det = a * k00 + b * k01 + c * k02
    adj = xp.stack([xp.stack([k00, k01, k02], -1), xp.stack([k01, k11, k12], -1),
                    xp.stack([k02, k12, k22], -1)], -2)
    return adj / det[..., None, None]


def farbrauschen_koeffizienten(fuehrung, chroma, staerke, form):
    """Farbgefuehrter Filter (He, Sun, Tang) als Fast Guided Filter auf einer Verkleinerung.

    fuehrung (H, W, 3) ist das sRGB-kodierte Bild selbst, chroma (H, W, 3) der
    Abstand jedes Kanals zur Helligkeit. Weil die Fuehrung farbig ist, bleiben
    Kanten zwischen Farben gleicher Helligkeit erhalten - eine Fuehrung nur
    ueber die Helligkeit wuerde dort die Farbe verschmieren. eps trennt Rauschen
    (kleine Streuung) von echten Farbkanten (grosse Streuung).

    Rueckgabe: a (h, w, 3, 3) und b (h, w, 3) auf der Verkleinerung; das
    Ergebnis ist chroma_k = sum_j a[k, j] * fuehrung_j + b[k].
    """
    xp = xp_von(chroma)
    radius = max(2.0, FARBRAUSCHEN_RADIUS * max(form[:2]))
    faktor = max(1, int(radius // 4))
    fuehrung_k = verkleinern_box(fuehrung, faktor)
    chroma_k = verkleinern_box(chroma, faktor)
    r = max(1, round(radius / faktor))
    eps = xp.float32(1e-4 + (staerke / 100) ** 2 * 0.004)

    mittel_f = box_mittel(fuehrung_k, r)                                   # (h, w, 3)
    mittel_c = box_mittel(chroma_k, r)                                     # (h, w, 3)
    produkte = fuehrung_k[..., :, None] * fuehrung_k[..., None, :]         # (h, w, 3, 3)
    streuung = (box_mittel(produkte.reshape(*produkte.shape[:2], 9), r)
                .reshape(produkte.shape) - mittel_f[..., :, None] * mittel_f[..., None, :])
    streuung = streuung + eps * xp.eye(3, dtype=xp.float32)
    kreuz = chroma_k[..., :, None] * fuehrung_k[..., None, :]              # [k, j]
    kovarianz = (box_mittel(kreuz.reshape(*kreuz.shape[:2], 9), r).reshape(kreuz.shape)
                 - mittel_c[..., :, None] * mittel_f[..., None, :])
    a = kovarianz @ _inverse_3x3(streuung)                                 # (h, w, 3, 3)
    b = mittel_c - (a @ mittel_f[..., None])[..., 0]
    a = box_mittel(a.reshape(*a.shape[:2], 9), r).reshape(a.shape)
    return a.astype(xp.float32), box_mittel(b, r).astype(xp.float32)


def entrauschen(rgb, werte: Einstellungen):
    """Luminanzrauschen mit Non-Local Means, Farbrauschen mit Fast Guided Filter."""
    if not werte.rauschen_aktiv():
        return rgb
    xp = xp_von(rgb)
    y = luminanz(rgb)
    v = linear_zu_srgb(y)
    if werte.rauschen_luminanz > 0:
        v_neu = nlm(v, werte.rauschen_luminanz)
        rgb = rgb * (srgb_zu_linear(xp.clip(v_neu, 0, None)) / xp.maximum(y, EPS))[..., None]
        v = v_neu
    if werte.rauschen_farbe > 0:
        kodiert = linear_zu_srgb(rgb)
        chroma = kodiert - v[..., None]
        a, b = farbrauschen_koeffizienten(kodiert, chroma, werte.rauschen_farbe, rgb.shape)
        hoehe, breite = v.shape
        geglaettet = ((vergroessern(a, hoehe, breite) @ kodiert[..., None])[..., 0]
                      + vergroessern(b, hoehe, breite))
        staerke = xp.float32(min(1.0, werte.rauschen_farbe / 50))
        chroma = chroma + staerke * (geglaettet - chroma)
        rgb = srgb_zu_linear(xp.maximum(v[..., None] + chroma, 0))
    return rgb.astype(xp.float32)


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
    bild = entrauschen(bild, werte)
    bild = dunst(bild, werte.dunst)
    bild = tonwerte(bild, werte.kontrast, werte.lichter, werte.tiefen)
    bild = klarheit(bild, werte.klarheit)
    bild = gradation(bild, werte)
    bild = hsl(bild, werte)
    bild = farbe(bild, werte.dynamik, werte.saettigung)
    bild = lut_anwenden(bild, werte)
    return schaerfen(bild, werte.schaerfe, werte.schaerfe_radius * massstab)


def anwenden_ausgabe(rgb_linear, werte: Einstellungen, massstab: float = 1.0,
                     speicher: dict | None = None, bits: int = 8):
    """Ganze Kette bis zum sRGB-Ergebnis mit 8 oder 16 Bit - fuer Anzeige und Export.

    Liegt das Bild auf der Grafikkarte, rechnen die zusammengefassten Kernel
    aus filter_gpu.py - mit denselben Formeln wie oben, aber in wenigen statt
    rund hundert Durchlaeufen. `speicher` ist ein Woerterbuch der Sitzung, in
    dem die GPU-Fassung Zwischenergebnisse fuer das naechste Mal ablegt.
    """
    if xp_von(rgb_linear) is not np:
        from . import filter_gpu
        return filter_gpu.anwenden_ausgabe(rgb_linear, werte, massstab, speicher, bits)
    ergebnis = anwenden(rgb_linear, werte, massstab)
    return nach_16bit(ergebnis) if bits == 16 else nach_8bit(ergebnis)
