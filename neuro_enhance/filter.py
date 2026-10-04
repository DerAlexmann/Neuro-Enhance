"""
Klassische Filter

Alle Funktionen arbeiten mit NumPy- oder CuPy-Arrays gleichermassen: xp_von()
waehlt das passende Modul nach dem Array. Im Programm liegen die Bilder als
CuPy-Arrays auf der Grafikkarte; die Tests rechnen dieselben Formeln mit
NumPy, ganz ohne Grafikkarte.

Gerechnet wird in linearem Licht mit float32 (Bildwerte 0..1, sRGB-Primaer-
farben). Erst fuer Anzeige und Export wird zurueck nach sRGB gewandelt.

Die Reihenfolge der Schritte ist fest, wie in der Bildentwicklung ueblich:
Weissabgleich -> Belichtung -> Dunst entfernen -> Tonwerte (Kontrast,
Lichter, Tiefen) -> Klarheit -> Gradationskurven -> Farbe (Dynamik,
Saettigung) -> Schaerfen.

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

    def ist_neutral(self) -> bool:
        for feld in fields(self):
            wert = getattr(self, feld.name)
            if feld.name in KURVEN:
                if not kurven.ist_identitaet(wert):
                    return False
            elif feld.name != "schaerfe_radius" and wert != REGLER_NACH_NAME[feld.name].vorgabe:
                return False
        return True


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
    if bild.ndim == 3:
        fy, fx = fy[..., None], fx[..., None]
    zeile0, zeile1 = bild[y0], bild[y1]
    oben = zeile0[:, x0] * (1 - fx) + zeile0[:, x1] * fx
    unten = zeile1[:, x0] * (1 - fx) + zeile1[:, x1] * fx
    return (oben * (1 - fy) + unten * fy).astype(xp.float32)


def box_mittel(bild, radius: int):
    """Mittelwert ueber ein (2r+1)-Quadrat per Summentabelle; am Rand ueber das Vorhandene."""
    xp = xp_von(bild)
    hoehe, breite = bild.shape[:2]

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
    bild = dunst(bild, werte.dunst)
    bild = tonwerte(bild, werte.kontrast, werte.lichter, werte.tiefen)
    bild = klarheit(bild, werte.klarheit)
    bild = gradation(bild, werte)
    bild = farbe(bild, werte.dynamik, werte.saettigung)
    return schaerfen(bild, werte.schaerfe, werte.schaerfe_radius * massstab)


def anwenden_8bit(rgb_linear, werte: Einstellungen, massstab: float = 1.0,
                  speicher: dict | None = None):
    """Ganze Kette bis zum sRGB-uint8 fuer Anzeige und Export.

    Liegt das Bild auf der Grafikkarte, rechnen die zusammengefassten Kernel
    aus filter_gpu.py - mit denselben Formeln wie oben, aber in wenigen statt
    rund hundert Durchlaeufen. `speicher` ist ein Woerterbuch der Sitzung, in
    dem die GPU-Fassung Zwischenergebnisse fuer das naechste Mal ablegt.
    """
    if xp_von(rgb_linear) is not np:
        from . import filter_gpu
        return filter_gpu.anwenden_8bit(rgb_linear, werte, massstab, speicher)
    return nach_8bit(anwenden(rgb_linear, werte, massstab))
