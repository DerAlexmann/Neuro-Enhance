"""
Anonymisieren: Flaechen verpixeln, weichzeichnen oder fuellen, Gesichter finden,
Standort und Seriennummern aus den Metadaten nehmen

Eine Flaeche ist (form, x0, y0, x1, y1): "rechteck" oder "ellipse", die Ecken auf
0..1 bezogen auf das Original. So folgen die Flaechen Drehen, Zuschnitt und
KI-Erweitern, und Vorschau und Export zeigen dasselbe. Gerechnet wird auf dem
Ausgangsbild vor allen Reglern; KI-Netze sehen weiterhin das Original.

Das Mosaik hat eine feste Zahl Bloecke ueber die Breite der Flaeche - in der
Vorschau so grob wie im gespeicherten Bild. Feine Mosaike lassen sich teilweise
zurueckrechnen; deshalb ist es standardmaessig grob.

Gesichter findet YuNet (MIT-Lizenz, Shiqi Yu; silberkorn/daten/). Es erkennt
Gesichter von etwa 10 bis 300 Pixeln - gerechnet wird darum in mehreren
Massstaeben und grosse Bilder in Kacheln. Es laeuft auf dem Prozessor: das Netz
ist winzig und nimmt so der Bearbeitung keinen Grafikspeicher.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image

from . import filter

FORMEN = ("rechteck", "ellipse")
ARTEN = ("mosaik", "weich", "fuellen")
BLOECKE = 8.0                 # Vorgabe: Bloecke ueber die Breite einer Flaeche
BLOECKE_MIN, BLOECKE_MAX = 4.0, 24.0

YUNET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daten",
                     "face_detection_yunet_2026may.onnx")
YUNET_SHA256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"
SCHRITTE = (8, 16, 32)        # Raster der drei Ausgaenge
# Mindestwert eines Gesichts. OpenCV nimmt 0,9; echte Gesichter lagen im Test bei
# 0,92-0,93, Autoreifen und Felgen bei 0,74-0,85. Was fehlt, ergaenzt man von Hand.
SCHWELLE = 0.88
NMS = 0.3                     # gleiche Gesichter: Ueberlappung (IoU) ab hier
KLEINSTE_KANTE = 640          # erster Massstab: laengste Kante
STUFE = 2.5                   # naechster Massstab: so viel groesser
KACHEL = 1600                 # groessere Bilder in Kacheln dieser Kante ...
UEBERLAPP = 320               # ... die sich so weit ueberlappen (groesste Gesichter)
RAND = 0.25                   # Gesichtsrahmen je Seite um diesen Anteil vergroessert

# EXIF: was beim Speichern ohne Standort wegfaellt
GPS_IFD = 0x8825
EXIF_IFD = 0x8769
PERSOENLICH = {
    0xA430: "CameraOwnerName",
    0xA431: "BodySerialNumber",
    0xA435: "LensSerialNumber",
    0x927C: "MakerNote",       # herstellereigen, enthaelt oft Seriennummern
}


# --------------------------------------------------------------------------
# Flaechen auf das Bild rechnen
# --------------------------------------------------------------------------

def _pixel(flaeche, hoehe: int, breite: int, ox: int, oy: int):
    """Flaeche -> ganze Pixel (xa, ya, xe, ye) im Bild, Original bei (ox, oy)."""
    _form, x0, y0, x1, y1 = flaeche
    xa, xe = sorted((x0 * breite, x1 * breite))
    ya, ye = sorted((y0 * hoehe, y1 * hoehe))
    return (int(math.floor(xa)) + ox, int(math.floor(ya)) + oy,
            int(math.ceil(xe)) + ox, int(math.ceil(ye)) + oy)


def _ellipse(xp, hoehe: int, breite: int):
    """Maske (h, w, 1) der eingeschriebenen Ellipse."""
    y = (xp.arange(hoehe, dtype=xp.float32) + 0.5) / hoehe * 2 - 1
    x = (xp.arange(breite, dtype=xp.float32) + 0.5) / breite * 2 - 1
    return ((y[:, None] ** 2 + x[None, :] ** 2) <= 1)[..., None]


def _mosaik(xp, teil, block: int):
    h, w = teil.shape[:2]
    ph, pw = -h % block, -w % block
    gross = xp.pad(teil, ((0, ph), (0, pw), (0, 0)), mode="edge")
    bloecke = gross.reshape((h + ph) // block, block, (w + pw) // block, block, 3).mean(
        axis=(1, 3))
    return xp.repeat(xp.repeat(bloecke, block, axis=0), block, axis=1)[:h, :w]


def anwenden(bild, flaechen, art: str = "mosaik", bloecke: float = BLOECKE,
             original: tuple[int, int] | None = None, versatz: tuple[int, int] = (0, 0)):
    """Flaechen auf ein lineares Bild (H, W, 3) rechnen - NumPy oder CuPy, als Kopie.

    original: Hoehe und Breite des Originals in diesem Bild (ohne die Raender einer
    KI-Erweiterung), versatz: seine linke obere Ecke. Ohne Angabe ist das ganze
    Bild das Original."""
    if not flaechen:
        return bild
    xp = filter.xp_von(bild)
    hoehe, breite = bild.shape[:2]
    oh, ow = original or (hoehe, breite)
    ox, oy = versatz
    bild = bild.copy()
    for flaeche in flaechen:
        xa, ya, xe, ye = _pixel(flaeche, oh, ow, ox, oy)
        xa, ya, xe, ye = max(xa, 0), max(ya, 0), min(xe, breite), min(ye, hoehe)
        if xe - xa < 2 or ye - ya < 2:
            continue
        teil = bild[ya:ye, xa:xe]
        # Blockgroesse aus der ganzen Flaeche, nicht dem sichtbaren Rest
        _f, x0, _y0, x1, _y1 = flaeche
        kante = max(abs(x1 - x0) * ow, 2.0)
        block = max(2, int(round(kante / bloecke)))
        if art == "mosaik":
            neu = _mosaik(xp, teil, block)
        elif art == "weich":
            # Mit Umgebung weichzeichnen, sonst dunkelt der Rand ab
            r = 3 * block
            ua, va = max(xa - r, 0), max(ya - r, 0)
            ue, ve = min(xe + r, breite), min(ye + r, hoehe)
            umgebung = bild[va:ve, ua:ue]
            weich = xp.stack([filter.gauss(xp.ascontiguousarray(umgebung[..., k]), block)
                              for k in range(3)], axis=-1)
            neu = weich[ya - va:ye - va, xa - ua:xe - ua]
        else:
            neu = xp.zeros_like(teil)
        if flaeche[0] == "ellipse":
            # Die Ellipse der ganzen Flaeche, auf den sichtbaren Teil beschnitten
            ga, gb, gc, gd = _pixel(flaeche, oh, ow, ox, oy)
            maske = _ellipse(xp, gd - gb, gc - ga)[ya - gb:ye - gb, xa - ga:xe - ga]
            neu = xp.where(maske, neu, teil)
        bild[ya:ye, xa:xe] = neu
    return bild


def flaeche_begrenzen(flaeche):
    """Ecken sortieren und auf das Original begrenzen; zu kleine Flaechen -> None."""
    form, x0, y0, x1, y1 = flaeche
    x0, x1 = sorted((min(max(x0, 0.0), 1.0), min(max(x1, 0.0), 1.0)))
    y0, y1 = sorted((min(max(y0, 0.0), 1.0), min(max(y1, 0.0), 1.0)))
    if x1 - x0 < 1e-3 or y1 - y0 < 1e-3:
        return None
    return (form if form in FORMEN else FORMEN[0], x0, y0, x1, y1)


# --------------------------------------------------------------------------
# Gesichter finden (YuNet)
# --------------------------------------------------------------------------

class Gesichtsfinder:
    """YuNet ueber ONNX Runtime auf dem Prozessor."""

    def __init__(self, pfad: str = YUNET):
        import onnxruntime as ort
        optionen = ort.SessionOptions()
        optionen.log_severity_level = 3
        self.sitzung = ort.InferenceSession(pfad, optionen, providers=["CPUExecutionProvider"])

    def _netz(self, bgr: np.ndarray):
        """BGR (h, w, 3) 0..255, Kanten Vielfache von 32 -> Gesichter (n, 5) x0 y0 x1 y1 Wert."""
        hoehe, breite = bgr.shape[:2]
        eingabe = np.ascontiguousarray(bgr.transpose(2, 0, 1)[None], dtype=np.float32)
        ausgaben = dict(zip([a.name for a in self.sitzung.get_outputs()],
                            self.sitzung.run(None, {"input": eingabe}), strict=True))
        return dekodieren(ausgaben, hoehe, breite)

    def _massstab(self, rgb: np.ndarray, massstab: float):
        """Ein Massstab, grosse Bilder in Kacheln -> Gesichter im Original."""
        hoehe, breite = rgb.shape[:2]
        h, w = max(32, round(hoehe * massstab)), max(32, round(breite * massstab))
        bild = rgb if (h, w) == (hoehe, breite) else np.asarray(
            Image.fromarray(rgb).resize((w, h), Image.BILINEAR))
        bgr = bild[..., ::-1].astype(np.float32)
        funde = []
        schritt = KACHEL - UEBERLAPP
        for ya in range(0, max(h - UEBERLAPP, 1), schritt) if h > KACHEL else (0,):
            for xa in range(0, max(w - UEBERLAPP, 1), schritt) if w > KACHEL else (0,):
                teil = bgr[ya:ya + KACHEL, xa:xa + KACHEL]
                th, tw = teil.shape[:2]
                teil = np.pad(teil, ((0, -th % 32), (0, -tw % 32), (0, 0)))
                gefunden = self._netz(teil)
                if len(gefunden):
                    gefunden[:, [0, 2]] += xa
                    gefunden[:, [1, 3]] += ya
                    funde.append(gefunden)
        if not funde:
            return np.zeros((0, 5), np.float32)
        alle = np.concatenate(funde)
        alle[:, :4] /= massstab
        return alle

    def finden(self, rgb: np.ndarray, fortschritt=None) -> list[tuple[float, ...]]:
        """sRGB uint8 (H, W, 3) -> Gesichter als (x0, y0, x1, y1, Wert) in Pixeln."""
        hoehe, breite = rgb.shape[:2]
        massstaebe = []
        m = min(1.0, KLEINSTE_KANTE / max(hoehe, breite))
        while True:
            massstaebe.append(m)
            if m >= 1.0:
                break
            m = min(1.0, m * STUFE)
        funde = []
        for nummer, massstab in enumerate(massstaebe):
            funde.append(self._massstab(rgb, massstab))
            if fortschritt is not None and fortschritt(nummer + 1, len(massstaebe)) is False:
                return []
        alle = np.concatenate(funde)
        return [tuple(float(v) for v in zeile) for zeile in alle[nms(alle)]]


def dekodieren(ausgaben: dict, hoehe: int, breite: int, schwelle: float = SCHWELLE):
    """Die zwoelf Ausgaben von YuNet -> (n, 5): x0, y0, x1, y1, Wert - wie
    cv::FaceDetectorYN (Mittelpunkt im Raster, Groesse exponentiell)."""
    funde = []
    for schritt in SCHRITTE:
        zeilen, spalten = hoehe // schritt, breite // schritt
        wert = np.sqrt(np.clip(ausgaben[f"cls_{schritt}"][0, :, 0], 0, 1)
                       * np.clip(ausgaben[f"obj_{schritt}"][0, :, 0], 0, 1))
        rahmen = ausgaben[f"bbox_{schritt}"][0]
        treffer = np.flatnonzero(wert >= schwelle)
        if not treffer.size:
            continue
        r, c = treffer // spalten, treffer % spalten
        cx = (c + rahmen[treffer, 0]) * schritt
        cy = (r + rahmen[treffer, 1]) * schritt
        w = np.exp(rahmen[treffer, 2]) * schritt
        h = np.exp(rahmen[treffer, 3]) * schritt
        funde.append(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2,
                               wert[treffer]], axis=1))
        assert zeilen * spalten == len(wert), "Raster passt nicht zur Eingabe"
    if not funde:
        return np.zeros((0, 5), np.float32)
    return np.concatenate(funde).astype(np.float32)


def nms(funde: np.ndarray, grenze: float = NMS) -> list[int]:
    """Indizes der Gesichter, die bleiben: je Gruppe ueberlappender das sicherste."""
    if not len(funde):
        return []
    x0, y0, x1, y1, wert = funde.T
    flaeche = np.maximum(x1 - x0, 0) * np.maximum(y1 - y0, 0)
    reihe = list(np.argsort(-wert))
    bleiben = []
    while reihe:
        i = reihe.pop(0)
        bleiben.append(int(i))
        rest = np.array(reihe, dtype=int)
        if not rest.size:
            break
        bx = np.maximum(0, np.minimum(x1[i], x1[rest]) - np.maximum(x0[i], x0[rest]))
        by = np.maximum(0, np.minimum(y1[i], y1[rest]) - np.maximum(y0[i], y0[rest]))
        schnitt = bx * by
        iou = schnitt / np.maximum(flaeche[i] + flaeche[rest] - schnitt, 1e-6)
        # Auch ein kleiner Rahmen ganz in einem grossen ist dasselbe Gesicht
        innen = schnitt / np.maximum(np.minimum(flaeche[i], flaeche[rest]), 1e-6)
        reihe = [int(j) for j, u, v in zip(rest, iou, innen, strict=True)
                 if u < grenze and v < 0.8]
    return bleiben


def als_flaeche(gesicht, hoehe: int, breite: int, form: str = "ellipse"):
    """Gesicht in Pixeln -> Flaeche, je Seite um RAND vergroessert (Haare, Kinn)."""
    x0, y0, x1, y1 = gesicht[:4]
    dx, dy = (x1 - x0) * RAND, (y1 - y0) * RAND
    return flaeche_begrenzen((form, (x0 - dx) / breite, (y0 - dy * 1.4) / hoehe,
                              (x1 + dx) / breite, (y1 + dy) / hoehe))


# --------------------------------------------------------------------------
# Metadaten
# --------------------------------------------------------------------------

def metadaten_bereinigen(exif: bytes) -> bytes:
    """EXIF ohne Standort (GPS), Seriennummern, Besitzername und Herstellerdaten."""
    if not exif:
        return exif
    daten = Image.Exif()
    daten.load(exif[6:] if exif.startswith(b"Exif\x00\x00") else exif)
    if GPS_IFD in daten:
        del daten[GPS_IFD]
    unter = daten.get_ifd(EXIF_IFD)
    for tag in PERSOENLICH:
        unter.pop(tag, None)
    return daten.tobytes()
