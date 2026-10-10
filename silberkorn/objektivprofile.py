"""
Objektivprofile aus der lensfun-Datenbank: Verzeichnung, Farbsaeume, Vignette

Die Datenbank (https://lensfun.github.io, CC BY-SA 3.0) wird nicht mitgeliefert,
sondern auf Wunsch des Anwenders vom Update-Server des Projekts geladen und im
Ordner `modelle/lensfun` abgelegt - dort, wo auch die KI-Modelle liegen. Ob es
eine neuere gibt, verraet die kleine Datei versions.json; nachgesehen wird nur
auf Knopfdruck.

Gesucht wird wie bei lensfun: die Kamera ueber Hersteller und Modell aus dem
EXIF, das Objektiv ueber Brennweite, Lichtstaerke und die uebrigen Woerter des
Namens, passend zum Bajonett der Kamera (auch ueber Adapter, etwa F an Z).

Die Formeln sind die von lensfun (alle bilden den korrigierten Radius r_u auf
den Radius im Original r_d ab - genau die Richtung, in der geometrie.py
rueckwaerts rechnet):

  Verzeichnung  ptlens  r_d = r_u (a r_u^3 + b r_u^2 + c r_u + 1 - a - b - c)
                poly3   r_d = r_u (1 - k1 + k1 r_u^2)
                poly5   r_d = r_u (1 + k1 r_u^2 + k2 r_u^4)
  Farbsaeume    linear  r_d = r_u k              (je Rot und Blau)
                poly3   r_d = r_u (b r_u^2 + c r_u + v)
  Vignette      pa      Pixel / (1 + k1 r^2 + k2 r^4 + k3 r^6)

Verzeichnung und Farbsaeume messen r in halben Bildhoehen des Sensors, mit dem
kalibriert wurde, die Vignette in halben Diagonalen. Wie lensfun wird der Faktor
d = 1 - a - b - c herausgerechnet, damit die Bildmitte ihren Massstab behaelt.
Zwischen kalibrierten Brennweiten (und Blenden) wird linear interpoliert.

Ergebnis ist ein Tupel von 13 Zahlen, bezogen auf die halbe Diagonale des
Bildes - unabhaengig von seiner Aufloesung, so passt es zu Vorschau und Export:

  (p1, p2, p3, p4)    Verzeichnung: Faktor 1 + p1 r + p2 r^2 + p3 r^3 + p4 r^4
  (v, c, b) Rot       Farbsaum: Faktor v + c r + b r^2 am Ort im Original
  (v, c, b) Blau
  (k1, k2, k3)        Vignette: 1 / (1 + k1 r^2 + k2 r^4 + k3 r^6)

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from . import einstellungen

QUELLE = "https://lensfun.github.io/db/"
FORMAT = 2                                   # Datenbankformat von lensfun 0.3
ARCHIV = f"version_{FORMAT}.tar.bz2"
VERSIONEN = "versions.json"
HERKUNFT = "lensfun (https://lensfun.github.io), CC BY-SA 3.0"
STAND = "stand.json"
MAX_ARCHIV = 20 * 2**20                      # Bytes; heute rund 0,4 MB
MAX_DATEI = 10 * 2**20
MAX_GESAMT = 60 * 2**20

DIAGONALE_KB = math.hypot(36.0, 24.0)        # Kleinbild, mm
NEUTRAL = (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class DatenbankFehler(Exception):
    """Die Datenbank laesst sich nicht laden oder lesen."""


class Abbruch(Exception):
    """Der Anwender hat das Herunterladen abgebrochen."""


@dataclass(frozen=True)
class Aufnahme:
    """Was EXIF bzw. LibRaw ueber Kamera und Objektiv verraten."""
    hersteller: str = ""
    kamera: str = ""
    objektiv: str = ""
    objektiv_hersteller: str = ""
    brennweite: float = 0.0              # mm
    blende: float = 0.0                  # Blendenzahl
    brennweiten: tuple[float, float] = (0.0, 0.0)   # des Objektivs, falls bekannt


# --------------------------------------------------------------------------
# Ablage, Herunterladen, Aktualisieren
# --------------------------------------------------------------------------

def ordner_kandidaten() -> list[str]:
    """Wie bei den KI-Modellen: neben dem Programm, sonst im Benutzerordner."""
    benutzer = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return [os.path.join(einstellungen.programm_ordner(), "modelle", "lensfun"),
            os.path.join(benutzer, "Silberkorn", "modelle", "lensfun")]


def ordner() -> str | None:
    """Ordner der geladenen Datenbank - oder None."""
    for kandidat in ordner_kandidaten():
        if os.path.isfile(os.path.join(kandidat, STAND)):
            return kandidat
    return None


def ziel_ordner() -> str:
    for kandidat in ordner_kandidaten():
        try:
            os.makedirs(os.path.dirname(kandidat), exist_ok=True)
            probe = kandidat + ".schreibprobe"
            with open(probe, "w", encoding="ascii") as datei:
                datei.write("x")
            os.remove(probe)
            return kandidat
        except OSError:
            continue
    return ordner_kandidaten()[-1]


def stand(pfad: str | None = None) -> dict | None:
    """{"zeitstempel": Sekunden, "geladen": Sekunden} der geladenen Datenbank."""
    pfad = pfad or ordner()
    if pfad is None:
        return None
    try:
        with open(os.path.join(pfad, STAND), encoding="utf-8") as datei:
            daten = json.load(datei)
        return daten if isinstance(daten.get("zeitstempel"), int) else None
    except (OSError, ValueError, AttributeError):
        return None


def _lesen(url: str, grenze: int, fortschritt=None) -> bytes:
    anfrage = urllib.request.Request(url, headers={"User-Agent": "Silberkorn"})
    teile, geladen = [], 0
    try:
        with urllib.request.urlopen(anfrage, timeout=30) as antwort:
            gesamt = int(antwort.headers.get("Content-Length") or 0)
            while True:
                block = antwort.read(1 << 16)
                if not block:
                    break
                geladen += len(block)
                if geladen > grenze:
                    raise DatenbankFehler(f"Antwort zu groß: {url}")
                teile.append(block)
                if fortschritt is not None and fortschritt(geladen, gesamt) is False:
                    raise Abbruch()
    except (urllib.error.URLError, OSError, TimeoutError) as fehler:
        raise DatenbankFehler(f"{url}: {fehler}") from fehler
    return b"".join(teile)


def neuester_stand() -> int:
    """Zeitstempel der neuesten Datenbank auf dem Server (Sekunden seit 1970)."""
    try:
        daten = json.loads(_lesen(QUELLE + VERSIONEN, 1 << 16).decode("utf-8"))
        zeitstempel, formate = int(daten[0]), list(daten[1])
    except (ValueError, TypeError, IndexError, KeyError, UnicodeDecodeError) as fehler:
        raise DatenbankFehler(f"{VERSIONEN} unlesbar: {fehler}") from fehler
    if FORMAT not in formate:
        raise DatenbankFehler(f"Der Server bietet Format {FORMAT} nicht mehr an.")
    return zeitstempel


def aktualisierung() -> int | None:
    """Zeitstempel einer neueren Datenbank als der geladenen - oder None."""
    neu = neuester_stand()
    alt = stand()
    return neu if alt is None or neu > alt["zeitstempel"] else None


_XML_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.xml$")


def _auspacken(archiv: bytes, ziel: str) -> int:
    """Nur flache XML-Dateien aus dem Archiv - keine Pfade, keine Verweise."""
    anzahl = gesamt = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(archiv), mode="r:bz2") as tar:
            for eintrag in tar.getmembers():
                name = eintrag.name.removeprefix("./")
                if not eintrag.isfile() or not _XML_NAME.match(name):
                    continue
                if eintrag.size > MAX_DATEI or gesamt + eintrag.size > MAX_GESAMT:
                    raise DatenbankFehler(f"Datei zu groß: {name}")
                daten = tar.extractfile(eintrag).read()
                gesamt += len(daten)
                with open(os.path.join(ziel, name), "wb") as datei:
                    datei.write(daten)
                anzahl += 1
    except (tarfile.TarError, EOFError, OSError, ValueError) as fehler:
        raise DatenbankFehler(f"Archiv unlesbar: {fehler}") from fehler
    return anzahl


def herunterladen(fortschritt=None) -> str:
    """Neueste Datenbank laden, pruefen und an ihren Platz bringen; Rueckgabe: Ordner.

    Ausgepackt wird in einen Nebenordner; erst wenn sich daraus Objektive lesen
    lassen, ersetzt er die bisherige Datenbank. fortschritt(geladen, gesamt)
    wie bei den KI-Modellen - False bricht ab."""
    zeitstempel = neuester_stand()
    archiv = _lesen(QUELLE + ARCHIV, MAX_ARCHIV, fortschritt)
    ziel = ziel_ordner()
    neu, alt = ziel + ".neu", ziel + ".alt"
    for rest in (neu, alt):
        shutil.rmtree(rest, ignore_errors=True)
    try:
        os.makedirs(neu)
        if _auspacken(archiv, neu) == 0:
            raise DatenbankFehler("Das Archiv enthält keine Datenbank.")
        if not Datenbank.laden(neu).objektive:
            raise DatenbankFehler("Die Datenbank enthält keine Objektive.")
        with open(os.path.join(neu, STAND), "w", encoding="utf-8") as datei:
            json.dump({"zeitstempel": zeitstempel, "geladen": int(time.time())}, datei)
        if os.path.isdir(ziel):
            os.replace(ziel, alt)
        os.replace(neu, ziel)
    except OSError as fehler:
        raise DatenbankFehler(str(fehler)) from fehler
    finally:
        shutil.rmtree(neu, ignore_errors=True)
        shutil.rmtree(alt, ignore_errors=True)
    _CACHE.clear()
    return ziel


# --------------------------------------------------------------------------
# Datenbank lesen
# --------------------------------------------------------------------------

@dataclass
class Kamera:
    hersteller: list[str]
    modelle: list[str]
    bajonett: str
    crop: float


@dataclass
class Objektiv:
    hersteller: list[str]
    modelle: list[str]
    bajonette: list[str]
    crop: float
    seitenverhaeltnis: float = 1.5
    verzeichnung: list[tuple[float, tuple[float, float, float, float]]] = field(
        default_factory=list)                  # (Brennweite, p1..p4 ohne d, kalibriert)
    farbsaum: list[tuple[float, tuple[float, ...]]] = field(
        default_factory=list)                  # (Brennweite, (vr, cr, br, vb, cb, bb))
    vignette: list[tuple[float, float, float, tuple[float, float, float]]] = field(
        default_factory=list)                  # (Brennweite, Blende, Entfernung, k1..k3)

    @property
    def name(self) -> str:
        return self.modelle[-1] if self.modelle else ""


def _zahl(text, vorgabe: float = 0.0) -> float:
    try:
        wert = float(text)
    except (TypeError, ValueError):
        return vorgabe
    return wert if math.isfinite(wert) else vorgabe


def _texte(element, name: str) -> list[str]:
    """Alle Texte eines Kindelements, die ohne Sprache zuerst."""
    teile = element.findall(name)
    teile.sort(key=lambda e: e.get("lang") is not None)
    return [" ".join((e.text or "").split()) for e in teile if (e.text or "").strip()]


def _verzeichnung(attr) -> tuple[float, float, float, float] | None:
    """lensfun-Modell -> 1 + p1 r + p2 r^2 + p3 r^3 + p4 r^4, d herausgerechnet."""
    modell = attr.get("model")
    if modell == "ptlens":
        a, b, c = (_zahl(attr.get(n)) for n in "abc")
        d = 1 - a - b - c
        if abs(d) < 1e-6:
            return None
        return c / d ** 2, b / d ** 3, a / d ** 4, 0.0
    if modell == "poly3":
        k1 = _zahl(attr.get("k1"))
        d = 1 - k1
        if abs(d) < 1e-6:
            return None
        return 0.0, k1 / d ** 3, 0.0, 0.0
    if modell == "poly5":
        return 0.0, _zahl(attr.get("k1")), 0.0, _zahl(attr.get("k2"))
    return None


def _farbsaum(attr) -> tuple[float, ...] | None:
    modell = attr.get("model")
    if modell == "linear":
        return _zahl(attr.get("kr"), 1.0), 0.0, 0.0, _zahl(attr.get("kb"), 1.0), 0.0, 0.0
    if modell == "poly3":
        return (_zahl(attr.get("vr"), 1.0), _zahl(attr.get("cr")), _zahl(attr.get("br")),
                _zahl(attr.get("vb"), 1.0), _zahl(attr.get("cb")), _zahl(attr.get("bb")))
    return None


@dataclass
class Datenbank:
    kameras: list[Kamera] = field(default_factory=list)
    objektive: list[Objektiv] = field(default_factory=list)
    bajonette: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def laden(cls, pfad: str) -> Datenbank:
        db = cls()
        for name in sorted(os.listdir(pfad)):
            if not _XML_NAME.match(name):
                continue
            try:
                with open(os.path.join(pfad, name), "rb") as datei:
                    daten = datei.read(MAX_DATEI + 1)
                if len(daten) > MAX_DATEI or b"<!ENTITY" in daten:
                    continue                        # keine Entitaeten: kein Aufblaehen
                wurzel = ET.fromstring(daten)
            except (OSError, ET.ParseError):
                continue
            db._lesen(wurzel)
        return db

    def _lesen(self, wurzel):
        for element in wurzel.findall("mount"):
            namen = _texte(element, "name")
            if namen:
                self.bajonette.setdefault(namen[0], set()).update(_texte(element, "compat"))
        for element in wurzel.findall("camera"):
            hersteller, modelle = _texte(element, "maker"), _texte(element, "model")
            bajonett = _texte(element, "mount")
            crop = _zahl(element.findtext("cropfactor"), 0.0)
            if hersteller and modelle and crop > 0:
                self.kameras.append(Kamera(hersteller, modelle,
                                           bajonett[0] if bajonett else "", crop))
        for element in wurzel.findall("lens"):
            objektiv = self._objektiv(element)
            if objektiv is not None:
                self.objektive.append(objektiv)

    @staticmethod
    def _objektiv(element) -> Objektiv | None:
        modelle = _texte(element, "model")
        crop = _zahl(element.findtext("cropfactor"), 0.0)
        if not modelle or crop <= 0:
            return None
        verhaeltnis = element.findtext("aspect-ratio")
        if verhaeltnis and ":" in verhaeltnis:
            breite, _, hoehe = verhaeltnis.partition(":")
            verhaeltnis = _zahl(breite, 3) / max(_zahl(hoehe, 2), 1e-6)
        objektiv = Objektiv(_texte(element, "maker"), modelle, _texte(element, "mount"), crop,
                            _zahl(verhaeltnis, 1.5) if verhaeltnis else 1.5)
        for kalibrierung in element.findall("calibration"):
            for eintrag in kalibrierung:
                brennweite = _zahl(eintrag.get("focal"))
                if brennweite <= 0:
                    continue
                if eintrag.tag == "distortion":
                    terme = _verzeichnung(eintrag.attrib)
                    if terme is not None:
                        objektiv.verzeichnung.append((brennweite, terme))
                elif eintrag.tag == "tca":
                    terme = _farbsaum(eintrag.attrib)
                    if terme is not None:
                        objektiv.farbsaum.append((brennweite, terme))
                elif eintrag.tag == "vignetting" and eintrag.get("model") == "pa":
                    blende = _zahl(eintrag.get("aperture"))
                    if blende > 0:
                        objektiv.vignette.append((
                            brennweite, blende, _zahl(eintrag.get("distance"), 1000.0),
                            tuple(_zahl(eintrag.get(k)) for k in ("k1", "k2", "k3"))))
        for liste in (objektiv.verzeichnung, objektiv.farbsaum, objektiv.vignette):
            liste.sort(key=lambda e: e[:2] if len(e) > 2 else e[0])
        if not (objektiv.verzeichnung or objektiv.farbsaum or objektiv.vignette):
            return None
        return objektiv


_CACHE: dict[tuple[str, float], Datenbank] = {}


def datenbank() -> Datenbank | None:
    """Die geladene Datenbank (einmal je Stand gelesen) - oder None."""
    pfad = ordner()
    if pfad is None:
        return None
    try:
        schluessel = (pfad, os.path.getmtime(os.path.join(pfad, STAND)))
    except OSError:
        return None
    if schluessel not in _CACHE:
        _CACHE.clear()
        _CACHE[schluessel] = Datenbank.laden(pfad)
    return _CACHE[schluessel]


# --------------------------------------------------------------------------
# Kamera und Objektiv finden
# --------------------------------------------------------------------------

def _normal(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").split())


# Woerter, die nur den Hersteller nennen - sie sagen nichts ueber die Bauart
_HERSTELLERWORTE = {"canon", "nikon", "nikkor", "fujifilm", "fujinon", "fuji", "sony", "zeiss",
                    "carl", "olympus", "panasonic", "lumix", "leica", "pentax", "smc", "hd",
                    "sigma", "tamron", "tokina", "samyang", "rokinon", "ricoh", "corporation",
                    "imaging", "optical", "co", "ltd", "inc", "om", "system", "zoom", "lens"}

_BRENNWEITE = re.compile(r"(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*mm")
_LICHTSTAERKE = re.compile(r"(?<![a-z])(?:f\s*/?|1\s*:)\s*(\d+(?:\.\d+)?)"
                           r"(?:\s*-\s*(\d+(?:\.\d+)?))?")


def name_zerlegen(name: str):
    """Objektivname -> (Brennweiten, Lichtstaerke bei kurzer Brennweite, Woerter)."""
    text = _normal(name)
    brennweiten = None
    treffer = _BRENNWEITE.search(text)
    if treffer:
        kurz = float(treffer.group(1))
        brennweiten = (kurz, float(treffer.group(2) or kurz))
        text = text[:treffer.start()] + " " + text[treffer.end():]
    lichtstaerke = None
    treffer = _LICHTSTAERKE.search(text)
    if treffer:
        lichtstaerke = float(treffer.group(1))
        text = text[:treffer.start()] + " " + text[treffer.end():]
    return brennweiten, lichtstaerke, set(re.findall(r"[a-z]+|\d+", text))


def kamera_finden(db: Datenbank, hersteller: str, modell: str) -> Kamera | None:
    if not modell:
        return None
    hersteller_n, modell_n = _normal(hersteller), _normal(modell)
    for kamera in db.kameras:
        herst = {_normal(h) for h in kamera.hersteller}
        if hersteller_n and not any(h in hersteller_n or hersteller_n in h for h in herst):
            continue
        for name in kamera.modelle:
            name_n = _normal(name)
            if name_n == modell_n:
                return kamera
            # "NIKON Z 6_2" gegen "Nikon Z 6_2" bzw. "Canon EOS R5" gegen "EOS R5"
            for h in herst | {hersteller_n}:
                for wort in h.split():
                    if modell_n.removeprefix(wort + " ") == name_n.removeprefix(wort + " "):
                        return kamera
    return None


def _gleich(a: float, b: float, toleranz: float) -> bool:
    return abs(a - b) <= toleranz * max(abs(a), abs(b), 1e-6)


def objektive_finden(db: Datenbank, aufnahme: Aufnahme,
                     kamera: Kamera | None) -> list[Objektiv]:
    """Alle Eintraege des besten Treffers (dasselbe Objektiv, oft mit mehreren Crops)."""
    if not aufnahme.objektiv:
        return []
    brennweiten, lichtstaerke, worte = name_zerlegen(aufnahme.objektiv)
    if brennweiten is None and aufnahme.brennweiten[0] > 0:
        brennweiten = aufnahme.brennweiten
    if brennweiten is None:
        return []
    erlaubt = None
    if kamera is not None and kamera.bajonett:
        erlaubt = {kamera.bajonett, *db.bajonette.get(kamera.bajonett, ())}
    hersteller_worte = set(re.findall(r"[a-z]+", _normal(
        aufnahme.objektiv_hersteller or aufnahme.hersteller)))
    worte -= _HERSTELLERWORTE | hersteller_worte

    bewertet = []
    for objektiv in db.objektive:
        if erlaubt is not None and objektiv.bajonette \
                and not erlaubt.intersection(objektiv.bajonette):
            continue
        bester = None
        for name in objektiv.modelle:
            b, licht, eigene = name_zerlegen(name)
            if b is None or not (_gleich(b[0], brennweiten[0], 0.01)
                                 and _gleich(b[1], brennweiten[1], 0.01)):
                continue
            if lichtstaerke is not None and licht is not None \
                    and not _gleich(licht, lichtstaerke, 0.04):
                continue
            eigene = eigene - _HERSTELLERWORTE - {w for h in objektiv.hersteller
                                                  for w in re.findall(r"[a-z]+", _normal(h))}
            gemeinsam = len(worte & eigene)
            if worte and not gemeinsam and eigene:
                continue                # "XF" gegen "Z S": kein Wort passt
            punkte = 2 * gemeinsam - 0.5 * len(eigene - worte) - 0.25 * len(worte - eigene)
            bester = punkte if bester is None else max(bester, punkte)
        if bester is not None:
            bewertet.append((bester, objektiv))
    if not bewertet:
        return []
    hoechst = max(p for p, _o in bewertet)
    namen = {_normal(o.name) for p, o in bewertet if p == hoechst}
    return [o for p, o in bewertet if p == hoechst and _normal(o.name) in namen]


# --------------------------------------------------------------------------
# Profil fuer ein Bild
# --------------------------------------------------------------------------

def _linear(punkte, x: float):
    """Lineare Interpolation in einer nach x sortierten Liste (x, Werte), aussen gehalten."""
    if x <= punkte[0][0]:
        return punkte[0][1]
    if x >= punkte[-1][0]:
        return punkte[-1][1]
    for (x0, w0), (x1, w1) in zip(punkte, punkte[1:], strict=False):
        if x0 <= x <= x1:
            t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
            return tuple(a + t * (b - a) for a, b in zip(w0, w1, strict=True))
    return punkte[-1][1]


def _je_brennweite(eintraege) -> list:
    """Mehrere Eintraege derselben Brennweite: der erste zaehlt."""
    ergebnis = []
    for brennweite, terme in eintraege:
        if not ergebnis or ergebnis[-1][0] != brennweite:
            ergebnis.append((brennweite, terme))
    return ergebnis


def _vignette(eintraege, brennweite: float, blende: float):
    """k1..k3: je Brennweite ueber den Kehrwert der Blende, dann ueber die Brennweite.
    Von mehreren Entfernungen gilt die groesste - die Aufnahmeentfernung ist unbekannt."""
    je_brennweite: dict[float, dict[float, tuple[float, float]]] = {}
    for f, n, entfernung, terme in eintraege:
        bisher = je_brennweite.setdefault(f, {}).get(n)
        if bisher is None or entfernung > bisher[0]:
            je_brennweite[f][n] = (entfernung, terme)
    punkte = []
    for f in sorted(je_brennweite):
        blenden = sorted(((1 / n, terme) for n, (_e, terme) in je_brennweite[f].items()),
                         key=lambda e: e[0])
        punkte.append((f, _linear(blenden, 1 / blende)))
    return _linear(punkte, brennweite)


def _wahl(objektive: list[Objektiv], art: str, crop_kamera: float | None) -> Objektiv | None:
    """Eintrag mit dieser Korrektur, dessen Crop am besten zur Kamera passt: der groesste,
    der nicht ueber dem der Kamera liegt - sonst der kleinste."""
    mit = [o for o in objektive if getattr(o, art)]
    if not mit:
        return None
    if crop_kamera is None:
        return mit[0]
    passend = [o for o in mit if o.crop <= crop_kamera * 1.02]
    if passend:
        return max(passend, key=lambda o: o.crop)
    return min(mit, key=lambda o: o.crop)


@dataclass(frozen=True)
class Profil:
    objektiv: str
    kamera: str                          # leer, wenn die Kamera unbekannt ist
    terme: tuple[float, ...]             # 13 Zahlen, siehe oben
    arten: tuple[str, ...]               # "verzeichnung", "farbsaum", "vignette"


def profil(db: Datenbank, aufnahme: Aufnahme) -> Profil | None:
    """Profil fuer die Aufnahme - oder None, wenn Objektiv oder Brennweite fehlen."""
    kamera = kamera_finden(db, aufnahme.hersteller, aufnahme.kamera)
    objektive = objektive_finden(db, aufnahme, kamera)
    if not objektive:
        return None
    brennweite = aufnahme.brennweite
    if brennweite <= 0:
        b = name_zerlegen(aufnahme.objektiv)[0] or aufnahme.brennweiten
        if not b or b[0] != b[1] or b[0] <= 0:
            return None                  # Zoom ohne Brennweite: nicht zu raten
        brennweite = b[0]
    crop_kamera = kamera.crop if kamera is not None else None
    terme = list(NEUTRAL)
    arten = []

    # Bezug: halbe Bilddiagonale in mm. Ohne Kamera gilt der Sensor der Kalibrierung.
    def halbdiagonale(objektiv: Objektiv) -> float:
        return DIAGONALE_KB / (crop_kamera or objektiv.crop) / 2

    def halbe_hoehe(objektiv: Objektiv) -> float:
        return DIAGONALE_KB / objektiv.crop / math.hypot(objektiv.seitenverhaeltnis, 1) / 2

    objektiv = _wahl(objektive, "verzeichnung", crop_kamera)
    if objektiv is not None:
        s = halbdiagonale(objektiv) / halbe_hoehe(objektiv)
        p = _linear(_je_brennweite(objektiv.verzeichnung), brennweite)
        terme[0:4] = [p[i] * s ** (i + 1) for i in range(4)]
        arten.append("verzeichnung")
    objektiv = _wahl(objektive, "farbsaum", crop_kamera)
    if objektiv is not None:
        s = halbdiagonale(objektiv) / halbe_hoehe(objektiv)
        vr, cr, br, vb, cb, bb = _linear(_je_brennweite(objektiv.farbsaum), brennweite)
        terme[4:10] = [vr, cr * s, br * s * s, vb, cb * s, bb * s * s]
        arten.append("farbsaum")
    objektiv = _wahl(objektive, "vignette", crop_kamera)
    if objektiv is not None and aufnahme.blende > 0:
        s = halbdiagonale(objektiv) / (DIAGONALE_KB / objektiv.crop / 2)
        k1, k2, k3 = _vignette(objektiv.vignette, brennweite, aufnahme.blende)
        terme[10:13] = [k1 * s ** 2, k2 * s ** 4, k3 * s ** 6]
        arten.append("vignette")
    if not arten:
        return None
    kamera_name = kamera.modelle[-1] if kamera is not None else ""
    return Profil(objektive[0].name, kamera_name, tuple(float(t) for t in terme), tuple(arten))


@dataclass(frozen=True)
class Suche:
    """Ergebnis fuer die Oberflaeche: Datenbank da? Objektiv erkannt? Profil?"""
    datenbank: bool
    aufnahme: Aufnahme | None
    profil: Profil | None


def suchen(aufnahme: Aufnahme | None) -> Suche:
    db = datenbank()
    if db is None or aufnahme is None:
        return Suche(db is not None, aufnahme, None)
    return Suche(True, aufnahme, profil(db, aufnahme))
