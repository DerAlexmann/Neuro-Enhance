"""
Bearbeitungssitzung: ein geoeffnetes Bild auf der Grafikkarte

Das Original liegt zweimal im Grafikspeicher - in voller Groesse fuer den
Export und verkleinert fuer die Vorschau, beide schon in linearem Licht und
in sRGB-Primaerfarben, gleich ob die Datei 8 Bit, 16 Bit oder RAW war und in
welchem Farbraum sie stand.
Jede Aenderung an einem Regler rechnet die Vorschau aus dem unveraenderten
Original neu; nichts wird ueberschrieben, jeder Regler bleibt jederzeit
umkehrbar.

KI-Entrauschen und KI-Schaerfen sind zu teuer, um sie bei jeder
Reglerbewegung zu rechnen. Sie laufen einmal ueber das ganze Bild und liegen
danach im Grafikspeicher - das Entrauschen als zweites Original, das
Schaerfen als Korrektur, die aufs Ergebnis des Entrauschens gelegt wird. Die
Staerkeregler mischen nur noch; das kostet je Vorschau Bruchteile einer
Millisekunde.

Ebenso die Maske des Motivs (KI-Freistellen oder per Klick ausgewaehlt):
einmal berechnet, liegt sie in der Geometrie des Originals im Grafikspeicher.
Fuer den Hintergrund rechnet die Kette das Bild ein zweites Mal mit dessen
eigenen Werten; die Maske - mit derselben Geometrie wie das Bild verformt -
mischt beides.

Entfernte Objekte (LaMa) liegen als Flicken ueber dem Ausgangsbild: je ein
Ausschnitt mit Fuellung und Deckkraft, aufgelegt nach Entrauschen und Schaerfen.
Jeder laesst sich einzeln zuruecknehmen. Was entfernt werden soll, markieren
Klicks (SAM) und ein Pinsel; die Markierung liegt wie die Maske in der
Geometrie des Originals.

Dieses Modul importiert CuPy und darf deshalb erst nach der Startprufung
geladen werden.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field

import numpy as np

from . import bilddatei, filter, geometrie
from .cuda import cupy as cp
from .filter import Einstellungen

# a + s * (b - a) in einem Durchlauf, ohne Zwischenpuffer
_mischen = cp.ElementwiseKernel("float32 a, float32 b, float32 s", "float32 c",
                                "c = a + s * (b - a)", "ne_mischen")
SCHAERFE_GROB = 10.0           # Sigma in Pixeln: groebere Anteile der KI-Schaerfung fallen weg
MASKE_KANTE = 20.0             # Kante weicher: Sigma bei 100 % in Pixeln des Originals
MASKE_VERSCHIEBEN = 20.0       # Kante verschieben: Pixel des Originals bei +-100 %
HG_UNSCHAERFE = 40.0           # Hintergrund weichzeichnen: Sigma bei 100 % in Pixeln
MASKE_FARBE = (1.0, 0.25, 0.2)  # Maskenansicht: so wird der Hintergrund eingefaerbt
MARKIERUNG_FARBE = (0.2, 0.55, 1.0)   # was entfernt werden soll
MARKIERUNG_DECKUNG = 0.55

# a + s * d - eine Korrektur mit Staerke auflegen
_auflegen = cp.ElementwiseKernel("float32 a, float32 d, float32 s", "float32 c",
                                 "c = a + s * d", "ne_auflegen")


def histogramm_von(rgb_uint8):
    """Histogramm der Helligkeit (Rec. 709 auf den sRGB-Werten), 256 Stufen.

    Jeder vierte Pixel genuegt fuer die Form und spart drei Viertel der Arbeit.
    """
    gewichte = cp.asarray(filter.LUMA, dtype=cp.float32)
    stichprobe = rgb_uint8[::2, ::2]
    helligkeit = (stichprobe.astype(cp.float32) @ gewichte + 0.5).astype(cp.int32)
    return cp.bincount(helligkeit.ravel(), minlength=256)[:256]


@dataclass
class _Klicks:
    """Klicks fuer ein Ziel (Maske oder Markierung) samt SAM-Logits nach jedem Klick."""
    punkte: list = field(default_factory=list)      # (x, y, dazu) im Original
    logits: list = field(default_factory=list)
    vorher: object = None                           # Stand des Ziels vor dem ersten Klick


@dataclass
class _Flicken:
    """Ein entferntes Objekt: Ausschnitt, Deckkraft und Fuellung mal Deckkraft (linear)."""
    voll: tuple                     # (y0, y1, x0, x1, deckkraft (h, w), fuellung (h, w, 3))
    vorschau: tuple                 # dasselbe in der Groesse der Vorschau


class Sitzung:
    def __init__(self, daten: bilddatei.Bilddaten, vorschau_kante: int):
        self.daten = daten
        self.werte = Einstellungen()
        self.gespeicherte_werte = Einstellungen()
        self._speicher: dict = {}            # Zwischenergebnisse der Filter, je Bildgroesse
        self._vollbild = None                 # (Einstellungen, fertiges Bild auf der GPU)
        self._ki_rauschfrei = None            # (voll, Vorschau): KI-entrauschtes Original
        self._ki_schaerfe = None              # (voll, Vorschau): Korrektur des KI-Schaerfens
        self._ki_maske = None                 # (voll, Vorschau): Maske des Motivs, 0..1
        self.maske_zeigen = False             # Vorschau faerbt den Hintergrund ein
        self.maske_per_klick = False          # Maske stammt aus der Auswahl per Klick
        self._klickstand = {"maske": _Klicks(), "entfernen": _Klicks()}
        self._entfern_klick = None            # (voll, Vorschau): per Klick markiert
        self._pinsel = None                   # (voll, Vorschau): mit dem Pinsel markiert
        self._flicken: list[_Flicken] = []    # entfernte Objekte, in Reihenfolge
        self._flicken_stand = 0               # zaehlt jede Aenderung an den Flicken
        self._flicken_gespeichert = 0

        self.original = daten.linear(cp)
        # Zwischenpuffer des Demosaicing an den Grafikspeicher zurueckgeben
        cp.get_default_memory_pool().free_all_blocks()
        self.vorschau_original, self.vorschau_massstab = filter.verkleinern_auf(
            self.original, vorschau_kante)

    @property
    def geaendert(self) -> bool:
        return (self.werte != self.gespeicherte_werte
                or self._flicken_stand != self._flicken_gespeichert)

    @property
    def klicks(self) -> list[tuple[float, float, bool]]:
        """Klicks der Maske (Motiv & Hintergrund)."""
        return self._klickstand["maske"].punkte

    def klicks_von(self, ziel: str) -> list[tuple[float, float, bool]]:
        return self._klickstand[ziel].punkte

    @property
    def markierung_da(self) -> bool:
        return self._entfern_klick is not None or self._pinsel is not None

    @property
    def entfernt(self) -> int:
        """Wie viele Objekte entfernt sind."""
        return len(self._flicken)

    @property
    def ki_entrauscht(self) -> bool:
        return self._ki_rauschfrei is not None

    @property
    def ki_geschaerft(self) -> bool:
        return self._ki_schaerfe is not None

    @property
    def ki_maske_da(self) -> bool:
        return self._ki_maske is not None

    def ki_freistellen(self, freisteller) -> float:
        """Die Maske des Motivs einmal mit KI berechnen; Rueckgabe: Rechenzeit in ms.

        Das Netz sieht das Bild, wie es gerade entrauscht und geschaerft ist.
        """
        beginn = time.perf_counter()
        srgb = filter.linear_zu_srgb(cp.clip(self._ausgang(self.werte, True), 0, 1))
        voll = freisteller.maske(srgb)
        del srgb
        self._ki_maske = (voll, self._vorschau_von(voll))
        self.maske_per_klick = False
        self._klickstand["maske"] = _Klicks()
        return self._ki_fertig(beginn)

    def ki_auswahl_beginnen(self, auswaehler, ziel: str = "maske") -> float:
        """Das Bild fuer die Auswahl per Klick vorbereiten (SAM-Encoder); Rueckgabe in ms.

        Wie beim Freistellen sieht das Netz das Bild, wie es gerade entrauscht und
        geschaerft ist. Die Maske von vorher bleibt, bis der erste Klick kommt;
        stammt sie schon aus Klicks, geht es mit ihnen weiter.
        """
        beginn = time.perf_counter()
        srgb = filter.linear_zu_srgb(cp.clip(self._ausgang(self.werte, True), 0, 1))
        auswaehler.bild_setzen(srgb)
        del srgb
        stand = self._klickstand[ziel]
        if not stand.punkte:
            stand.vorher = self._ziel_lesen(ziel)
        cp.get_default_memory_pool().free_all_blocks()
        return (time.perf_counter() - beginn) * 1000

    def quelle_von(self, x: float, y: float) -> tuple[float, float] | None:
        """Punkt im fertigen Bild (Pixel) -> Punkt im Original, oder None ausserhalb."""
        sx, sy = geometrie.zur_quelle(self.original.shape, geometrie.aus(self.werte), x, y)
        hoehe, breite = self.original.shape[:2]
        if not (0 <= sx < breite and 0 <= sy < hoehe):
            return None
        return sx, sy

    def ki_klick(self, auswaehler, x: float, y: float, dazu: bool,
                 ziel: str = "maske") -> float:
        """Einen Klick (im Original) dazunehmen und die Maske neu bestimmen; Rueckgabe in ms.

        SAM bekommt alle Klicks und seine letzte Maske als Hinweis - so verfeinert
        jeder weitere Klick das Bisherige, statt neu zu beginnen.
        """
        beginn = time.perf_counter()
        stand = self._klickstand[ziel]
        vorige = stand.logits[-1] if stand.logits else None
        logits = auswaehler.roh([*stand.punkte, (x, y, dazu)], vorige)
        stand.punkte.append((x, y, dazu))
        stand.logits.append(logits)
        self._aus_klicks(auswaehler, ziel)
        return (time.perf_counter() - beginn) * 1000

    def ki_klick_zurueck(self, auswaehler, ziel: str = "maske") -> bool:
        """Den letzten Klick zuruecknehmen; ohne Klicks gilt wieder der Stand von vorher."""
        stand = self._klickstand[ziel]
        if not stand.punkte:
            return False
        stand.punkte.pop()
        stand.logits.pop()
        if stand.punkte:
            self._aus_klicks(auswaehler, ziel)
        else:
            self._ziel_setzen(ziel, stand.vorher)
        return True

    def _ziel_lesen(self, ziel: str):
        if ziel == "maske":
            return self._ki_maske, self.maske_per_klick
        return self._entfern_klick

    def _ziel_setzen(self, ziel: str, wert):
        if ziel == "maske":
            self._ki_maske, self.maske_per_klick = wert
        else:
            self._entfern_klick = wert
        self._vollbild = None                    # das Vollbild zeigte den alten Stand

    def _aus_klicks(self, auswaehler, ziel: str):
        voll = auswaehler.maske(self._klickstand[ziel].logits[-1])
        paar = (voll, self._vorschau_von(voll))
        self._ziel_setzen(ziel, (paar, True) if ziel == "maske" else paar)

    # ------------------------------------------------------------------
    # Objekte entfernen: Markierung, LaMa, Flicken
    # ------------------------------------------------------------------

    def pinseln(self, punkte, radius: float, dazu: bool):
        """Kreise mit radius (Pixel des Originals) um die Punkte (im Original) markieren
        oder - mit dazu=False - die Markierung dort wegnehmen, auch die per Klick."""
        hoehe, breite = self.original.shape[:2]
        if self._pinsel is None and dazu:
            self._pinsel = (cp.zeros((hoehe, breite), dtype=cp.float32),
                            cp.zeros(self.vorschau_original.shape[:2], dtype=cp.float32))
        ebenen = [self._pinsel] if self._pinsel is not None else []
        if not dazu and self._entfern_klick is not None:
            ebenen.append(self._entfern_klick)
        massstaebe = (1.0, self.vorschau_massstab)
        for ebene in ebenen:
            for teil, m in zip(ebene, massstaebe, strict=True):
                for x, y in punkte:
                    _kreis(teil, x * m, y * m, max(radius * m, 0.5), dazu)
        self._vollbild = None

    def markierung_verwerfen(self):
        self._entfern_klick = None
        self._pinsel = None
        self._klickstand["entfernen"] = _Klicks()
        self._vollbild = None

    def _markierung(self, voll: bool):
        """Was entfernt werden soll (H, W) 0..1 im Original - oder None."""
        stelle = 0 if voll else 1
        ebenen = [e[stelle] for e in (self._entfern_klick, self._pinsel) if e is not None]
        if not ebenen:
            return None
        return ebenen[0] if len(ebenen) == 1 else cp.maximum(ebenen[0], ebenen[1])

    def ki_entfernen(self, entferner) -> float | None:
        """Das Markierte mit LaMa entfernen; Rueckgabe: Rechenzeit in ms, None ohne
        Markierung. LaMa sieht das Bild, wie es gerade entrauscht, geschaerft und
        retuschiert ist - so lassen sich Entfernungen stapeln."""
        markierung = self._markierung(True)
        if markierung is None:
            return None
        beginn = time.perf_counter()
        srgb = filter.linear_zu_srgb(cp.clip(self._ausgang(self.werte, True), 0, 1))
        ergebnis = entferner.fuellen(srgb, markierung)
        del srgb
        if ergebnis is None:
            return None
        y0, y1, x0, x1, fuellung, deckkraft = ergebnis
        gefuellt = filter.srgb_zu_linear(fuellung) * deckkraft[..., None]
        voll = (y0, y1, x0, x1, deckkraft, gefuellt)
        self._flicken.append(_Flicken(voll, self._flicken_vorschau(voll)))
        self._flicken_stand += 1
        self.markierung_verwerfen()
        return self._ki_fertig(beginn)

    def _flicken_vorschau(self, voll):
        """Derselbe Flicken fuer die Vorschau: Box-Mittel wie beim Verkleinern des
        Originals, damit Vorschau und Export dasselbe zeigen."""
        y0, y1, x0, x1, deckkraft, gefuellt = voll
        faktor = round(1 / self.vorschau_massstab)
        if faktor == 1:
            return voll
        vh, vb = self.vorschau_original.shape[:2]
        ya, xa = y0 // faktor, x0 // faktor
        ye, xe = min(vh, -(-y1 // faktor)), min(vb, -(-x1 // faktor))
        if ye <= ya or xe <= xa:
            return (ya, ya, xa, xa, cp.zeros((0, 0), cp.float32), cp.zeros((0, 0, 3), cp.float32))
        hoehe, breite = (ye - ya) * faktor, (xe - xa) * faktor
        oy, ox = y0 - ya * faktor, x0 - xa * faktor
        d = cp.zeros((hoehe, breite), dtype=cp.float32)
        g = cp.zeros((hoehe, breite, 3), dtype=cp.float32)
        h, w = min(y1 - y0, hoehe - oy), min(x1 - x0, breite - ox)
        d[oy:oy + h, ox:ox + w] = deckkraft[:h, :w]
        g[oy:oy + h, ox:ox + w] = gefuellt[:h, :w]
        return (ya, ye, xa, xe, filter.verkleinern_box(d, faktor).astype(cp.float32),
                filter.verkleinern_box(g, faktor).astype(cp.float32))

    def entfernen_zurueck(self) -> bool:
        """Die letzte Entfernung zuruecknehmen."""
        if not self._flicken:
            return False
        self._flicken.pop()
        self._flicken_stand += 1
        self._speicher.clear()
        self._vollbild = None
        return True

    def _flicken_auflegen(self, bild, stelle: int):
        if bild is self.original or bild is self.vorschau_original:
            bild = bild.copy()                    # das Original bleibt unberuehrt
        for flicken in self._flicken:
            y0, y1, x0, x1, deckkraft, gefuellt = flicken.vorschau if stelle else flicken.voll
            if y1 <= y0 or x1 <= x0:
                continue
            teil = bild[y0:y1, x0:x1]
            teil *= (1 - deckkraft)[..., None]
            teil += gefuellt
        return bild

    @staticmethod
    def _in_geometrie(maske, werte: Einstellungen):
        """Eine Maske im Original (H, W) in die Geometrie des fertigen Bildes bringen."""
        geo = geometrie.aus(werte)
        if not geo.ist_neutral():
            nur_form = dataclasses.replace(geo, vignette=0.0, ca_rot=0.0, ca_blau=0.0)
            maske = geometrie.anwenden(cp.repeat(maske[..., None], 3, axis=2), nur_form)[..., 0]
        return cp.ascontiguousarray(cp.clip(maske, 0, 1), dtype=cp.float32)

    def _maske_fuer(self, werte: Einstellungen, voll: bool):
        """Gewicht des Motivs (H, W) in der Geometrie des fertigen Bildes: Kante
        verschoben und weich, wenn gewuenscht umgekehrt. 1 heisst: Werte des ganzen
        Bildes, 0: die des Hintergrunds."""
        maske = self._ki_maske[0 if voll else 1]
        massstab = 1.0 if voll else self.vorschau_massstab
        if werte.maske_verschieben:
            from cupyx.scipy import ndimage
            weite = max(1, round(abs(werte.maske_verschieben) / 100 * MASKE_VERSCHIEBEN
                                 * massstab))
            art = ndimage.maximum_filter if werte.maske_verschieben > 0 else ndimage.minimum_filter
            maske = art(maske, size=2 * weite + 1, mode="nearest")
        if werte.maske_kante:
            maske = filter.gauss(maske, werte.maske_kante / 100 * MASKE_KANTE * massstab)
        if werte.maske_umkehren:
            maske = 1 - maske
        return self._in_geometrie(maske, werte)

    def _rendern(self, werte: Einstellungen, voll: bool, speicher, bits: int = 8,
                 zeigen: bool = False):
        """Die ganze Kette bis zum sRGB-Bild - mit eigenem Hintergrund, falls eingestellt."""
        massstab = 1.0 if voll else self.vorschau_massstab
        eingang = self._ausgang(werte, voll)
        bild = filter.anwenden_ausgabe(eingang, werte, massstab, speicher, bits=bits)
        markierung = self._markierung(voll) if zeigen else None
        zeigen = zeigen and self.maske_zeigen
        hintergrund = self._ki_maske is not None and (werte.hintergrund_aktiv() or zeigen)
        if not hintergrund and markierung is None:
            return bild
        hoechst = 65535.0 if bits == 16 else 255.0
        ergebnis = bild.astype(cp.float32)
        if markierung is not None:
            m = self._in_geometrie(markierung, werte)[..., None] * MARKIERUNG_DECKUNG
            farbe = cp.asarray(MARKIERUNG_FARBE, dtype=cp.float32) * hoechst
            ergebnis = ergebnis * (1 - m) + m * farbe
        typ = cp.uint16 if bits == 16 else cp.uint8
        if not hintergrund:
            return cp.clip(ergebnis + 0.5, 0, hoechst).astype(typ)
        maske = self._maske_fuer(werte, voll)[..., None]
        if werte.hintergrund_aktiv():
            hg = filter.anwenden_ausgabe(eingang, werte.fuer_hintergrund(), massstab, speicher,
                                         bits=bits).astype(cp.float32)
            if werte.hg_unschaerfe:
                # Nur der Hintergrund wird verwischt: das Motiv darf nicht als Schein
                # in ihn hineinlaufen (normierte Faltung mit dem Hintergrund als Gewicht)
                sigma = werte.hg_unschaerfe / 100 * HG_UNSCHAERFE * massstab
                gewicht = 1 - maske[..., 0]
                nenner = cp.maximum(filter.gauss(gewicht, sigma), 1e-4)
                for kanal in range(3):
                    hg[..., kanal] = filter.gauss(hg[..., kanal] * gewicht, sigma) / nenner
            ergebnis = hg + maske * (ergebnis - hg)
        if zeigen:
            farbe = cp.asarray(MASKE_FARBE, dtype=cp.float32) * hoechst
            ergebnis = maske * ergebnis + (1 - maske) * (0.4 * ergebnis + 0.6 * farbe)
        return cp.clip(ergebnis + 0.5, 0, hoechst).astype(typ)

    def _ki_korrektur(self, netz, bild, kachel: int, fortschritt):
        """Was das Netz an einem Bild in linearem Licht aendert, als Korrektur.

        Das Netz kennt Bilder in sRGB von 0 bis 1. Was heller ist - Lichter
        einer RAW -, bekommt es begrenzt zu sehen; zurueck kommt nur seine
        Korrektur, die aufs ungekuerzte Bild gelegt wird. So bleiben die
        Lichter erhalten.
        """
        begrenzt = cp.clip(bild, 0, 1)
        neu = netz.rechnen(filter.linear_zu_srgb(begrenzt), kachel, fortschritt)
        korrektur = filter.srgb_zu_linear(neu)
        del neu
        korrektur -= begrenzt
        return korrektur

    def _vorschau_von(self, voll):
        faktor = round(1 / self.vorschau_massstab)
        return voll if faktor == 1 else filter.verkleinern_box(voll, faktor).astype(cp.float32)

    def _ki_fertig(self, beginn: float) -> float:
        # Zwischenergebnisse beruhten auf dem alten Ausgangsbild
        self._speicher.clear()
        self._vollbild = None
        cp.get_default_memory_pool().free_all_blocks()
        return (time.perf_counter() - beginn) * 1000

    def ki_entrauschen(self, entrauscher, kachel: int, fortschritt=None) -> float:
        """Das Original einmal mit KI entrauschen; Rueckgabe: Rechenzeit in ms."""
        beginn = time.perf_counter()
        voll = self._ki_korrektur(entrauscher, self.original, kachel, fortschritt)
        voll += self.original
        self._ki_rauschfrei = (voll, self._vorschau_von(voll))
        return self._ki_fertig(beginn)

    def ki_schaerfen(self, schaerfer, kachel: int, fortschritt=None) -> float:
        """Das Bild einmal mit KI schaerfen; Rueckgabe: Rechenzeit in ms.

        Geschaerft wird das Original, wie es gerade entrauscht ist - Rauschen
        wuerde das Netz sonst mitschaerfen. Gespeichert wird nur die Korrektur;
        sie bleibt gueltig, wenn sich die Staerke des Entrauschens danach
        noch etwas aendert.

        Von der Korrektur bleibt nur der feine Anteil. Restormer hellt das Bild
        nebenbei auf und verschiebt die Farbe ein wenig, je Kachel verschieden;
        Schaerfe aber steckt in den feinen Strukturen. Was groeber ist als etwa
        zehn Pixel, wird deshalb abgezogen.
        """
        beginn = time.perf_counter()
        korrektur = self._ki_korrektur(schaerfer, self._ausgang(self.werte, True, schaerfen=False),
                                       kachel, fortschritt)
        for kanal in range(3):
            korrektur[..., kanal] -= filter.gauss(korrektur[..., kanal], SCHAERFE_GROB)
        self._ki_schaerfe = (korrektur, self._vorschau_von(korrektur))
        return self._ki_fertig(beginn)

    def _ausgang(self, werte: Einstellungen, voll: bool, schaerfen: bool = True):
        """Original - je nach Staerke mit dem KI-entrauschten gemischt und KI-geschaerft."""
        stelle = 0 if voll else 1
        bild = self.original if voll else self.vorschau_original
        if self._ki_rauschfrei is not None and werte.ki_rauschen > 0:
            glatt = self._ki_rauschfrei[stelle]
            bild = glatt if werte.ki_rauschen >= 100 else \
                _mischen(bild, glatt, cp.float32(werte.ki_rauschen / 100))
        if schaerfen and self._ki_schaerfe is not None and werte.ki_schaerfe > 0:
            bild = _auflegen(bild, self._ki_schaerfe[stelle], cp.float32(werte.ki_schaerfe / 100))
        if self._flicken:
            bild = self._flicken_auflegen(bild, stelle)
        return bild

    def vorschau(self, unbearbeitet: bool = False,
                 werte: Einstellungen | None = None) -> tuple[np.ndarray, float, np.ndarray]:
        """Vorschau als sRGB-uint8, Rechenzeit in ms und Helligkeitshistogramm (256 Stufen)."""
        beginn = time.perf_counter()
        if werte is None:
            werte = Einstellungen() if unbearbeitet else self.werte
        bild = self._rendern(werte, False, self._speicher, zeigen=not unbearbeitet)
        histogramm = cp.asnumpy(histogramm_von(bild))
        ergebnis = cp.asnumpy(bild)                     # wartet auf die GPU
        return ergebnis, (time.perf_counter() - beginn) * 1000, histogramm

    def ausgabe_form(self, werte: Einstellungen | None = None) -> tuple[int, int]:
        """Hoehe und Breite des fertigen Bildes in voller Aufloesung."""
        werte = self.werte if werte is None else werte
        return geometrie.ausgabe_form(self.original.shape, geometrie.aus(werte))

    def ausschnitt(self, x0: int, y0: int, breite: int, hoehe: int,
                   werte: Einstellungen | None = None) -> tuple[np.ndarray, float, np.ndarray]:
        """Ausschnitt des fertigen Bildes in voller Aufloesung - fuer die 100-%-Ansicht.

        Gerechnet wird das ganze Bild, damit der Ausschnitt genau dem
        gespeicherten Bild entspricht - auch bei Filtern, die auf das ganze
        Bild schauen (Dunst, Klarheit). Das Ergebnis bleibt auf der GPU liegen,
        solange sich die Einstellungen nicht aendern; beim Verschieben wird
        nur noch ausgeschnitten und heruntergeladen.
        """
        beginn = time.perf_counter()
        werte = self.werte if werte is None else werte
        schluessel = (dataclasses.replace(werte), self.maske_zeigen)
        if self._vollbild is None or self._vollbild[0] != schluessel:
            self._vollbild = None                 # das alte Vollbild zuerst freigeben
            bild = self._rendern(werte, True, self._speicher, zeigen=True)
            self._vollbild = (schluessel, bild)
        teil = self._vollbild[1][y0:y0 + hoehe, x0:x0 + breite]
        histogramm = cp.asnumpy(histogramm_von(teil))
        ergebnis = cp.asnumpy(teil)
        return ergebnis, (time.perf_counter() - beginn) * 1000, histogramm

    def vollbild_vergessen(self):
        """Das zwischengespeicherte Vollbild freigeben (etwa beim Wechsel zur Einpassung)."""
        self._vollbild = None

    def _alpha(self, faktor: int = 1):
        """Alphakanal mit derselben Geometrie wie das Bild - ohne Vignette und Farbsaeume."""
        alpha = self.daten.alpha
        if alpha is None:
            return None
        geo = geometrie.aus(self.werte)
        if geo.ist_neutral() and faktor == 1:
            return alpha
        hoechst = 65535.0 if alpha.dtype == np.uint16 else 255.0
        ebene = cp.asarray(alpha, dtype=cp.float32) / hoechst
        if not geo.ist_neutral():
            nur_form = dataclasses.replace(geo, vignette=0.0, ca_rot=0.0, ca_blau=0.0)
            ebene = geometrie.anwenden(cp.repeat(ebene[..., None], 3, axis=2), nur_form)[..., 0]
        if faktor > 1:
            ebene = filter.vergroessern(ebene, ebene.shape[0] * faktor, ebene.shape[1] * faktor)
        return cp.asnumpy((cp.clip(ebene, 0, 1) * hoechst + 0.5).astype(alpha.dtype))

    def _alpha_gesamt(self, faktor: int = 1):
        """Alphakanal fuers Speichern - beim Freistellen mit der Maske verrechnet."""
        alpha = self._alpha(faktor)
        if not (self.werte.freistellen and self._ki_maske is not None):
            return alpha
        maske = self._maske_fuer(self.werte, True)
        if faktor > 1:
            maske = filter.vergroessern(maske, maske.shape[0] * faktor, maske.shape[1] * faktor)
        if alpha is None:
            return cp.asnumpy((cp.clip(maske, 0, 1) * 65535 + 0.5).astype(cp.uint16))
        hoechst = 65535.0 if alpha.dtype == np.uint16 else 255.0
        gesamt = cp.asarray(alpha, dtype=cp.float32) * maske
        return cp.asnumpy(cp.clip(gesamt + 0.5, 0, hoechst).astype(alpha.dtype))

    def exportieren(self, pfad: str, bits: int = 8, ki_auftrag=None, fortschritt=None) -> float:
        """Rechnet das Bild in voller Groesse und speichert es; Rueckgabe in ms.

        ki_auftrag ist (Hochskalierer, Faktor, Kachelgroesse) oder None. Mit KI
        wird das fertige Bild in 16 Bit gerechnet und danach in Kacheln
        vergroessert; fortschritt(i, n) meldet jede fertige Kachel.
        """
        beginn = time.perf_counter()
        if ki_auftrag is None:
            rgb = cp.asnumpy(self._rendern(self.werte, True, None, bits=bits))
            faktor = 1
        else:
            hochskalierer, faktor, kachel = ki_auftrag
            fertig = self._rendern(self.werte, True, None, bits=16)
            srgb = fertig.astype(cp.float32) / 65535
            del fertig
            cp.get_default_memory_pool().free_all_blocks()
            rgb = hochskalierer.hochskalieren(srgb, faktor, kachel, bits, fortschritt)
            del srgb
        # Zwischenergebnisse der vollen Groesse sofort zurueckgeben - der
        # Speicherpool von CuPy hielte sie sonst fuer das naechste Mal fest.
        cp.get_default_memory_pool().free_all_blocks()
        bilddatei.speichern(pfad, rgb, self._alpha_gesamt(faktor), self.daten.exif)
        self.gespeicherte_werte = dataclasses.replace(self.werte)
        self._flicken_gespeichert = self._flicken_stand
        return (time.perf_counter() - beginn) * 1000

    def schliessen(self):
        """Grafikspeicher sofort freigeben, nicht erst beim Aufraeumen von Python."""
        self.original = None
        self.vorschau_original = None
        self._vollbild = None
        self._ki_rauschfrei = None
        self._ki_schaerfe = None
        self._ki_maske = None
        self._klickstand = {"maske": _Klicks(), "entfernen": _Klicks()}
        self._entfern_klick = None
        self._pinsel = None
        self._flicken = []
        self._speicher.clear()
        cp.get_default_memory_pool().free_all_blocks()


def _kreis(ebene, x: float, y: float, radius: float, dazu: bool):
    """Eine weiche Kreisscheibe in eine Ebene (H, W) stempeln oder herausnehmen."""
    hoehe, breite = ebene.shape
    y0, y1 = max(0, int(y - radius - 1)), min(hoehe, int(y + radius + 2))
    x0, x1 = max(0, int(x - radius - 1)), min(breite, int(x + radius + 2))
    if y1 <= y0 or x1 <= x0:
        return
    yy = cp.arange(y0, y1, dtype=cp.float32)[:, None] + 0.5
    xx = cp.arange(x0, x1, dtype=cp.float32)[None, :] + 0.5
    abstand = cp.sqrt((yy - y) ** 2 + (xx - x) ** 2)
    scheibe = cp.clip(radius + 0.5 - abstand, 0, 1)      # ein Pixel weicher Rand
    teil = ebene[y0:y1, x0:x1]
    if dazu:
        cp.maximum(teil, scheibe, out=teil)
    else:
        cp.minimum(teil, 1 - scheibe, out=teil)
