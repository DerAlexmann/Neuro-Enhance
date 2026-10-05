"""
Bearbeitungssitzung: ein geoeffnetes Bild auf der Grafikkarte

Das Original liegt zweimal im Grafikspeicher - in voller Groesse fuer den
Export und verkleinert fuer die Vorschau, beide schon in linearem Licht und
in sRGB-Primaerfarben, gleich ob die Datei 8 Bit, 16 Bit oder RAW war und in
welchem Farbraum sie stand.
Jede Aenderung an einem Regler rechnet die Vorschau aus dem unveraenderten
Original neu; nichts wird ueberschrieben, jeder Regler bleibt jederzeit
umkehrbar.

Dieses Modul importiert CuPy und darf deshalb erst nach der Startprufung
geladen werden.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np

from . import bilddatei, filter, geometrie
from .cuda import cupy as cp
from .filter import Einstellungen


def histogramm_von(rgb_uint8):
    """Histogramm der Helligkeit (Rec. 709 auf den sRGB-Werten), 256 Stufen.

    Jeder vierte Pixel genuegt fuer die Form und spart drei Viertel der Arbeit.
    """
    gewichte = cp.asarray(filter.LUMA, dtype=cp.float32)
    stichprobe = rgb_uint8[::2, ::2]
    helligkeit = (stichprobe.astype(cp.float32) @ gewichte + 0.5).astype(cp.int32)
    return cp.bincount(helligkeit.ravel(), minlength=256)[:256]


class Sitzung:
    def __init__(self, daten: bilddatei.Bilddaten, vorschau_kante: int):
        self.daten = daten
        self.werte = Einstellungen()
        self.gespeicherte_werte = Einstellungen()
        self._speicher: dict = {}            # Zwischenergebnisse der Filter, je Bildgroesse
        self._vollbild = None                 # (Einstellungen, fertiges Bild auf der GPU)

        self.original = daten.linear(cp)
        # Zwischenpuffer des Demosaicing an den Grafikspeicher zurueckgeben
        cp.get_default_memory_pool().free_all_blocks()
        self.vorschau_original, self.vorschau_massstab = filter.verkleinern_auf(
            self.original, vorschau_kante)

    @property
    def geaendert(self) -> bool:
        return self.werte != self.gespeicherte_werte

    def vorschau(self, unbearbeitet: bool = False,
                 werte: Einstellungen | None = None) -> tuple[np.ndarray, float, np.ndarray]:
        """Vorschau als sRGB-uint8, Rechenzeit in ms und Helligkeitshistogramm (256 Stufen)."""
        beginn = time.perf_counter()
        if werte is None:
            werte = Einstellungen() if unbearbeitet else self.werte
        bild = filter.anwenden_ausgabe(self.vorschau_original, werte, self.vorschau_massstab,
                                       self._speicher)
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
        if self._vollbild is None or self._vollbild[0] != werte:
            bild = filter.anwenden_ausgabe(self.original, werte, 1.0, self._speicher)
            self._vollbild = (dataclasses.replace(werte), bild)
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

    def exportieren(self, pfad: str, bits: int = 8, ki_auftrag=None, fortschritt=None) -> float:
        """Rechnet das Bild in voller Groesse und speichert es; Rueckgabe in ms.

        ki_auftrag ist (Hochskalierer, Faktor, Kachelgroesse) oder None. Mit KI
        wird das fertige Bild in 16 Bit gerechnet und danach in Kacheln
        vergroessert; fortschritt(i, n) meldet jede fertige Kachel.
        """
        beginn = time.perf_counter()
        if ki_auftrag is None:
            rgb = cp.asnumpy(filter.anwenden_ausgabe(self.original, self.werte, 1.0, bits=bits))
            faktor = 1
        else:
            hochskalierer, faktor, kachel = ki_auftrag
            fertig = filter.anwenden_ausgabe(self.original, self.werte, 1.0, bits=16)
            srgb = fertig.astype(cp.float32) / 65535
            del fertig
            cp.get_default_memory_pool().free_all_blocks()
            rgb = hochskalierer.hochskalieren(srgb, faktor, kachel, bits, fortschritt)
            del srgb
        # Zwischenergebnisse der vollen Groesse sofort zurueckgeben - der
        # Speicherpool von CuPy hielte sie sonst fuer das naechste Mal fest.
        cp.get_default_memory_pool().free_all_blocks()
        bilddatei.speichern(pfad, rgb, self._alpha(faktor), self.daten.exif)
        self.gespeicherte_werte = dataclasses.replace(self.werte)
        return (time.perf_counter() - beginn) * 1000

    def schliessen(self):
        """Grafikspeicher sofort freigeben, nicht erst beim Aufraeumen von Python."""
        self.original = None
        self.vorschau_original = None
        self._vollbild = None
        self._speicher.clear()
        cp.get_default_memory_pool().free_all_blocks()
