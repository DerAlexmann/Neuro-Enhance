"""
Bearbeitungssitzung: ein geoeffnetes Bild auf der Grafikkarte

Das Original liegt zweimal im Grafikspeicher - in voller Groesse fuer den
Export und verkleinert fuer die Vorschau, beide schon in linearem Licht.
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

from . import bilddatei, filter
from .cuda import cupy as cp
from .filter import Einstellungen


class Sitzung:
    def __init__(self, daten: bilddatei.Bilddaten, vorschau_kante: int):
        self.daten = daten
        self.werte = Einstellungen()
        self.gespeicherte_werte = Einstellungen()

        self.original = filter.von_8bit(cp.asarray(daten.rgb))
        klein, self.vorschau_massstab = bilddatei.verkleinern(daten.rgb, vorschau_kante)
        self.vorschau_original = filter.von_8bit(cp.asarray(klein))

    @property
    def geaendert(self) -> bool:
        return self.werte != self.gespeicherte_werte

    def vorschau(self, unbearbeitet: bool = False) -> tuple[np.ndarray, float]:
        """Vorschau als sRGB-uint8 und die Rechenzeit auf der GPU in Millisekunden."""
        beginn = time.perf_counter()
        werte = Einstellungen() if unbearbeitet else self.werte
        bild = filter.anwenden_8bit(self.vorschau_original, werte, self.vorschau_massstab)
        ergebnis = cp.asnumpy(bild)                     # wartet auf die GPU
        return ergebnis, (time.perf_counter() - beginn) * 1000

    def exportieren(self, pfad: str) -> float:
        """Rechnet das Bild in voller Groesse und speichert es; Rueckgabe in ms."""
        beginn = time.perf_counter()
        rgb = cp.asnumpy(filter.anwenden_8bit(self.original, self.werte, 1.0))
        # Zwischenergebnisse der vollen Groesse sofort zurueckgeben - der
        # Speicherpool von CuPy hielte sie sonst fuer das naechste Mal fest.
        cp.get_default_memory_pool().free_all_blocks()
        bilddatei.speichern(pfad, rgb, self.daten.alpha, self.daten.exif)
        self.gespeicherte_werte = dataclasses.replace(self.werte)
        return (time.perf_counter() - beginn) * 1000

    def schliessen(self):
        """Grafikspeicher sofort freigeben, nicht erst beim Aufraeumen von Python."""
        self.original = None
        self.vorschau_original = None
        cp.get_default_memory_pool().free_all_blocks()
