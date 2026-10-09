"""Hauptfenster ohne Bild: jeder sichtbare Knopf ist beschriftet."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def fenster():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("keine Grafikkarte")
    from PySide6.QtWidgets import QApplication

    from silberkorn import gpu_pruefung
    from silberkorn.hauptfenster import Hauptfenster
    QApplication.instance() or QApplication([])
    befund = gpu_pruefung.ermitteln()
    if not befund.ok:
        pytest.skip("keine RTX-Karte")
    fenster = Hauptfenster(befund)
    fenster.show()
    QApplication.processEvents()
    yield fenster
    fenster.close()


def test_alle_sichtbaren_knoepfe_beschriftet(fenster):
    """Beim Start setzte niemand die Texte der KI-Karten, wenn ihre Modelle schon
    da waren - die Knoepfe blieben leer, bis ein Bild geoeffnet wurde."""
    from PySide6.QtWidgets import QPushButton
    leer = [knopf for knopf in fenster.findChildren(QPushButton)
            if knopf.isVisible() and not knopf.text().strip() and knopf.icon().isNull()]
    assert leer == []
