"""Tests der Geometrie: Drehen, Spiegeln, Zuschnitt, Begradigen, Perspektive, Objektiv."""

from __future__ import annotations

import numpy as np
import pytest

from neuro_enhance import filter as f
from neuro_enhance import geometrie as g
from neuro_enhance.geometrie import Geometrie


@pytest.fixture
def bild():
    zufall = np.random.default_rng(21)
    return zufall.random((40, 80, 3), dtype=np.float32)


def test_neutral_ist_dasselbe_bild(bild):
    assert g.anwenden(bild, Geometrie()) is bild


@pytest.mark.parametrize("drehung, erwartet", [
    (1, lambda a: np.rot90(a, -1)),            # im Uhrzeigersinn
    (2, lambda a: a[::-1, ::-1]),
    (3, lambda a: np.rot90(a, 1)),
])
def test_vierteldrehungen_sind_exakt(bild, drehung, erwartet):
    assert np.allclose(g.anwenden(bild, Geometrie(drehung90=drehung)), erwartet(bild), atol=1e-6)


def test_spiegeln_ist_exakt(bild):
    assert np.allclose(g.anwenden(bild, Geometrie(spiegeln=True)), bild[:, ::-1], atol=1e-6)


def test_zuschnitt_ist_exakt(bild):
    ausschnitt = g.anwenden(bild, Geometrie(zuschnitt=(0.25, 0.25, 0.75, 0.75)))
    assert np.allclose(ausschnitt, bild[10:30, 20:60], atol=1e-6)


def test_ausgabe_form():
    assert g.ausgabe_form((40, 80), Geometrie(drehung90=1)) == (80, 40)
    assert g.ausgabe_form((40, 80), Geometrie(zuschnitt=(0, 0, 0.5, 0.5))) == (20, 40)


def punkt_finden(bild):
    y, x = np.unravel_index(np.argmax(bild[..., 0]), bild.shape[:2])
    return y, x


def test_begradigen_dreht_im_uhrzeigersinn():
    """Ein Lichtpunkt rechts der Mitte wandert bei positivem Winkel nach unten."""
    bild = np.zeros((101, 101, 3), dtype=np.float32)
    bild[50, 80] = 1.0
    y, x = punkt_finden(g.anwenden(bild, Geometrie(begradigen=30)))
    assert y > 55 and x > 50


def test_begradigen_ohne_leere_ecken():
    """Nach dem Begradigen muss jeder Ergebnispixel aus dem Bild stammen."""
    form = (60, 90)
    geo = Geometrie(begradigen=12, perspektive_v=40, verzeichnung=-50)
    zoom = g.automatischer_zoom(form, geo)
    assert zoom > 1
    # Ein Bild mit Rand 0 und Innenflaeche 1: Randpixel duerfen nicht hereinlaufen
    bild = np.zeros((*form, 3), dtype=np.float32)
    bild[2:-2, 2:-2] = 1.0
    ergebnis = g.anwenden(bild, geo)
    assert ergebnis[3:-3, 3:-3].min() > 0.99


def test_vignette_hellt_die_ecken_auf():
    grau = np.full((40, 60, 3), 0.2, dtype=np.float32)
    ergebnis = g.anwenden(grau, Geometrie(vignette=50))
    assert ergebnis[0, 0, 0] > ergebnis[20, 30, 0] * 1.2
    assert np.isclose(ergebnis[20, 30, 0], 0.2, atol=2e-3)


def test_farbsaeume_skalieren_nur_rot_und_blau(bild):
    ergebnis = g.anwenden(bild, Geometrie(ca_rot=100, ca_blau=-100))
    assert np.allclose(ergebnis[..., 1], bild[..., 1], atol=1e-6)
    assert not np.allclose(ergebnis[..., 0], bild[..., 0], atol=1e-3)


def test_verzeichnung_positiv_zieht_den_rand_nach_aussen():
    """Positive Werte gleichen tonnenfoermige Verzeichnung aus: Randpunkte wandern nach aussen."""
    bild = np.zeros((101, 101, 3), dtype=np.float32)
    bild[50, 90] = 1.0
    y, x = punkt_finden(g.anwenden(bild, Geometrie(verzeichnung=100)))
    assert x > 90


def test_ganze_kette_mit_zuschnitt(bild):
    werte = f.Einstellungen(zuschnitt=(0.1, 0.2, 0.6, 0.9), drehung90=1, kontrast=20)
    ergebnis = f.anwenden_ausgabe(bild, werte)
    assert ergebnis.shape[:2] == g.ausgabe_form(bild.shape, g.aus(werte))


@pytest.mark.parametrize("geo", [
    Geometrie(drehung90=1, spiegeln=True, zuschnitt=(0.1, 0.05, 0.9, 0.8)),
    Geometrie(begradigen=7.5, perspektive_v=30, perspektive_h=-20),
    Geometrie(verzeichnung=40, vignette=60, ca_rot=50, ca_blau=-30),
])
def test_cuda_wie_referenz(bild, geo):
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("keine Grafikkarte")
    referenz = g.anwenden(bild, geo)
    gpu = cp.asnumpy(g.anwenden(cp.asarray(bild), geo))
    assert gpu.shape == referenz.shape
    assert np.abs(gpu - referenz).max() < 2e-3


@pytest.mark.parametrize("gespiegelt", [False, True])
@pytest.mark.parametrize("richtung", [1, -1])
def test_drehknopf_dreht_sichtbar_in_pfeilrichtung(bild, gespiegelt, richtung):
    """Rechts drehen dreht das sichtbare Bild im Uhrzeigersinn - auch wenn es gespiegelt ist.

    Gespiegelt kehrt sich die Drehrichtung der Quelle um; die Seite gleicht das
    aus (BearbeitenSeite.drehen). Hier wird dieselbe Rechnung geprueft.
    """
    vorher = Geometrie(drehung90=1, spiegeln=gespiegelt)
    nachher = Geometrie(drehung90=(1 + (-richtung if gespiegelt else richtung)) % 4,
                        spiegeln=gespiegelt)
    sichtbar = g.anwenden(bild, vorher)
    erwartet = np.rot90(sichtbar, -richtung)
    assert np.allclose(g.anwenden(bild, nachher), erwartet, atol=1e-6)


@pytest.mark.parametrize("geo", [
    Geometrie(),
    Geometrie(drehung90=1, spiegeln=True),
    Geometrie(drehung90=3, zuschnitt=(0.1, 0.2, 0.9, 0.7)),
    Geometrie(begradigen=7.0, perspektive_v=30, verzeichnung=20, zuschnitt=(0.1, 0.1, 0.8, 0.9)),
])
def test_klick_findet_den_punkt_im_original(geo):
    """zur_quelle ist die Abbildung, mit der anwenden() liest - ein Klick auf einen
    Punkt im fertigen Bild trifft ihn im Original."""
    quelle = np.zeros((60, 90, 3), dtype=np.float32)
    quelle[34, 51] = 1.0
    ergebnis = g.anwenden(quelle, geo)
    y, x = punkt_finden(ergebnis)
    sx, sy = g.zur_quelle(quelle.shape, geo, x + 0.5, y + 0.5)
    assert abs(sx - 51.5) < 1.0 and abs(sy - 34.5) < 1.0
