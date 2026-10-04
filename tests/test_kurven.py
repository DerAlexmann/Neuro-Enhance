"""Tests der Gradationskurven: Punkte bearbeiten und Tabellen bilden."""

from __future__ import annotations

import numpy as np

from neuro_enhance import kurven as k


def test_identitaet_ist_gerade():
    tabelle = k.tabelle(k.IDENTITAET)
    assert tabelle.shape == (k.TABELLE,) and tabelle.dtype == np.float32
    assert np.allclose(tabelle, np.linspace(0, 1, k.TABELLE), atol=1e-6)
    assert k.ist_identitaet(k.IDENTITAET)


def test_punkt_auf_der_diagonale_bleibt_identitaet():
    punkte, _ = k.einfuegen(k.IDENTITAET, 0.5, 0.5)
    assert k.ist_identitaet(punkte)
    assert np.allclose(k.tabelle(punkte), np.linspace(0, 1, k.TABELLE), atol=1e-6)


def test_kurve_laeuft_durch_ihre_punkte():
    punkte = ((0.0, 0.0), (0.25, 0.15), (0.75, 0.9), (1.0, 1.0))
    tabelle = k.tabelle(punkte)
    for x, y in punkte:
        assert abs(tabelle[round(x * (k.TABELLE - 1))] - y) < 2e-3


def test_monoton_ohne_ueberschwingen():
    """Steile S-Kurve mit eng liegenden Punkten: keine Umkehr, kein Ueberlauf."""
    punkte = ((0.0, 0.0), (0.4, 0.05), (0.45, 0.95), (1.0, 1.0))
    tabelle = k.tabelle(punkte)
    assert (np.diff(tabelle) >= -1e-7).all()
    assert tabelle.min() >= 0 and tabelle.max() <= 1


def test_einfuegen_sortiert_und_meldet_index():
    punkte, index = k.einfuegen(k.IDENTITAET, 0.3, 0.6)
    assert punkte == ((0.0, 0.0), (0.3, 0.6), (1.0, 1.0)) and index == 1


def test_einfuegen_zu_nah_wird_abgelehnt():
    punkte, _ = k.einfuegen(k.IDENTITAET, 0.5, 0.5)
    gleich, index = k.einfuegen(punkte, 0.505, 0.7)
    assert gleich == punkte and index is None


def test_verschieben_ueberholt_keinen_nachbarn():
    punkte = ((0.0, 0.0), (0.3, 0.3), (0.6, 0.6), (1.0, 1.0))
    neu = k.verschieben(punkte, 1, 0.9, 0.8)
    assert neu[1][0] < neu[2][0]
    assert neu[1][1] == 0.8


def test_endpunkte_bleiben_am_rand():
    neu = k.verschieben(k.IDENTITAET, 0, 0.4, 0.2)
    assert neu[0] == (0.0, 0.2)
    neu = k.verschieben(k.IDENTITAET, 1, 0.4, 1.5)
    assert neu[1] == (1.0, 1.0)


def test_entfernen_laesst_endpunkte_stehen():
    punkte = ((0.0, 0.0), (0.5, 0.7), (1.0, 1.0))
    assert k.entfernen(punkte, 1) == k.IDENTITAET
    assert k.entfernen(punkte, 0) == punkte
    assert k.entfernen(punkte, 2) == punkte
