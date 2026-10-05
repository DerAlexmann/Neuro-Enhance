"""Tests des KI-Hochskalierens.

Die Stufenlogik laeuft ueberall. Alles, was rechnet, braucht eine Grafikkarte,
ONNX Runtime und die Modelldateien im Ordner `modelle` - fehlt etwas davon,
ueberspringen sich die Tests von selbst (so in der CI).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from neuro_enhance import ki


def test_modelle_je_stufe():
    assert ki.angeboten(ki.MODELLE["schnell"], "S")
    assert not ki.angeboten(ki.MODELLE["qualitaet"], "S")
    assert ki.angeboten(ki.MODELLE["qualitaet"], "M")
    assert not ki.angeboten(ki.MODELLE["schnell"], "-")


def test_kachelgroesse_waechst_mit_der_stufe():
    for modell in ki.MODELLE.values():
        groessen = [ki.kachelgroesse(modell, s) for s in ki.STUFEN if ki.angeboten(modell, s)]
        assert groessen == sorted(groessen)


def test_beschaedigte_datei_wird_abgelehnt(tmp_path, monkeypatch):
    monkeypatch.setattr(ki, "modell_ordner", lambda: str(tmp_path))
    (tmp_path / "x.onnx").write_bytes(b"nicht das Modell")
    with pytest.raises(ki.KiFehler):
        ki.datei_pruefen("x.onnx", "0" * 64)
    with pytest.raises(ki.KiFehler):
        ki.datei_pruefen("fehlt.onnx", "0" * 64)


def bereit(schluessel):
    cp = pytest.importorskip("cupy")
    pytest.importorskip("onnxruntime")
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("keine Grafikkarte")
    if not ki.vorhanden(ki.MODELLE[schluessel]):
        pytest.skip(f"Modelldateien fehlen in {ki.modell_ordner()}")
    return cp


@pytest.fixture(scope="module")
def schnell():
    bereit("schnell")
    return ki.Hochskalierer(ki.MODELLE["schnell"], 0.5)


@pytest.fixture
def bild():
    zufall = np.random.default_rng(31)
    flaechen = np.kron(zufall.random((6, 8, 3)), np.ones((16, 16, 1)))      # 96 x 128
    return (flaechen * 0.8 + zufall.random((96, 128, 3)) * 0.1).astype(np.float32)


@pytest.mark.parametrize("faktor", [2, 4])
def test_ergebnisgroesse(schnell, bild, faktor):
    cp = bereit("schnell")
    aus = schnell.hochskalieren(cp.asarray(bild), faktor, 64, bits=16)
    assert aus.shape == (96 * faktor, 128 * faktor, 3) and aus.dtype == np.uint16


def test_kachelgrenzen_unsichtbar(schnell, bild):
    cp = bereit("schnell")
    ganz = schnell.hochskalieren(cp.asarray(bild), 4, 1024).astype(int)
    kacheln = schnell.hochskalieren(cp.asarray(bild), 4, 40).astype(int)
    assert np.abs(ganz - kacheln).mean() < 0.5


def test_vergroesserung_bleibt_beim_bild(schnell, bild):
    """Verkleinert man das Ergebnis wieder, muss das Original herauskommen."""
    cp = bereit("schnell")
    aus = schnell.hochskalieren(cp.asarray(bild), 4, 256).astype(np.float32) / 255
    zurueck = aus.reshape(96, 4, 128, 4, 3).mean(axis=(1, 3))
    assert np.abs(zurueck - bild).mean() < 0.03


def test_entrauschregler_wirkt(bild):
    cp = bereit("schnell")
    schwach = ki.Hochskalierer(ki.MODELLE["schnell"], 0.0)
    stark = ki.Hochskalierer(ki.MODELLE["schnell"], 1.0)
    a = schwach.hochskalieren(cp.asarray(bild), 4, 256).astype(int)
    b = stark.hochskalieren(cp.asarray(bild), 4, 256).astype(int)
    assert np.abs(a - b).mean() > 0.5


def test_abbruch(schnell, bild):
    cp = bereit("schnell")
    with pytest.raises(ki.KiAbbruch):
        schnell.hochskalieren(cp.asarray(bild), 4, 32, fortschritt=lambda i, n: i < 2)


def test_modellordner_liegt_neben_dem_programm():
    assert os.path.basename(ki.modell_ordner()) == "modelle"
