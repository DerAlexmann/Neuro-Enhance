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
    monkeypatch.setattr(ki, "ordner_kandidaten", lambda: [str(tmp_path)])
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


def test_katalog_vollstaendig():
    for modell in ki.MODELLE.values():
        for name in ki.dateien(modell):
            assert name in ki.DATEIEN
        assert modell.sha256 == ki.DATEIEN[modell.datei][0]


# ----------------------------------------------------------------------
# Herunterladen - mit einem file://-Ordner statt GitHub
# ----------------------------------------------------------------------

@pytest.fixture
def quelle_und_ziel(tmp_path, monkeypatch):
    """Ein Quellordner mit den Dateien des schnellen Modells (Inhalt erfunden)."""
    import hashlib
    quelle, ziel = tmp_path / "quelle", tmp_path / "ziel"
    quelle.mkdir()
    katalog = {}
    for name in ki.dateien(ki.MODELLE["schnell"]):
        inhalt = (name * 50000).encode()
        (quelle / name).write_bytes(inhalt)
        katalog[name] = (hashlib.sha256(inhalt).hexdigest(), len(inhalt))
    monkeypatch.setattr(ki, "DATEIEN", katalog)
    monkeypatch.setattr(ki, "ordner_kandidaten", lambda: [str(ziel)])
    return quelle.as_uri() + "/", ziel


def test_herunterladen(quelle_und_ziel):
    url, ziel = quelle_und_ziel
    modell = ki.MODELLE["schnell"]
    assert not ki.vorhanden(modell)
    meldungen = []
    ki.herunterladen(modell, lambda g, n: meldungen.append((g, n)) or True, quelle=url)
    assert ki.vorhanden(modell) and ki.fehlende(modell) == []
    assert meldungen[-1][0] == meldungen[-1][1] == sum(v[1] for v in ki.DATEIEN.values())
    assert not list(ziel.glob("*.teil"))


def test_falsche_pruefsumme_wird_verworfen(quelle_und_ziel, monkeypatch):
    url, ziel = quelle_und_ziel
    name = ki.MODELLE["schnell"].datei
    monkeypatch.setitem(ki.DATEIEN, name, ("0" * 64, ki.DATEIEN[name][1]))
    with pytest.raises(ki.KiFehler):
        ki.herunterladen(ki.MODELLE["schnell"], quelle=url)
    assert not (ziel / name).exists() and not list(ziel.glob("*.teil"))


def test_abbruch_hinterlaesst_nichts(quelle_und_ziel):
    url, ziel = quelle_und_ziel
    with pytest.raises(ki.KiAbbruch):
        ki.herunterladen(ki.MODELLE["schnell"], lambda g, n: False, quelle=url)
    assert not list(ziel.glob("*.teil"))
    assert not ki.vorhanden(ki.MODELLE["schnell"])


def test_fehlende_quelle(quelle_und_ziel):
    url, _ziel = quelle_und_ziel
    with pytest.raises(ki.KiFehler):
        ki.herunterladen(ki.MODELLE["schnell"], quelle=url + "gibt-es-nicht/")
