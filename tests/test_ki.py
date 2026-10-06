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


@pytest.mark.parametrize("schluessel", ["schnell", "qualitaet"])
def test_fp16_wie_fp32(bild, schluessel):
    """Halbe Genauigkeit darf sich in 16 Bit kaum vom FP32-Ergebnis unterscheiden."""
    cp = bereit(schluessel)
    halb = ki.Hochskalierer(ki.MODELLE[schluessel], 0.5)
    voll = ki.Hochskalierer(ki.MODELLE[schluessel], 0.5, fp16=False)
    a = halb.hochskalieren(cp.asarray(bild), 4, 64, bits=16).astype(np.float64) / 65535
    b = voll.hochskalieren(cp.asarray(bild), 4, 64, bits=16).astype(np.float64) / 65535
    assert np.abs(a - b).mean() < 0.001 and np.abs(a - b).max() < 0.02


def test_abbruch(schnell, bild):
    cp = bereit("schnell")
    with pytest.raises(ki.KiAbbruch):
        schnell.hochskalieren(cp.asarray(bild), 4, 32, fortschritt=lambda i, n: i < 2)


# ----------------------------------------------------------------------
# TensorRT - optional


def test_ohne_tensorrt(monkeypatch):
    monkeypatch.setattr(ki.importlib.util, "find_spec", lambda name: None)
    assert ki.tensorrt_ordner() is None and ki.tensorrt_fassung() is None


def test_tensorrt_ordner_ohne_bibliotheken(tmp_path, monkeypatch):
    """Ein Paket ohne nvinfer zaehlt nicht als installiert."""
    spec = ki.importlib.util.spec_from_file_location(
        "tensorrt_libs", tmp_path / "__init__.py", submodule_search_locations=[str(tmp_path)])
    monkeypatch.setattr(ki.importlib.util, "find_spec", lambda name: spec)
    assert ki.tensorrt_ordner() is None
    (tmp_path / "nvinfer_10.dll").write_bytes(b"")
    assert ki.tensorrt_ordner() == str(tmp_path)


def test_ohne_kachel_kein_tensorrt(bild):
    bereit("schnell")
    assert ki.Hochskalierer(ki.MODELLE["schnell"], 0.5).beschleuniger == "CUDA"


def test_tensorrt_scheitert_beim_laden(bild, monkeypatch):
    cp = bereit("schnell")

    def kaputt(self):
        raise ki.KiFehler("kaputt")

    monkeypatch.setattr(ki.Hochskalierer, "_tensorrt_sitzung", kaputt)
    h = ki.Hochskalierer(ki.MODELLE["schnell"], 0.5, kachel=64, tensorrt=True)
    assert h.beschleuniger == "CUDA" and h.tensorrt_fehler == "kaputt" and not h.erster_lauf
    assert h.hochskalieren(cp.asarray(bild), 2, 64).shape == (192, 256, 3)


def test_tensorrt_scheitert_beim_rechnen(bild):
    """Bricht TensorRT mitten im Lauf ab, rechnet CUDA das Bild fertig."""
    cp = bereit("schnell")
    h = ki.Hochskalierer(ki.MODELLE["schnell"], 0.5, kachel=64, tensorrt=False)
    erwartet = h.hochskalieren(cp.asarray(bild), 2, 64)
    echt = h.sitzung

    class Scheitert:
        io_binding = echt.io_binding

        def run_with_iobinding(self, bindung):
            raise RuntimeError("TensorRT EP execution context enqueue failed")

    h.sitzung, h.tensorrt, h.kachel = Scheitert(), True, 64
    aus = h.hochskalieren(cp.asarray(bild), 2, 64)
    assert h.beschleuniger == "CUDA" and "enqueue" in h.tensorrt_fehler
    assert np.array_equal(aus, erwartet)


@pytest.mark.parametrize("schluessel", ["schnell", "qualitaet"])
def test_tensorrt_wie_cuda(bild, schluessel, monkeypatch, pytestconfig):
    """Nur wo TensorRT installiert ist. Der erste Lauf baut die Engines (Minuten)."""
    cp = bereit(schluessel)
    if ki.tensorrt_ordner() is None:
        pytest.skip("TensorRT nicht installiert")
    monkeypatch.setattr(ki, "tensorrt_cache", lambda: str(pytestconfig.cache.mkdir("tensorrt")))
    trt = ki.Hochskalierer(ki.MODELLE[schluessel], 0.5, kachel=64)
    assert trt.beschleuniger == "TensorRT", trt.tensorrt_fehler
    assert not ki.tensorrt_baut(ki.MODELLE[schluessel], 64)
    cuda = ki.Hochskalierer(ki.MODELLE[schluessel], 0.5)
    a = trt.hochskalieren(cp.asarray(bild), 4, 64, bits=16).astype(np.float64) / 65535
    assert trt.beschleuniger == "TensorRT", trt.tensorrt_fehler
    b = cuda.hochskalieren(cp.asarray(bild), 4, 64, bits=16).astype(np.float64) / 65535
    assert np.abs(a - b).mean() < 0.002 and np.abs(a - b).max() < 0.05


def test_tensorrt_baut_in_eigenem_prozess(tmp_path, monkeypatch):
    """Der Bau laeuft in einem eigenen Prozess; danach liegt die Engine bereit."""
    bereit("schnell")
    if ki.tensorrt_ordner() is None:
        pytest.skip("TensorRT nicht installiert")
    monkeypatch.setattr(ki, "tensorrt_cache", lambda: str(tmp_path))
    modell = ki.MODELLE["schnell"]
    assert ki.tensorrt_baut(modell, 48)
    ki.tensorrt_vorbereiten(modell, 48).result(timeout=600)
    assert not ki.tensorrt_baut(modell, 48)
    h = ki.Hochskalierer(modell, 0.5, kachel=48)
    assert h.beschleuniger == "TensorRT" and not h.erster_lauf


def test_modellordner_liegt_neben_dem_programm():
    assert os.path.basename(ki.modell_ordner()) == "modelle"


def test_katalog_vollstaendig():
    for modell in ki.ALLE_MODELLE.values():
        for name in ki.dateien(modell):
            assert name in ki.DATEIEN
        assert modell.sha256 == ki.DATEIEN[modell.datei][0]
        assert modell.lizenzdatei in ki.dateien(modell)
        assert modell.release.startswith("https://github.com/DerAlexmann/Neuro-Enhance/")
        assert modell.mindeststufe in ki.STUFEN and set(modell.kacheln) <= set(ki.STUFEN)
    assert set(ki.ALLE_MODELLE) == (set(ki.MODELLE) | set(ki.ENTRAUSCH_MODELLE)
                                    | set(ki.SCHAERF_MODELLE))


def test_entrauschmodell_aus_eigenem_release():
    """modelle-1 bleibt unveraendert - neue Modelle kommen in ein neues Release."""
    scunet = ki.ENTRAUSCH_MODELLE["scunet"]
    assert scunet.release == ki.MODELL_RELEASE_2 != ki.MODELL_RELEASE
    assert all(m.release == ki.MODELL_RELEASE for m in ki.MODELLE.values())
    assert ki.SCHAERF_MODELLE["restormer"].release == ki.MODELL_RELEASE_3


def test_feste_kachel_rueckt_randkacheln_ins_bild():
    """Restormer bekommt nur Kacheln einer Groesse zu sehen - auch am Bildrand."""
    import dataclasses
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("keine Grafikkarte")
    modell = dataclasses.replace(ki.SCHAERF_MODELLE["restormer"], rand=4, vielfaches=8)
    netz = object.__new__(ki._Netz)
    netz.modell = modell
    formen = []

    def kachel(eingabe):
        formen.append(eingabe.shape[2:])
        return eingabe

    netz._kachel = kachel
    bild = cp.random.rand(1, 3, 50, 70).astype(cp.float32)
    ziel = cp.zeros_like(bild[0])

    def ablegen(y0, y1, x0, x1, kern):
        ziel[:, y0:y1, x0:x1] = kern

    netz._kacheln(bild, 16, None, ablegen)
    assert set(formen) == {(24, 24)}
    assert cp.array_equal(ziel, bild[0])


def test_eingabe_groesse_ist_vielfaches():
    modell = ki.ENTRAUSCH_MODELLE["scunet"]
    for kachel in modell.kacheln.values():
        assert ki.eingabe_groesse(modell, kachel) % 64 == 0
    assert ki.eingabe_groesse(modell, 100) == 192
    assert ki.eingabe_groesse(ki.MODELLE["schnell"], 384) == 404


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


def test_herunterladen_nimmt_das_release_des_modells(tmp_path, monkeypatch):
    """Ohne ausdrueckliche Quelle laedt jedes Modell aus seinem eigenen Release."""
    angefragt = []

    def urlopen(anfrage, timeout=None):
        angefragt.append(anfrage.full_url)
        raise ki.urllib.error.URLError("kein Netz im Test")

    monkeypatch.setattr(ki.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(ki, "ordner_kandidaten", lambda: [str(tmp_path)])
    with pytest.raises(ki.KiFehler):
        ki.herunterladen(ki.ENTRAUSCH_MODELLE["scunet"])
    assert angefragt[0] == ki.MODELL_RELEASE_2 + "scunet-color-real-psnr.onnx"
