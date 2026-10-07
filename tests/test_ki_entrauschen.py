"""KI-Entrauschen: Sitzung (mit Platzhalter-Netz) und SCUNet selbst - nur mit Grafikkarte."""

from __future__ import annotations

import numpy as np
import pytest

from silberkorn import ki

cp = pytest.importorskip("cupy")


def grafikkarte():
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("keine Grafikkarte")


class Platzhalter:
    """Statt SCUNet: macht jedes Bild flach grau - leicht zu erkennen."""
    beschleuniger = "Test"

    def __init__(self):
        self.aufrufe = []

    def rechnen(self, srgb, kachel, fortschritt=None):
        self.aufrufe.append((srgb.shape, kachel))
        return cp.full_like(srgb, 0.5)


@pytest.fixture
def sitzung():
    grafikkarte()
    from silberkorn import bilddatei
    from silberkorn.bearbeitung import Sitzung
    zufall = np.random.default_rng(4)
    pixel = (zufall.random((120, 160, 3)) * 255).astype(np.uint8)
    daten = bilddatei.Bilddaten(pixel=pixel, profil=None, alpha=None, exif=b"", pfad="t.png")
    s = Sitzung(daten, vorschau_kante=80)
    yield s
    s.schliessen()


def test_ohne_berechnung_wirkt_der_regler_nicht(sitzung):
    vorher, _ms, _h = sitzung.vorschau()
    sitzung.werte.ki_rauschen = 100
    nachher, _ms, _h = sitzung.vorschau()
    assert not sitzung.ki_entrauscht and np.array_equal(vorher, nachher)


def test_staerke_mischt(sitzung):
    platzhalter = Platzhalter()
    sitzung.ki_entrauschen(platzhalter, 64)
    assert sitzung.ki_entrauscht and platzhalter.aufrufe == [((120, 160, 3), 64)]
    grau = round(0.5 * 255)
    sitzung.werte.ki_rauschen = 100
    voll, _ms, _h = sitzung.vorschau()
    assert np.abs(voll.astype(int) - grau).max() <= 1
    sitzung.werte.ki_rauschen = 0
    original, _ms, _h = sitzung.vorschau()
    sitzung.werte.ki_rauschen = 50
    halb, _ms, _h = sitzung.vorschau()
    # halb liegt zwischen Original und Grau (in linearem Licht gemischt)
    zwischen = (np.minimum(original, grau) - 1 <= halb) & (halb <= np.maximum(original, grau) + 1)
    assert zwischen.all() and not np.array_equal(halb, original)


def test_vollbild_und_vorschau_passen(sitzung):
    sitzung.ki_entrauschen(Platzhalter(), 64)
    sitzung.werte.ki_rauschen = 100
    teil, _ms, _h = sitzung.ausschnitt(0, 0, 160, 120)
    assert teil.shape == (120, 160, 3) and np.abs(teil.astype(int) - 128).max() <= 1


def test_lichter_ueber_eins_bleiben(sitzung):
    """Was heller als 1 ist, bekommt das Netz begrenzt - zurueck kommt nur die Korrektur."""
    sitzung.original[:10] = 3.0

    class Unveraendert(Platzhalter):
        def rechnen(self, srgb, kachel, fortschritt=None):
            return srgb.copy()

    sitzung.ki_entrauschen(Unveraendert(), 64)
    voll = sitzung._ki_rauschfrei[0]
    assert float(cp.abs(voll - sitzung.original).max()) < 1e-4


def test_berechnung_leert_die_zwischenspeicher(sitzung):
    sitzung.werte.rauschen_luminanz = 30
    sitzung.vorschau()
    assert sitzung._speicher
    sitzung.ki_entrauschen(Platzhalter(), 64)
    assert not sitzung._speicher


def test_export_nimmt_das_entrauschte_bild(sitzung, tmp_path):
    from PIL import Image
    sitzung.ki_entrauschen(Platzhalter(), 64)
    sitzung.werte.ki_rauschen = 100
    pfad = str(tmp_path / "aus.png")
    sitzung.exportieren(pfad)
    assert np.abs(np.asarray(Image.open(pfad)).astype(int) - 128).max() <= 1


# ----------------------------------------------------------------------
# SCUNet


@pytest.fixture(scope="module")
def scunet():
    grafikkarte()
    pytest.importorskip("onnxruntime")
    modell = ki.ENTRAUSCH_MODELLE["scunet"]
    if not ki.vorhanden(modell):
        pytest.skip(f"Modelldateien fehlen in {ki.modell_ordner()}")
    return ki.Entrauscher(modell)


@pytest.fixture
def verrauscht():
    zufall = np.random.default_rng(8)
    sauber = np.kron(zufall.random((5, 7, 3)), np.ones((40, 40, 1)))        # 200 x 280
    rauschen = zufall.normal(0, 0.08, sauber.shape)
    return sauber.astype(np.float32), np.clip(sauber + rauschen, 0, 1).astype(np.float32)


def test_scunet_entrauscht(scunet, verrauscht):
    sauber, laut = verrauscht
    glatt = cp.asnumpy(scunet.rechnen(cp.asarray(laut), 512))
    assert glatt.shape == laut.shape
    vorher = np.sqrt(((laut - sauber) ** 2).mean())
    nachher = np.sqrt(((glatt - sauber) ** 2).mean())
    assert nachher < vorher / 2


def test_scunet_kachelgrenzen_unsichtbar(scunet, verrauscht):
    """Groesse kein Vielfaches von 64, Kacheln kleiner als das Bild."""
    _sauber, laut = verrauscht
    ganz = cp.asnumpy(scunet.rechnen(cp.asarray(laut), 512))
    kacheln = cp.asnumpy(scunet.rechnen(cp.asarray(laut), 64))
    assert np.abs(ganz - kacheln).mean() < 0.003


def test_scunet_fp16_wie_fp32(verrauscht):
    grafikkarte()
    pytest.importorskip("onnxruntime")
    modell = ki.ENTRAUSCH_MODELLE["scunet"]
    if not ki.vorhanden(modell):
        pytest.skip("Modelldateien fehlen")
    _sauber, laut = verrauscht
    a = cp.asnumpy(ki.Entrauscher(modell).rechnen(cp.asarray(laut), 256))
    b = cp.asnumpy(ki.Entrauscher(modell, fp16=False).rechnen(cp.asarray(laut), 256))
    assert np.abs(a - b).mean() < 0.001


def test_scunet_abbruch(scunet, verrauscht):
    _sauber, laut = verrauscht
    with pytest.raises(ki.KiAbbruch):
        scunet.rechnen(cp.asarray(laut), 64, fortschritt=lambda i, n: i < 2)


def test_scunet_mit_tensorrt(verrauscht, monkeypatch, pytestconfig):
    """Nur wo TensorRT installiert ist; der erste Lauf baut die Engine."""
    grafikkarte()
    pytest.importorskip("onnxruntime")
    modell = ki.ENTRAUSCH_MODELLE["scunet"]
    if not ki.vorhanden(modell) or ki.tensorrt_ordner() is None:
        pytest.skip("Modell oder TensorRT fehlt")
    monkeypatch.setattr(ki, "tensorrt_cache", lambda: str(pytestconfig.cache.mkdir("tensorrt")))
    _sauber, laut = verrauscht
    trt = ki.Entrauscher(modell, kachel=128)
    assert trt.beschleuniger == "TensorRT", trt.tensorrt_fehler
    a = cp.asnumpy(trt.rechnen(cp.asarray(laut), 128))
    assert trt.beschleuniger == "TensorRT", trt.tensorrt_fehler
    b = cp.asnumpy(ki.Entrauscher(modell).rechnen(cp.asarray(laut), 128))
    assert np.abs(a - b).mean() < 0.002
