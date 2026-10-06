"""KI-Schaerfen: Sitzung (mit Platzhalter-Netzen) und Restormer selbst - nur mit Grafikkarte."""

from __future__ import annotations

import numpy as np
import pytest

from neuro_enhance import ki

cp = pytest.importorskip("cupy")


def grafikkarte():
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("keine Grafikkarte")


class Grau:
    """Statt SCUNet: macht jedes Bild flach grau."""
    beschleuniger = "Test"

    def rechnen(self, srgb, kachel, fortschritt=None):
        return cp.full_like(srgb, 0.5)


class Muster:
    """Statt Restormer: legt ein feines Schachbrett (+-0,05) auf und merkt sich, was es
    gesehen hat - feine Strukturen wie beim Schaerfen."""
    beschleuniger = "Test"

    def __init__(self):
        self.gesehen = None

    def rechnen(self, srgb, kachel, fortschritt=None):
        self.gesehen = srgb.copy()
        yy, xx = cp.indices(srgb.shape[:2])
        schach = ((yy + xx) % 2 * 2 - 1).astype(cp.float32)[..., None]
        return cp.clip(srgb + 0.05 * schach, 0, 1)


class Heller(Muster):
    """Hebt nur die Helligkeit an - das soll wegfallen."""

    def rechnen(self, srgb, kachel, fortschritt=None):
        self.gesehen = srgb.copy()
        return cp.clip(srgb + 0.1, 0, 1)


@pytest.fixture
def sitzung():
    grafikkarte()
    from neuro_enhance import bilddatei
    from neuro_enhance.bearbeitung import Sitzung
    zufall = np.random.default_rng(5)
    pixel = (zufall.random((120, 160, 3)) * 200).astype(np.uint8)
    daten = bilddatei.Bilddaten(pixel=pixel, profil=None, alpha=None, exif=b"", pfad="t.png")
    s = Sitzung(daten, vorschau_kante=80)
    yield s
    s.schliessen()


def test_ohne_berechnung_wirkt_der_regler_nicht(sitzung):
    vorher, _ms, _h = sitzung.vorschau()
    sitzung.werte.ki_schaerfe = 100
    nachher, _ms, _h = sitzung.vorschau()
    assert not sitzung.ki_geschaerft and np.array_equal(vorher, nachher)


def voll(sitzung):
    bild, _ms, _h = sitzung.ausschnitt(0, 0, 160, 120)
    return bild.astype(int)


def test_staerke_skaliert_die_korrektur(sitzung):
    sitzung.ki_schaerfen(Muster(), 64)
    assert sitzung.ki_geschaerft
    original = voll(sitzung)
    sitzung.werte.ki_schaerfe = 100
    ganz = np.abs(voll(sitzung) - original).mean()
    sitzung.werte.ki_schaerfe = 50
    halb = np.abs(voll(sitzung) - original).mean()
    assert ganz > 5 and 0.4 < halb / ganz < 0.6


def test_aufhellen_faellt_weg(sitzung):
    """Nur der feine Anteil der Korrektur zaehlt - Helligkeit und Farbe bleiben."""
    verlauf = cp.linspace(0.05, 0.6, 160, dtype=cp.float32)
    sitzung.original[:] = verlauf[None, :, None]
    sitzung.ki_schaerfen(Heller(), 64)
    original = voll(sitzung)
    sitzung.werte.ki_schaerfe = 100
    assert np.abs(voll(sitzung) - original).mean() < 1


def test_geschaerft_wird_das_entrauschte_bild(sitzung):
    sitzung.ki_entrauschen(Grau(), 64)
    sitzung.werte.ki_rauschen = 100
    muster = Muster()
    sitzung.ki_schaerfen(muster, 64)
    assert float(cp.abs(muster.gesehen - 0.5).max()) < 1e-3
    # beides zusammen: Grau mit feinem Schachbrett
    sitzung.werte.ki_schaerfe = 100
    bild = voll(sitzung)
    assert abs(bild.mean() - 128) < 2 and bild.std() > 8


def test_ohne_entrauschen_wird_das_original_geschaerft(sitzung):
    from neuro_enhance import filter
    muster = Muster()
    sitzung.ki_schaerfen(muster, 64)
    soll = filter.linear_zu_srgb(cp.clip(sitzung.original, 0, 1))
    assert float(cp.abs(muster.gesehen - soll).max()) < 1e-5


def test_lichter_ueber_eins_bleiben(sitzung):
    sitzung.original[:10] = 3.0

    class Unveraendert(Muster):
        def rechnen(self, srgb, kachel, fortschritt=None):
            return srgb.copy()

    sitzung.ki_schaerfen(Unveraendert(), 64)
    assert float(cp.abs(sitzung._ki_schaerfe[0]).max()) < 1e-4


def test_export_nimmt_das_geschaerfte_bild(sitzung, tmp_path):
    from PIL import Image
    sitzung.ki_entrauschen(Grau(), 64)
    sitzung.werte.ki_rauschen = 100
    sitzung.ki_schaerfen(Muster(), 64)
    sitzung.werte.ki_schaerfe = 100
    pfad = str(tmp_path / "aus.png")
    sitzung.exportieren(pfad)
    gespeichert = np.asarray(Image.open(pfad)).astype(int)
    assert np.array_equal(gespeichert, voll(sitzung))
    assert gespeichert.std() > 8


# ----------------------------------------------------------------------
# Restormer


@pytest.fixture(scope="module")
def restormer():
    grafikkarte()
    pytest.importorskip("onnxruntime")
    modell = ki.SCHAERF_MODELLE["restormer"]
    if not ki.vorhanden(modell):
        pytest.skip(f"Modelldateien fehlen in {ki.modell_ordner()}")
    return ki.Schaerfer(modell, tensorrt=False)


def kanten(bild):
    return float(np.abs(np.diff(bild, axis=0)).mean() + np.abs(np.diff(bild, axis=1)).mean())


@pytest.fixture
def unscharf():
    """Kanten und feine Streifen, mit einer Kreisscheibe (Radius 2) weichgezeichnet."""
    zufall = np.random.default_rng(9)
    scharf = np.kron(zufall.random((12, 17, 3)), np.ones((16, 16, 1)))      # 192 x 272
    scharf[:, ::6] *= 0.6
    yy, xx = np.mgrid[-2:3, -2:3]
    kern = ((yy ** 2 + xx ** 2) <= 4).astype(np.float64)
    kern /= kern.sum()
    rand = np.pad(scharf, ((2, 2), (2, 2), (0, 0)), mode="edge")
    weich = sum(kern[i, j] * rand[i:i + 192, j:j + 272] for i in range(5) for j in range(5))
    return scharf.astype(np.float32), weich.astype(np.float32)


def test_restormer_schaerft(restormer, unscharf):
    scharf, weich = unscharf
    neu = cp.asnumpy(restormer.rechnen(cp.asarray(weich), 256))
    assert neu.shape == weich.shape
    assert kanten(neu) > kanten(weich) * 1.1
    # keine Fehlmuster wie bei NAFNet: das Bild bleibt das Bild
    assert np.isfinite(neu).all() and np.abs(neu - weich).mean() < 0.1
    assert np.corrcoef(neu.ravel(), scharf.ravel())[0, 1] > 0.9


def test_restormer_kachelgrenzen_unsichtbar(restormer, unscharf):
    _scharf, weich = unscharf
    ganz = cp.asnumpy(restormer.rechnen(cp.asarray(weich), 512))
    kacheln = cp.asnumpy(restormer.rechnen(cp.asarray(weich), 96))
    assert np.abs(ganz - kacheln).mean() < 0.01


def test_restormer_fp16_wie_fp32(unscharf):
    grafikkarte()
    pytest.importorskip("onnxruntime")
    modell = ki.SCHAERF_MODELLE["restormer"]
    if not ki.vorhanden(modell):
        pytest.skip("Modelldateien fehlen")
    _scharf, weich = unscharf
    a = cp.asnumpy(ki.Schaerfer(modell, tensorrt=False).rechnen(cp.asarray(weich), 256))
    b = cp.asnumpy(ki.Schaerfer(modell, fp16=False, tensorrt=False).rechnen(cp.asarray(weich),
                                                                            256))
    assert np.abs(a - b).mean() < 0.002


def test_restormer_mit_tensorrt(unscharf, monkeypatch, pytestconfig):
    """Nur wo TensorRT installiert ist; der erste Lauf baut die Engine."""
    grafikkarte()
    pytest.importorskip("onnxruntime")
    modell = ki.SCHAERF_MODELLE["restormer"]
    if not ki.vorhanden(modell) or ki.tensorrt_ordner() is None:
        pytest.skip("Modell oder TensorRT fehlt")
    monkeypatch.setattr(ki, "tensorrt_cache", lambda: str(pytestconfig.cache.mkdir("tensorrt")))
    _scharf, weich = unscharf
    trt = ki.Schaerfer(modell, kachel=128)
    assert trt.beschleuniger == "TensorRT", trt.tensorrt_fehler
    a = cp.asnumpy(trt.rechnen(cp.asarray(weich), 128))
    assert trt.beschleuniger == "TensorRT", trt.tensorrt_fehler
    b = cp.asnumpy(ki.Schaerfer(modell, tensorrt=False).rechnen(cp.asarray(weich), 128))
    assert np.abs(a - b).mean() < 0.003
