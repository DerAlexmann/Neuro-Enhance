"""Tests der RAW-Entwicklung auf der GPU (Demosaicing nach Malvar, He und Cutler).

Geprueft wird gegen drei Massstaebe: die bekannte Szene einer kuenstlichen
DNG, LibRaw selbst und - fuer den CUDA-Kernel - die NumPy-Referenz.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuro_enhance import bilddatei as b
from neuro_enhance import demosaik as d

SRGB_NACH_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]])
# Kamera mit eigenen Grundfarben und einem Farbstich der Lichtquelle
KAMERA = np.array([[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.25, 0.7]])
STICH = np.array([1.6, 1.0, 0.7])
# ((Zeilen, Spalten), lineares sRGB) - eine Liste, weil slice erst ab
# Python 3.12 als Schluessel eines Woerterbuchs taugt
FLAECHEN = [
    ((slice(0, 32), slice(0, 48)), (0.40, 0.10, 0.05)),
    ((slice(0, 32), slice(48, 96)), (0.05, 0.30, 0.08)),
    ((slice(32, 64), slice(0, 48)), (0.06, 0.08, 0.45)),
    ((slice(32, 64), slice(48, 96)), (0.18, 0.18, 0.18)),
]
# DNG-Ausrichtung -> Drehung in NumPy (k fuer np.rot90)
AUSRICHTUNG = {1: 0, 3: 2, 6: 3, 8: 1}


def gpu_vorhanden():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("keine Grafikkarte")
    return cp


def schiefe_dng(pfad: str, ausrichtung: int = 1, muster=(0, 1, 1, 2)):
    """DNG einer Kamera mit fremden Grundfarben, Farbstich und Schwarzwert 512."""
    tifffile = pytest.importorskip("tifffile")
    hoehe, breite = 64, 96
    szene = np.zeros((hoehe, breite, 3))
    for flaeche, farbe in FLAECHEN:
        szene[flaeche] = farbe
    kamera = (szene @ KAMERA.T) * STICH / STICH.max()
    kanal = np.array(muster).reshape(2, 2)[np.arange(hoehe)[:, None] % 2,
                                           np.arange(breite)[None, :] % 2]
    schwarz, weiss = 512, 16383
    werte = np.take_along_axis(kamera, kanal[..., None], axis=2)[..., 0]
    mosaik = np.round(schwarz + werte * (weiss - schwarz)).astype(np.uint16)

    def rational(werte):
        return [x for w in np.ravel(werte) for x in (int(round(w * 10000)), 10000)]

    tags = [
        (50706, "B", 4, (1, 4, 0, 0), True),
        (50708, "s", 0, "Testkamera", True),
        (33421, "H", 2, (2, 2), True),
        (33422, "B", 4, tuple(muster), True),
        (50714, "H", 1, schwarz, True),
        (50717, "H", 1, weiss, True),
        (50721, 10, 9, rational(KAMERA @ np.linalg.inv(SRGB_NACH_XYZ)), True),
        (50778, "H", 1, 21, True),
        (50728, 5, 3, rational(STICH / STICH.max()), True),
        (274, "H", 1, ausrichtung, True),
    ]
    tifffile.imwrite(pfad, mosaik, photometric=32803, extratags=tags, metadata=None)


def flaechen_pruefen(linear, ausrichtung, toleranz=0.01):
    """Mittelwert jeder Farbflaeche (ohne Kanten) gegen die Szene."""
    zurueck = np.rot90(linear, -AUSRICHTUNG[ausrichtung])   # zurueck in Aufnahmelage
    for (zeilen, spalten), farbe in FLAECHEN:
        mitte = zurueck[zeilen, spalten][6:-6, 6:-6]
        assert np.allclose(mitte.mean(axis=(0, 1)), farbe, atol=toleranz), farbe


@pytest.mark.parametrize("ausrichtung", [1, 3, 6, 8])
def test_szene_wird_farbrichtig_entwickelt(tmp_path, ausrichtung):
    pytest.importorskip("rawpy")
    pfad = str(tmp_path / "schief.dng")
    schiefe_dng(pfad, ausrichtung)
    daten = b.laden(pfad)
    assert daten.mosaik is not None and daten.pixel is None
    linear = daten.linear()                     # NumPy-Referenz
    assert linear.shape[:2] == daten.form
    flaechen_pruefen(linear, ausrichtung)


def test_mosaik_ist_eine_eigene_kopie(tmp_path):
    """raw_image_visible zeigt in LibRaws Speicher, der beim Schliessen frei wird."""
    pytest.importorskip("rawpy")
    pfad = str(tmp_path / "kopie.dng")
    schiefe_dng(pfad)
    assert b.laden(pfad).mosaik.daten.flags.owndata


@pytest.mark.parametrize("muster", [(0, 1, 1, 2), (2, 1, 1, 0), (1, 0, 2, 1), (1, 2, 0, 1)])
def test_alle_bayer_anordnungen(tmp_path, muster):
    pytest.importorskip("rawpy")
    pfad = str(tmp_path / "muster.dng")
    schiefe_dng(pfad, muster=muster)
    flaechen_pruefen(b.laden(pfad).linear(), 1)


def test_wie_libraw(tmp_path):
    """GPU-Weg und LibRaw muessen dieselben Farben liefern."""
    pytest.importorskip("rawpy")
    pfad = str(tmp_path / "vergleich.dng")
    schiefe_dng(pfad, 6)
    eigen = b.laden(pfad).linear()
    libraw = b.laden(pfad, raw_auf_gpu=False)
    assert libraw.mosaik is None
    fremd = libraw.linear()
    assert eigen.shape == fremd.shape
    innen = (slice(8, -8), slice(8, -8))
    assert np.abs(eigen[innen] - fremd[innen]).mean() < 0.005


def test_grauer_verlauf_bleibt_erhalten():
    """Bei einem linearen Grauverlauf ist Malvar-He-Cutler exakt."""
    hoehe, breite = 20, 30
    verlauf = (np.linspace(0.1, 0.9, breite)[None, :]
               + np.linspace(0, 0.05, hoehe)[:, None]).astype(np.float32)
    muster = np.array([[0, 1], [1, 2]])
    rgb = d.malvar(verlauf, muster)
    innen = rgb[2:-2, 2:-2]
    for kanal in range(3):
        assert np.allclose(innen[..., kanal], verlauf[2:-2, 2:-2], atol=1e-5)


def test_ausbrennende_lichter_werden_weiss():
    """Ein voll ausgesteuerter Sensor darf trotz Weissabgleich nicht magenta werden."""
    mosaik = d.RawMosaik(daten=np.full((8, 8), 16383, np.uint16),
                         muster=np.array([[0, 1], [1, 2]]), schwarz=np.full((2, 2), 512.0),
                         weiss=16383.0, weissabgleich=np.array([2.0, 1.0, 1.5]),
                         matrix=np.eye(3), drehung=0)
    assert np.allclose(d.entwickeln(mosaik.daten, mosaik), 1.0)


class _FalscheRawDatei:
    """Gerade genug von rawpy, um aus_rawpy() abzulehnen."""

    def __init__(self, muster, beschreibung=b"RGBG", farben=3):
        import rawpy
        self.raw_type = rawpy.RawType.Flat
        self.raw_pattern = np.asarray(muster)
        self.color_desc = beschreibung
        self.num_colors = farben


def test_x_trans_und_fremde_sensoren_bleiben_bei_libraw():
    pytest.importorskip("rawpy")
    assert d.aus_rawpy(_FalscheRawDatei(np.zeros((6, 6), int))) is None     # X-Trans
    assert d.aus_rawpy(_FalscheRawDatei([[0, 1], [3, 2]], b"CMYG", 4)) is None


@pytest.mark.parametrize("muster", [[[0, 1], [1, 2]], [[2, 1], [1, 0]], [[1, 0], [2, 1]],
                                    [[1, 2], [0, 1]]])
@pytest.mark.parametrize("drehung", [0, 3, 5, 6])
def test_cuda_kernel_wie_referenz(muster, drehung):
    cp = gpu_vorhanden()
    zufall = np.random.default_rng(8)
    mosaik = d.RawMosaik(
        daten=zufall.integers(300, 16383, (37, 52)).astype(np.uint16),
        muster=np.array(muster), schwarz=np.array([[512.0, 500.0], [505.0, 510.0]]),
        weiss=16383.0, weissabgleich=np.array([2.1, 1.0, 1.4]),
        matrix=np.array([[1.3, -0.2, -0.1], [-0.15, 1.3, -0.15], [0.0, -0.4, 1.4]]),
        drehung=drehung)
    referenz = d.entwickeln(mosaik.daten, mosaik)
    gpu = cp.asnumpy(d.entwickeln(cp.asarray(mosaik.daten), mosaik))
    assert gpu.shape == referenz.shape
    assert np.abs(gpu - referenz).max() < 1e-4
