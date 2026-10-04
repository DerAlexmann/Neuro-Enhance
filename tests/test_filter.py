"""Tests der klassischen Filter - mit NumPy, also ohne Grafikkarte.

Ein Test vergleicht zusaetzlich CuPy mit NumPy; er laeuft nur, wenn CuPy und
eine Grafikkarte vorhanden sind.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from neuro_enhance import filter as f


@pytest.fixture
def bild():
    """Kleines lineares Testbild mit Verlauf, Farbflaechen, Schwarz und Weiss."""
    zufall = np.random.default_rng(1)
    rgb = zufall.random((48, 64, 3), dtype=np.float32) * 0.8
    rgb[:8] = 0.0                             # schwarz
    rgb[-8:] = 1.0                            # weiss
    rgb[8:16, :, :] = np.linspace(0, 1, 64, dtype=np.float32)[None, :, None]
    return rgb


def test_srgb_hin_und_zurueck():
    werte = np.linspace(0, 1, 256, dtype=np.float32)
    assert np.allclose(f.linear_zu_srgb(f.srgb_zu_linear(werte)), werte, atol=1e-5)


def test_8bit_hin_und_zurueck_verlustfrei():
    alle = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
    assert np.array_equal(f.nach_8bit(f.von_8bit(alle)), alle)


def test_neutrale_einstellungen_aendern_nichts(bild):
    assert np.array_equal(f.anwenden(bild, f.Einstellungen()), bild)
    assert f.Einstellungen().ist_neutral()


def test_regler_und_einstellungen_passen_zusammen():
    namen = {feld.name for feld in dataclasses.fields(f.Einstellungen)}
    assert namen == set(f.REGLER_NACH_NAME)
    for feld in dataclasses.fields(f.Einstellungen):
        assert feld.default == f.REGLER_NACH_NAME[feld.name].vorgabe


def test_belichtung_verdoppelt_licht(bild):
    assert np.allclose(f.belichtung(bild, 1.0), bild * 2)
    assert np.allclose(f.belichtung(bild, -1.0), bild / 2)


def test_weissabgleich_erhaelt_helligkeit_von_grau():
    """Normiert ist auf Neutralgrau; farbige Pixel duerfen sich leicht aendern."""
    grau = np.linspace(0, 1, 30, dtype=np.float32)[:, None, None].repeat(3, axis=2)
    for temperatur, toenung in ((100, 0), (-100, 0), (0, 100), (60, -40)):
        neu = f.weissabgleich(grau, temperatur, toenung)
        assert np.allclose(f.luminanz(neu), f.luminanz(grau), atol=1e-5)


def test_waermer_heisst_mehr_rot_weniger_blau():
    grau = np.full((2, 2, 3), 0.18, dtype=np.float32)
    warm = f.weissabgleich(grau, 50, 0)
    assert warm[0, 0, 0] > 0.18 > warm[0, 0, 2]


def test_saettigung_null_ergibt_grau(bild):
    grau = f.farbe(bild, 0, -100)
    assert np.allclose(grau[..., 0], grau[..., 1], atol=1e-6)
    assert np.allclose(grau[..., 1], grau[..., 2], atol=1e-6)


def test_saettigung_erhaelt_helligkeit(bild):
    assert np.allclose(f.luminanz(f.farbe(bild, 0, 50)), f.luminanz(bild), atol=1e-5)


def test_dynamik_wirkt_auf_blasse_farben_staerker():
    blass = np.array([[[0.20, 0.18, 0.17]]], dtype=np.float32)
    kraeftig = np.array([[[0.40, 0.05, 0.02]]], dtype=np.float32)

    def zuwachs(px):
        y = f.luminanz(px)[..., None]
        vorher = np.abs(px - y).sum()
        return np.abs(f.farbe(px, 100, 0) - y).sum() / vorher

    assert zuwachs(blass) > zuwachs(kraeftig)


def test_kontrast_spreizt_um_die_mitte():
    grau = np.array([[[0.05] * 3, [0.6] * 3]], dtype=np.float32)
    neu = f.tonwerte(grau, 60, 0, 0)
    assert neu[0, 0, 0] < 0.05                # dunkles dunkler
    assert neu[0, 1, 0] > 0.6                 # helles heller


def test_tonwerte_erhalten_farbton(bild):
    neu = f.tonwerte(bild, 40, -50, 50)
    farbig = bild[16:-8].reshape(-1, 3)
    neu_farbig = neu[16:-8].reshape(-1, 3)
    # Gleiches Kanalverhaeltnis = gleicher Farbton und gleiche Saettigung
    verhaeltnis_alt = farbig / farbig.sum(axis=1, keepdims=True)
    verhaeltnis_neu = neu_farbig / neu_farbig.sum(axis=1, keepdims=True)
    assert np.allclose(verhaeltnis_alt, verhaeltnis_neu, atol=1e-4)


def test_tiefen_hellen_schwarz_auf_und_lichter_daempfen_weiss(bild):
    assert f.tonwerte(bild, 0, 0, 100)[:8].mean() > 0
    assert f.tonwerte(bild, 0, -100, 0)[-8:].mean() < 1


def test_keine_ungueltigen_werte(bild):
    extrem = f.Einstellungen(temperatur=100, toenung=-100, belichtung=5, kontrast=100,
                             lichter=-100, tiefen=100, dynamik=100, saettigung=100,
                             schaerfe=150, schaerfe_radius=3)
    neu = f.anwenden(bild, extrem)
    assert np.isfinite(neu).all()
    assert f.nach_8bit(neu).dtype == np.uint8


def test_gauss_erhaelt_mittelwert_und_flaechen():
    flaeche = np.full((20, 30), 0.4, dtype=np.float32)
    assert np.allclose(f.gauss(flaeche, 2.0), 0.4, atol=1e-6)
    assert np.isclose(f.gauss_kern(1.5).sum(), 1.0)


def test_schaerfen_verstaerkt_kanten():
    kante = np.zeros((10, 20, 3), dtype=np.float32)
    kante[:, 10:] = 0.5
    neu = f.schaerfen(kante, 100, 1.0)
    assert neu[5, 11, 0] > 0.5                # Ueberschwinger auf der hellen Seite
    assert np.allclose(neu[5, 0], 0) and np.allclose(neu[5, 19], 0.5, atol=1e-4)


def test_gpu_rechnet_wie_cpu(bild):
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("keine Grafikkarte")
    werte = f.Einstellungen(temperatur=30, toenung=-10, belichtung=0.7, kontrast=40,
                            lichter=-30, tiefen=40, dynamik=30, saettigung=-10, schaerfe=80)
    cpu = f.anwenden(bild, werte, 0.5)
    gpu = cp.asnumpy(f.anwenden(cp.asarray(bild), werte, 0.5))
    assert np.allclose(cpu, gpu, atol=1e-4)


@pytest.mark.parametrize("werte", [
    f.Einstellungen(),
    f.Einstellungen(belichtung=0.7, temperatur=30, toenung=-10),
    f.Einstellungen(kontrast=40, lichter=-30, tiefen=40),
    f.Einstellungen(kontrast=-60, dynamik=30, saettigung=-10),
    f.Einstellungen(schaerfe=80, schaerfe_radius=2.0),
    f.Einstellungen(temperatur=100, toenung=-100, belichtung=5, kontrast=100, lichter=-100,
                    tiefen=100, dynamik=100, saettigung=100, schaerfe=150, schaerfe_radius=3),
])
def test_cuda_kernel_rechnen_wie_die_referenz(bild, werte):
    """Die zusammengefassten Kernel muessen dieselben Formeln rechnen wie filter.py."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("keine Grafikkarte")
    referenz = f.anwenden_8bit(bild, werte, 0.5)
    gpu = cp.asnumpy(f.anwenden_8bit(cp.asarray(bild), werte, 0.5))
    abweichung = np.abs(referenz.astype(int) - gpu.astype(int))
    assert abweichung.max() <= 1              # Rundung von float32 auf der GPU
