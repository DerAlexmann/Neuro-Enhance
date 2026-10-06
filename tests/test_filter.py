"""Tests der klassischen Filter - mit NumPy, also ohne Grafikkarte.

Ein Test vergleicht zusaetzlich CuPy mit NumPy; er laeuft nur, wenn CuPy und
eine Grafikkarte vorhanden sind.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from neuro_enhance import filter as f
from neuro_enhance import kurven


@pytest.fixture
def bild():
    """Kleines lineares Testbild mit Verlauf, Farbflaechen, Schwarz und Weiss."""
    zufall = np.random.default_rng(1)
    rgb = zufall.random((96, 128, 3), dtype=np.float32) * 0.8
    rgb[:8] = 0.0                             # schwarz
    rgb[-8:] = 1.0                            # weiss
    rgb[8:16, :, :] = np.linspace(0, 1, 128, dtype=np.float32)[None, :, None]
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
    assert namen == (set(f.REGLER_NACH_NAME) | set(f.KURVEN) | set(f.HSL) | {"lut"}
                     | set(f.GEOMETRIE_FELDER) | set(f.MASKEN_SCHALTER))
    for feld in dataclasses.fields(f.Einstellungen):
        if feld.name in f.KURVEN:
            assert feld.default == kurven.IDENTITAET
        elif feld.name in f.HSL:
            assert feld.default == f.HSL_NEUTRAL
        elif feld.name == "lut":
            assert feld.default == ""
        elif feld.name in f.GEOMETRIE_FELDER:
            assert feld.default == f.GEOMETRIE_FELDER[feld.name]
        elif feld.name in f.MASKEN_SCHALTER:
            assert feld.default == f.MASKEN_SCHALTER[feld.name]
        else:
            assert feld.default == f.REGLER_NACH_NAME[feld.name].vorgabe


def test_geaenderte_kurve_ist_nicht_neutral():
    werte = f.Einstellungen(kurve_rot=((0.0, 0.0), (0.5, 0.6), (1.0, 1.0)))
    assert not werte.ist_neutral()


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


# ----------------------------------------------------------------------
# Werkzeuge fuer grosse Nachbarschaften
# ----------------------------------------------------------------------

def test_verkleinern_box_mittelt():
    bild = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert np.allclose(f.verkleinern_box(bild, 2), [[2.5, 4.5], [10.5, 12.5]])


def test_vergroessern_form_und_flaechen():
    flaeche = np.full((5, 7, 3), 0.3, dtype=np.float32)
    gross = f.vergroessern(flaeche, 40, 31)
    assert gross.shape == (40, 31, 3) and np.allclose(gross, 0.3)


def test_vergroessern_bewahrt_einen_verlauf():
    verlauf = np.linspace(0, 1, 10, dtype=np.float32)[None, :].repeat(4, axis=0)
    gross = f.vergroessern(verlauf, 8, 40)
    assert (np.diff(gross[0]) >= -1e-6).all()


def test_box_mittel_wie_von_hand():
    zufall = np.random.default_rng(3).random((9, 11), dtype=np.float32)
    ergebnis = f.box_mittel(zufall, 2)
    for y, x in ((0, 0), (4, 5), (8, 10), (2, 9)):
        ausschnitt = zufall[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        assert np.isclose(ergebnis[y, x], ausschnitt.mean(), atol=1e-6)


def test_min_filter_wie_von_hand():
    zufall = np.random.default_rng(4).random((7, 8), dtype=np.float32)
    ergebnis = f.min_filter(zufall, 1)
    assert ergebnis[3, 3] == zufall[2:5, 2:5].min()
    assert ergebnis[0, 0] == zufall[0:2, 0:2].min()


def test_gefuehrter_filter_glaettet_konstante_fuehrung_zum_mittel():
    fuehrung = np.full((20, 20), 0.5, dtype=np.float32)
    eingabe = np.random.default_rng(5).random((20, 20), dtype=np.float32)
    ergebnis = f.gefuehrter_filter(fuehrung, eingabe, 3, 1e-3)
    assert ergebnis.std() < eingabe.std()


# ----------------------------------------------------------------------
# Dunst, Klarheit, Kurven
# ----------------------------------------------------------------------

HIMMEL = 24                                    # Zeilen ganz im Dunst


def dunstige_szene():
    """Landschaft unter gleichmaessigem Dunst, oben ein Streifen Himmel.

    Das Verfahren liest die Lichtfarbe des Dunstes an den dunstigsten Stellen
    ab - in echten Dunstfotos ist das fast immer der Himmel.
    """
    zufall = np.random.default_rng(6)
    szene = zufall.random((96, 128, 3), dtype=np.float32) * 0.5
    szene[..., 2] *= 0.2                       # jedes Fleckchen hat einen dunklen Kanal
    licht = np.array([0.8, 0.82, 0.85], dtype=np.float32)
    dunstig = szene * 0.5 + licht * 0.5
    dunstig[:HIMMEL] = licht
    return szene, dunstig


def test_dunst_entfernen_hebt_den_kontrast():
    szene, dunstig = dunstige_szene()
    klar = f.dunst(dunstig, 100)
    land, land_klar, land_szene = dunstig[HIMMEL:], klar[HIMMEL:], szene[HIMMEL:]
    assert land_klar.std() > land.std() * 1.4
    # und kommt der dunstfreien Szene deutlich naeher als das dunstige Bild
    assert np.abs(land_klar - land_szene).mean() < np.abs(land - land_szene).mean() * 0.5


def test_dunst_hinzufuegen_senkt_den_kontrast(bild):
    assert f.dunst(bild, -80).std() < bild.std()


def test_dunst_null_aendert_nichts(bild):
    assert f.dunst(bild, 0) is bild


def test_klarheit_hebt_lokalen_kontrast():
    """Weiches Muster in Mitteltoenen: Klarheit verstaerkt, negative Klarheit glaettet."""
    y, x = np.mgrid[0:120, 0:160].astype(np.float32)
    muster = 0.18 + 0.06 * np.sin(x / 6) * np.sin(y / 6)
    bild = np.repeat(muster[..., None], 3, axis=2)
    assert f.klarheit(bild, 100).std() > bild.std()
    assert f.klarheit(bild, -100).std() < bild.std()


def test_helligkeitskurve_hebt_mitten_farbtreu(bild):
    werte = f.Einstellungen(kurve_hell=((0.0, 0.0), (0.5, 0.7), (1.0, 1.0)))
    neu = f.gradation(bild, werte)
    assert f.luminanz(neu)[16:-8].mean() > f.luminanz(bild)[16:-8].mean()
    farbig, neu_farbig = bild[16:-8].reshape(-1, 3), neu[16:-8].reshape(-1, 3)
    assert np.allclose(farbig / farbig.sum(1, keepdims=True),
                       neu_farbig / neu_farbig.sum(1, keepdims=True), atol=1e-4)


def test_kanalkurve_wirkt_nur_auf_ihren_kanal(bild):
    werte = f.Einstellungen(kurve_rot=((0.0, 0.0), (0.5, 0.3), (1.0, 1.0)))
    neu = f.gradation(bild, werte)
    assert neu[..., 0].mean() < bild[..., 0].mean()
    assert np.array_equal(neu[..., 1:], bild[..., 1:])


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
    f.Einstellungen(dunst=70),
    f.Einstellungen(dunst=-50, kontrast=20),
    f.Einstellungen(dunst=40, kontrast=30, klarheit=60),
    f.Einstellungen(klarheit=-80),
    f.Einstellungen(kurve_hell=((0.0, 0.1), (0.4, 0.55), (1.0, 0.95))),
    f.Einstellungen(kurve_rot=((0.0, 0.0), (0.5, 0.6), (1.0, 1.0)),
                    kurve_blau=((0.0, 0.05), (0.6, 0.5), (1.0, 1.0)), saettigung=20),
    f.Einstellungen(temperatur=100, toenung=-100, belichtung=5, kontrast=100, lichter=-100,
                    tiefen=100, dynamik=100, saettigung=100, schaerfe=150, schaerfe_radius=3,
                    klarheit=100, dunst=100,
                    kurve_hell=((0.0, 0.0), (0.3, 0.6), (1.0, 1.0))),
])
def test_cuda_kernel_rechnen_wie_die_referenz(bild, werte):
    """Die zusammengefassten Kernel muessen dieselben Formeln rechnen wie filter.py."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("keine Grafikkarte")
    referenz = f.anwenden_ausgabe(bild, werte, 0.5)
    gpu = cp.asnumpy(f.anwenden_ausgabe(cp.asarray(bild), werte, 0.5))
    abweichung = np.abs(referenz.astype(int) - gpu.astype(int))
    assert abweichung.max() <= 1              # Rundung von float32 auf der GPU


# ----------------------------------------------------------------------
# 16 Bit und Vorschau
# ----------------------------------------------------------------------

def test_16bit_ausgabe_feiner_als_8bit():
    verlauf = np.linspace(0, 1, 4096, dtype=np.float32)[None, :, None].repeat(3, axis=2)
    acht = f.anwenden_ausgabe(verlauf, f.Einstellungen(), bits=8)
    sechzehn = f.anwenden_ausgabe(verlauf, f.Einstellungen(), bits=16)
    assert acht.dtype == np.uint8 and sechzehn.dtype == np.uint16
    assert len(np.unique(sechzehn[0, :, 0])) > 10 * len(np.unique(acht[0, :, 0]))
    # beide sagen dasselbe, nur feiner
    assert np.abs(sechzehn.astype(int) / 257 - acht.astype(int)).max() <= 1


def test_verkleinern_auf():
    bild = np.random.default_rng(7).random((1000, 1500, 3), dtype=np.float32)
    klein, massstab = f.verkleinern_auf(bild, 400)
    assert max(klein.shape[:2]) <= 400 and massstab == pytest.approx(1 / 4)
    assert klein.dtype == np.float32
    gleich, massstab = f.verkleinern_auf(bild, 2000)
    assert gleich is bild and massstab == 1.0


def test_cuda_16bit_wie_die_referenz(bild):
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("keine Grafikkarte")
    werte = f.Einstellungen(kontrast=30, klarheit=40, dunst=30, schaerfe=60,
                            kurve_hell=((0.0, 0.0), (0.4, 0.5), (1.0, 1.0)))
    referenz = f.anwenden_ausgabe(bild, werte, 0.5, bits=16)
    gpu = cp.asnumpy(f.anwenden_ausgabe(cp.asarray(bild), werte, 0.5, bits=16))
    assert gpu.dtype == np.uint16
    # 1/255 Abstand in 8 Bit entspricht 257 Stufen in 16 Bit
    assert np.abs(referenz.astype(int) - gpu.astype(int)).max() <= 257
