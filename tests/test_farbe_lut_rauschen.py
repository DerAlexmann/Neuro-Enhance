"""Tests fuer HSL je Farbbereich, LUTs und Entrauschen - Referenz und CUDA."""

from __future__ import annotations

import numpy as np
import pytest

from neuro_enhance import filter as f
from neuro_enhance import lut

N = len(f.FARBBEREICHE)


def gpu_vorhanden():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("keine Grafikkarte")
    return cp


def farbflaeche(rgb, groesse=(8, 8)):
    return np.full((*groesse, 3), rgb, dtype=np.float32)


def bereich(name, wert):
    werte = [0.0] * N
    werte[f.FARBBEREICHE.index(name)] = wert
    return tuple(werte)


@pytest.fixture
def bild():
    zufall = np.random.default_rng(11)
    rgb = zufall.random((64, 80, 3), dtype=np.float32) * 0.9
    rgb[:8] = np.linspace(0, 1, 80, dtype=np.float32)[None, :, None]   # Grauverlauf
    return rgb


# ----------------------------------------------------------------------
# HSL
# ----------------------------------------------------------------------

def test_oklab_hin_und_zurueck(bild):
    assert np.allclose(f.von_oklab(f.nach_oklab(bild)), bild, atol=1e-5)


def test_bereichsgewichte_ergeben_eins():
    winkel = np.linspace(0, 359.9, 2000, dtype=np.float32)
    gewichte = f.bereichsgewichte(winkel)
    assert np.allclose(gewichte.sum(axis=-1), 1, atol=1e-5)
    # an jeder Bereichsmitte wirkt genau dieser Bereich
    mitten = f.bereichsgewichte(np.array(f.FARBBEREICH_MITTE, dtype=np.float32))
    assert np.allclose(mitten, np.eye(N), atol=1e-5)


def test_grau_bleibt_grau():
    grau = np.linspace(0, 1, 50, dtype=np.float32)[:, None, None].repeat(3, axis=2)
    werte = f.Einstellungen(hsl_farbton=(100.0,) * N, hsl_saettigung=(100.0,) * N,
                            hsl_luminanz=(100.0,) * N)
    assert np.allclose(f.hsl(grau, werte), grau, atol=1e-4)


def test_rot_aendert_nur_rot():
    rot, blau = farbflaeche((0.6, 0.05, 0.04)), farbflaeche((0.03, 0.06, 0.6))
    werte = f.Einstellungen(hsl_saettigung=bereich("rot", -100))
    rot_neu, blau_neu = f.hsl(rot, werte), f.hsl(blau, werte)
    assert np.ptp(rot_neu[0, 0]) < 0.02                      # entsaettigt
    assert np.allclose(blau_neu, blau, atol=1e-4)            # blau unberuehrt


def test_farbton_dreht_rot_richtung_orange():
    rot = farbflaeche((0.6, 0.05, 0.04))
    neu = f.hsl(rot, f.Einstellungen(hsl_farbton=bereich("rot", 100)))
    assert neu[0, 0, 1] > rot[0, 0, 1] + 0.02                 # mehr Gruen = Richtung Orange


def test_luminanz_hellt_auf_ohne_farbton_zu_aendern():
    gruen = farbflaeche((0.05, 0.4, 0.05))
    neu = f.hsl(gruen, f.Einstellungen(hsl_luminanz=bereich("gruen", 100)))
    lab_alt, lab_neu = f.nach_oklab(gruen)[0, 0], f.nach_oklab(neu)[0, 0]
    assert lab_neu[0] > lab_alt[0] * 1.2
    winkel = np.degrees(np.arctan2([lab_alt[2], lab_neu[2]], [lab_alt[1], lab_neu[1]]))
    assert abs(winkel[0] - winkel[1]) < 0.5


# ----------------------------------------------------------------------
# LUT
# ----------------------------------------------------------------------

def cube_text(funktion, n=17, titel="Test"):
    zeilen = [f'TITLE "{titel}"', f"LUT_3D_SIZE {n}"]
    stellen = np.linspace(0, 1, n)
    for b in stellen:
        for g in stellen:
            for r in stellen:
                zeilen.append(" ".join(f"{w:.6f}" for w in funktion(r, g, b)))
    return "\n".join(zeilen)


def test_identitaet_laesst_das_bild_unveraendert(bild):
    tabelle = lut.lesen_text(cube_text(lambda r, g, b: (r, g, b)))
    v = f.linear_zu_srgb(np.clip(bild, 0, 1))
    assert np.allclose(lut.anwenden(v, tabelle), v, atol=1e-5)


def test_tetraedrisch_ist_exakt_fuer_lineare_luts(bild):
    """Eine lineare Abbildung gibt jede tetraedrische Interpolation exakt wieder."""
    def mischen(r, g, b):
        return (0.7 * r + 0.2 * g + 0.1 * b, 0.1 * r + 0.8 * g + 0.1 * b, 0.3 * b + 0.2)
    tabelle = lut.lesen_text(cube_text(mischen, n=9))
    v = f.linear_zu_srgb(np.clip(bild, 0, 1))
    erwartet = np.stack(mischen(v[..., 0], v[..., 1], v[..., 2]), axis=-1)
    assert np.allclose(lut.anwenden(v, tabelle), erwartet, atol=1e-5)


def test_1d_lut():
    text = "LUT_1D_SIZE 3\n0 0 0\n0.25 0.5 0.75\n1 1 1\n"
    tabelle = lut.lesen_text(text)
    v = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
    assert np.allclose(lut.anwenden(v, tabelle), [[[0.25, 0.5, 0.75]]], atol=1e-3)


@pytest.mark.parametrize("text", ["", "LUT_3D_SIZE 2\n0 0 0\n", "LUT_3D_SIZE 2\nkaputt\n",
                                  "LUT_3D_SIZE 2\nDOMAIN_MIN 1 1 1\nDOMAIN_MAX 0 0 0\n"])
def test_kaputte_luts(text):
    with pytest.raises(lut.LutFehler):
        lut.lesen_text(text)


def test_lut_staerke_mischt(tmp_path, bild):
    pfad = tmp_path / "invers.cube"
    pfad.write_text(cube_text(lambda r, g, b: (1 - r, 1 - g, 1 - b)))
    voll = f.lut_anwenden(bild, f.Einstellungen(lut=str(pfad)))
    halb = f.lut_anwenden(bild, f.Einstellungen(lut=str(pfad), lut_staerke=50))
    v = f.linear_zu_srgb(np.clip(bild, 0, 1))
    assert np.allclose(f.linear_zu_srgb(voll), 1 - v, atol=1e-3)
    assert np.allclose(f.linear_zu_srgb(halb), 0.5, atol=1e-3)


# ----------------------------------------------------------------------
# Entrauschen
# ----------------------------------------------------------------------

def verrauschte_szene():
    zufall = np.random.default_rng(12)
    sauber = np.zeros((64, 64, 3), dtype=np.float32)
    sauber[:, :32] = (0.10, 0.12, 0.30)
    sauber[:, 32:] = (0.40, 0.30, 0.10)
    hell = zufall.normal(0, 0.012, (64, 64, 1)).astype(np.float32)      # Luminanzrauschen
    farbig = zufall.normal(0, 0.012, (64, 64, 3)).astype(np.float32)    # Farbrauschen
    return sauber, np.clip(sauber + hell + farbig, 0, None)


def test_luminanzrauschen_sinkt_und_kante_bleibt():
    sauber, verrauscht = verrauschte_szene()
    neu = f.entrauschen(verrauscht, f.Einstellungen(rauschen_luminanz=80))
    flaeche = (slice(8, -8), slice(4, 26))
    y_alt, y_neu = f.luminanz(verrauscht), f.luminanz(neu)
    assert y_neu[flaeche].std() < 0.6 * y_alt[flaeche].std()
    # Kante: der Sprung zwischen den Flaechen bleibt erhalten
    sprung = f.luminanz(sauber)[:, 34].mean() - f.luminanz(sauber)[:, 29].mean()
    assert abs((y_neu[:, 34].mean() - y_neu[:, 29].mean()) - sprung) < 0.2 * abs(sprung)


def test_farbrauschen_sinkt():
    sauber, verrauscht = verrauschte_szene()
    neu = f.entrauschen(verrauscht, f.Einstellungen(rauschen_farbe=80))

    def farbstreuung(rgb):
        chroma = rgb - f.luminanz(rgb)[..., None]
        return chroma[8:-8, 4:26].std(axis=(0, 1)).mean()

    assert farbstreuung(neu) < 0.6 * farbstreuung(verrauscht)


def test_farbrauschen_erhaelt_kanten_gleicher_helligkeit():
    """Blau neben Gruen bei gleicher Helligkeit: die Farbkante darf nicht verlaufen.

    Ein nur von der Helligkeit gefuehrter Filter sieht hier keine Kante und
    verschmiert die Farbe ueber Dutzende Pixel.
    """
    zufall = np.random.default_rng(13)
    sauber = np.zeros((64, 96, 3), dtype=np.float32)
    blau, gruen = np.array([0.05, 0.08, 0.60]), np.array([0.05, 0.14, 0.03])
    blau = blau * f.luminanz(gruen) / f.luminanz(blau)        # gleiche Helligkeit
    sauber[:, :48], sauber[:, 48:] = blau, gruen
    verrauscht = np.clip(sauber + zufall.normal(0, 0.01, sauber.shape), 0, None)
    neu = f.entrauschen(verrauscht.astype(np.float32), f.Einstellungen(rauschen_farbe=100))
    # drei Pixel neben der Kante muss jede Seite ihre Farbe behalten
    for spalte, soll in ((44, blau), (51, gruen)):
        ist = neu[16:-16, spalte].mean(axis=0)
        assert np.abs(ist - soll).max() < 0.02, (spalte, ist, soll)


def test_box_mittel_gpu_wie_summentabelle():
    cp = gpu_vorhanden()
    zufall = np.random.default_rng(14).random((37, 53, 3), dtype=np.float32)
    for radius in (1, 4, 9):
        cpu = f.box_mittel(zufall, radius)
        gpu = cp.asnumpy(f.box_mittel(cp.asarray(zufall), radius))
        assert np.abs(cpu - gpu).max() < 1e-5


def test_inverse_3x3():
    zufall = np.random.default_rng(15)
    m = zufall.random((5, 3, 3))
    m = m @ np.swapaxes(m, -1, -2) + 0.1 * np.eye(3)          # symmetrisch, invertierbar
    assert np.allclose(f._inverse_3x3(m) @ m, np.eye(3), atol=1e-8)


def test_entrauschen_aus_aendert_nichts(bild):
    assert f.entrauschen(bild, f.Einstellungen()) is bild


# ----------------------------------------------------------------------
# GPU gegen Referenz
# ----------------------------------------------------------------------

@pytest.mark.parametrize("werte", [
    f.Einstellungen(hsl_farbton=(30, -20, 60, 0, -40, 80, 10, -100),
                    hsl_saettigung=(50, 0, -30, 100, 20, -60, 0, 40),
                    hsl_luminanz=(-50, 20, 0, 60, -30, 10, 100, -20)),
    f.Einstellungen(rauschen_luminanz=60),
    f.Einstellungen(rauschen_farbe=70),
    f.Einstellungen(rauschen_luminanz=40, rauschen_farbe=40, dunst=30, kontrast=20),
])
def test_cuda_wie_referenz(bild, werte):
    cp = gpu_vorhanden()
    referenz = f.anwenden_ausgabe(bild, werte, 0.5)
    gpu = cp.asnumpy(f.anwenden_ausgabe(cp.asarray(bild), werte, 0.5))
    abweichung = np.abs(referenz.astype(int) - gpu.astype(int))
    assert abweichung.max() <= 1


def test_cuda_lut_wie_referenz(tmp_path, bild):
    cp = gpu_vorhanden()
    pfad = tmp_path / "look.cube"
    pfad.write_text(cube_text(lambda r, g, b: (r ** 1.2, 0.9 * g + 0.1 * b, b ** 0.8 * 0.95),
                              n=17))
    werte = f.Einstellungen(lut=str(pfad), lut_staerke=80, saettigung=10)
    referenz = f.anwenden_ausgabe(bild, werte, 0.5)
    speicher = {}
    gpu = cp.asnumpy(f.anwenden_ausgabe(cp.asarray(bild), werte, 0.5, speicher))
    assert np.abs(referenz.astype(int) - gpu.astype(int)).max() <= 1


def test_zwischenspeicher_liefert_dasselbe(bild):
    cp = gpu_vorhanden()
    werte = f.Einstellungen(rauschen_luminanz=50, rauschen_farbe=30, klarheit=40)
    speicher = {}
    erstes = cp.asnumpy(f.anwenden_ausgabe(cp.asarray(bild), werte, 1.0, speicher))
    zweites = cp.asnumpy(f.anwenden_ausgabe(cp.asarray(bild), werte, 1.0, speicher))
    assert any(k[0] == "rauschen" for k in speicher)
    assert np.array_equal(erstes, zweites)                    # der Speicher bleibt unberuehrt
