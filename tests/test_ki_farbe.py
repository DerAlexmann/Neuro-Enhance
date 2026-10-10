"""KI-Kolorieren: Farbe einsetzen, Netzaufruf, Sitzung mit Platzhalter-Farbe und DDColor
selbst - nur mit Grafikkarte."""

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


# Lineares sRGB <-> Lab (D65), unabhaengig vom CUDA-Kern
M = np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]])
WEISS = np.array([0.950456, 1.0, 1.088754])


def lab(linear):
    xyz = linear @ M.T / WEISS
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def test_farbe_einsetzen_setzt_a_und_b_und_haelt_die_helligkeit():
    grafikkarte()
    zufall = np.random.default_rng(4)
    bild = zufall.uniform(0.05, 0.8, (20, 30, 3)).astype(np.float32)
    ab = zufall.uniform(-20, 20, (20, 30, 2)).astype(np.float32)
    farbig = cp.asnumpy(ki.farbe_einsetzen(cp.asarray(bild), cp.asarray(ab), 1.0))
    soll, ist = lab(bild.astype(np.float64)), lab(farbig.astype(np.float64))
    assert np.abs(ist[..., 0] - soll[..., 0]).max() < 0.05          # L bleibt
    assert np.abs(ist[..., 1:] - ab).max() < 0.1                    # a, b wie geschaetzt


def test_staerke_mischt_und_null_aendert_nichts():
    grafikkarte()
    bild = cp.full((8, 8, 3), 0.3, dtype=cp.float32)
    bild[..., 0] = 0.4                                               # leichter Sepiaton
    ab = cp.zeros((8, 8, 2), dtype=cp.float32)
    ab[..., 0] = 30
    null = ki.farbe_einsetzen(bild, ab, 0.0)
    assert cp.allclose(null, bild)
    voll = ki.farbe_einsetzen(bild, ab, 1.0)
    halb = ki.farbe_einsetzen(bild, ab, 0.5)
    assert cp.allclose(halb, (bild + voll) / 2, atol=1e-6)


def test_ohne_farbanteile_wird_sepia_grau():
    grafikkarte()
    sepia = cp.asarray(np.array([[[0.45, 0.33, 0.20]]], np.float32))
    grau = cp.asnumpy(ki.farbe_einsetzen(sepia, cp.zeros((1, 1, 2), cp.float32), 1.0))[0, 0]
    assert np.ptp(grau) < 1e-3
    helligkeit = M[1] @ np.array([0.45, 0.33, 0.20])
    assert grau[0] == pytest.approx(helligkeit, abs=1e-3)


class FalscheSitzung:
    """Statt ONNX Runtime: merkt sich die Eingabe und liefert a = 10, b = -5."""

    def __init__(self):
        self.eingabe = None


def test_netz_sieht_ein_graubild_in_fester_groesse(monkeypatch):
    grafikkarte()
    kolorierer = object.__new__(ki.Kolorierer)
    kolorierer.modell = ki.FARB_MODELLE["ddcolor"]
    kolorierer.sitzung = FalscheSitzung()

    def binden(sitzung, eingaben, ausgaben, schrumpfen=False):
        sitzung.eingabe = cp.asnumpy(eingaben["eingabe"])
        ausgaben["ausgabe"][0, 0] = 10
        ausgaben["ausgabe"][0, 1] = -5

    monkeypatch.setattr(ki, "_binden", binden)
    bild = cp.zeros((700, 1100, 3), dtype=cp.float32)
    bild[..., 0] = 0.5                                               # roetlich
    ab = kolorierer.farben(bild)
    eingabe = kolorierer.sitzung.eingabe
    assert eingabe.shape == (1, 3, 512, 512)
    assert np.allclose(eingabe[0, 0], eingabe[0, 1]) and np.allclose(eingabe[0, 1], eingabe[0, 2])
    assert ab.shape == (512, 512, 2) and float(ab[..., 0].mean()) == 10


class Gruen:
    """Statt DDColor: alles gruenlich (a = -30)."""

    def farben(self, linear):
        ab = cp.zeros((512, 512, 2), dtype=cp.float32)
        ab[..., 0] = -30
        return ab


@pytest.fixture
def sitzung():
    grafikkarte()
    from silberkorn import bilddatei
    from silberkorn.bearbeitung import Sitzung
    pixel = np.full((120, 160, 3), 140, dtype=np.uint8)
    daten = bilddatei.Bilddaten(pixel=pixel, profil=None, alpha=None, exif=b"", pfad="t.png")
    s = Sitzung(daten, vorschau_kante=80)
    yield s
    s.schliessen()


def test_sitzung_mischt_die_farbe_ein(sitzung):
    grau, _ms, _h = sitzung.vorschau()
    assert sitzung.ki_kolorieren(Gruen()) >= 0 and sitzung.ki_koloriert
    ohne, _ms, _h = sitzung.vorschau()
    assert np.array_equal(ohne, grau) and not sitzung.ki_inhalt     # Regler noch auf 0
    sitzung.werte.ki_farbe = 100
    farbig, _ms, _h = sitzung.vorschau()
    assert farbig[..., 1].mean() > farbig[..., 0].mean() + 10
    assert sitzung.ki_inhalt                     # wird beim Veroeffentlichen vermerkt
    voll, _ms, _h = sitzung.ausschnitt(0, 0, 160, 120)
    assert voll[..., 1].mean() > voll[..., 0].mean() + 10


# ----------------------------------------------------------------------
# DDColor


@pytest.fixture(scope="module")
def kolorierer():
    grafikkarte()
    pytest.importorskip("onnxruntime")
    modell = ki.FARB_MODELLE["ddcolor"]
    if not ki.vorhanden(modell):
        pytest.skip(f"Modelldateien fehlen in {ki.modell_ordner()}")
    return ki.Kolorierer(modell)


def test_ddcolor_liefert_farbanteile(kolorierer):
    """Was DDColor fuer ein Graubild erfindet, ist nicht vorherzusagen - wohl aber, dass
    es endliche Farbanteile im Lab-Bereich sind, nicht ueberall gleich, und dass
    dasselbe Bild dieselbe Farbe bekommt."""
    zufall = np.random.default_rng(1)
    bild = np.empty((400, 600, 3), np.float32)
    bild[:200] = 0.55
    bild[200:] = (0.12 + 0.08 * zufall.random((200, 600)))[..., None]
    ab = cp.asnumpy(kolorierer.farben(cp.asarray(bild)))
    assert ab.shape == (512, 512, 2) and np.isfinite(ab).all()
    assert np.abs(ab).max() < 128 and ab.std() > 0.5
    assert np.allclose(cp.asnumpy(kolorierer.farben(cp.asarray(bild))), ab, atol=0.05)
