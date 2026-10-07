"""Tiefe & Bokeh: Sitzung mit Platzhalter-Tiefe und Depth Anything selbst - nur mit Grafikkarte."""

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


class LinksNah:
    """Statt Depth Anything: die linke Haelfte ist nah (1), die rechte fern (0)."""

    def tiefe(self, srgb):
        t = cp.zeros((60, 80), dtype=cp.float32)
        t[:, :40] = 1
        return t


class RechtsMotiv:
    def maske(self, srgb):
        m = cp.zeros(srgb.shape[:2], dtype=cp.float32)
        m[:, srgb.shape[1] // 2:] = 1
        return m


@pytest.fixture
def sitzung():
    grafikkarte()
    from neuro_enhance import bilddatei
    from neuro_enhance.bearbeitung import Sitzung
    pixel = np.zeros((120, 160, 3), dtype=np.uint8)
    pixel[(np.indices((120, 160)).sum(axis=0) % 2) == 0] = 200     # feines Schachbrett
    daten = bilddatei.Bilddaten(pixel=pixel, profil=None, alpha=None, exif=b"", pfad="t.png")
    s = Sitzung(daten, vorschau_kante=80)
    yield s
    s.schliessen()


def voll(sitzung):
    bild, _ms, _h = sitzung.ausschnitt(0, 0, *sitzung.ausgabe_form()[::-1])
    return bild.astype(int)


def struktur(teil):
    """Wie viel Schachbrett noch zu sehen ist: Streuung der Helligkeit."""
    return float(teil.mean(axis=-1).std())


def test_ohne_bokeh_bleibt_das_bild(sitzung):
    vorher = voll(sitzung)
    sitzung.ki_tiefe(LinksNah())
    assert sitzung.ki_tiefe_da and np.array_equal(voll(sitzung), vorher)


def test_bokeh_verwischt_was_ausserhalb_des_fokus_liegt(sitzung):
    scharf = voll(sitzung)
    sitzung.ki_tiefe(LinksNah())
    sitzung.werte.bokeh = 100
    sitzung.werte.fokus = 100                      # auf das Nahe scharf gestellt
    bild = voll(sitzung)
    assert np.array_equal(bild[:, :70], scharf[:, :70])
    assert struktur(bild[:, 100:]) < 0.2 * struktur(scharf[:, 100:])
    sitzung.werte.fokus = 0                        # jetzt auf das Ferne
    bild = voll(sitzung)
    assert np.array_equal(bild[:, 90:], scharf[:, 90:])
    assert struktur(bild[:, :60]) < 0.2 * struktur(scharf[:, :60])
    vorschau, _ms, _h = sitzung.vorschau()
    assert vorschau.shape[:2] == sitzung.vorschau_original.shape[:2]


def test_scharfes_laeuft_nicht_ins_unscharfe(sitzung):
    sitzung.original[:, :80] = 1.0                 # links hell und nah, rechts dunkel und fern
    sitzung.original[:, 80:] = 0.02
    sitzung.vorschau_original[:, :40] = 1.0
    sitzung.vorschau_original[:, 40:] = 0.02
    sitzung.ki_tiefe(LinksNah())
    sitzung.werte.bokeh = 100
    sitzung.werte.fokus = 100
    bild = voll(sitzung)
    assert bild[:, 84:].max() < 60                 # kein heller Schein hinter der Kante


def test_motiv_bleibt_scharf(sitzung):
    scharf = voll(sitzung)
    sitzung.ki_freistellen(RechtsMotiv())
    sitzung.ki_tiefe(LinksNah())
    sitzung.werte.bokeh = 100
    sitzung.werte.fokus = 100
    assert np.array_equal(voll(sitzung)[:, 90:], scharf[:, 90:])
    sitzung.werte.bokeh_motiv = False
    assert struktur(voll(sitzung)[:, 100:]) < 0.2 * struktur(scharf[:, 100:])


def test_fokus_per_klick_und_vorschlag(sitzung):
    sitzung.ki_tiefe(LinksNah())
    assert sitzung.tiefe_an(10.0, 60.0) == 100 and sitzung.tiefe_an(150.0, 60.0) == 0
    assert sitzung.fokus_vorschlag() == 100        # ohne Maske: das Naechste
    sitzung.ki_freistellen(RechtsMotiv())
    assert sitzung.fokus_vorschlag() == 0          # mit Maske: das Motiv


def test_tiefenkarte_nur_in_der_anzeige(sitzung, tmp_path):
    from PIL import Image
    sitzung.ki_tiefe(LinksNah())
    sitzung.tiefe_zeigen = True
    bild = voll(sitzung)
    assert bild[:, :70].min() >= 254 and bild[:, 90:].max() <= 1
    pfad = str(tmp_path / "aus.png")
    sitzung.exportieren(pfad)
    assert struktur(np.asarray(Image.open(pfad)).astype(int)) > 50


# ----------------------------------------------------------------------
# Depth Anything V2


@pytest.fixture(scope="module")
def schaetzer():
    grafikkarte()
    pytest.importorskip("onnxruntime")
    modell = ki.TIEFEN_MODELLE["tiefe"]
    if not ki.vorhanden(modell):
        pytest.skip(f"Modelldateien fehlen in {ki.modell_ordner()}")
    return ki.Tiefenschaetzer(modell)


def boden(hoehe, breite):
    """Ein Schachbrettboden, der nach oben in die Ferne laeuft, darueber Himmel."""
    yy, xx = np.mgrid[0:hoehe, 0:breite].astype(np.float32)
    horizont = hoehe * 0.35
    unten = np.maximum(yy - horizont, 1)
    z = hoehe / unten                                   # Entfernung auf dem Boden
    u = (xx - breite / 2) * z / breite
    feld = ((np.floor(u * 4) + np.floor(z * 2)) % 2).astype(np.float32)
    bild = np.where((yy > horizont)[..., None], (0.3 + 0.5 * feld)[..., None] * [1, 0.9, 0.8],
                    [0.6, 0.75, 0.95])
    return bild.astype(np.float32)


@pytest.mark.parametrize("form", [(600, 900), (900, 600)])
def test_depth_anything_sieht_den_boden_nach_hinten_laufen(schaetzer, form):
    tiefe = cp.asnumpy(schaetzer.tiefe(cp.asarray(boden(*form))))
    h, w = tiefe.shape
    assert abs(h / w - form[0] / form[1]) < 0.02 and max(h, w) <= ki.Tiefenschaetzer.ARBEIT
    assert 0 <= tiefe.min() and tiefe.max() <= 1
    assert tiefe[int(h * 0.9)].mean() > tiefe[int(h * 0.5)].mean() + 0.2    # unten ist nah
