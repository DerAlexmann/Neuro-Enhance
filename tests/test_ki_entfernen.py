"""Objekte entfernen: Sitzung mit Platzhalter-Netz und LaMa selbst - nur mit Grafikkarte."""

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


class Weiss:
    """Statt LaMa: fuellt alles Markierte weiss."""
    beschleuniger = "Test"

    def __init__(self):
        self.gesehen = None

    def fuellen(self, srgb, maske):
        self.gesehen = srgb.copy()
        zeilen = cp.flatnonzero((maske > 0.5).any(axis=1))
        spalten = cp.flatnonzero((maske > 0.5).any(axis=0))
        if zeilen.size == 0:
            return None
        y0, y1, x0, x1 = int(zeilen[0]), int(zeilen[-1]) + 1, int(spalten[0]), int(spalten[-1]) + 1
        return (y0, y1, x0, x1, cp.ones((y1 - y0, x1 - x0, 3), cp.float32),
                cp.ascontiguousarray(maske[y0:y1, x0:x1]))


class LinksVomKlick:
    def __init__(self):
        self.form = None

    def bild_setzen(self, srgb):
        self.form = srgb.shape[:2]

    def roh(self, klicks, vorige=None):
        x = klicks[-1][0]
        spalte = cp.arange(256, dtype=cp.float32)[None, :] * self.form[1] / 256
        return cp.broadcast_to(cp.where(spalte < x, 10.0, -10.0), (256, 256)).astype(cp.float32)

    def maske(self, logits):
        from neuro_enhance import filter
        return (filter.vergroessern(logits, *self.form) > 0).astype(cp.float32)


@pytest.fixture
def sitzung():
    grafikkarte()
    from neuro_enhance import bilddatei
    from neuro_enhance.bearbeitung import Sitzung
    pixel = np.full((120, 160, 3), 64, dtype=np.uint8)
    daten = bilddatei.Bilddaten(pixel=pixel, profil=None, alpha=None, exif=b"", pfad="t.png")
    s = Sitzung(daten, vorschau_kante=80)
    yield s
    s.schliessen()


def voll(sitzung):
    bild, _ms, _h = sitzung.ausschnitt(0, 0, *sitzung.ausgabe_form()[::-1])
    return bild.astype(int)


def test_pinsel_markiert_und_radiert(sitzung):
    assert not sitzung.markierung_da
    sitzung.pinseln([(40.0, 60.0)], 10.0, True)
    assert sitzung.markierung_da
    m = cp.asnumpy(sitzung._markierung(True))
    assert m[60, 40] == 1 and m[60, 55] == 0 and m[60, 49] > 0
    sitzung.pinseln([(40.0, 60.0)], 4.0, False)
    m = cp.asnumpy(sitzung._markierung(True))
    assert m[60, 40] == 0 and m[60, 47] == 1
    vorschau = cp.asnumpy(sitzung._markierung(False))
    assert vorschau.shape == sitzung.vorschau_original.shape[:2] and vorschau.max() == 1


def test_markierung_nur_in_der_anzeige(sitzung, tmp_path):
    from PIL import Image
    sitzung.pinseln([(40.0, 60.0)], 10.0, True)
    bild = voll(sitzung)
    assert bild[60, 40, 2] > bild[60, 40, 0] + 50             # blau eingefaerbt
    assert np.abs(bild[10, 150] - 64).max() <= 1
    pfad = str(tmp_path / "aus.png")
    sitzung.exportieren(pfad)
    assert np.abs(np.asarray(Image.open(pfad)).astype(int) - 64).max() <= 1


def test_entfernen_legt_einen_flicken_auf(sitzung, tmp_path):
    from PIL import Image
    sitzung.pinseln([(40.0, 60.0)], 10.0, True)
    netz = Weiss()
    assert sitzung.ki_entfernen(netz) is not None
    assert netz.gesehen.shape == (120, 160, 3)
    assert not sitzung.markierung_da and sitzung.entfernt == 1 and sitzung.geaendert
    bild = voll(sitzung)
    assert bild[60, 40].min() >= 254 and np.abs(bild[10, 150] - 64).max() <= 1
    vorschau, _ms, _h = sitzung.vorschau()
    assert vorschau[30, 20].min() > 200 and np.abs(vorschau[5, 75].astype(int) - 64).max() <= 1
    pfad = str(tmp_path / "aus.png")
    sitzung.exportieren(pfad)
    assert not sitzung.geaendert
    assert np.asarray(Image.open(pfad))[60, 40].min() >= 254
    assert sitzung.entfernen_zurueck() and sitzung.entfernt == 0 and sitzung.geaendert
    assert np.abs(voll(sitzung) - 64).max() <= 1
    assert not sitzung.entfernen_zurueck()


def test_ohne_markierung_passiert_nichts(sitzung):
    assert sitzung.ki_entfernen(Weiss()) is None and sitzung.entfernt == 0


def test_flicken_ueberstehen_den_geometriewechsel(sitzung):
    sitzung.pinseln([(20.0, 20.0)], 6.0, True)
    sitzung.ki_entfernen(Weiss())
    sitzung.werte.drehung90 = 2                    # das Original bleibt, der Flicken auch
    bild = voll(sitzung)
    assert bild[120 - 21, 160 - 21].min() >= 250


def test_klicks_markieren_ohne_die_maske_anzufassen(sitzung):
    netz = LinksVomKlick()
    sitzung.ki_auswahl_beginnen(netz, "entfernen")
    sitzung.ki_klick(netz, 40.0, 60.0, True, "entfernen")
    assert sitzung.markierung_da and not sitzung.ki_maske_da
    assert sitzung.klicks == [] and len(sitzung.klicks_von("entfernen")) == 1
    assert sitzung.ki_klick_zurueck(netz, "entfernen")
    assert not sitzung.markierung_da


# ----------------------------------------------------------------------
# LaMa


@pytest.fixture(scope="module")
def entferner():
    grafikkarte()
    pytest.importorskip("onnxruntime")
    modell = ki.ENTFERN_MODELLE["lama"]
    if not ki.vorhanden(modell):
        pytest.skip(f"Modelldateien fehlen in {ki.modell_ordner()}")
    return ki.Entferner(modell, 512)


def szene(hoehe, breite):
    """Ein dunkles Quadrat vor einem sanften Verlauf."""
    yy, xx = np.mgrid[0:hoehe, 0:breite].astype(np.float32)
    bild = np.stack([0.5 + 0.3 * xx / breite, 0.6 + 0.2 * yy / hoehe,
                     np.full_like(xx, 0.7)], -1)
    hinter = bild.copy()
    quadrat = (np.abs(yy - hoehe / 2) < hoehe / 10) & (np.abs(xx - breite / 2) < hoehe / 10)
    bild[quadrat] = (0.1, 0.1, 0.1)
    return bild.astype(np.float32), hinter.astype(np.float32), quadrat


@pytest.mark.parametrize("form", [(300, 420), (1500, 2100)])      # das zweite verkleinert
def test_lama_fuellt_mit_dem_hintergrund(entferner, form):
    bild, hinter, quadrat = szene(*form)
    ergebnis = entferner.fuellen(cp.asarray(bild), cp.asarray(quadrat.astype(np.float32)))
    y0, y1, x0, x1, fuellung, deckkraft = ergebnis
    assert fuellung.shape == (y1 - y0, x1 - x0, 3) and deckkraft.shape == (y1 - y0, x1 - x0)
    d = cp.asnumpy(deckkraft)
    assert d[quadrat[y0:y1, x0:x1]].min() == 1 and d.min() == 0     # ganz drin, Rand frei
    neu = bild.copy()
    d3 = d[..., None]
    neu[y0:y1, x0:x1] = bild[y0:y1, x0:x1] * (1 - d3) + cp.asnumpy(fuellung) * d3
    assert np.abs(neu[quadrat] - hinter[quadrat]).mean() < 0.04


def test_lama_ohne_markierung(entferner):
    assert entferner.fuellen(cp.zeros((64, 64, 3), cp.float32), cp.zeros((64, 64), cp.float32)) \
        is None
