"""Auswahl per Klick: Sitzung mit Platzhalter-Netz und SAM 2 selbst - nur mit Grafikkarte."""

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


class LinksVomKlick:
    """Statt SAM: alles links vom letzten Klick ist ausgewaehlt (oder rechts davon,
    wenn er wegnimmt)."""
    beschleuniger = "Test"

    def __init__(self):
        self.form = None
        self.vorige = []

    def bild_setzen(self, srgb):
        self.form = srgb.shape[:2]

    def roh(self, klicks, vorige=None):
        self.vorige.append(vorige)
        x, _y, dazu = klicks[-1]
        spalte = cp.arange(256, dtype=cp.float32)[None, :] * self.form[1] / 256
        logits = cp.where(spalte < x, 10.0, -10.0) * (1 if dazu else -1)
        return cp.broadcast_to(logits, (256, 256)).astype(cp.float32)

    def maske(self, logits):
        from silberkorn import filter
        return (filter.vergroessern(logits, *self.form) > 0).astype(cp.float32)


class LinksMotiv:
    beschleuniger = "Test"

    def maske(self, srgb):
        m = cp.zeros(srgb.shape[:2], dtype=cp.float32)
        m[:, : srgb.shape[1] // 2] = 1
        return m


@pytest.fixture
def sitzung():
    grafikkarte()
    from silberkorn import bilddatei
    from silberkorn.bearbeitung import Sitzung
    pixel = np.full((120, 160, 3), 128, dtype=np.uint8)
    daten = bilddatei.Bilddaten(pixel=pixel, profil=None, alpha=None, exif=b"", pfad="t.png")
    s = Sitzung(daten, vorschau_kante=80)
    yield s
    s.schliessen()


def voll(sitzung):
    bild, _ms, _h = sitzung.ausschnitt(0, 0, *sitzung.ausgabe_form()[::-1])
    return bild.astype(int)


def test_klick_waehlt_aus_und_regler_wirken_auf_den_rest(sitzung):
    netz = LinksVomKlick()
    sitzung.ki_auswahl_beginnen(netz)
    assert netz.form == (120, 160) and not sitzung.ki_maske_da
    sitzung.werte.hg_belichtung = -2
    sitzung.ki_klick(netz, 40.0, 60.0, True)
    assert sitzung.ki_maske_da and sitzung.maske_per_klick
    assert sitzung.klicks == [(40.0, 60.0, True)]
    bild = voll(sitzung)
    assert np.abs(bild[:, :35] - 128).max() <= 1 and bild[:, 45:].max() < 80
    # Ein zweiter Klick bekommt die Maske des ersten als Hinweis
    sitzung.ki_klick(netz, 100.0, 60.0, True)
    assert netz.vorige[0] is None and netz.vorige[1] is not None
    assert voll(sitzung)[:, :95].min() > 120


def test_klick_zurueck_bringt_die_alte_maske(sitzung):
    sitzung.ki_freistellen(LinksMotiv())
    netz = LinksVomKlick()
    sitzung.ki_auswahl_beginnen(netz)
    sitzung.werte.hg_belichtung = -2
    vorher = voll(sitzung)
    sitzung.ki_klick(netz, 20.0, 60.0, True)
    assert not np.array_equal(voll(sitzung), vorher)
    assert sitzung.ki_klick_zurueck(netz)
    assert not sitzung.maske_per_klick and np.array_equal(voll(sitzung), vorher)
    assert not sitzung.ki_klick_zurueck(netz)


def test_klicks_gehen_nach_neuem_beginn_weiter(sitzung):
    netz = LinksVomKlick()
    sitzung.ki_auswahl_beginnen(netz)
    sitzung.ki_klick(netz, 40.0, 60.0, True)
    sitzung.ki_auswahl_beginnen(LinksVomKlick())  # Klickmodus verlassen und wieder an
    assert sitzung.klicks == [(40.0, 60.0, True)]
    sitzung.ki_klick(netz, 60.0, 60.0, True)
    assert len(sitzung.klicks) == 2 and netz.vorige[-1] is not None
    assert sitzung.ki_klick_zurueck(netz) and sitzung.ki_klick_zurueck(netz)
    assert not sitzung.ki_maske_da                # wieder wie vor dem ersten Klick


def test_freistellen_ersetzt_die_klicks(sitzung):
    netz = LinksVomKlick()
    sitzung.ki_auswahl_beginnen(netz)
    sitzung.ki_klick(netz, 20.0, 60.0, True)
    sitzung.ki_freistellen(LinksMotiv())
    assert not sitzung.maske_per_klick and sitzung.klicks == []


def test_klick_im_gedrehten_bild_trifft_das_original(sitzung):
    sitzung.werte.drehung90 = 1                  # Ergebnis 120 breit, 160 hoch
    x, y = sitzung.quelle_von(30.0, 10.0)
    assert abs(x - 10.0) < 0.5 and abs(y - 90.0) < 0.5
    assert sitzung.quelle_von(-5.0, 10.0) is None


def test_mehrere_klicks_nehmen_die_eigene_maske_ausser_sie_ist_instabil():
    netz = object.__new__(ki.Auswaehler)
    masken = cp.full((4, 256, 256), -5.0, dtype=cp.float32)
    masken[0, :100] = 5.0
    guete = cp.asarray([0.5, 0.2, 0.9, 0.4], dtype=cp.float32)
    assert netz._waehlen(masken, guete, 1) == 2     # ein Klick: beste Deutung
    assert netz._waehlen(masken, guete, 2) == 0     # stabil: die eigene
    masken[0, 100:150] = 0.0                       # breiter unsicherer Saum
    assert netz._waehlen(masken, guete, 2) == 2


# ----------------------------------------------------------------------
# SAM 2


@pytest.fixture(scope="module")
def auswaehler():
    grafikkarte()
    pytest.importorskip("onnxruntime")
    modell = ki.AUSWAHL_MODELLE["sam2"]
    if not ki.vorhanden(modell):
        pytest.skip(f"Modelldateien fehlen in {ki.modell_ordner()}")
    return ki.Auswaehler(modell)


def szene(hoehe, breite):
    """Ein dunkler Kreis links und ein dunkles Quadrat rechts vor hellem Verlauf."""
    yy, xx = np.mgrid[0:hoehe, 0:breite].astype(np.float32)
    bild = np.stack([0.6 + 0.3 * xx / breite, 0.7 + 0.2 * yy / hoehe,
                     np.full_like(xx, 0.85)], -1)
    r = min(hoehe, breite) / 6
    kreis = (yy - hoehe / 2) ** 2 + (xx - breite / 4) ** 2 < r ** 2
    quadrat = (np.abs(yy - hoehe / 2) < r) & (np.abs(xx - 3 * breite / 4) < r)
    bild[kreis] = (0.15, 0.1, 0.1)
    bild[quadrat] = (0.1, 0.15, 0.3)
    return bild.astype(np.float32), kreis, quadrat


@pytest.mark.parametrize("form", [(480, 720), (900, 600)])
def test_sam_waehlt_das_angeklickte_objekt(auswaehler, form):
    hoehe, breite = form
    bild, kreis, quadrat = szene(hoehe, breite)
    auswaehler.bild_setzen(cp.asarray(bild))
    klick = (breite / 4, hoehe / 2, True)
    logits = auswaehler.roh([klick])
    maske = cp.asnumpy(auswaehler.maske(logits))
    assert maske.shape == form and 0 <= maske.min() and maske.max() <= 1
    assert maske[kreis].mean() > 0.9 and maske[quadrat].mean() < 0.1
    assert maske[~kreis & ~quadrat].mean() < 0.05
    # Zweiter Klick aufs Quadrat nimmt es dazu
    beide = cp.asnumpy(auswaehler.maske(auswaehler.roh(
        [klick, (3 * breite / 4, hoehe / 2, True)], logits)))
    assert beide[kreis].mean() > 0.9 and beide[quadrat].mean() > 0.9
