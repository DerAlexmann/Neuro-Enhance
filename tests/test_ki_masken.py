"""Motiv & Hintergrund: Sitzung mit Platzhalter-Maske und BiRefNet selbst - nur mit Grafikkarte."""

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


class LinksMotiv:
    """Statt BiRefNet: die linke Haelfte ist Motiv, die rechte Hintergrund."""
    beschleuniger = "Test"

    def __init__(self):
        self.gesehen = None

    def maske(self, srgb):
        self.gesehen = srgb.copy()
        m = cp.zeros(srgb.shape[:2], dtype=cp.float32)
        m[:, : srgb.shape[1] // 2] = 1
        return m


@pytest.fixture
def sitzung():
    grafikkarte()
    from neuro_enhance import bilddatei
    from neuro_enhance.bearbeitung import Sitzung
    pixel = np.full((120, 160, 3), 128, dtype=np.uint8)
    daten = bilddatei.Bilddaten(pixel=pixel, profil=None, alpha=None, exif=b"", pfad="t.png")
    s = Sitzung(daten, vorschau_kante=80)
    yield s
    s.schliessen()


def voll(sitzung):
    bild, _ms, _h = sitzung.ausschnitt(0, 0, *sitzung.ausgabe_form()[::-1])
    return bild.astype(int)


def test_ohne_maske_wirkt_der_hintergrund_nicht(sitzung):
    vorher = voll(sitzung)
    sitzung.werte.hg_belichtung = -2
    assert not sitzung.ki_maske_da and np.array_equal(voll(sitzung), vorher)


def test_hintergrund_eigene_belichtung(sitzung):
    platzhalter = LinksMotiv()
    sitzung.ki_freistellen(platzhalter)
    assert sitzung.ki_maske_da and platzhalter.gesehen.shape == (120, 160, 3)
    sitzung.werte.hg_belichtung = -2
    bild = voll(sitzung)
    assert np.abs(bild[:, :70] - 128).max() <= 1          # Motiv unveraendert
    assert bild[:, 90:].max() < 80                        # Hintergrund dunkler
    vorschau, _ms, _h = sitzung.vorschau()
    assert vorschau[:, :30].mean() > 120 and vorschau[:, 50:].mean() < 80


def test_umkehren(sitzung):
    sitzung.ki_freistellen(LinksMotiv())
    sitzung.werte.hg_belichtung = -2
    sitzung.werte.maske_umkehren = True
    bild = voll(sitzung)
    assert bild[:, :70].max() < 80 and np.abs(bild[:, 90:] - 128).max() <= 1


def test_werte_des_ganzen_bildes_gelten_auch_im_hintergrund(sitzung):
    sitzung.ki_freistellen(LinksMotiv())
    sitzung.werte.belichtung = 1
    ohne = voll(sitzung)
    sitzung.werte.hg_saettigung = 1                    # Grau bleibt grau, aber aktiv
    assert np.abs(voll(sitzung) - ohne).max() <= 1


def test_kante_weich_und_verschoben(sitzung):
    sitzung.ki_freistellen(LinksMotiv())
    sitzung.werte.hg_belichtung = -2
    hart = voll(sitzung)[60]
    sitzung.werte.maske_kante = 50
    weich = voll(sitzung)[60]
    uebergang = lambda zeile: int(((zeile > 40) & (zeile < 120)).any(axis=-1).sum())  # noqa: E731
    assert uebergang(weich) > uebergang(hart) + 4
    sitzung.werte.maske_kante = 0
    sitzung.werte.maske_verschieben = 50                # Motiv waechst um 10 px
    gewachsen = voll(sitzung)[60]
    assert gewachsen[85].mean() > 120 and hart[85].mean() < 80


def test_hintergrund_unschaerfe_laesst_das_motiv(sitzung):
    sitzung.original[:, 100:110] = 1.0                  # helle Linie im Hintergrund
    sitzung.ki_freistellen(LinksMotiv())
    scharf = voll(sitzung)
    sitzung.werte.hg_unschaerfe = 30
    weich = voll(sitzung)
    assert np.array_equal(weich[:, :70], scharf[:, :70])
    assert weich[:, 100:110].max() < scharf[:, 100:110].max()


def test_maskenansicht_nur_in_der_anzeige(sitzung, tmp_path):
    from PIL import Image
    sitzung.ki_freistellen(LinksMotiv())
    sitzung.maske_zeigen = True
    bild = voll(sitzung)
    assert bild[:, 90:, 0].mean() > bild[:, 90:, 2].mean() + 50   # rot eingefaerbt
    pfad = str(tmp_path / "aus.png")
    sitzung.exportieren(pfad)
    assert np.abs(np.asarray(Image.open(pfad)).astype(int) - 128).max() <= 1


def test_freistellen_speichert_durchsichtig(sitzung, tmp_path):
    from PIL import Image
    sitzung.ki_freistellen(LinksMotiv())
    sitzung.werte.freistellen = True
    pfad = str(tmp_path / "frei.png")
    sitzung.exportieren(pfad)
    gespeichert = np.asarray(Image.open(pfad))
    assert gespeichert.shape == (120, 160, 4)
    assert gespeichert[:, :70, 3].min() == 255 and gespeichert[:, 90:, 3].max() == 0


def test_freistellen_folgt_der_geometrie(sitzung, tmp_path):
    from PIL import Image
    sitzung.ki_freistellen(LinksMotiv())
    sitzung.werte.freistellen = True
    sitzung.werte.drehung90 = 1
    pfad = str(tmp_path / "gedreht.png")
    sitzung.exportieren(pfad)
    alpha = np.asarray(Image.open(pfad))[..., 3]
    assert alpha.shape == (160, 120)
    # die linke Haelfte liegt nach einer Vierteldrehung oben oder unten
    oben, unten = alpha[:70].mean(), alpha[90:].mean()
    assert {round(oben), round(unten)} == {0, 255}


# ----------------------------------------------------------------------
# BiRefNet


@pytest.fixture
def freisteller():
    """Je Test neu: Eine Sitzung belegt mehrere GB, zwei passen nicht zugleich auf 8 GB."""
    grafikkarte()
    pytest.importorskip("onnxruntime")
    modell = ki.MASKEN_MODELLE["birefnet"]
    if not ki.vorhanden(modell):
        pytest.skip(f"Modelldateien fehlen in {ki.modell_ordner()}")
    return ki.Freisteller(modell, tensorrt=False)


def szene(hoehe, breite):
    """Ein dunkler Kreis vor hellem Verlauf - ein deutliches Motiv."""
    yy, xx = np.mgrid[0:hoehe, 0:breite].astype(np.float32)
    bild = np.stack([0.6 + 0.3 * xx / breite, 0.7 + 0.2 * yy / hoehe,
                     np.full_like(xx, 0.85)], -1)
    kreis = (yy - hoehe / 2) ** 2 + (xx - breite / 2) ** 2 < (min(hoehe, breite) / 4) ** 2
    bild[kreis] = (0.15, 0.1, 0.1)
    return bild.astype(np.float32), kreis


@pytest.mark.parametrize("form", [(360, 640), (640, 360)])
def test_birefnet_findet_das_motiv(freisteller, form):
    bild, kreis = szene(*form)
    maske = cp.asnumpy(freisteller.maske(cp.asarray(bild)))
    assert maske.shape == form and 0 <= maske.min() and maske.max() <= 1
    assert maske[kreis].mean() > 0.8 and maske[~kreis].mean() < 0.2


def test_birefnet_mit_tensorrt(monkeypatch, pytestconfig):
    """Nur wo TensorRT installiert ist; der erste Lauf baut die Engine."""
    grafikkarte()
    pytest.importorskip("onnxruntime")
    modell = ki.MASKEN_MODELLE["birefnet"]
    if not ki.vorhanden(modell) or ki.tensorrt_ordner() is None:
        pytest.skip("Modell oder TensorRT fehlt")
    monkeypatch.setattr(ki, "tensorrt_cache", lambda: str(pytestconfig.cache.mkdir("tensorrt")))
    bild, _kreis = szene(360, 640)
    trt = ki.Freisteller(modell)
    assert trt.beschleuniger == "TensorRT", trt.tensorrt_fehler
    a = cp.asnumpy(trt.maske(cp.asarray(bild)))
    assert trt.beschleuniger == "TensorRT", trt.tensorrt_fehler
    del trt                                  # beide Netze zugleich passen nicht auf 8 GB
    b = cp.asnumpy(ki.Freisteller(modell, tensorrt=False).maske(cp.asarray(bild)))
    assert np.abs(a - b).mean() < 0.01
