"""Anonymisieren: Flaechen, Gesichter, Metadaten, Rueckweg der Geometrie."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from silberkorn import anonym, geometrie


def streifen(hoehe=60, breite=80):
    """Ein Bild, in dem jeder Pixel anders ist."""
    y, x = np.mgrid[0:hoehe, 0:breite].astype(np.float32)
    return np.stack([x / breite, y / hoehe, (x * y) % 7 / 7], axis=-1)


def test_ohne_flaechen_unveraendert():
    bild = streifen()
    assert anonym.anwenden(bild, ()) is bild


def test_mosaik_nur_in_der_flaeche():
    bild = streifen()
    erg = anonym.anwenden(bild, [("rechteck", 0.25, 0.25, 0.75, 0.75)], "mosaik", 4)
    # ausserhalb gleich
    assert np.array_equal(erg[:15], bild[:15]) and np.array_equal(erg[:, :20], bild[:, :20])
    teil = erg[15:45, 20:60]
    # 40 Pixel breit, 4 Bloecke -> Bloecke von 10 Pixeln, je Block eine Farbe
    assert np.allclose(teil[:10, :10], teil[0, 0]) and not np.allclose(teil[0, 0], teil[0, 10])
    assert np.allclose(teil[0, 0], bild[15:25, 20:30].mean(axis=(0, 1)), atol=1e-6)
    assert not np.array_equal(bild, erg)                        # Kopie, nicht im Bild selbst


def test_ellipse_laesst_ecken_stehen():
    bild = streifen()
    erg = anonym.anwenden(bild, [("ellipse", 0.0, 0.0, 1.0, 1.0)], "fuellen")
    assert np.array_equal(erg[0, 0], bild[0, 0])                # Ecke ausserhalb
    assert np.array_equal(erg[30, 40], [0, 0, 0])               # Mitte gefuellt


def test_weich_ohne_dunklen_rand():
    bild = np.full((60, 80, 3), 0.5, np.float32)
    erg = anonym.anwenden(bild, [("rechteck", 0.1, 0.1, 0.9, 0.9)], "weich", 8)
    assert np.allclose(erg, 0.5, atol=1e-5)


def test_versatz_bei_erweiterung():
    """Original bei (10, 5) in einer groesseren Leinwand: die Flaeche wandert mit."""
    bild = np.zeros((70, 100, 3), np.float32)
    bild[5:65, 10:90] = streifen()
    erg = anonym.anwenden(bild, [("rechteck", 0.0, 0.0, 0.5, 0.5)], "fuellen",
                          original=(60, 80), versatz=(10, 5))
    assert np.array_equal(erg[5:35, 10:50], np.zeros((30, 40, 3)))
    assert np.array_equal(erg[36:, 51:], bild[36:, 51:])


def test_flaeche_begrenzen():
    assert anonym.flaeche_begrenzen(("ellipse", 0.8, 1.2, -0.1, 0.5)) == \
        ("ellipse", 0.0, 0.5, 0.8, 1.0)
    assert anonym.flaeche_begrenzen(("rechteck", 0.3, 0.3, 0.3, 0.6)) is None


def test_nms_haelt_das_sicherste():
    funde = np.array([[0, 0, 10, 10, 0.9], [1, 1, 11, 11, 0.95], [50, 50, 60, 60, 0.9],
                      [52, 52, 56, 56, 0.99]], np.float32)
    assert sorted(anonym.nms(funde)) == [1, 3]                 # der kleine in 2 zaehlt mit


def test_dekodieren_wie_opencv():
    """Ein Treffer im Raster 8 an Zeile 2, Spalte 3: Mitte (3,5 * 8, 2,25 * 8)."""
    hoehe, breite = 64, 96
    ausgaben = {}
    for schritt in anonym.SCHRITTE:
        n = (hoehe // schritt) * (breite // schritt)
        ausgaben[f"cls_{schritt}"] = np.zeros((1, n, 1), np.float32)
        ausgaben[f"obj_{schritt}"] = np.zeros((1, n, 1), np.float32)
        ausgaben[f"bbox_{schritt}"] = np.zeros((1, n, 4), np.float32)
    i = 2 * (breite // 8) + 3
    ausgaben["cls_8"][0, i, 0] = 0.81
    ausgaben["obj_8"][0, i, 0] = 1.0
    ausgaben["bbox_8"][0, i] = (0.5, 0.25, np.log(2), np.log(3))
    funde = anonym.dekodieren(ausgaben, hoehe, breite)
    assert funde.shape == (1, 5)
    x0, y0, x1, y1, wert = funde[0]
    assert (x0 + x1) / 2 == pytest.approx(28) and (y0 + y1) / 2 == pytest.approx(18)
    assert x1 - x0 == pytest.approx(16) and y1 - y0 == pytest.approx(24)
    assert wert == pytest.approx(0.9)


def test_gesicht_als_flaeche():
    form, x0, y0, x1, y1 = anonym.als_flaeche((40, 40, 60, 60, 0.9), 100, 100)
    assert form == "ellipse" and x0 == pytest.approx(0.35) and x1 == pytest.approx(0.65)
    assert y0 < 0.35 and y1 == pytest.approx(0.65)            # oben mehr Rand (Haare)


def test_yunet_liegt_bei():
    import hashlib
    with open(anonym.YUNET, "rb") as datei:
        assert hashlib.sha256(datei.read()).hexdigest() == anonym.YUNET_SHA256


def test_gesichtsfinder_laeuft():
    pytest.importorskip("onnxruntime")
    finder = anonym.Gesichtsfinder()
    leer = np.full((300, 400, 3), 128, np.uint8)
    assert finder.finden(leer) == []


def test_metadaten_ohne_standort_und_seriennummern():
    exif = Image.Exif()
    exif[0x010F] = "Hersteller"
    exif[0x0110] = "Kamera"
    unter = exif.get_ifd(anonym.EXIF_IFD)
    unter[0x829A] = 0.01                                        # Belichtungszeit
    unter[0xA431] = "SN123"
    unter[0xA430] = "Besitzer"
    gps = exif.get_ifd(anonym.GPS_IFD)
    gps[2] = (52.0, 31.0, 0.0)
    roh = exif.tobytes()
    neu = Image.Exif()
    neu.load(anonym.metadaten_bereinigen(roh)[6:])
    assert neu[0x0110] == "Kamera" and neu[0x010F] == "Hersteller"
    assert anonym.GPS_IFD not in neu and not neu.get_ifd(anonym.GPS_IFD)
    rest = neu.get_ifd(anonym.EXIF_IFD)
    assert 0xA431 not in rest and 0xA430 not in rest and 0x829A in rest
    assert anonym.metadaten_bereinigen(b"") == b""


@pytest.fixture
def sitzung():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("keine Grafikkarte")
    from silberkorn import bilddatei
    from silberkorn.bearbeitung import Sitzung
    pixel = (streifen(120, 160) * 255).astype(np.uint8)
    exif = Image.Exif()
    exif[0x0110] = "Kamera"
    exif.get_ifd(anonym.GPS_IFD)[2] = (52.0, 31.0, 0.0)
    daten = bilddatei.Bilddaten(pixel=pixel, profil=None, alpha=None, exif=exif.tobytes(),
                                pfad="t.png")
    s = Sitzung(daten, vorschau_kante=80)
    yield s
    s.schliessen()


def test_flaechen_in_vorschau_und_export(sitzung, tmp_path):
    vorher, _ms, _h = sitzung.vorschau()
    sitzung.werte.anonym_flaechen = (("rechteck", 0.0, 0.0, 0.5, 0.5),)
    sitzung.werte.anonym_art = "fuellen"
    nachher, _ms, _h = sitzung.vorschau()
    assert nachher[:30, :40].max() == 0                       # Vorschau halb so gross
    assert np.array_equal(nachher[35:, 45:], vorher[35:, 45:])
    sitzung.werte.metadaten_entfernen = True
    pfad = str(tmp_path / "aus.jpg")
    sitzung.exportieren(pfad)
    with Image.open(pfad) as bild:
        gespeichert = np.asarray(bild)
        exif = bild.getexif()
    assert gespeichert[:55, :75].max() < 10                   # JPEG: fast schwarz
    assert exif.get(0x0110) == "Kamera" and anonym.GPS_IFD not in exif


def test_flaechen_wandern_mit_der_erweiterung(sitzung):
    """Original bei x = 40 in einer Leinwand von 240 x 120: die Flaeche bleibt im Original."""
    import cupy as cp

    class Weiss:
        def raender(self, srgb, gross_b, gross_h, x, y, seed, fortschritt=None):
            return cp.ones((gross_h, gross_b, 3), dtype=cp.float32)

    sitzung.ki_erweitern(Weiss(), 2.0, seed=1)
    sitzung.werte.anonym_flaechen = (("rechteck", 0.0, 0.0, 1.0, 1.0),)
    sitzung.werte.anonym_art = "fuellen"
    bild, _ms, _h = sitzung.ausschnitt(0, 0, 240, 120)
    assert bild[60, 5].min() > 200                             # Rand bleibt weiss
    assert bild[60, 120].max() == 0                            # Original gefuellt
    # Rueckweg: die Ecke des Originals liegt im fertigen Bild bei x = 40
    assert sitzung.ziel_von([[0.0, 0.0]])[0] == pytest.approx([40, 0], abs=1e-3)


@pytest.mark.parametrize("werte", [
    {},
    {"drehung90": 1},
    {"drehung90": 3, "spiegeln": True, "zuschnitt": (0.1, 0.2, 0.8, 0.9)},
    {"begradigen": 7.0, "perspektive_v": 30.0, "verzeichnung": 20.0},
])
def test_von_quelle_umkehrt_zur_quelle(werte):
    g = geometrie.Geometrie(**werte)
    form = (300, 400)
    punkte = np.array([[200.0, 150.0], [120.0, 90.0], [330.0, 240.0]])
    ziel = geometrie.von_quelle(form, g, punkte)
    for (sx, sy), (x, y) in zip(punkte, ziel, strict=True):
        qx, qy = geometrie.zur_quelle(form, g, x, y)
        assert qx == pytest.approx(sx, abs=1e-3) and qy == pytest.approx(sy, abs=1e-3)
