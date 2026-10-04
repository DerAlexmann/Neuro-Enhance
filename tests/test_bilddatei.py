"""Tests fuer Laden und Speichern - Ausrichtung, Alphakanal, Farbprofile, Formate."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageCms

from neuro_enhance import bilddatei as b


def muster(breite=40, hoehe=24):
    rgb = np.zeros((hoehe, breite, 3), dtype=np.uint8)
    rgb[..., 0] = np.linspace(0, 255, breite, dtype=np.uint8)[None, :]
    rgb[..., 1] = np.linspace(0, 255, hoehe, dtype=np.uint8)[:, None]
    rgb[..., 2] = 128
    return rgb


@pytest.mark.parametrize("endung", [".png", ".tif", ".webp", ".jpg"])
def test_speichern_und_laden(tmp_path, endung):
    rgb = muster()
    pfad = str(tmp_path / f"bild{endung}")
    b.speichern(pfad, rgb)
    geladen = b.laden(pfad)
    assert geladen.rgb.shape == rgb.shape
    toleranz = 0 if endung in (".png", ".tif") else 12
    assert np.abs(geladen.rgb.astype(int) - rgb.astype(int)).max() <= toleranz


def test_exif_ausrichtung_wird_angewendet(tmp_path):
    rgb = muster(40, 24)
    exif = Image.Exif()
    exif[b.EXIF_AUSRICHTUNG] = 6               # 90 Grad im Uhrzeigersinn
    pfad = str(tmp_path / "gedreht.jpg")
    Image.fromarray(rgb).save(pfad, exif=exif.tobytes(), quality=95)
    geladen = b.laden(pfad)
    assert (geladen.breite, geladen.hoehe) == (24, 40)
    # Die mitgefuehrten EXIF-Daten duerfen beim Speichern nicht ein zweites Mal drehen
    gelesen = Image.Exif()
    gelesen.load(geladen.exif)
    assert gelesen.get(b.EXIF_AUSRICHTUNG) == 1


def test_alphakanal_bleibt_erhalten(tmp_path):
    rgba = np.dstack([muster(), np.full((24, 40), 77, dtype=np.uint8)])
    quelle = str(tmp_path / "durchsichtig.png")
    Image.fromarray(rgba, "RGBA").save(quelle)
    geladen = b.laden(quelle)
    assert geladen.alpha is not None and (geladen.alpha == 77).all()
    ziel = str(tmp_path / "ziel.png")
    b.speichern(ziel, geladen.rgb, geladen.alpha)
    assert Image.open(ziel).mode == "RGBA"


def test_jpeg_verwirft_alpha_ohne_fehler(tmp_path):
    ziel = str(tmp_path / "ohne_alpha.jpg")
    b.speichern(ziel, muster(), np.full((24, 40), 0, dtype=np.uint8))
    assert Image.open(ziel).mode == "RGB"


def test_srgb_profil_laesst_werte_unveraendert(tmp_path):
    rgb = np.full((8, 8, 3), (200, 60, 40), dtype=np.uint8)
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    pfad = str(tmp_path / "srgb.png")
    Image.fromarray(rgb).save(pfad, icc_profile=srgb.tobytes())
    assert np.abs(b.laden(pfad).rgb.astype(int) - rgb.astype(int)).max() <= 1


def test_eingebettetes_profil_wird_umgerechnet(tmp_path, monkeypatch):
    """Mit Profil muss LittleCMS nach sRGB rechnen - ohne Profil nicht."""
    aufrufe = []
    original = ImageCms.profileToProfile

    def mitschreiben(bild, *args, **kwargs):
        aufrufe.append(bild.mode)
        return original(bild, *args, **kwargs)

    monkeypatch.setattr(ImageCms, "profileToProfile", mitschreiben)
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    mit = str(tmp_path / "mit.png")
    ohne = str(tmp_path / "ohne.png")
    Image.fromarray(muster()).save(mit, icc_profile=srgb.tobytes())
    Image.fromarray(muster()).save(ohne)
    b.laden(ohne)
    assert aufrufe == []
    b.laden(mit)
    assert aufrufe == ["RGB"]


def test_cmyk_wird_rgb(tmp_path):
    pfad = str(tmp_path / "cmyk.jpg")
    Image.new("CMYK", (10, 6), (0, 255, 255, 0)).save(pfad)
    geladen = b.laden(pfad)
    assert geladen.rgb.shape == (6, 10, 3)


def test_unlesbare_datei(tmp_path):
    pfad = tmp_path / "kaputt.jpg"
    pfad.write_bytes(b"kein bild")
    with pytest.raises(b.BildFehler):
        b.laden(str(pfad))


def test_unbekanntes_zielformat(tmp_path):
    with pytest.raises(b.BildFehler):
        b.speichern(str(tmp_path / "bild.xyz"), muster())


def test_verkleinern():
    rgb = muster(400, 200)
    klein, massstab = b.verkleinern(rgb, 100)
    assert klein.shape[:2] == (50, 100) and massstab == pytest.approx(0.25)
    gleich, massstab = b.verkleinern(rgb, 1000)
    assert gleich is rgb and massstab == 1.0
