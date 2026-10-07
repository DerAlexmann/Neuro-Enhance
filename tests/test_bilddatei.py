"""Tests fuer Laden und Speichern - 8 Bit, 16 Bit, RAW, Ausrichtung, Alpha, Farbprofile."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageCms, ImageOps

from silberkorn import bilddatei as b
from silberkorn import icc


def muster(breite=40, hoehe=24, bits=8):
    hoechst = 65535 if bits == 16 else 255
    typ = np.uint16 if bits == 16 else np.uint8
    rgb = np.zeros((hoehe, breite, 3), dtype=typ)
    rgb[..., 0] = np.linspace(0, hoechst, breite).astype(typ)[None, :]
    rgb[..., 1] = np.linspace(0, hoechst, hoehe).astype(typ)[:, None]
    rgb[..., 2] = hoechst // 2
    return rgb


# ----------------------------------------------------------------------
# 8 Bit
# ----------------------------------------------------------------------

@pytest.mark.parametrize("endung", [".png", ".tif", ".webp", ".jpg"])
def test_speichern_und_laden_8bit(tmp_path, endung):
    rgb = muster()
    pfad = str(tmp_path / f"bild{endung}")
    b.speichern(pfad, rgb)
    geladen = b.laden(pfad)
    assert geladen.pixel.shape == rgb.shape and geladen.bits == 8
    assert geladen.profil is None             # gespeichert wird mit sRGB-Profil
    toleranz = 0 if endung in (".png", ".tif") else 12
    assert np.abs(geladen.pixel.astype(int) - rgb.astype(int)).max() <= toleranz


def test_exif_ausrichtung_wird_angewendet(tmp_path):
    exif = Image.Exif()
    exif[b.EXIF_AUSRICHTUNG] = 6               # 90 Grad im Uhrzeigersinn
    pfad = str(tmp_path / "gedreht.jpg")
    Image.fromarray(muster(40, 24)).save(pfad, exif=exif.tobytes(), quality=95)
    geladen = b.laden(pfad)
    assert (geladen.breite, geladen.hoehe) == (24, 40)
    # Die mitgefuehrten EXIF-Daten duerfen beim Speichern nicht ein zweites Mal drehen
    gelesen = Image.Exif()
    gelesen.load(geladen.exif)
    assert gelesen.get(b.EXIF_AUSRICHTUNG) == 1


@pytest.mark.parametrize("wert", range(1, 9))
def test_ausrichten_wie_pillow(tmp_path, wert):
    """Die eigene Drehung fuer 16 Bit muss genau das tun, was Pillow bei 8 Bit tut."""
    rgb = muster(7, 5)
    exif = Image.Exif()
    exif[b.EXIF_AUSRICHTUNG] = wert
    pfad = str(tmp_path / "ausrichtung.png")
    Image.fromarray(rgb).save(pfad, exif=exif.tobytes())
    erwartet = np.asarray(ImageOps.exif_transpose(Image.open(pfad)))
    assert np.array_equal(b.ausrichten(rgb, wert), erwartet)


def test_alphakanal_bleibt_erhalten(tmp_path):
    rgba = np.dstack([muster(), np.full((24, 40), 77, dtype=np.uint8)])
    quelle = str(tmp_path / "durchsichtig.png")
    Image.fromarray(rgba, "RGBA").save(quelle)
    geladen = b.laden(quelle)
    assert geladen.alpha is not None and (geladen.alpha == 77).all()
    ziel = str(tmp_path / "ziel.png")
    b.speichern(ziel, geladen.pixel, geladen.alpha)
    assert Image.open(ziel).mode == "RGBA"


def test_jpeg_verwirft_alpha_ohne_fehler(tmp_path):
    ziel = str(tmp_path / "ohne_alpha.jpg")
    b.speichern(ziel, muster(), np.full((24, 40), 0, dtype=np.uint8))
    assert Image.open(ziel).mode == "RGB"


def test_cmyk_wird_rgb(tmp_path):
    pfad = str(tmp_path / "cmyk.jpg")
    Image.new("CMYK", (10, 6), (0, 255, 255, 0)).save(pfad)
    geladen = b.laden(pfad)
    assert geladen.pixel.shape == (6, 10, 3) and geladen.profil is None


def test_unlesbare_datei(tmp_path):
    pfad = tmp_path / "kaputt.jpg"
    pfad.write_bytes(b"kein bild")
    with pytest.raises(b.BildFehler):
        b.laden(str(pfad))


def test_unbekanntes_zielformat(tmp_path):
    with pytest.raises(b.BildFehler):
        b.speichern(str(tmp_path / "bild.xyz"), muster())


def test_jpeg_kann_keine_16_bit(tmp_path):
    with pytest.raises(b.BildFehler):
        b.speichern(str(tmp_path / "bild.jpg"), muster(bits=16))


# ----------------------------------------------------------------------
# 16 Bit
# ----------------------------------------------------------------------

@pytest.mark.parametrize("endung", [".png", ".tif"])
def test_16bit_verlustfrei(tmp_path, endung):
    pytest.importorskip("tifffile")
    pytest.importorskip("imagecodecs")
    rgb = muster(bits=16)
    rgb[3, 5] = (1, 2, 65534)                  # Werte, die 8 Bit nicht unterscheiden kann
    pfad = str(tmp_path / f"bild16{endung}")
    b.speichern(pfad, rgb)
    geladen = b.laden(pfad)
    assert geladen.bits == 16 and geladen.profil is None
    assert np.array_equal(geladen.pixel, rgb)


@pytest.mark.parametrize("endung", [".png", ".tif"])
def test_16bit_mit_alpha(tmp_path, endung):
    pytest.importorskip("tifffile")
    pytest.importorskip("imagecodecs")
    alpha = np.full((24, 40), 40000, dtype=np.uint16)
    pfad = str(tmp_path / f"alpha16{endung}")
    b.speichern(pfad, muster(bits=16), alpha)
    geladen = b.laden(pfad)
    assert geladen.alpha is not None and (geladen.alpha == 40000).all()


def test_16bit_png_behaelt_exif(tmp_path):
    pytest.importorskip("imagecodecs")
    exif = Image.Exif()
    exif[0x010F] = "Testkamera"                # Make
    pfad = str(tmp_path / "exif16.png")
    b.speichern(pfad, muster(bits=16), exif=exif.tobytes())
    assert Image.open(pfad).getexif().get(0x010F) == "Testkamera"


def test_alpha_bittiefe_umrechnen():
    alpha = np.array([0, 128, 255], dtype=np.uint8)
    hoch = b.alpha_umrechnen(alpha, 16)
    assert hoch.tolist() == [0, 32896, 65535]
    assert b.alpha_umrechnen(hoch, 8).tolist() == [0, 128, 255]


# ----------------------------------------------------------------------
# Farbprofile
# ----------------------------------------------------------------------

def test_srgb_profil_ergibt_kein_eigenes_profil(tmp_path):
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    pfad = str(tmp_path / "srgb.png")
    Image.fromarray(muster()).save(pfad, icc_profile=srgb.tobytes())
    assert b.laden(pfad).profil is None


def test_matrixprofil_wird_mitgegeben(tmp_path):
    from test_icc import adobe_rgb_profil
    pfad = str(tmp_path / "adobe.png")
    Image.fromarray(muster()).save(pfad, icc_profile=adobe_rgb_profil())
    geladen = b.laden(pfad)
    assert geladen.profil is not None and not geladen.profil.ist_srgb()
    # die Pixel selbst bleiben unveraendert - umgerechnet wird erst auf der GPU
    assert np.array_equal(geladen.pixel, muster())


# ----------------------------------------------------------------------
# RAW
# ----------------------------------------------------------------------

# (Flaeche, Farbe in linearem sRGB) - eine Liste, weil slice erst ab
# Python 3.12 als Schluessel eines Woerterbuchs taugt
SZENE = [
    ((slice(0, 32), slice(0, 48)), (0.40, 0.10, 0.05)),
    ((slice(0, 32), slice(48, 96)), (0.05, 0.30, 0.08)),
    ((slice(32, 64), slice(0, 48)), (0.06, 0.08, 0.45)),
    ((slice(32, 64), slice(48, 96)), (0.18, 0.18, 0.18)),
]


def bayer_dng(pfad: str):
    """Kleine DNG mit RGGB-Mosaik; die Kamera sieht genau lineares sRGB."""
    tifffile = pytest.importorskip("tifffile")
    hoehe, breite = 64, 96
    szene = np.zeros((hoehe, breite, 3))
    for flaeche, farbe in SZENE:
        szene[flaeche] = farbe
    kanal = np.zeros((hoehe, breite), int)
    kanal[0::2, 1::2] = 1
    kanal[1::2, 0::2] = 1
    kanal[1::2, 1::2] = 2
    schwarz, weiss = 256, 16383
    werte = np.take_along_axis(szene, kanal[..., None], axis=2)[..., 0]
    mosaik = np.round(schwarz + werte * (weiss - schwarz)).astype(np.uint16)
    srgb_nach_xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                              [0.2126729, 0.7151522, 0.0721750],
                              [0.0193339, 0.1191920, 0.9503041]])
    farbmatrix = [x for w in np.linalg.inv(srgb_nach_xyz).ravel()
                  for x in (int(round(w * 10000)), 10000)]
    tags = [
        (50706, "B", 4, (1, 4, 0, 0), True),          # DNGVersion
        (50708, "s", 0, "Testkamera", True),          # UniqueCameraModel
        (33421, "H", 2, (2, 2), True),                # CFARepeatPatternDim
        (33422, "B", 4, (0, 1, 1, 2), True),          # CFAPattern RGGB
        (50714, "H", 1, schwarz, True),               # BlackLevel
        (50717, "H", 1, weiss, True),                 # WhiteLevel
        (50721, 10, 9, farbmatrix, True),             # ColorMatrix1
        (50778, "H", 1, 21, True),                    # CalibrationIlluminant1 = D65
        (50728, 5, 3, [1, 1, 1, 1, 1, 1], True),      # AsShotNeutral
    ]
    tifffile.imwrite(pfad, mosaik, photometric=32803, extratags=tags, metadata=None)


def test_raw_wird_farbrichtig_entwickelt(tmp_path):
    pytest.importorskip("rawpy")
    pfad = str(tmp_path / "test.dng")
    bayer_dng(pfad)
    geladen = b.laden(pfad)
    assert geladen.raw and geladen.bits == 16 and geladen.profil is icc.LINEAR_SRGB
    linear = geladen.linear()
    for (zeilen, spalten), farbe in SZENE:
        mitte = linear[zeilen, spalten][8:-8, 8:-8]   # Rand des Mosaiks auslassen
        assert np.allclose(mitte.mean(axis=(0, 1)), farbe, atol=0.01)


def test_kaputtes_raw(tmp_path):
    pytest.importorskip("rawpy")
    pfad = tmp_path / "kaputt.nef"
    pfad.write_bytes(b"kein raw")
    with pytest.raises(b.BildFehler):
        b.laden(str(pfad))
