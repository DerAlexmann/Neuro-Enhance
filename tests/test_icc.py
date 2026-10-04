"""Tests des ICC-Lesers und der Umrechnung nach linearem sRGB."""

from __future__ import annotations

import struct

import numpy as np
from PIL import ImageCms

from neuro_enhance import filter as f
from neuro_enhance import icc

# Adobe RGB (1998) nach XYZ, an D50 angepasst - wie im Profil von Adobe
ADOBE_NACH_XYZ_D50 = np.array([
    [0.6097559, 0.2052401, 0.1492240],
    [0.3111242, 0.6256560, 0.0632197],
    [0.0194811, 0.0608902, 0.7448387],
])
ADOBE_GAMMA = 563 / 256                        # 2.19921875, so steht es im Profil


def adobe_rgb_profil() -> bytes:
    """Minimales ICC-Matrixprofil fuer Adobe RGB - nur die noetigen Eintraege."""
    def xyz(werte):
        return b"XYZ \x00\x00\x00\x00" + b"".join(
            struct.pack(">i", round(w * 65536)) for w in werte)

    kurve = b"curv\x00\x00\x00\x00" + struct.pack(">IH", 1, round(ADOBE_GAMMA * 256)) + b"\x00\x00"
    eintraege = [(b"rXYZ", xyz(ADOBE_NACH_XYZ_D50[:, 0])), (b"gXYZ", xyz(ADOBE_NACH_XYZ_D50[:, 1])),
                 (b"bXYZ", xyz(ADOBE_NACH_XYZ_D50[:, 2])),
                 (b"rTRC", kurve), (b"gTRC", kurve), (b"bTRC", kurve)]
    tabelle = struct.pack(">I", len(eintraege))
    daten = b""
    stelle = 128 + 4 + 12 * len(eintraege)
    for kennung, inhalt in eintraege:
        tabelle += struct.pack(">4sII", kennung, stelle + len(daten), len(inhalt))
        daten += inhalt
    kopf = bytearray(128)
    struct.pack_into(">I", kopf, 0, 128 + len(tabelle) + len(daten))
    kopf[8:12] = b"\x02\x10\x00\x00"           # Version 2.1
    kopf[12:16] = b"mntr"
    kopf[16:20] = b"RGB "
    kopf[20:24] = b"XYZ "
    kopf[36:40] = b"acsp"
    return bytes(kopf) + tabelle + daten


def test_littlecms_srgb_wird_als_srgb_erkannt():
    profil = icc.lesen(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    assert profil is not None and profil.ist_srgb()


def test_adobe_rgb_wird_gelesen():
    profil = icc.lesen(adobe_rgb_profil())
    assert profil is not None and not profil.ist_srgb()
    assert np.allclose(profil.matrix, ADOBE_NACH_XYZ_D50, atol=1e-4)
    x = np.linspace(0, 1, icc.TABELLE)
    assert np.allclose(profil.kurven[0], x ** ADOBE_GAMMA, atol=1e-6)


def test_keine_matrixprofile_werden_abgelehnt():
    assert icc.lesen(None) is None
    assert icc.lesen(b"zu kurz") is None
    lab = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
    assert icc.lesen(lab) is None


def test_weiss_bleibt_weiss():
    """Beide Profile beziehen sich auf D50 - Weiss muss Weiss bleiben."""
    weiss = np.full((1, 1, 3), 255, dtype=np.uint8)
    linear = icc.linearisieren(weiss, icc.lesen(adobe_rgb_profil()))
    assert np.allclose(linear, 1.0, atol=2e-3)


def test_adobe_gruen_liegt_ausserhalb_von_srgb():
    """Reines Adobe-Gruen hat in sRGB negatives Rot - es wird nicht abgeschnitten."""
    gruen = np.array([[[0, 255, 0]]], dtype=np.uint8)
    linear = icc.linearisieren(gruen, icc.lesen(adobe_rgb_profil()))
    assert linear[0, 0, 0] < -0.1 and linear[0, 0, 1] > 0.9


def test_ohne_profil_wie_srgb():
    alle = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
    assert np.allclose(icc.linearisieren(alle, None), f.von_8bit(alle))


def test_16bit_linear():
    werte = np.array([[[0, 32768, 65535]]], dtype=np.uint16)
    linear = icc.linearisieren(werte, icc.LINEAR_SRGB)
    assert np.allclose(linear, [[[0.0, 32768 / 65535, 1.0]]], atol=1e-6)
