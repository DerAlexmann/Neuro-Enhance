"""
Bilder laden und speichern

Geladen wird mit Pillow. Die Ausrichtung aus den EXIF-Daten wird gleich beim
Laden angewendet, ein eingebettetes Farbprofil (etwa Adobe RGB) nach sRGB
umgerechnet - die Filter rechnen durchweg in sRGB-Primaerfarben. Ein
Alphakanal wird unveraendert durchgereicht.

Diese erste Fassung arbeitet mit 8 Bit je Kanal; 16 Bit und RAW folgen.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageCms, ImageOps

try:                                          # HEIC/HEIF, falls installiert
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF = True
except ImportError:
    HEIF = False

LESBAR = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"]
if HEIF:
    LESBAR += [".heic", ".heif"]

# Endung -> (Pillow-Format, kann Alphakanal)
SCHREIBBAR = {
    ".jpg": ("JPEG", False),
    ".jpeg": ("JPEG", False),
    ".png": ("PNG", True),
    ".tif": ("TIFF", True),
    ".tiff": ("TIFF", True),
    ".webp": ("WEBP", True),
}

JPEG_QUALITAET = 95
EXIF_AUSRICHTUNG = 0x0112


class BildFehler(Exception):
    """Die Datei laesst sich nicht als Bild lesen oder schreiben."""


@dataclass
class Bilddaten:
    rgb: np.ndarray                  # (H, W, 3) uint8, sRGB
    alpha: np.ndarray | None         # (H, W) uint8 oder None
    exif: bytes                      # Ausrichtung bereits auf 1 gesetzt
    pfad: str

    @property
    def breite(self) -> int:
        return self.rgb.shape[1]

    @property
    def hoehe(self) -> int:
        return self.rgb.shape[0]

    @property
    def name(self) -> str:
        return os.path.basename(self.pfad)


def _srgb_profil() -> ImageCms.ImageCmsProfile:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))


def _nach_srgb(bild: Image.Image) -> Image.Image:
    """Bild (ohne Alphakanal) nach sRGB-RGB umrechnen.

    Mit eingebettetem Profil rechnet LittleCMS um - auch aus CMYK oder Graustufen.
    Ohne Profil gilt das Bild als sRGB.
    """
    icc = bild.info.get("icc_profile")
    if icc and bild.mode in ("RGB", "CMYK", "L"):
        try:
            quelle = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            return ImageCms.profileToProfile(bild, quelle, _srgb_profil(), outputMode="RGB")
        except (ImageCms.PyCMSError, OSError, ValueError):
            pass                              # unlesbares Profil: wie sRGB behandeln
    return bild.convert("RGB")


def laden(pfad: str) -> Bilddaten:
    try:
        with Image.open(pfad) as roh:
            roh.load()
            exif = roh.getexif()
            bild = ImageOps.exif_transpose(roh)
    except (OSError, ValueError, Image.DecompressionBombError) as fehler:
        raise BildFehler(str(fehler)) from fehler

    alpha = None
    if bild.mode in ("RGBA", "LA", "PA") or (bild.mode == "P" and "transparency" in bild.info):
        icc = bild.info.get("icc_profile")
        bild = bild.convert("RGBA")
        alpha = np.asarray(bild.getchannel("A")).copy()
        bild = bild.convert("RGB")
        if icc:
            bild.info["icc_profile"] = icc
    rgb_bild = _nach_srgb(bild)

    if EXIF_AUSRICHTUNG in exif:
        exif[EXIF_AUSRICHTUNG] = 1            # bereits gedreht
    return Bilddaten(np.asarray(rgb_bild).copy(), alpha, exif.tobytes(), pfad)


def speichern(pfad: str, rgb: np.ndarray, alpha: np.ndarray | None = None,
              exif: bytes = b"") -> None:
    endung = os.path.splitext(pfad)[1].lower()
    if endung not in SCHREIBBAR:
        raise BildFehler(f"unbekanntes Format: {endung}")
    format_name, kann_alpha = SCHREIBBAR[endung]

    bild = Image.fromarray(rgb, "RGB")
    if alpha is not None and kann_alpha:
        bild.putalpha(Image.fromarray(alpha, "L"))

    optionen = {"icc_profile": _srgb_profil().tobytes()}
    if exif and format_name in ("JPEG", "PNG", "TIFF", "WEBP"):
        optionen["exif"] = exif
    if format_name == "JPEG":
        optionen.update(quality=JPEG_QUALITAET, subsampling=0)
    elif format_name == "WEBP":
        optionen.update(quality=JPEG_QUALITAET)
    elif format_name == "TIFF":
        optionen.update(compression="tiff_lzw")

    try:
        bild.save(pfad, format_name, **optionen)
    except (OSError, ValueError) as fehler:
        raise BildFehler(str(fehler)) from fehler


def verkleinern(rgb: np.ndarray, laengste_kante: int) -> tuple[np.ndarray, float]:
    """Verkleinerte Kopie fuer die Vorschau und ihr Massstab zum Original."""
    hoehe, breite = rgb.shape[:2]
    massstab = min(1.0, laengste_kante / max(hoehe, breite))
    if massstab >= 1.0:
        return rgb, 1.0
    groesse = (max(1, round(breite * massstab)), max(1, round(hoehe * massstab)))
    klein = Image.fromarray(rgb, "RGB").resize(groesse, Image.Resampling.LANCZOS)
    return np.asarray(klein), massstab
