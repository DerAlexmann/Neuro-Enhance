"""
Bilder laden und speichern - 8 Bit, 16 Bit und RAW

Woher die Pixel kommen:
  RAW (Bayer)    LibRaw liest das Mosaik, entwickelt wird auf der GPU (demosaik.py)
  RAW (sonst)    LibRaw entwickelt linear in 16 Bit, sRGB-Primaerfarben
  TIFF           tifffile - 8 und 16 Bit, auch komprimiert
  PNG mit 16 Bit imagecodecs
  alles andere   Pillow, 8 Bit

Die Pixel bleiben so kodiert, wie sie in der Datei stehen; ein eingebettetes
RGB-Matrixprofil (Adobe RGB, ProPhoto, Display P3 ...) wird mitgegeben und
erst auf der Grafikkarte in lineares sRGB umgerechnet (icc.linearisieren).
Nur Profile anderer Bauart - CMYK, Graustufen, Tabellenprofile - rechnet
LittleCMS ueber Pillow schon beim Laden nach sRGB, dann mit 8 Bit.

Die EXIF-Ausrichtung wird beim Laden angewendet. EXIF-Daten und Alphakanal
werden beim Speichern weitergegeben, soweit das Zielformat sie traegt.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import io
import os
import struct
import zlib
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageCms, ImageOps

from . import demosaik, icc

try:                                          # HEIC/HEIF, falls installiert
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF = True
except ImportError:
    HEIF = False

RAW = [".3fr", ".arw", ".cr2", ".cr3", ".dng", ".erf", ".iiq", ".kdc", ".mef", ".mos",
       ".mrw", ".nef", ".nrw", ".orf", ".pef", ".raf", ".rw2", ".rwl", ".sr2", ".srf",
       ".srw", ".x3f"]
BILDER = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"]
if HEIF:
    BILDER += [".heic", ".heif"]
LESBAR = BILDER + RAW

# Endung -> (Format, kann Alphakanal, kann 16 Bit)
SCHREIBBAR = {
    ".jpg": ("JPEG", False, False),
    ".jpeg": ("JPEG", False, False),
    ".png": ("PNG", True, True),
    ".tif": ("TIFF", True, True),
    ".tiff": ("TIFF", True, True),
    ".webp": ("WEBP", True, False),
}

JPEG_QUALITAET = 95
EXIF_AUSRICHTUNG = 0x0112
TIFF_ICC = 34675


class BildFehler(Exception):
    """Die Datei laesst sich nicht als Bild lesen oder schreiben."""


@dataclass
class Bilddaten:
    pixel: np.ndarray | None         # (H, W, 3) uint8 oder uint16, kodiert wie in der Datei
    profil: icc.Matrixprofil | None  # Farbraum der Pixel; None heisst sRGB
    alpha: np.ndarray | None         # (H, W) in derselben Bittiefe wie pixel
    exif: bytes                      # Ausrichtung bereits auf 1 gesetzt
    pfad: str
    raw: bool = False
    mosaik: demosaik.RawMosaik | None = None   # statt pixel: Bayer-RAW fuer die GPU

    @property
    def form(self) -> tuple[int, int]:
        return self.mosaik.form if self.mosaik is not None else self.pixel.shape[:2]

    @property
    def breite(self) -> int:
        return self.form[1]

    @property
    def hoehe(self) -> int:
        return self.form[0]

    @property
    def bits(self) -> int:
        if self.mosaik is not None:
            return 16
        return 16 if self.pixel.dtype == np.uint16 else 8

    def linear(self, xp=np):
        """Das Bild in linearem sRGB als float32 - mit NumPy oder CuPy (xp)."""
        if self.mosaik is not None:
            return demosaik.entwickeln(xp.asarray(self.mosaik.daten), self.mosaik)
        return icc.linearisieren(xp.asarray(self.pixel), self.profil)

    @property
    def name(self) -> str:
        return os.path.basename(self.pfad)


# --------------------------------------------------------------------------
# Hilfen
# --------------------------------------------------------------------------

def _srgb_profil() -> ImageCms.ImageCmsProfile:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))


def ausrichten(pixel: np.ndarray, wert: int) -> np.ndarray:
    """EXIF-Ausrichtung 1..8 auf ein Array anwenden - wie ImageOps.exif_transpose."""
    if wert == 2:
        return pixel[:, ::-1]
    if wert == 3:
        return pixel[::-1, ::-1]
    if wert == 4:
        return pixel[::-1]
    if wert == 5:
        return np.swapaxes(pixel, 0, 1)
    if wert == 6:
        return np.rot90(pixel, 3)
    if wert == 7:
        return np.rot90(np.swapaxes(pixel, 0, 1), 2)
    if wert == 8:
        return np.rot90(pixel, 1)
    return pixel


def _kanaele_trennen(pixel: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Graustufen zu RGB ausweiten und einen Alphakanal abtrennen."""
    if pixel.ndim == 2:
        return np.repeat(pixel[..., None], 3, axis=2), None
    kanaele = pixel.shape[2]
    if kanaele == 1:
        return np.repeat(pixel, 3, axis=2), None
    if kanaele == 2:
        return np.repeat(pixel[..., :1], 3, axis=2), pixel[..., 1].copy()
    if kanaele == 4:
        return pixel[..., :3], pixel[..., 3].copy()
    return pixel[..., :3], None


def _bittiefe_angleichen(pixel: np.ndarray) -> np.ndarray:
    """Alles, was nicht 8 oder 16 Bit ist, auf 16 Bit bringen."""
    if pixel.dtype in (np.uint8, np.uint16):
        return pixel
    if np.issubdtype(pixel.dtype, np.floating):
        return (np.clip(pixel, 0, 1) * 65535 + 0.5).astype(np.uint16)
    if pixel.dtype == np.uint32:
        return (pixel >> 16).astype(np.uint16)
    raise BildFehler(f"nicht unterstützte Bittiefe: {pixel.dtype}")


def _metadaten(pfad: str) -> tuple[bytes | None, Image.Exif]:
    """Farbprofil und EXIF ueber Pillow lesen, ohne die Pixel zu dekodieren."""
    try:
        with Image.open(pfad) as kopf:
            return kopf.info.get("icc_profile"), kopf.getexif()
    except (OSError, ValueError):
        return None, Image.Exif()


def _exif_ohne_ausrichtung(exif: Image.Exif) -> bytes:
    if EXIF_AUSRICHTUNG in exif:
        exif[EXIF_AUSRICHTUNG] = 1            # bereits gedreht
    return exif.tobytes() if len(exif) else b""


def _profil_oder_fehler(daten: bytes | None) -> icc.Matrixprofil | None:
    """Matrixprofil oder None (= sRGB). Andere Profile kann nur der 8-Bit-Weg."""
    profil = icc.lesen(daten)
    if profil is not None and profil.ist_srgb():
        return None
    return profil


# --------------------------------------------------------------------------
# Laden
# --------------------------------------------------------------------------

def _raw_laden(pfad: str, gpu: bool = True) -> Bilddaten:
    import rawpy
    try:
        with rawpy.imread(pfad) as roh:
            mosaik = demosaik.aus_rawpy(roh) if gpu else None
            if mosaik is not None:
                return Bilddaten(None, icc.LINEAR_SRGB, None, b"", pfad, raw=True, mosaik=mosaik)
            # Linear (Gamma 1), 16 Bit, sRGB-Primaerfarben, Weissabgleich der
            # Kamera - und ohne die automatische Aufhellung von LibRaw. Die
            # liesse 1 % der Pixel ausbrennen; so bleibt der volle Umfang des
            # Sensors erhalten, und die Helligkeit regelt die Belichtung.
            pixel = roh.postprocess(gamma=(1, 1), output_bps=16, use_camera_wb=True,
                                    output_color=rawpy.ColorSpace.sRGB, no_auto_bright=True)
    except (rawpy.LibRawError, OSError, ValueError) as fehler:
        raise BildFehler(str(fehler)) from fehler
    return Bilddaten(np.ascontiguousarray(pixel), icc.LINEAR_SRGB, None, b"", pfad, raw=True)


def _tiff_laden(pfad: str) -> Bilddaten | None:
    """TIFF mit tifffile; None fuer Farbmodelle, die Pillow besser kann (CMYK, Palette ...)."""
    import tifffile
    try:
        with tifffile.TiffFile(pfad) as datei:
            seite = datei.pages[0]
            if seite.photometric not in (tifffile.PHOTOMETRIC.RGB,
                                         tifffile.PHOTOMETRIC.MINISBLACK):
                return None
            pixel = seite.asarray()
            if seite.planarconfig == tifffile.PLANARCONFIG.SEPARATE and pixel.ndim == 3:
                pixel = np.moveaxis(pixel, 0, -1)
            icc_tag = seite.tags.get(TIFF_ICC)
            icc_daten = bytes(icc_tag.value) if icc_tag is not None else None
            tag = seite.tags.get(EXIF_AUSRICHTUNG)
            ausrichtung = int(tag.value) if tag is not None else 1
    except (tifffile.TiffFileError, OSError, ValueError, IndexError) as fehler:
        raise BildFehler(str(fehler)) from fehler

    if pixel.dtype == np.uint8 and icc_daten and icc.lesen(icc_daten) is None:
        return None                           # Tabellenprofil: LittleCMS ueber Pillow
    rgb, alpha = _kanaele_trennen(_bittiefe_angleichen(pixel))
    rgb, alpha = ausrichten(rgb, ausrichtung), (ausrichten(alpha, ausrichtung)
                                                if alpha is not None else None)
    _icc, exif = _metadaten(pfad)
    return Bilddaten(np.ascontiguousarray(rgb), _profil_oder_fehler(icc_daten),
                     None if alpha is None else np.ascontiguousarray(alpha),
                     _exif_ohne_ausrichtung(exif), pfad)


def _png_bittiefe(pfad: str) -> int:
    with open(pfad, "rb") as datei:
        kopf = datei.read(26)
    if len(kopf) < 26 or kopf[:8] != b"\x89PNG\r\n\x1a\n":
        return 0
    return kopf[24]


def _png16_laden(pfad: str) -> Bilddaten:
    import imagecodecs
    try:
        with open(pfad, "rb") as datei:
            pixel = imagecodecs.png_decode(datei.read())
    except (imagecodecs.PngError, OSError, ValueError) as fehler:
        raise BildFehler(str(fehler)) from fehler
    icc_daten, exif = _metadaten(pfad)
    rgb, alpha = _kanaele_trennen(_bittiefe_angleichen(pixel))
    ausrichtung = exif.get(EXIF_AUSRICHTUNG, 1)
    rgb = ausrichten(rgb, ausrichtung)
    if alpha is not None:
        alpha = np.ascontiguousarray(ausrichten(alpha, ausrichtung))
    if icc_daten and icc.lesen(icc_daten) is None:
        raise BildFehler("16-Bit-PNG mit einem Farbprofil, das kein RGB-Matrixprofil ist")
    return Bilddaten(np.ascontiguousarray(rgb), _profil_oder_fehler(icc_daten), alpha,
                     _exif_ohne_ausrichtung(exif), pfad)


def _pillow_laden(pfad: str) -> Bilddaten:
    try:
        with Image.open(pfad) as roh:
            roh.load()
            exif = roh.getexif()
            bild = ImageOps.exif_transpose(roh)
    except (OSError, ValueError, Image.DecompressionBombError) as fehler:
        raise BildFehler(str(fehler)) from fehler

    icc_daten = bild.info.get("icc_profile")
    alpha = None
    if bild.mode in ("RGBA", "LA", "PA") or (bild.mode == "P" and "transparency" in bild.info):
        bild = bild.convert("RGBA")
        alpha = np.asarray(bild.getchannel("A")).copy()
        bild = bild.convert("RGB")

    profil = icc.lesen(icc_daten)
    if bild.mode == "RGB" and (not icc_daten or profil is not None):
        # sRGB oder Matrixprofil: Pixel unveraendert, umgerechnet wird auf der GPU
        pixel, profil = np.asarray(bild), _profil_oder_fehler(icc_daten)
    else:
        # CMYK, Graustufen, Tabellenprofile: LittleCMS rechnet nach sRGB
        profil = None
        if icc_daten and bild.mode in ("RGB", "CMYK", "L"):
            try:
                quelle = ImageCms.ImageCmsProfile(io.BytesIO(icc_daten))
                bild = ImageCms.profileToProfile(bild, quelle, _srgb_profil(), outputMode="RGB")
            except (ImageCms.PyCMSError, OSError, ValueError):
                pass                          # unlesbares Profil: wie sRGB behandeln
        pixel = np.asarray(bild.convert("RGB"))
    return Bilddaten(np.ascontiguousarray(pixel), profil, alpha,
                     _exif_ohne_ausrichtung(exif), pfad)


def laden(pfad: str, raw_auf_gpu: bool = True) -> Bilddaten:
    """Laedt ein Bild. raw_auf_gpu=False laesst auch Bayer-RAWs von LibRaw entwickeln."""
    endung = os.path.splitext(pfad)[1].lower()
    if endung in RAW:
        return _raw_laden(pfad, raw_auf_gpu)
    if endung in (".tif", ".tiff"):
        daten = _tiff_laden(pfad)
        if daten is not None:
            return daten
    if endung == ".png" and _png_bittiefe(pfad) == 16:
        return _png16_laden(pfad)
    return _pillow_laden(pfad)


# --------------------------------------------------------------------------
# Speichern
# --------------------------------------------------------------------------

def _png_block(art: bytes, daten: bytes) -> bytes:
    pruefsumme = zlib.crc32(art + daten) & 0xFFFFFFFF
    return struct.pack(">I", len(daten)) + art + daten + struct.pack(">I", pruefsumme)


def _png16_schreiben(pfad: str, pixel: np.ndarray, exif: bytes) -> None:
    """16-Bit-PNG mit sRGB-Profil (iCCP) und EXIF (eXIf) direkt hinter dem Kopf."""
    import imagecodecs
    roh = imagecodecs.png_encode(pixel, level=6)
    kopf_ende = 8 + 25                        # Signatur + IHDR-Block
    zusatz = _png_block(b"iCCP", b"sRGB\x00\x00" + zlib.compress(_srgb_profil().tobytes()))
    if exif:
        zusatz += _png_block(b"eXIf", exif[6:] if exif.startswith(b"Exif\x00\x00") else exif)
    with open(pfad, "wb") as datei:
        datei.write(roh[:kopf_ende] + zusatz + roh[kopf_ende:])


def _tiff16_schreiben(pfad: str, pixel: np.ndarray) -> None:
    import tifffile
    zusatz = {"extrasamples": [2]} if pixel.shape[2] == 4 else {}   # 2 = freier Alphakanal
    tifffile.imwrite(pfad, pixel, photometric="rgb", compression="zlib",
                     iccprofile=_srgb_profil().tobytes(), **zusatz)


def alpha_umrechnen(alpha: np.ndarray, bits: int) -> np.ndarray:
    if bits == 16 and alpha.dtype == np.uint8:
        return alpha.astype(np.uint16) * 257
    if bits == 8 and alpha.dtype == np.uint16:
        return ((alpha.astype(np.uint32) + 128) // 257).astype(np.uint8)
    return alpha


def speichern(pfad: str, rgb: np.ndarray, alpha: np.ndarray | None = None,
              exif: bytes = b"") -> None:
    """Speichert rgb (uint8 oder uint16, sRGB) im Format der Dateiendung."""
    endung = os.path.splitext(pfad)[1].lower()
    if endung not in SCHREIBBAR:
        raise BildFehler(f"unbekanntes Format: {endung}")
    format_name, kann_alpha, kann_16 = SCHREIBBAR[endung]
    bits = 16 if rgb.dtype == np.uint16 else 8
    if bits == 16 and not kann_16:
        raise BildFehler(f"{format_name} kann keine 16 Bit speichern")

    if alpha is not None and kann_alpha:
        rgb = np.dstack([rgb, alpha_umrechnen(alpha, bits)])

    try:
        if bits == 16:
            if format_name == "PNG":
                _png16_schreiben(pfad, rgb, exif)
            else:
                _tiff16_schreiben(pfad, rgb)
            return
        bild = Image.fromarray(rgb, "RGBA" if rgb.shape[2] == 4 else "RGB")
        optionen = {"icc_profile": _srgb_profil().tobytes()}
        if exif:
            optionen["exif"] = exif
        if format_name == "JPEG":
            optionen.update(quality=JPEG_QUALITAET, subsampling=0)
        elif format_name == "WEBP":
            optionen.update(quality=JPEG_QUALITAET)
        elif format_name == "TIFF":
            optionen.update(compression="tiff_lzw")
        bild.save(pfad, format_name, **optionen)
    except (OSError, ValueError) as fehler:
        raise BildFehler(str(fehler)) from fehler
