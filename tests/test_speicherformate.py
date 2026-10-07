"""Formatwahl im Speichern-Dialog - braucht Qt und CuPy, laeuft daher nur lokal."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("cupy")

from silberkorn.bearbeiten_seite import speicherformate, ziel_bestimmen  # noqa: E402


def filter_mit(endung, bits):
    return next(b for b, e, x in speicherformate() if e == endung and x == bits)


def test_endung_wird_ergaenzt():
    assert ziel_bestimmen("C:/bild", filter_mit(".tif", 16)) == ("C:/bild.tif", 16)
    assert ziel_bestimmen("C:/bild", filter_mit(".png", 8)) == ("C:/bild.png", 8)


def test_passende_endung_behaelt_bits():
    assert ziel_bestimmen("C:/bild.tiff", filter_mit(".tif", 16)) == ("C:/bild.tiff", 16)
    assert ziel_bestimmen("C:/bild.jpeg", speicherformate()[0][0]) == ("C:/bild.jpeg", 8)


def test_getippte_endung_gewinnt():
    # PNG 16 Bit gewaehlt, aber .jpg getippt: JPEG kann nur 8 Bit
    assert ziel_bestimmen("C:/bild.jpg", filter_mit(".png", 16)) == ("C:/bild.jpg", 8)
    # TIFF 16 Bit gewaehlt, .png getippt: PNG kann 16 Bit
    assert ziel_bestimmen("C:/bild.png", filter_mit(".tif", 16)) == ("C:/bild.png", 16)
