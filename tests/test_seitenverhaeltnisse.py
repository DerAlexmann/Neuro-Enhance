"""Seitenverhaeltnisse fuer den Zuschnitt."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("cupy")

from silberkorn.bearbeiten_seite import seitenverhaeltnisse  # noqa: E402


def test_feste_formate_stimmen_mit_ihrer_beschriftung():
    for text, wert in seitenverhaeltnisse():
        if ":" in text:
            breite, hoehe = (int(z) for z in text.split(":"))
            assert wert == pytest.approx(breite / hoehe), text


def test_jedes_format_gibt_es_quer_und_hoch():
    formate = {text for text, _wert in seitenverhaeltnisse() if ":" in text}
    for text in formate:
        breite, hoehe = text.split(":")
        assert f"{hoehe}:{breite}" in formate, text


def test_instagram_hochformat():
    assert dict(seitenverhaeltnisse())["4:5"] == pytest.approx(0.8)
