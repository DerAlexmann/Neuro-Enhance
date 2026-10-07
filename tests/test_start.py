"""Startmeldungen: Jeder Ablehnungsgrund hat einen Titel und einen Text."""

from __future__ import annotations

import pytest

from silberkorn import gpu_pruefung as g
from silberkorn import start
from silberkorn.uebersetzung import _


@pytest.fixture(params=["de", "en"])
def sprache(request):
    vorher = _.language
    _.language = request.param
    yield request.param
    _.language = vorher


@pytest.mark.parametrize("grund", ["kein_treiber", "keine_nvidia", "keine_rtx", "treiber_alt"])
def test_jede_ablehnung_hat_eine_meldung(grund, sprache):
    befund = start.simulierter_befund(grund)
    assert not befund.ok
    titel, text = start.meldungstext(befund)
    assert titel and text
    assert "{" not in text                   # alle Platzhalter eingesetzt


def test_treiber_meldung_nennt_beide_versionen():
    titel, text = start.meldungstext(start.simulierter_befund("treiber_alt"))
    assert "552.22" in text
    assert g.treiber_anzeige() in text


def test_wenig_vram_startet():
    befund = start.simulierter_befund("wenig_vram")
    assert befund.ok
    assert befund.stufe == g.STUFE_OHNE_KI
