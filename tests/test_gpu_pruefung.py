"""Tests der Startpruefung - ohne Grafikkarte, nur die Bewertung."""

from __future__ import annotations

import pytest

from silberkorn import gpu_pruefung as g

GIB = 1024 ** 3


def karte(name, vram_gib, cc, index=0):
    return g.Grafikkarte(index, name, int(vram_gib * GIB), cc)


@pytest.mark.parametrize("name, cc, erwartet", [
    ("NVIDIA GeForce RTX 4060", (8, 9), True),
    ("NVIDIA GeForce RTX 2060", (7, 5), True),
    ("NVIDIA GeForce RTX 5090", (12, 0), True),
    ("NVIDIA RTX A2000", (8, 6), True),
    ("NVIDIA RTX 4000 Ada Generation", (8, 9), True),
    ("NVIDIA GeForce GTX 1660 SUPER", (7, 5), False),     # Turing ohne Tensor Cores
    ("NVIDIA GeForce GTX 1080", (6, 1), False),
    ("Quadro RTX 4000", (7, 5), True),
    ("NVIDIA GeForce RTX 9999 Fake", (6, 1), False),      # Name allein genuegt nicht
    ("NVIDIA GeForce RTXA", (8, 6), False),               # RTX nur als eigenes Wort
])
def test_rtx_erkennung(name, cc, erwartet):
    assert karte(name, 8, cc).ist_rtx is erwartet


@pytest.mark.parametrize("vram, stufe", [
    (2, g.STUFE_OHNE_KI),
    (3.9, "S"),         # 4-GB-Karten melden etwas weniger
    (5.8, "M"),
    (7.6, "M"),
    (11.6, "L"),
    (15.9, "L"),
    (23.6, "XL"),
    (31.8, "XL"),
])
def test_stufen(vram, stufe):
    assert g.stufe_fuer_vram(int(vram * GIB)) == stufe


@pytest.mark.parametrize("text, tupel", [
    ("617.14", (617, 14)),
    ("570.65", (570, 65)),
    ("552", (552, 0)),
    ("", (0, 0)),
    ("unbekannt", (0, 0)),
])
def test_treiber_tupel(text, tupel):
    assert g.treiber_tupel(text) == tupel


def test_keine_karten():
    assert g.bewerten([], "617.14").grund == g.KEINE_NVIDIA


def test_nur_gtx():
    befund = g.bewerten([karte("NVIDIA GeForce GTX 1660", 6, (7, 5))], "617.14")
    assert befund.grund == g.KEINE_RTX
    assert befund.karte.name == "NVIDIA GeForce GTX 1660"


def test_treiber_zu_alt():
    befund = g.bewerten([karte("NVIDIA GeForce RTX 3060", 12, (8, 6))], "570.64")
    assert befund.grund == g.TREIBER_ALT
    assert not befund.ok


def test_mindesttreiber_genuegt():
    assert g.bewerten([karte("NVIDIA GeForce RTX 3060", 12, (8, 6))], "570.65").ok


def test_beste_karte_nach_vram():
    karten = [
        karte("NVIDIA GeForce GTX 1080 Ti", 11, (6, 1), 0),
        karte("NVIDIA GeForce RTX 4060", 8, (8, 9), 1),
        karte("NVIDIA GeForce RTX 3090", 24, (8, 6), 2),
    ]
    befund = g.bewerten(karten, "617.14")
    assert befund.ok
    assert befund.karte.index == 2
    assert befund.stufe == "XL"


@pytest.mark.parametrize("cc, fp8, fp4", [
    ((7, 5), False, False),
    ((8, 6), False, False),
    ((8, 9), True, False),
    ((12, 0), True, True),
])
def test_rechengenauigkeit(cc, fp8, fp4):
    befund = g.bewerten([karte("NVIDIA GeForce RTX X", 8, cc)], "617.14")
    assert (befund.fp8, befund.fp4) == (fp8, fp4)


def test_kleine_karte_startet_ohne_ki():
    befund = g.bewerten([karte("NVIDIA GeForce RTX 2050", 2, (8, 6))], "617.14")
    assert befund.ok
    assert befund.stufe == g.STUFE_OHNE_KI


def test_ermitteln_liefert_immer_einen_befund():
    """Auf jedem Rechner - mit oder ohne NVIDIA - darf ermitteln() nicht abstuerzen."""
    befund = g.ermitteln()
    assert befund.grund in {g.OK, g.KEIN_TREIBER, g.KEINE_NVIDIA, g.KEINE_RTX, g.TREIBER_ALT}
