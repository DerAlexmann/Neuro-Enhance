"""Jeder Text, der durch _() laeuft, braucht einen Eintrag in jeder Sprache.

Die Texte werden aus dem Quelltext gelesen (ast), nicht durch Ausfuehren -
der Test braucht also weder Qt noch eine Grafikkarte.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from silberkorn.uebersetzung import TRANSLATIONS

PAKET = Path(__file__).resolve().parent.parent / "silberkorn"


def texte_im_quelltext() -> set[str]:
    texte = set()
    for datei in PAKET.glob("*.py"):
        baum = ast.parse(datei.read_text(encoding="utf-8"))
        for knoten in ast.walk(baum):
            if (isinstance(knoten, ast.Call) and isinstance(knoten.func, ast.Name)
                    and knoten.func.id == "_" and knoten.args
                    and isinstance(knoten.args[0], ast.Constant)
                    and isinstance(knoten.args[0].value, str)):
                texte.add(knoten.args[0].value)
    return texte


def test_texte_gefunden():
    assert len(texte_im_quelltext()) > 20


@pytest.mark.parametrize("sprache", sorted(TRANSLATIONS))
def test_alle_texte_uebersetzt(sprache):
    fehlend = sorted(texte_im_quelltext() - set(TRANSLATIONS[sprache]))
    assert not fehlend, f"In '{sprache}' fehlen: {fehlend}"


@pytest.mark.parametrize("sprache", sorted(TRANSLATIONS))
def test_keine_verwaisten_eintraege(sprache):
    verwaist = sorted(set(TRANSLATIONS[sprache]) - texte_im_quelltext())
    assert not verwaist, f"In '{sprache}' unbenutzt: {verwaist}"


@pytest.mark.parametrize("sprache", sorted(TRANSLATIONS))
def test_platzhalter_gleich(sprache):
    for quelle, ziel in TRANSLATIONS[sprache].items():
        assert set(re.findall(r"\{\w+\}", quelle)) == set(re.findall(r"\{\w+\}", ziel)), quelle
