"""
Einstellungen: eine JSON-Datei neben dem Programm

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import json
import locale
import os
import sys

from . import farben
from .uebersetzung import LANGUAGE_NAMES, SOURCE_LANGUAGE, TRANSLATIONS

CONFIG_NAME = "neuro-enhance.json"


def ist_eingefroren() -> bool:
    """Laeuft das Programm als gebuendelte EXE (PyInstaller & Co.)?"""
    return getattr(sys, "frozen", False)


def programm_ordner() -> str:
    """Ordner, in dem das Programm fuer den Anwender sichtbar liegt.

    Als PyInstaller-EXE zeigt __file__ in den Entpackordner, der den
    Programmlauf nicht ueberdauert. Alles Bleibende gehoert deshalb neben die
    EXE. Als Skript ist es der Ordner ueber dem Paket, also der, in dem
    Neuro-Enhance.pyw liegt.
    """
    if ist_eingefroren():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def config_path() -> str:
    return os.path.join(programm_ordner(), CONFIG_NAME)


def load_config() -> dict:
    try:
        with open(config_path(), encoding="utf-8") as datei:
            daten = json.load(datei)
        return daten if isinstance(daten, dict) else {}
    except (OSError, ValueError):
        return {}


def save_config(daten: dict) -> bool:
    try:
        with open(config_path(), "w", encoding="utf-8") as datei:
            json.dump(daten, datei, indent=2)
        return True
    except OSError:
        return False


def detect_language() -> str:
    """Sprache des Betriebssystems, falls dafuer eine Tabelle vorliegt."""
    try:
        locale.setlocale(locale.LC_CTYPE, "")
        code = (locale.getlocale()[0] or "").lower()
    except (locale.Error, ValueError):
        code = ""
    bekannt = {SOURCE_LANGUAGE, *TRANSLATIONS}
    kurz = code.split("_")[0]
    if kurz in bekannt:
        return kurz
    for name, sprache in (("german", "de"), ("deutsch", "de"), ("english", "en")):
        if kurz.startswith(name) and sprache in bekannt:
            return sprache
    return SOURCE_LANGUAGE


def startup_language() -> str:
    """Gespeicherte Sprache, sonst die des Betriebssystems."""
    gespeichert = load_config().get("language")
    if gespeichert in LANGUAGE_NAMES and (gespeichert == SOURCE_LANGUAGE
                                          or gespeichert in TRANSLATIONS):
        return gespeichert
    return detect_language()


def raw_qualitaet() -> str:
    """Gespeicherte Wahl fuer die RAW-Entwicklung: "schnell" (GPU) oder "beste" (LibRaw)."""
    wert = load_config().get("raw_qualitaet")
    return wert if wert in ("schnell", "beste") else "schnell"


def startup_theme() -> str:
    """Gespeichertes Farbschema, sonst das helle."""
    gespeichert = load_config().get("theme")
    return gespeichert if gespeichert in farben.THEMES else farben.DEFAULT_THEME
