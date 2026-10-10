"""Gemeinsame Vorbereitung der Tests.

Ohne Bildschirm (QT_QPA_PLATFORM=offscreen) findet Qt die Windows-Schriften nicht
und setzt eine Ersatzschrift - Texte werden dann breiter als im Programm, und
Messungen der Oberflaeche stimmen nicht. Unter Windows bekommt Qt deshalb den
Schriftordner, bevor irgendein Test eine QApplication anlegt.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

import os

if os.name == "nt":
    _schriften = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    if os.path.isdir(_schriften):
        os.environ.setdefault("QT_QPA_FONTDIR", _schriften)
