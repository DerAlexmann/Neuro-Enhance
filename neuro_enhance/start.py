"""
Programmstart: Grafikkarte pruefen, dann das Hauptfenster oeffnen

Ohne geeignete NVIDIA-RTX-Karte gibt es keinen Notbetrieb: Die App meldet
klar, was gefunden wurde und was fehlt, und beendet sich.

Zum Ansehen der Startmeldungen ohne passende Hardware:
    python Neuro-Enhance.pyw --simulieren=keine_rtx
(ebenso kein_treiber, keine_nvidia, treiber_alt, wenig_vram)

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import os
import sys

from . import PROGRAMM, VERSION, einstellungen, farben, gpu_pruefung
from .gpu_pruefung import Befund, Grafikkarte
from .uebersetzung import _

TREIBER_SEITE = "https://www.nvidia.com/drivers"

# Rueckgabewerte des Programms
ENDE_OK = 0
ENDE_OHNE_QT = 2
ENDE_OHNE_RTX = 3


def meldungstext(befund: Befund) -> tuple[str, str]:
    """Titel und Text der Startmeldung zu einem Befund."""
    karte = befund.karte.name if befund.karte else ""
    if befund.grund == gpu_pruefung.KEIN_TREIBER:
        return (_("Keine NVIDIA-Grafikkarte gefunden"),
                _("Neuro-Enhance benötigt eine NVIDIA-RTX-Grafikkarte (ab der RTX-2000-Serie) "
                  "mit installiertem NVIDIA-Treiber. Auf diesem Rechner wurde weder eine "
                  "NVIDIA-Grafikkarte noch ein NVIDIA-Treiber gefunden."))
    if befund.grund == gpu_pruefung.KEINE_NVIDIA:
        return (_("Keine NVIDIA-Grafikkarte gefunden"),
                _("Der NVIDIA-Treiber ist installiert, meldet aber keine Grafikkarte. "
                  "Neuro-Enhance benötigt eine NVIDIA-RTX-Grafikkarte (ab der RTX-2000-Serie)."))
    if befund.grund == gpu_pruefung.KEINE_RTX:
        return (_("Keine RTX-Grafikkarte gefunden"),
                _("Gefunden: {karte}. Neuro-Enhance benötigt eine RTX-Grafikkarte mit "
                  "Tensor Cores (ab der RTX-2000-Serie). GTX-Karten und ältere Modelle "
                  "werden nicht unterstützt.").format(karte=karte))
    if befund.grund == gpu_pruefung.TREIBER_ALT:
        return (_("Grafiktreiber zu alt"),
                _("Gefunden: {karte} mit Treiber {treiber}. Neuro-Enhance benötigt den "
                  "NVIDIA-Treiber {mindestens} oder neuer. Bitte den Grafiktreiber "
                  "aktualisieren und das Programm danach erneut starten.")
                .format(karte=karte, treiber=befund.treiber,
                        mindestens=gpu_pruefung.treiber_anzeige()))
    return ("", "")


def simulierter_befund(grund: str) -> Befund:
    """Erfundene Befunde, um die Startmeldungen ohne passende Hardware zu sehen."""
    gtx = Grafikkarte(0, "NVIDIA GeForce GTX 1660 SUPER", 6 * 1024 ** 3, (7, 5))
    rtx = Grafikkarte(0, "NVIDIA GeForce RTX 3050 Laptop GPU", 4 * 1024 ** 3, (8, 6))
    klein = Grafikkarte(0, "NVIDIA GeForce RTX 2050", 2 * 1024 ** 3, (8, 6))
    if grund == "keine_rtx":
        return gpu_pruefung.bewerten([gtx], "617.14")
    if grund == "treiber_alt":
        return gpu_pruefung.bewerten([rtx], "552.22")
    if grund == "keine_nvidia":
        return gpu_pruefung.bewerten([], "617.14")
    if grund == "wenig_vram":
        return gpu_pruefung.bewerten([klein], "617.14")
    return Befund(gpu_pruefung.KEIN_TREIBER, fehler="simuliert")


def befund_holen(argumente: list[str]) -> Befund:
    for argument in argumente:
        if argument.startswith("--simulieren="):
            return simulierter_befund(argument.split("=", 1)[1])
    return gpu_pruefung.ermitteln()


def startfehler_melden(text: str) -> None:
    """Fehler vor dem Laden von Qt: Konsole und - wenn moeglich - ein Tk-Fenster.

    Unter pythonw.exe gibt es keine Konsole, sys.stderr ist dann None.
    """
    if sys.stderr is not None:
        print(text, file=sys.stderr)
    try:
        import tkinter as tk
        from tkinter import messagebox
        wurzel = tk.Tk()
        wurzel.withdraw()
        messagebox.showerror(f"{PROGRAMM} {VERSION}", text)
        wurzel.destroy()
    except Exception:                       # ohne Tkinter bleibt nur die Konsole
        pass


def ohne_rtx_beenden(befund: Befund) -> int:
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import QMessageBox

    titel, text = meldungstext(befund)
    box = QMessageBox()
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle(f"{PROGRAMM} {VERSION}")
    box.setText(f"<b>{titel}</b>")
    box.setInformativeText(
        text + "\n\n" + _("Die vollständigen Systemvoraussetzungen stehen in der README."))
    if befund.fehler:
        box.setDetailedText(befund.fehler)
    treiber_knopf = None
    if befund.grund == gpu_pruefung.TREIBER_ALT:
        treiber_knopf = box.addButton(_("Treiber herunterladen"),
                                      QMessageBox.ButtonRole.ActionRole)
    box.addButton(_("Beenden"), QMessageBox.ButtonRole.RejectRole)
    box.exec()
    if treiber_knopf is not None and box.clickedButton() is treiber_knopf:
        QDesktopServices.openUrl(QUrl(TREIBER_SEITE))
    return ENDE_OHNE_RTX


def main(argumente: list[str] | None = None) -> int:
    argumente = sys.argv[1:] if argumente is None else argumente

    # NVML zaehlt die Karten nach PCI-Bus, CUDA von sich aus nach Leistung.
    # Mit dieser Einstellung meinen beide mit derselben Nummer dieselbe Karte.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    try:
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import QApplication
    except ImportError:
        startfehler_melden("PySide6 fehlt: pip install -r requirements.txt\n"
                           "PySide6 is missing: pip install -r requirements.txt")
        return ENDE_OHNE_QT

    app = QApplication(sys.argv[:1])
    app.setApplicationName(PROGRAMM)
    app.setApplicationVersion(VERSION)
    app.setFont(QFont(*farben.FONT[:2]))

    _.language = einstellungen.startup_language()
    farben.apply_theme(einstellungen.startup_theme())
    app.setStyleSheet(farben.stylesheet())

    befund = befund_holen(argumente)
    if not befund.ok:
        return ohne_rtx_beenden(befund)

    from .hauptfenster import Hauptfenster
    fenster = Hauptfenster(befund)
    fenster.show()
    app.exec()
    return ENDE_OK
