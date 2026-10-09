"""
Programmstart: Grafikkarte pruefen, dann das Hauptfenster oeffnen

Ohne geeignete NVIDIA-RTX-Karte gibt es keinen Notbetrieb: Die App meldet
klar, was gefunden wurde und was fehlt, und beendet sich.

Zum Ansehen der Startmeldungen ohne passende Hardware:
    python Silberkorn.pyw --simulieren=keine_rtx
(ebenso kein_treiber, keine_nvidia, treiber_alt, wenig_vram)

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import os
import sys
import traceback

from . import PROGRAMM, VERSION, einstellungen, farben, gpu_pruefung, nvidia_laufzeit
from .gpu_pruefung import Befund, Grafikkarte
from .uebersetzung import _

TREIBER_SEITE = "https://www.nvidia.com/drivers"

# Rueckgabewerte des Programms
ENDE_OK = 0
ENDE_OHNE_QT = 2
ENDE_OHNE_RTX = 3
ENDE_OHNE_NVIDIA_BIBLIOTHEKEN = 4
ENDE_STARTFEHLER = 5

FEHLER_LOG = "silberkorn-fehler.log"


def meldungstext(befund: Befund) -> tuple[str, str]:
    """Titel und Text der Startmeldung zu einem Befund."""
    karte = befund.karte.name if befund.karte else ""
    if befund.grund == gpu_pruefung.KEIN_TREIBER:
        return (_("Keine NVIDIA-Grafikkarte gefunden"),
                _("Silberkorn benötigt eine NVIDIA-RTX-Grafikkarte (ab der RTX-2000-Serie) "
                  "mit installiertem NVIDIA-Treiber. Auf diesem Rechner wurde weder eine "
                  "NVIDIA-Grafikkarte noch ein NVIDIA-Treiber gefunden."))
    if befund.grund == gpu_pruefung.KEINE_NVIDIA:
        return (_("Keine NVIDIA-Grafikkarte gefunden"),
                _("Der NVIDIA-Treiber ist installiert, meldet aber keine Grafikkarte. "
                  "Silberkorn benötigt eine NVIDIA-RTX-Grafikkarte (ab der RTX-2000-Serie)."))
    if befund.grund == gpu_pruefung.KEINE_RTX:
        return (_("Keine RTX-Grafikkarte gefunden"),
                _("Gefunden: {karte}. Silberkorn benötigt eine RTX-Grafikkarte mit "
                  "Tensor Cores (ab der RTX-2000-Serie). GTX-Karten und ältere Modelle "
                  "werden nicht unterstützt.").format(karte=karte))
    if befund.grund == gpu_pruefung.TREIBER_ALT:
        return (_("Grafiktreiber zu alt"),
                _("Gefunden: {karte} mit Treiber {treiber}. Silberkorn benötigt den "
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


_qt_uebersetzer = None


def qt_sprache_setzen(sprache: str) -> None:
    """Qts eigene Texte (Ja/Nein/Abbrechen, Kontextmenues) in der Programmsprache.

    Ohne das bleiben sie englisch, auch wenn das Programm deutsch spricht. Die
    Dateien qtbase_<sprache>.qm bringt PySide6 mit; fuer Englisch braucht es keine.
    """
    global _qt_uebersetzer
    from PySide6.QtCore import QLibraryInfo, QTranslator
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if _qt_uebersetzer is not None:
        app.removeTranslator(_qt_uebersetzer)
        _qt_uebersetzer = None
    uebersetzer = QTranslator(app)
    ordner = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    if uebersetzer.load(f"qtbase_{sprache}", ordner):
        app.installTranslator(uebersetzer)
        _qt_uebersetzer = uebersetzer


def startfehler_zeigen(titel: str, details: str) -> int:
    """Fehler nach dem Laden von Qt: Fenster mit Details, dazu eine Logdatei
    neben dem Programm - die EXE hat keine Konsole."""
    from PySide6.QtWidgets import QMessageBox

    log = os.path.join(einstellungen.programm_ordner(), FEHLER_LOG)
    try:
        with open(log, "w", encoding="utf-8") as datei:
            datei.write(f"{PROGRAMM} {VERSION}\n{titel}\n\n{details}\n")
    except OSError:
        log = ""
    box = QMessageBox(QMessageBox.Icon.Critical, f"{PROGRAMM} {VERSION}", f"<b>{titel}</b>")
    box.setInformativeText(
        _("Die Einzelheiten stehen unter „Details“ und in {log}.").format(log=log) if log
        else _("Die Einzelheiten stehen unter „Details“."))
    box.setDetailedText(details)
    box.exec()
    return ENDE_STARTFEHLER


def main(argumente: list[str] | None = None) -> int:
    argumente = sys.argv[1:] if argumente is None else argumente

    # NVML zaehlt die Karten nach PCI-Bus, CUDA von sich aus nach Leistung.
    # Mit dieser Einstellung meinen beide mit derselben Nummer dieselbe Karte.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    try:
        from PySide6.QtCore import Qt
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
    # Windows blendet Aufklapplisten sonst ein und malt dabei kurz den eigenen
    # Hintergrund - im hellen Schema ein sichtbares Flackern
    app.setEffectEnabled(Qt.UIEffect.UI_AnimateCombo, False)

    _.language = einstellungen.startup_language()
    qt_sprache_setzen(_.language)
    farben.apply_theme(einstellungen.startup_theme())
    app.setStyleSheet(farben.stylesheet())
    farben.palette_setzen(app)
    from . import mausrad
    app.mausrad = mausrad.einrichten(app)

    befund = befund_holen(argumente)
    if not befund.ok:
        return ohne_rtx_beenden(befund)

    # Die EXE bringt CUDA und cuDNN nicht mit - beim ersten Start laden
    if nvidia_laufzeit.noetig() and nvidia_laufzeit.gefunden() is None:
        from .einrichten import nvidia_einrichten
        if not nvidia_einrichten():
            return ENDE_OHNE_NVIDIA_BIBLIOTHEKEN

    from . import cuda
    if cuda.cupy is None:
        return startfehler_zeigen(_("CuPy ließ sich nicht laden"), cuda.fehler or "")
    try:
        from .hauptfenster import Hauptfenster
        fenster = Hauptfenster(befund)
    except Exception:
        return startfehler_zeigen(_("Das Hauptfenster ließ sich nicht öffnen"),
                                  traceback.format_exc())
    fenster.show()
    app.exec()
    return ENDE_OK
