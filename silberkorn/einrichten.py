"""
Erster Start der EXE: NVIDIA-Bibliotheken nach Zustimmung laden

Fragt mit Quelle, Groesse, Ablage und NVIDIAs Lizenzen nach, laedt dann mit
Fortschrittsanzeige (nvidia_laufzeit.einrichten) und meldet Fehler. Ohne
Zustimmung oder nach einem Fehler beendet sich das Programm; beim naechsten
Start wird erneut gefragt.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import html

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from . import PROGRAMM, VERSION, nvidia_laufzeit
from .uebersetzung import _


def _gb(zahl: int) -> str:
    return f"{zahl / 1e9:.1f}".replace(".", "," if _.language == "de" else ".")


def zustimmung() -> bool:
    ziel = nvidia_laufzeit.ziel_ordner()
    lizenzen = "<br>".join(f'<a href="{url}">{name}</a>' for name, url in nvidia_laufzeit.LIZENZEN)
    box = QMessageBox()
    box.setIcon(QMessageBox.Icon.Information)
    box.setWindowTitle(f"{PROGRAMM} {VERSION}")
    box.setTextFormat(Qt.TextFormat.RichText)
    box.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    box.setText("<b>" + _("NVIDIA-Bibliotheken einrichten") + "</b>")
    box.setInformativeText(
        _("Silberkorn rechnet mit CUDA und cuDNN von NVIDIA. Diese Bibliotheken "
          "werden nicht mit dem Programm ausgeliefert, sondern einmalig aus NVIDIAs "
          "offiziellen Paketen vom Python Package Index (pypi.org) geladen – "
          "{laden} GB, ausgepackt rund {platz} GB. Jede Datei wird gegen ihre "
          "Prüfsumme geprüft.").format(laden=_gb(nvidia_laufzeit.download_groesse()),
                                       platz=_gb(nvidia_laufzeit.AUSGEPACKT))
        + "<br><br>"
        + _("Für diese Bibliotheken gelten die Lizenzbedingungen von NVIDIA:")
        + "<br>" + lizenzen + "<br><br>"
        + _("Mit „Zustimmen und herunterladen“ werden sie anerkannt.")
        + "<br><br>" + _("Ablage: {ordner}").format(ordner=html.escape(ziel)))
    ja = box.addButton(_("Zustimmen und herunterladen"), QMessageBox.ButtonRole.AcceptRole)
    box.addButton(_("Beenden"), QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(ja)
    box.exec()
    return box.clickedButton() is ja


def nvidia_einrichten() -> bool:
    """True, wenn die Bibliotheken danach bereitstehen."""
    if not zustimmung():
        return False
    anzeige = QProgressDialog(_("NVIDIA-Bibliotheken werden heruntergeladen …"),
                              _("Abbrechen"), 0, 1000)
    anzeige.setWindowTitle(f"{PROGRAMM} {VERSION}")
    anzeige.setWindowModality(Qt.WindowModality.ApplicationModal)
    anzeige.setMinimumDuration(0)
    anzeige.setMinimumWidth(420)

    def fortschritt(geladen, gesamt):
        anzeige.setValue(round(1000 * geladen / max(gesamt, 1)))
        anzeige.setLabelText(_("NVIDIA-Bibliotheken werden heruntergeladen …")
                             + f"\n{geladen / 2**20:,.0f} / {gesamt / 2**20:,.0f} MB"
                             .replace(",", "."))
        QApplication.processEvents()
        return not anzeige.wasCanceled()

    try:
        ordner = nvidia_laufzeit.einrichten(fortschritt)
    except nvidia_laufzeit.Abbruch:
        return False
    except nvidia_laufzeit.LaufzeitFehler as fehler:
        anzeige.close()
        box = QMessageBox(QMessageBox.Icon.Critical, f"{PROGRAMM} {VERSION}",
                          _("Die NVIDIA-Bibliotheken ließen sich nicht einrichten. "
                            "Beim nächsten Start versucht Silberkorn es erneut."))
        box.setDetailedText(str(fehler))
        box.exec()
        return False
    finally:
        anzeige.close()
    nvidia_laufzeit.aktivieren(ordner)
    return True
