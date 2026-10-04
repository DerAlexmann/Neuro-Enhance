"""
Hauptfenster: Kopfzeile, Reiter und Statuszeile

Farben setzt allein das Stylesheet aus farben.py; die Widgets tragen dafuer
nur Objektnamen. Beschriftungen laufen ueber beschriften(), damit sie einen
Sprachwechsel ohne Neuaufbau des Fensters mitmachen.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import base64
import os
import platform
import sys

from PySide6 import __version__ as PYSIDE_VERSION
from PySide6.QtCore import QByteArray, Qt, qVersion
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import PROGRAMM, VERSION, einstellungen, farben
from .gpu_pruefung import STUFE_OHNE_KI, Befund
from .uebersetzung import SOURCE_LANGUAGE, _

SYMBOL_NAME = "neuro_enhance.ico"


def symbol_pfad() -> str:
    """Programmsymbol: als EXE aus der EXE selbst, sonst die .ico im Projekt."""
    if einstellungen.ist_eingefroren():
        return sys.executable
    return os.path.join(einstellungen.programm_ordner(), SYMBOL_NAME)


def stufe_beschreibung(stufe: str) -> str:
    texte = {
        STUFE_OHNE_KI: _("Nur klassische Filter – für KI-Funktionen sind mindestens "
                         "4 GB Grafikspeicher nötig."),
        "S": _("Klassische Filter und kleine KI-Modelle, in kleinen Kacheln gerechnet."),
        "M": _("Klassische Filter und alle Restaurierungsmodelle: Hochskalieren, "
               "Entrauschen, Freistellen und Objektauswahl."),
        "L": _("Zusätzlich größere Modelle, größere Kacheln und generative Füllung."),
        "XL": _("Voller Umfang, auch große generative Modelle und mehrere Modelle "
                "gleichzeitig im Speicher."),
    }
    return texte[stufe]


def reitertext(text: str) -> str:
    """Qt liest ein einzelnes & in Reitertiteln als Tastenkuerzel und verschluckt es."""
    return text.replace("&", "&&")


def genauigkeit_anzeige(befund: Befund) -> str:
    formate = ["FP16"]
    if befund.fp8:
        formate.append("FP8")
    if befund.fp4:
        formate.append("FP4")
    return " · ".join(formate)


def vram_anzeige(befund: Befund) -> str:
    if not befund.karte:
        return "–"
    return f"{befund.karte.vram_gib:.1f} GiB"


class Hauptfenster(QMainWindow):
    def __init__(self, befund: Befund):
        super().__init__()
        self.befund = befund
        self._beschriftungen = []           # (setzen, Uebersetzt) fuer den Sprachwechsel

        self.setMinimumSize(900, 620)
        if os.path.exists(symbol_pfad()):
            self.setWindowIcon(QIcon(symbol_pfad()))
        self.setWindowTitle(f"{PROGRAMM} {VERSION}")

        seite = QWidget(objectName="seite")
        aufbau = QVBoxLayout(seite)
        aufbau.setContentsMargins(0, 0, 0, 0)
        aufbau.setSpacing(0)
        aufbau.addWidget(self._kopfzeile())

        self.reiter = QTabWidget()
        self.reiter.setDocumentMode(True)
        innen = QWidget(objectName="seite")
        innen_aufbau = QVBoxLayout(innen)
        innen_aufbau.setContentsMargins(16, 12, 16, 12)
        innen_aufbau.addWidget(self.reiter)
        aufbau.addWidget(innen, 1)

        self._reiter_bearbeiten()
        self._reiter_info()
        self._statuszeile()
        self.setCentralWidget(seite)
        self._fensterlage_laden()

    # ------------------------------------------------------------------
    # Beschriftungen
    # ------------------------------------------------------------------

    def beschriften(self, setzen, text):
        """Setzt eine Beschriftung und merkt sie fuer den Sprachwechsel."""
        setzen(text)
        self._beschriftungen.append((setzen, text))

    def texte_auffrischen(self):
        for setzen, text in self._beschriftungen:
            neu = _(text.schluessel)
            setzen(neu.format(**text.werte) if text.werte else neu)
        for nummer, text in enumerate(self._reitertitel):
            self.reiter.setTabText(nummer, reitertext(_(text.schluessel)))

    def _label(self, text="", name=None, umbruch=False) -> QLabel:
        label = QLabel()
        if name:
            label.setObjectName(name)
        if umbruch:
            label.setWordWrap(True)
        if text:
            if hasattr(text, "schluessel"):
                self.beschriften(label.setText, text)
            else:
                label.setText(text)
        return label

    def _karte(self, titel=None) -> tuple[QFrame, QVBoxLayout]:
        karte = QFrame(objectName="karte")
        aufbau = QVBoxLayout(karte)
        aufbau.setContentsMargins(18, 14, 18, 16)
        aufbau.setSpacing(6)
        if titel:
            aufbau.addWidget(self._label(titel, "kartentitel"))
        return karte, aufbau

    # ------------------------------------------------------------------
    # Kopfzeile
    # ------------------------------------------------------------------

    def _kopfzeile(self) -> QFrame:
        kopf = QFrame(objectName="kopf")
        aufbau = QHBoxLayout(kopf)
        aufbau.setContentsMargins(18, 10, 18, 10)

        aufbau.addWidget(self._label(PROGRAMM, "titel"))
        aufbau.addSpacing(10)
        aufbau.addWidget(self._label(_("Version {version}").format(version=VERSION)))
        aufbau.addStretch(1)

        aufbau.addWidget(self._label(_("Sprache & Darstellung"), "kopfgruppe"))
        self.sprachwahl = QComboBox()
        for code, name in _.available().items():
            self.sprachwahl.addItem(name, code)
        self.sprachwahl.setCurrentIndex(max(0, self.sprachwahl.findData(_.language)))
        self.sprachwahl.currentIndexChanged.connect(self._sprache_gewechselt)
        aufbau.addWidget(self.sprachwahl)

        self.dunkel = QCheckBox()
        self.beschriften(self.dunkel.setText, _("Dunkel"))
        self.dunkel.setChecked(farben.CURRENT_THEME == "dark")
        self.dunkel.toggled.connect(self._schema_gewechselt)
        aufbau.addWidget(self.dunkel)
        return kopf

    def _sprache_gewechselt(self):
        _.language = self.sprachwahl.currentData() or SOURCE_LANGUAGE
        self.texte_auffrischen()
        self._einstellungen_sichern()

    def _schema_gewechselt(self, dunkel: bool):
        farben.apply_theme("dark" if dunkel else "light")
        QApplication.instance().setStyleSheet(farben.stylesheet())
        self._einstellungen_sichern()

    # ------------------------------------------------------------------
    # Reiter
    # ------------------------------------------------------------------

    def _reiterseite(self) -> tuple[QScrollArea, QVBoxLayout]:
        flaeche = QScrollArea()
        flaeche.setWidgetResizable(True)
        inhalt = QWidget(objectName="seite")
        aufbau = QVBoxLayout(inhalt)
        aufbau.setContentsMargins(0, 12, 0, 0)
        aufbau.setSpacing(12)
        flaeche.setWidget(inhalt)
        return flaeche, aufbau

    def _reiter_bearbeiten(self):
        seite, aufbau = self._reiterseite()

        leinwand = QFrame(objectName="leinwand")
        leinwand.setMinimumHeight(320)
        innen = QVBoxLayout(leinwand)
        innen.addStretch(1)
        leer = self._label(_("Noch kein Bild geöffnet."), "leer")
        leer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        innen.addWidget(leer)
        hinweis = self._label(_("Das Grundgerüst steht – die Bearbeitungsfunktionen folgen."),
                              "nebentext")
        hinweis.setAlignment(Qt.AlignmentFlag.AlignCenter)
        innen.addWidget(hinweis)
        innen.addStretch(1)
        aufbau.addWidget(leinwand, 1)

        karte, innen = self._karte(_("Grafikkarte"))
        name = self.befund.karte.name if self.befund.karte else "–"
        innen.addWidget(self._label(_("{karte} mit {vram} Grafikspeicher – Funktionsstufe {stufe}")
                                    .format(karte=name, vram=vram_anzeige(self.befund),
                                            stufe=self.befund.stufe)))
        beschreibung = self._label(name="nebentext", umbruch=True)
        self.beschriften(beschreibung.setText, stufe_beschreibung(self.befund.stufe))
        if self.befund.stufe == STUFE_OHNE_KI:
            beschreibung.setObjectName("warnung")
        innen.addWidget(beschreibung)
        aufbau.addWidget(karte)

        self.reiter.addTab(seite, "")

    def _reiter_info(self):
        seite, aufbau = self._reiterseite()

        karte, innen = self._karte(f"{PROGRAMM} {VERSION}")
        innen.addWidget(self._label(
            _("Ein quelloffener, GPU-beschleunigter KI-Bild- und Fotoverbesserer "
              "auf Basis neuronaler Netze."), umbruch=True))
        innen.addWidget(self._label("Copyright 2026 Alexander Unverhau"))
        innen.addWidget(self._label(_("Erstellt mit Unterstützung von Claude AI")))
        innen.addWidget(self._label(_("Veröffentlicht unter der MIT-Lizenz.")))
        aufbau.addWidget(karte)

        karte, innen = self._karte(_("Technisches"))
        gitter = QGridLayout()
        gitter.setHorizontalSpacing(24)
        gitter.setVerticalSpacing(4)
        k = self.befund.karte
        zeilen = [
            (_("Grafikkarte"), k.name if k else "–"),
            (_("Grafikspeicher"), vram_anzeige(self.befund)),
            (_("Compute Capability"),
             f"{k.compute_capability[0]}.{k.compute_capability[1]}" if k else "–"),
            (_("Treiber"), self.befund.treiber or "–"),
            (_("Funktionsstufe"), self.befund.stufe),
            (_("Rechengenauigkeit"), genauigkeit_anzeige(self.befund)),
            (_("Python"), platform.python_version()),
            (_("PySide6 / Qt"), f"{PYSIDE_VERSION} / {qVersion()}"),
            (_("Einstellungen"), einstellungen.config_path()),
        ]
        for nummer, (bezeichnung, wert) in enumerate(zeilen):
            gitter.addWidget(self._label(bezeichnung, "nebentext"), nummer, 0,
                             Qt.AlignmentFlag.AlignTop)
            # Der Pfad bricht um, sonst zieht ein tief verschachtelter Ablageort
            # das ganze Fenster in die Breite.
            wertfeld = self._label(wert, "wert", umbruch=True)
            wertfeld.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            gitter.addWidget(wertfeld, nummer, 1)
        gitter.setColumnStretch(1, 1)
        innen.addLayout(gitter)
        aufbau.addWidget(karte)

        karte, innen = self._karte(_("Marken & Hinweise"))
        for text in (
            _("NVIDIA, RTX, GeForce, CUDA und TensorRT sind Marken oder eingetragene Marken "
              "der NVIDIA Corporation in den USA und anderen Ländern. Neuro-Enhance ist ein "
              "unabhängiges Projekt und steht in keiner Verbindung zur NVIDIA Corporation; "
              "es wird von ihr weder unterstützt noch gesponsert."),
            _("Qt ist eine Marke der The Qt Company Ltd., Python eine Marke der Python "
              "Software Foundation, Windows eine Marke der Microsoft Corporation. Alle "
              "weiteren Marken gehören ihren jeweiligen Inhabern."),
            _("KI-Verfahren ergänzen Bilddetails, die im Original nicht vorhanden waren. "
              "Bearbeitete Bilder eignen sich deshalb nicht als Beweis- oder "
              "Dokumentationsmittel."),
            _("Verwendete Fremdkomponenten und ihre Lizenzen stehen in der Datei NOTICE."),
        ):
            innen.addWidget(self._label(text, "nebentext", umbruch=True))
        aufbau.addWidget(karte)
        aufbau.addStretch(1)

        self.reiter.addTab(seite, "")
        self._reitertitel = [_("Bearbeiten"), _("Info & Copyright")]
        for nummer, text in enumerate(self._reitertitel):
            self.reiter.setTabText(nummer, reitertext(text))

    # ------------------------------------------------------------------
    # Statuszeile
    # ------------------------------------------------------------------

    def _statuszeile(self):
        zeile = self.statusBar()
        zeile.setSizeGripEnabled(False)
        self.statusmeldung = self._label(_("Bereit."))
        zeile.addWidget(self.statusmeldung, 1)
        k = self.befund.karte
        if k:
            text = f"{k.name} · {vram_anzeige(self.befund)} · {genauigkeit_anzeige(self.befund)}"
            zeile.addPermanentWidget(self._label(text, "wert"))
            zeile.addPermanentWidget(self._label(_("Stufe {stufe}")
                                                 .format(stufe=self.befund.stufe)))

    # ------------------------------------------------------------------
    # Einstellungen
    # ------------------------------------------------------------------

    def _fensterlage_laden(self):
        gespeichert = einstellungen.load_config().get("geometry")
        if gespeichert:
            try:
                self.restoreGeometry(QByteArray(base64.b64decode(gespeichert)))
                return
            except (ValueError, TypeError):
                pass
        self.resize(1100, 760)

    def _einstellungen_sichern(self, mit_fenster=False):
        daten = einstellungen.load_config()
        daten["language"] = _.language
        daten["theme"] = farben.CURRENT_THEME
        if mit_fenster:
            daten["geometry"] = base64.b64encode(bytes(self.saveGeometry())).decode("ascii")
        einstellungen.save_config(daten)

    def closeEvent(self, ereignis):                 # noqa: N802 - Qt-Name
        self._einstellungen_sichern(mit_fenster=True)
        super().closeEvent(ereignis)
