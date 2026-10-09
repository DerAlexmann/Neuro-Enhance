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

SYMBOL_NAME = "silberkorn.ico"


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


def cuda_anzeige() -> str:
    from .cuda import cupy
    if cupy is None:
        return "–"
    # Die Kernel uebersetzt NVRTC aus dem installierten cuda-toolkit; dessen
    # Fassung bestimmt, welchen Treiber es braucht - nicht die, gegen die
    # CuPy selbst gebaut wurde.
    haupt, neben = cupy.cuda.nvrtc.getVersion()
    return f"{cupy.__version__} / NVRTC {haupt}.{neben}"


def tensorrt_anzeige() -> str:
    from . import ki
    fassung = ki.tensorrt_fassung()
    if fassung is None:
        return _("nicht installiert (optional, siehe README)")
    return fassung


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
        self.auswahlbreiten_anpassen()
        self._fensterlage_laden()

    # ------------------------------------------------------------------
    # Beschriftungen
    # ------------------------------------------------------------------

    def beschriften(self, setzen, text):
        """Setzt eine Beschriftung und merkt sie fuer den Sprachwechsel.

        Je Ziel zaehlt nur die zuletzt gesetzte Beschriftung - Anzeigen, die sich
        oft erneuern, sammeln so keine alten Texte an. Feste Texte ohne
        Uebersetzung (etwa "100 %") werden gesetzt, aber nicht gemerkt.
        """
        setzen(text)
        self._beschriftungen = [(s, t) for s, t in self._beschriftungen if s != setzen]
        if hasattr(text, "schluessel"):
            self._beschriftungen.append((setzen, text))

    def texte_auffrischen(self):
        bleibend = []
        for setzen, text in self._beschriftungen:
            neu = _(text.schluessel)
            try:
                setzen(neu.format(**text.werte) if text.werte else neu)
            except RuntimeError:                 # Widget gibt es nicht mehr
                continue
            bleibend.append((setzen, text))
        self._beschriftungen = bleibend
        for nummer, text in enumerate(self._reitertitel):
            self.reiter.setTabText(nummer, reitertext(_(text.schluessel)))
        self.auswahlbreiten_anpassen()

    def auswahlbreiten_anpassen(self):
        """Jede Auswahlliste mindestens so breit wie ihr laengster Eintrag.

        Qt rechnet den Platz fuer den Pfeil aus dem Stylesheet (padding rechts)
        nicht in die Breite ein - der Text wuerde sonst abgeschnitten, etwa
        "Schnell (GPU)". Nach jedem Sprachwechsel neu, die Texte aendern sich.
        """
        for auswahl in self.findChildren(QComboBox):
            text = max((auswahl.itemText(i) for i in range(auswahl.count())),
                       key=auswahl.fontMetrics().horizontalAdvance, default="")
            # Text + Polster links (8) und rechts mit Pfeil (24) + Rahmen und Luft
            auswahl.setMinimumWidth(auswahl.fontMetrics().horizontalAdvance(text) + 8 + 24 + 8)

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

        aufbau.addSpacing(12)
        aufbau.addWidget(self._label(_("RAW"), "kopfgruppe"))
        self.raw_wahl = QComboBox()
        for nummer, (wert, text) in enumerate((("schnell", _("Schnell (GPU)")),
                                               ("beste", _("Beste Qualität")))):
            self.raw_wahl.addItem("", wert)
            self.beschriften(lambda t, i=nummer: self.raw_wahl.setItemText(i, t), text)
        self.raw_wahl.setCurrentIndex(
            max(0, self.raw_wahl.findData(einstellungen.raw_qualitaet())))
        self.beschriften(self.raw_wahl.setToolTip, _(
            "Schnell: Demosaicing nach Malvar, He und Cutler auf der Grafikkarte, eine "
            "24-MP-RAW in rund 0,2 s. Beste Qualität: LibRaw mit dem Verfahren DHT auf dem "
            "Prozessor, etwa 1,5 s, mit weniger Farbsäumen an feinen Mustern. Gilt beim "
            "nächsten Öffnen einer RAW."))
        self.raw_wahl.currentIndexChanged.connect(lambda _i: self._einstellungen_sichern())
        aufbau.addWidget(self.raw_wahl)
        return kopf

    def _sprache_gewechselt(self):
        _.language = self.sprachwahl.currentData() or SOURCE_LANGUAGE
        from .start import qt_sprache_setzen
        qt_sprache_setzen(_.language)
        self.texte_auffrischen()
        self._einstellungen_sichern()

    def _schema_gewechselt(self, dunkel: bool):
        farben.apply_theme("dark" if dunkel else "light")
        QApplication.instance().setStyleSheet(farben.stylesheet())
        farben.palette_setzen(QApplication.instance())
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
        # Erst hier geladen: das Modul bringt CuPy mit, und das darf erst nach
        # der Startpruefung geschehen.
        from .bearbeiten_seite import BearbeitenSeite
        self.bearbeiten = BearbeitenSeite(self)
        self.reiter.addTab(self.bearbeiten, "")

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
            (_("CUDA-Architektur"),
             f"{k.compute_capability[0]}.{k.compute_capability[1]}" if k else "–"),
            (_("Treiber"), self.befund.treiber or "–"),
            (_("Funktionsstufe"), self.befund.stufe),
            (_("Rechengenauigkeit"), genauigkeit_anzeige(self.befund)),
            (_("Python"), platform.python_version()),
            (_("PySide6 / Qt"), f"{PYSIDE_VERSION} / {qVersion()}"),
            (_("CuPy / CUDA"), cuda_anzeige()),
            ("TensorRT", tensorrt_anzeige()),
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
              "der NVIDIA Corporation in den USA und anderen Ländern. Silberkorn ist ein "
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
        self.rechenzeit = self._label(name="wert")
        # Feste Breite, damit die Statuszeile beim Ziehen nicht zappelt
        self.rechenzeit.setMinimumWidth(
            self.rechenzeit.fontMetrics().horizontalAdvance("GPU 9999.9 ms") + 24)
        zeile.addPermanentWidget(self.rechenzeit)
        k = self.befund.karte
        if k:
            text = f"{k.name} · {vram_anzeige(self.befund)} · {genauigkeit_anzeige(self.befund)}"
            zeile.addPermanentWidget(self._label(text, "wert"))
            stufe = self._label(_("Stufe {stufe}").format(stufe=self.befund.stufe))
            self.beschriften(stufe.setToolTip, stufe_beschreibung(self.befund.stufe))
            zeile.addPermanentWidget(stufe)

    def melden(self, text, fehler=False):
        """Meldung in der Statuszeile; uebersetzte Texte machen den Sprachwechsel mit."""
        self.statusmeldung.setObjectName("warnung" if fehler else "")
        self.statusmeldung.style().unpolish(self.statusmeldung)
        self.statusmeldung.style().polish(self.statusmeldung)
        # Nur die zuletzt gesetzte Meldung soll sich beim Sprachwechsel erneuern.
        self._beschriftungen = [(s, t) for s, t in self._beschriftungen
                                if s != self.statusmeldung.setText]
        if hasattr(text, "schluessel"):
            self.beschriften(self.statusmeldung.setText, text)
        else:
            self.statusmeldung.setText(text)

    def rechenzeit_zeigen(self, ms):
        self.rechenzeit.setText("" if ms is None else f"GPU {ms:.1f} ms")

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
        if hasattr(self, "raw_wahl"):
            daten["raw_qualitaet"] = self.raw_wahl.currentData()
        if mit_fenster:
            daten["geometry"] = base64.b64encode(bytes(self.saveGeometry())).decode("ascii")
        einstellungen.save_config(daten)

    def closeEvent(self, ereignis):                 # noqa: N802 - Qt-Name
        if not self.bearbeiten.aenderungen_verwerfen_ok():
            ereignis.ignore()
            return
        self._einstellungen_sichern(mit_fenster=True)
        self.bearbeiten.schliessen()
        super().closeEvent(ereignis)
