"""
Reiter "Bearbeiten": Leinwand, Werkzeugleiste und Reglerleiste

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import os

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from . import bilddatei, einstellungen, filter
from .bearbeitung import Sitzung
from .cuda import cupy as cp
from .leinwand import Leinwand
from .uebersetzung import _

REGLERLEISTE_BREITE = 320


def gruppen_titel(gruppe: str) -> str:
    return {
        "weissabgleich": _("Weißabgleich"),
        "licht": _("Licht"),
        "farbe": _("Farbe"),
        "details": _("Details"),
    }[gruppe]


def regler_titel(name: str) -> str:
    return {
        "temperatur": _("Temperatur"),
        "toenung": _("Tönung"),
        "belichtung": _("Belichtung"),
        "kontrast": _("Kontrast"),
        "lichter": _("Lichter"),
        "tiefen": _("Tiefen"),
        "dynamik": _("Dynamik"),
        "saettigung": _("Sättigung"),
        "schaerfe": _("Schärfen"),
        "schaerfe_radius": _("Radius"),
    }[name]


def wert_anzeige(regler: filter.Regler, wert: float) -> str:
    text = f"{wert:.{regler.nachkomma}f}"
    if regler.minimum < 0 and wert > 0:
        text = "+" + text
    text = text.replace("-", "−")
    if regler.name == "belichtung":
        text += " EV"
    elif regler.name == "schaerfe_radius":
        text += " px"
    return text


class Doppelklick(QObject):
    """Setzt einen Regler per Doppelklick auf seine Vorgabe zurueck."""

    def __init__(self, zuruecksetzen):
        super().__init__()
        self._zuruecksetzen = zuruecksetzen

    def eventFilter(self, objekt, ereignis):        # noqa: N802 - Qt-Name
        if ereignis.type() == QEvent.Type.MouseButtonDblClick:
            self._zuruecksetzen()
            return True
        return False


class ReglerZeile:
    """Beschriftung, Wertanzeige und Schieberegler fuer einen Filterwert."""

    def __init__(self, seite: BearbeitenSeite, regler: filter.Regler, gitter: QGridLayout,
                 zeile: int):
        self.regler = regler
        self.seite = seite
        self.stufen = round((regler.maximum - regler.minimum) / regler.schritt)

        self.titel = QLabel()
        seite.fenster.beschriften(self.titel.setText, regler_titel(regler.name))
        self.anzeige = QLabel(objectName="wert")
        self.anzeige.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.schieber = QSlider(Qt.Orientation.Horizontal)
        self.schieber.setRange(0, self.stufen)
        self.schieber.setPageStep(max(1, self.stufen // 20))
        self._filter = Doppelklick(self.zuruecksetzen)
        for widget in (self.titel, self.anzeige, self.schieber):
            widget.installEventFilter(self._filter)
            seite.fenster.beschriften(widget.setToolTip,
                                      _("Doppelklick setzt den Regler zurück."))

        gitter.addWidget(self.titel, zeile, 0)
        gitter.addWidget(self.anzeige, zeile, 1)
        gitter.addWidget(self.schieber, zeile + 1, 0, 1, 2)
        self.setzen(regler.vorgabe)
        self.schieber.valueChanged.connect(self._bewegt)

    def wert(self) -> float:
        roh = self.regler.minimum + self.schieber.value() * self.regler.schritt
        return round(roh, self.regler.nachkomma)

    def setzen(self, wert: float):
        stufe = round((wert - self.regler.minimum) / self.regler.schritt)
        self.schieber.blockSignals(True)
        self.schieber.setValue(stufe)
        self.schieber.blockSignals(False)
        self.anzeige.setText(wert_anzeige(self.regler, self.wert()))

    def zuruecksetzen(self):
        self.setzen(self.regler.vorgabe)
        self._bewegt()

    def _bewegt(self, *_args):
        self.anzeige.setText(wert_anzeige(self.regler, self.wert()))
        self.seite.wert_geaendert(self.regler.name, self.wert())


class BearbeitenSeite(QWidget):
    def __init__(self, fenster):
        super().__init__(objectName="seite")
        self.fenster = fenster
        self.sitzung: Sitzung | None = None
        self._vorher = False
        self._zeichnen_angefordert = False

        aufbau = QHBoxLayout(self)
        aufbau.setContentsMargins(0, 12, 0, 0)
        aufbau.setSpacing(12)

        links = QVBoxLayout()
        links.setSpacing(8)
        links.addLayout(self._werkzeugleiste())
        self.leinwand = Leinwand()
        fenster.beschriften(self.leinwand.leer.setText, _("Noch kein Bild geöffnet."))
        fenster.beschriften(self.leinwand.hinweis.setText,
                            _("Bild hierher ziehen oder „Öffnen …“ wählen."))
        self.leinwand.datei_abgelegt.connect(self.oeffnen)
        links.addWidget(self.leinwand, 1)
        aufbau.addLayout(links, 1)
        aufbau.addWidget(self._reglerleiste())

        QShortcut(QKeySequence.StandardKey.Open, self, self.oeffnen_dialog)
        QShortcut(QKeySequence.StandardKey.Save, self, self.speichern_dialog)
        self._knoepfe_freischalten()

    # ------------------------------------------------------------------
    # Aufbau
    # ------------------------------------------------------------------

    def _knopf(self, text, aktion, name=None) -> QPushButton:
        knopf = QPushButton()
        if name:
            knopf.setObjectName(name)
        self.fenster.beschriften(knopf.setText, text)
        if aktion:
            knopf.clicked.connect(aktion)
        return knopf

    def _werkzeugleiste(self) -> QHBoxLayout:
        leiste = QHBoxLayout()
        leiste.setSpacing(8)
        leiste.addWidget(self._knopf(_("Öffnen …"), self.oeffnen_dialog, "hauptschalter"))
        self.speichern_knopf = self._knopf(_("Speichern unter …"), self.speichern_dialog)
        leiste.addWidget(self.speichern_knopf)
        leiste.addStretch(1)
        self.vorher_knopf = self._knopf(_("Vorher"), None)
        self.fenster.beschriften(self.vorher_knopf.setToolTip,
                                 _("Gedrückt halten, um das unbearbeitete Bild zu sehen."))
        self.vorher_knopf.pressed.connect(lambda: self._vorher_zeigen(True))
        self.vorher_knopf.released.connect(lambda: self._vorher_zeigen(False))
        leiste.addWidget(self.vorher_knopf)
        self.zuruecksetzen_knopf = self._knopf(_("Alles zurücksetzen"), self.alles_zuruecksetzen)
        leiste.addWidget(self.zuruecksetzen_knopf)
        return leiste

    def _reglerleiste(self) -> QScrollArea:
        flaeche = QScrollArea()
        flaeche.setWidgetResizable(True)
        flaeche.setFixedWidth(REGLERLEISTE_BREITE)
        flaeche.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inhalt = QWidget(objectName="seite")
        spalte = QVBoxLayout(inhalt)
        spalte.setContentsMargins(0, 0, 4, 0)
        spalte.setSpacing(12)

        self.zeilen: dict[str, ReglerZeile] = {}
        gruppen: dict[str, QGridLayout] = {}
        zaehler: dict[str, int] = {}
        for regler in filter.REGLER:
            if regler.gruppe not in gruppen:
                karte = QFrame(objectName="karte")
                innen = QVBoxLayout(karte)
                innen.setContentsMargins(16, 12, 16, 14)
                titel = QLabel(objectName="kartentitel")
                self.fenster.beschriften(titel.setText, gruppen_titel(regler.gruppe))
                innen.addWidget(titel)
                gitter = QGridLayout()
                gitter.setVerticalSpacing(2)
                gitter.setColumnStretch(0, 1)
                innen.addLayout(gitter)
                spalte.addWidget(karte)
                gruppen[regler.gruppe] = gitter
                zaehler[regler.gruppe] = 0
            self.zeilen[regler.name] = ReglerZeile(self, regler, gruppen[regler.gruppe],
                                                   zaehler[regler.gruppe])
            zaehler[regler.gruppe] += 2
        spalte.addStretch(1)
        flaeche.setWidget(inhalt)
        return flaeche

    def _knoepfe_freischalten(self):
        offen = self.sitzung is not None
        for widget in (self.speichern_knopf, self.vorher_knopf, self.zuruecksetzen_knopf):
            widget.setEnabled(offen)
        for zeile in self.zeilen.values():
            zeile.schieber.setEnabled(offen)

    # ------------------------------------------------------------------
    # Regler und Vorschau
    # ------------------------------------------------------------------

    def wert_geaendert(self, name: str, wert: float):
        if self.sitzung is None:
            return
        setattr(self.sitzung.werte, name, wert)
        self.zeichnen_anfordern()

    def zeichnen_anfordern(self):
        """Faellt bei schnellem Ziehen mehrere Aenderungen in ein Neuzeichnen zusammen."""
        if not self._zeichnen_angefordert:
            self._zeichnen_angefordert = True
            QTimer.singleShot(0, self._zeichnen)

    def _zeichnen(self):
        self._zeichnen_angefordert = False
        if self.sitzung is None:
            self.leinwand.zeigen(None)
            self.fenster.rechenzeit_zeigen(None)
            return
        bild, ms = self.sitzung.vorschau(unbearbeitet=self._vorher)
        self.leinwand.zeigen(bild)
        self.fenster.rechenzeit_zeigen(ms)

    def _vorher_zeigen(self, an: bool):
        self._vorher = an
        self.zeichnen_anfordern()

    def alles_zuruecksetzen(self):
        for zeile in self.zeilen.values():
            zeile.setzen(zeile.regler.vorgabe)
        if self.sitzung is not None:
            self.sitzung.werte = filter.Einstellungen()
        self.zeichnen_anfordern()

    # ------------------------------------------------------------------
    # Oeffnen und Speichern
    # ------------------------------------------------------------------

    def _letzter_ordner(self) -> str:
        ordner = einstellungen.load_config().get("letzter_ordner", "")
        return ordner if os.path.isdir(ordner) else os.path.expanduser("~")

    def _letzten_ordner_merken(self, pfad: str):
        daten = einstellungen.load_config()
        daten["letzter_ordner"] = os.path.dirname(pfad)
        einstellungen.save_config(daten)

    def aenderungen_verwerfen_ok(self) -> bool:
        """Fragt nach, bevor ungespeicherte Aenderungen verloren gehen."""
        if self.sitzung is None or not self.sitzung.geaendert:
            return True
        antwort = QMessageBox.question(
            self, _("Ungespeicherte Änderungen"),
            _("Die Änderungen an {name} sind nicht gespeichert. Trotzdem fortfahren?")
            .format(name=self.sitzung.daten.name),
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        return antwort == QMessageBox.StandardButton.Discard

    def oeffnen_dialog(self):
        endungen = " ".join(f"*{e}" for e in bilddatei.LESBAR)
        pfad, _filter = QFileDialog.getOpenFileName(
            self, _("Bild öffnen"), self._letzter_ordner(),
            f"{_('Bilder')} ({endungen})")
        if pfad:
            self.oeffnen(pfad)

    def oeffnen(self, pfad: str):
        if os.path.splitext(pfad)[1].lower() not in bilddatei.LESBAR:
            self.fenster.melden(_("Dieses Dateiformat wird nicht unterstützt."), fehler=True)
            return
        if not self.aenderungen_verwerfen_ok():
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            daten = bilddatei.laden(pfad)
            if self.sitzung is not None:
                self.sitzung.schliessen()
                self.sitzung = None
            self.sitzung = Sitzung(daten, self._vorschau_kante())
        except bilddatei.BildFehler as fehler:
            self._fehler(_("Die Datei lässt sich nicht öffnen."), str(fehler))
            return
        except cp.cuda.memory.OutOfMemoryError:
            self._fehler(_("Für dieses Bild reicht der Grafikspeicher nicht."), "")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._letzten_ordner_merken(pfad)
        for zeile in self.zeilen.values():
            zeile.setzen(zeile.regler.vorgabe)
        self._knoepfe_freischalten()
        self.fenster.melden(_("{name} geöffnet – {breite} × {hoehe} Pixel")
                            .format(name=daten.name, breite=daten.breite, hoehe=daten.hoehe))
        self.zeichnen_anfordern()

    def speichern_dialog(self):
        if self.sitzung is None:
            return
        stamm, endung = os.path.splitext(self.sitzung.daten.pfad)
        if endung.lower() not in bilddatei.SCHREIBBAR:
            endung = ".jpg"
        vorschlag = f"{stamm}-bearbeitet{endung.lower()}"
        filterliste = ";;".join([
            "JPEG (*.jpg *.jpeg)", "PNG (*.png)", "TIFF (*.tif *.tiff)", "WebP (*.webp)"])
        pfad, gewaehlt = QFileDialog.getSaveFileName(
            self, _("Bild speichern"), vorschlag, filterliste)
        if not pfad:
            return
        if os.path.splitext(pfad)[1].lower() not in bilddatei.SCHREIBBAR:
            pfad += "." + gewaehlt.split("*.")[1].split(" ")[0].rstrip(")")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ms = self.sitzung.exportieren(pfad)
        except bilddatei.BildFehler as fehler:
            self._fehler(_("Das Bild lässt sich nicht speichern."), str(fehler))
            return
        except cp.cuda.memory.OutOfMemoryError:
            self._fehler(_("Für dieses Bild reicht der Grafikspeicher nicht."), "")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._letzten_ordner_merken(pfad)
        self.fenster.melden(_("Gespeichert: {name} ({ms} ms)")
                            .format(name=os.path.basename(pfad), ms=f"{ms:.0f}"))

    def _fehler(self, text: str, einzelheiten: str):
        box = QMessageBox(QMessageBox.Icon.Warning, self.fenster.windowTitle(), text,
                          QMessageBox.StandardButton.Ok, self)
        if einzelheiten:
            box.setDetailedText(einzelheiten)
        box.exec()
        self.fenster.melden(text, fehler=True)

    def _vorschau_kante(self) -> int:
        """Laengste Kante der Vorschau: so gross wie der Bildschirm in echten Pixeln."""
        bildschirm = QGuiApplication.primaryScreen()
        groesse = bildschirm.size() * bildschirm.devicePixelRatio()
        return max(1024, min(4096, max(groesse.width(), groesse.height())))

    def schliessen(self):
        if self.sitzung is not None:
            self.sitzung.schliessen()
            self.sitzung = None
