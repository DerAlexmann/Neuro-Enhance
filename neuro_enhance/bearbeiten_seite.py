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
    QButtonGroup,
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

from . import bilddatei, einstellungen, filter, lut
from .bearbeitung import Sitzung
from .cuda import cupy as cp
from .kurveneditor import KurvenEditor
from .leinwand import Leinwand
from .uebersetzung import _

REGLERLEISTE_BREITE = 320


def gruppen_titel(gruppe: str) -> str:
    return {
        "weissabgleich": _("Weißabgleich"),
        "licht": _("Licht"),
        "praesenz": _("Präsenz"),
        "farbe": _("Farbe"),
        "rauschen": _("Rauschminderung"),
        "details": _("Details"),
        "lut": _("LUT"),
    }[gruppe]


def farbbereich_titel(nummer: int) -> str:
    return (_("Rot"), _("Orange"), _("Gelb"), _("Grün"), _("Aqua"), _("Blau"), _("Lila"),
            _("Magenta"))[nummer]


# Regler der Farbbereichskarte - je nach gewaehlter Eigenschaft zeigen sie
# Farbton, Saettigung oder Luminanz desselben Bereichs
HSL_REGLER = tuple(filter.Regler(f"hsl_{i}", "hsl", -100, 100)
                   for i in range(len(filter.FARBBEREICHE)))


def regler_titel(name: str) -> str:
    return {
        "temperatur": _("Temperatur"),
        "toenung": _("Tönung"),
        "belichtung": _("Belichtung"),
        "kontrast": _("Kontrast"),
        "lichter": _("Lichter"),
        "tiefen": _("Tiefen"),
        "klarheit": _("Klarheit"),
        "dunst": _("Dunst entfernen"),
        "dynamik": _("Dynamik"),
        "saettigung": _("Sättigung"),
        "schaerfe": _("Schärfen"),
        "schaerfe_radius": _("Radius"),
        "rauschen_luminanz": _("Luminanz"),
        "rauschen_farbe": _("Farbe"),
        "lut_staerke": _("Stärke"),
    }[name] if not name.startswith("hsl_") else farbbereich_titel(int(name[4:]))


def speicherformate() -> list[tuple[str, str, int]]:
    """Dateifilter fuer den Speichern-Dialog: (Beschriftung, Endung, Bits)."""
    return [
        ("JPEG (*.jpg *.jpeg)", ".jpg", 8),
        (f"PNG {_('8 Bit')} (*.png)", ".png", 8),
        (f"PNG {_('16 Bit')} (*.png)", ".png", 16),
        (f"TIFF {_('8 Bit')} (*.tif *.tiff)", ".tif", 8),
        (f"TIFF {_('16 Bit')} (*.tif *.tiff)", ".tif", 16),
        ("WebP (*.webp)", ".webp", 8),
    ]


def ziel_bestimmen(pfad: str, gewaehlt: str) -> tuple[str, int]:
    """Endung ergaenzen und Bittiefe aus dem gewaehlten Filter ablesen.

    Hat der Anwender eine Endung getippt, die nicht zum Filter passt, gilt die
    Endung; 16 Bit gibt es dann nur, wenn das Format sie kann.
    """
    formate = {beschriftung: (endung, bits) for beschriftung, endung, bits in speicherformate()}
    endung_filter, bits = formate.get(gewaehlt, (".jpg", 8))
    endung = os.path.splitext(pfad)[1].lower()
    if endung not in bilddatei.SCHREIBBAR:
        return pfad + endung_filter, bits
    if endung_filter == endung or (endung, endung_filter) in ((".jpeg", ".jpg"),
                                                             (".tiff", ".tif")):
        return pfad, bits
    return pfad, 16 if bits == 16 and bilddatei.SCHREIBBAR[endung][2] else 8


def art_anzeige(daten: bilddatei.Bilddaten) -> str:
    if daten.raw:
        return "RAW"
    return _("16 Bit") if daten.bits == 16 else _("8 Bit")


def wert_anzeige(regler: filter.Regler, wert: float) -> str:
    text = f"{wert:.{regler.nachkomma}f}"
    if regler.minimum < 0 and wert > 0:
        text = "+" + text
    text = text.replace("-", "−")
    if regler.name == "belichtung":
        text += " EV"
    elif regler.name == "schaerfe_radius":
        text += " px"
    elif regler.name == "lut_staerke":
        text += " %"
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
        karten: dict[str, QFrame] = {}
        zaehler: dict[str, int] = {}
        for regler in filter.REGLER:
            if regler.gruppe not in gruppen:
                karte, innen = self._karte(gruppen_titel(regler.gruppe))
                if regler.gruppe == "lut":
                    self._lut_bedienung(innen)
                gitter = QGridLayout()
                gitter.setVerticalSpacing(2)
                gitter.setColumnStretch(0, 1)
                innen.addLayout(gitter)
                if regler.gruppe == "rauschen":
                    hinweis = QLabel(objectName="nebentext")
                    hinweis.setWordWrap(True)
                    self.fenster.beschriften(hinweis.setText, _(
                        "Die Vorschau ist verkleinert und zeigt Rauschen schwächer als das "
                        "gespeicherte Bild."))
                    innen.addWidget(hinweis)
                spalte.addWidget(karte)
                karten[regler.gruppe] = karte
                gruppen[regler.gruppe] = gitter
                zaehler[regler.gruppe] = 0
            self.zeilen[regler.name] = ReglerZeile(self, regler, gruppen[regler.gruppe],
                                                   zaehler[regler.gruppe])
            zaehler[regler.gruppe] += 2
        # Gradationskurve hinter die Tonwerte, Farbbereiche hinter Dynamik und Saettigung
        spalte.insertWidget(spalte.indexOf(karten["licht"]) + 1, self._kurvenkarte())
        spalte.insertWidget(spalte.indexOf(karten["farbe"]) + 1, self._hsl_karte())
        spalte.addStretch(1)
        flaeche.setWidget(inhalt)
        return flaeche

    def _karte(self, titeltext) -> tuple[QFrame, QVBoxLayout]:
        karte = QFrame(objectName="karte")
        innen = QVBoxLayout(karte)
        innen.setContentsMargins(16, 12, 16, 14)
        titel = QLabel(objectName="kartentitel")
        self.fenster.beschriften(titel.setText, titeltext)
        innen.addWidget(titel)
        return karte, innen

    def _hsl_karte(self) -> QFrame:
        karte, innen = self._karte(_("Farbbereiche"))
        leiste = QHBoxLayout()
        leiste.setSpacing(4)
        self.hsl_modus = filter.HSL[0]
        self._hsl_knoepfe: dict[str, QPushButton] = {}
        gruppe = QButtonGroup(karte)
        for name, text in zip(filter.HSL, (_("Farbton"), _("Sättigung"), _("Luminanz")),
                              strict=True):
            knopf = QPushButton(objectName="kanal")
            knopf.setCheckable(True)
            self.fenster.beschriften(knopf.setText, text)
            knopf.clicked.connect(lambda _an, n=name: self._hsl_modus_waehlen(n))
            gruppe.addButton(knopf)
            self._hsl_knoepfe[name] = knopf
            leiste.addWidget(knopf)
        self._hsl_knoepfe[self.hsl_modus].setChecked(True)
        leiste.addStretch(1)
        innen.addLayout(leiste)
        gitter = QGridLayout()
        gitter.setVerticalSpacing(2)
        gitter.setColumnStretch(0, 1)
        innen.addLayout(gitter)
        self.hsl_zeilen = [ReglerZeile(self, regler, gitter, 2 * i)
                           for i, regler in enumerate(HSL_REGLER)]
        return karte

    def _hsl_modus_waehlen(self, modus: str):
        self.hsl_modus = modus
        self._hsl_anzeigen()

    def _hsl_anzeigen(self):
        werte = (getattr(self.sitzung.werte, self.hsl_modus) if self.sitzung is not None
                 else filter.HSL_NEUTRAL)
        for zeile, wert in zip(self.hsl_zeilen, werte, strict=True):
            zeile.setzen(wert)

    def _lut_bedienung(self, innen: QVBoxLayout):
        self.lut_name = QLabel(objectName="nebentext")
        self.lut_name.setWordWrap(True)
        innen.addWidget(self.lut_name)
        leiste = QHBoxLayout()
        leiste.setSpacing(6)
        self.lut_knopf = self._knopf(_("LUT laden …"), self.lut_dialog)
        self.lut_entfernen_knopf = self._knopf(_("Entfernen"), self.lut_entfernen)
        leiste.addWidget(self.lut_knopf)
        leiste.addWidget(self.lut_entfernen_knopf)
        leiste.addStretch(1)
        innen.addLayout(leiste)
        self._lut_anzeigen()

    def _lut_anzeigen(self):
        pfad = self.sitzung.werte.lut if self.sitzung is not None else ""
        if pfad:
            try:
                titel = lut.laden(pfad).titel
            except lut.LutFehler:
                titel = os.path.basename(pfad)
            self.fenster.beschriften(self.lut_name.setText, _("Geladen: {name}").format(name=titel))
        else:
            self.fenster.beschriften(self.lut_name.setText, _("Keine LUT geladen."))
        self.lut_entfernen_knopf.setEnabled(bool(pfad))

    def lut_dialog(self):
        if self.sitzung is None:
            return
        ordner = einstellungen.load_config().get("lut_ordner", "")
        pfad, _filter = QFileDialog.getOpenFileName(
            self, _("LUT laden"), ordner if os.path.isdir(ordner) else self._letzter_ordner(),
            "Cube-LUT (*.cube)")
        if not pfad:
            return
        try:
            lut.laden(pfad)
        except lut.LutFehler as fehler:
            self._fehler(_("Die LUT lässt sich nicht lesen."), str(fehler))
            return
        daten = einstellungen.load_config()
        daten["lut_ordner"] = os.path.dirname(pfad)
        einstellungen.save_config(daten)
        self.sitzung.werte.lut = pfad
        self._lut_anzeigen()
        self.zeichnen_anfordern()

    def lut_entfernen(self):
        if self.sitzung is None:
            return
        self.sitzung.werte.lut = ""
        self._lut_anzeigen()
        self.zeichnen_anfordern()

    def _kurvenkarte(self) -> QFrame:
        karte = QFrame(objectName="karte")
        innen = QVBoxLayout(karte)
        innen.setContentsMargins(16, 12, 16, 14)
        titel = QLabel(objectName="kartentitel")
        self.fenster.beschriften(titel.setText, _("Gradationskurve"))
        innen.addWidget(titel)
        self.kurven = KurvenEditor(
            self.fenster.beschriften,
            {"kurve_hell": _("Hell"), "kurve_rot": _("R"), "kurve_gruen": _("G"),
             "kurve_blau": _("B")},
            _("Zurücksetzen"))
        self.fenster.beschriften(
            self.kurven.flaeche.setToolTip,
            _("Klicken setzt einen Punkt, Ziehen verschiebt ihn, Doppelklick entfernt ihn."))
        self.kurven.geaendert.connect(self.wert_geaendert)
        innen.addWidget(self.kurven)
        return karte

    def _knoepfe_freischalten(self):
        offen = self.sitzung is not None
        for widget in (self.speichern_knopf, self.vorher_knopf, self.zuruecksetzen_knopf):
            widget.setEnabled(offen)
        for zeile in [*self.zeilen.values(), *self.hsl_zeilen]:
            zeile.schieber.setEnabled(offen)
        for knopf in self._hsl_knoepfe.values():
            knopf.setEnabled(offen)
        self.kurven.setEnabled(offen)
        self.lut_knopf.setEnabled(offen)
        self._lut_anzeigen()

    # ------------------------------------------------------------------
    # Regler und Vorschau
    # ------------------------------------------------------------------

    def wert_geaendert(self, name: str, wert):
        if self.sitzung is None:
            return
        if name.startswith("hsl_") and name[4:].isdigit():
            # Ein Farbbereichsregler: gilt fuer die gerade gewaehlte Eigenschaft
            werte = list(getattr(self.sitzung.werte, self.hsl_modus))
            werte[int(name[4:])] = wert
            name, wert = self.hsl_modus, tuple(werte)
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
            self.kurven.histogramm_zeigen(None)
            self.fenster.rechenzeit_zeigen(None)
            return
        bild, ms, histogramm = self.sitzung.vorschau(unbearbeitet=self._vorher)
        self.leinwand.zeigen(bild)
        self.kurven.histogramm_zeigen(histogramm)
        self.fenster.rechenzeit_zeigen(ms)

    def _vorher_zeigen(self, an: bool):
        self._vorher = an
        self.zeichnen_anfordern()

    def alles_zuruecksetzen(self):
        if self.sitzung is not None:
            self.sitzung.werte = filter.Einstellungen()
        self._alles_anzeigen()
        self.zeichnen_anfordern()

    def _alles_anzeigen(self):
        """Alle Bedienelemente auf die Werte der Sitzung (oder die Vorgaben) setzen."""
        werte = self.sitzung.werte if self.sitzung is not None else filter.Einstellungen()
        for name, zeile in self.zeilen.items():
            zeile.setzen(getattr(werte, name))
        self.kurven.alle_setzen({name: getattr(werte, name) for name in filter.KURVEN})
        self._hsl_anzeigen()
        self._lut_anzeigen()

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
        def muster(endungen):
            return " ".join(f"*{e}" for e in endungen)
        filterliste = ";;".join([
            f"{_('Alle Bilder')} ({muster(bilddatei.LESBAR)})",
            f"{_('Bilder')} ({muster(bilddatei.BILDER)})",
            f"RAW ({muster(bilddatei.RAW)})",
        ])
        pfad, _filter = QFileDialog.getOpenFileName(
            self, _("Bild öffnen"), self._letzter_ordner(), filterliste)
        if pfad:
            self.oeffnen(pfad)

    def oeffnen(self, pfad: str):
        if os.path.splitext(pfad)[1].lower() not in bilddatei.LESBAR:
            self.fenster.melden(_("Dieses Dateiformat wird nicht unterstützt."), fehler=True)
            return
        if not self.aenderungen_verwerfen_ok():
            return
        if os.path.splitext(pfad)[1].lower() in bilddatei.RAW:
            # LibRaw entwickelt auf dem Prozessor; das dauert einige Sekunden.
            self.fenster.melden(_("RAW wird entwickelt …"))
            QApplication.processEvents()
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
        self._alles_anzeigen()
        self._knoepfe_freischalten()
        self.fenster.melden(_("{name} geöffnet – {breite} × {hoehe} Pixel, {art}")
                            .format(name=daten.name, breite=daten.breite, hoehe=daten.hoehe,
                                    art=art_anzeige(daten)))
        self.zeichnen_anfordern()

    def speichern_dialog(self):
        if self.sitzung is None:
            return
        daten = self.sitzung.daten
        stamm, endung = os.path.splitext(daten.pfad)
        endung = endung.lower()
        formate = speicherformate()
        if daten.raw or daten.bits == 16:
            # Mehr als 8 Bit im Original: als 16-Bit-TIFF vorschlagen, damit nichts verloren geht
            endung, vorauswahl = ".tif", formate[4][0]
        else:
            if endung not in bilddatei.SCHREIBBAR:
                endung = ".jpg"
            vorauswahl = next(b for b, e, bits in formate
                              if bits == 8 and e == {".jpeg": ".jpg", ".tiff": ".tif"}
                              .get(endung, endung))
        pfad, gewaehlt = QFileDialog.getSaveFileName(
            self, _("Bild speichern"), f"{stamm}-bearbeitet{endung}",
            ";;".join(b for b, _e, _bits in formate), vorauswahl)
        if not pfad:
            return
        pfad, bits = ziel_bestimmen(pfad, gewaehlt)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ms = self.sitzung.exportieren(pfad, bits)
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
