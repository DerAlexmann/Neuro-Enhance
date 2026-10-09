"""
Dialog "Für Veröffentlichung speichern": Größe, Wasserzeichen und Rechteangaben
mit Vorschau

Die Vorschau ist das Bild der Leinwand: Alle Groessen des Wasserzeichens sind
Anteile des Bildes, also sieht sie aus wie das gespeicherte Bild.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import dataclasses
import datetime
import os

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from . import veroeffentlichen
from .uebersetzung import _
from .veroeffentlichen import Vorlage

VORSCHAU = (520, 400)                 # groesste Flaeche der Vorschau in Pixeln


def position_titel() -> list[tuple[str, str]]:
    return [("unten_rechts", _("Unten rechts")), ("unten_links", _("Unten links")),
            ("oben_rechts", _("Oben rechts")), ("oben_links", _("Oben links")),
            ("mitte", _("Mitte"))]


def copyright_vorschlag(urheber: str) -> str:
    return f"© {datetime.date.today().year} {urheber.strip()}" if urheber.strip() else ""


class VeroeffentlichenDialog(QDialog):
    def __init__(self, eltern, vorschau: np.ndarray, volle_form: tuple[int, int],
                 vorlage: Vorlage, ki_inhalt: bool = False):
        super().__init__(eltern)
        self.setWindowTitle(_("Für Veröffentlichung speichern"))
        # Nur so gross wie gezeigt - jede Aenderung legt das Wasserzeichen neu auf
        bild = veroeffentlichen.als_float(vorschau)
        faktor = min(1.0, max(VORSCHAU) / max(bild.shape[:2]))
        self._vorschau = veroeffentlichen.verkleinern(
            bild, max(1, round(bild.shape[0] * faktor)), max(1, round(bild.shape[1] * faktor)))
        self._volle_form = volle_form
        self._logo_cache: tuple[str, object] = ("", None)

        aufbau = QVBoxLayout(self)
        oben = QHBoxLayout()
        oben.setSpacing(12)
        links = QVBoxLayout()
        links.setSpacing(12)
        links.addWidget(self._groesse_karte())
        links.addWidget(self._wasserzeichen_karte())
        links.addWidget(self._rechte_karte(ki_inhalt))
        links.addStretch(1)
        oben.addLayout(links)

        rechts = QVBoxLayout()
        self.bild = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.bild.setFixedSize(*VORSCHAU)
        rechts.addWidget(self.bild)
        self.groesse_info = QLabel(objectName="nebentext", alignment=Qt.AlignmentFlag.AlignCenter)
        rechts.addWidget(self.groesse_info)
        rechts.addStretch(1)
        oben.addLayout(rechts, 1)
        aufbau.addLayout(oben, 1)

        knoepfe = QHBoxLayout()
        knoepfe.addStretch(1)
        abbrechen = QPushButton(_("Abbrechen"))
        abbrechen.clicked.connect(self.reject)
        self.speichern_knopf = QPushButton(_("Speichern …"), objectName="hauptschalter")
        self.speichern_knopf.setDefault(True)
        self.speichern_knopf.clicked.connect(self.accept)
        knoepfe.addWidget(abbrechen)
        knoepfe.addWidget(self.speichern_knopf)
        aufbau.addLayout(knoepfe)

        self._setzen(vorlage)
        self._verbinden()
        self._aktualisieren()

    # ------------------------------------------------------------------
    # Aufbau
    # ------------------------------------------------------------------

    @staticmethod
    def _karte(titeltext: str) -> tuple[QFrame, QVBoxLayout]:
        karte = QFrame(objectName="karte")
        innen = QVBoxLayout(karte)
        innen.setContentsMargins(16, 12, 16, 14)
        titel = QLabel(titeltext, objectName="kartentitel")
        innen.addWidget(titel)
        return karte, innen

    def _groesse_karte(self) -> QFrame:
        karte, innen = self._karte(_("Größe"))
        zeile = QHBoxLayout()
        self.groesse_art = QComboBox()
        self.groesse_art.addItem(_("Längste Kante"), "kante")
        self.groesse_art.addItem(_("Prozent"), "prozent")
        zeile.addWidget(self.groesse_art)
        self.kante = QSpinBox(suffix=" px")
        self.kante.setRange(veroeffentlichen.KANTE_MIN, veroeffentlichen.KANTE_MAX)
        self.kante.setSingleStep(10)
        self.prozent = QSpinBox(suffix=" %")
        self.prozent.setRange(1, 100)
        for feld in (self.kante, self.prozent):
            feld.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)   # getippt oder Schnellwahl
            feld.setMinimumWidth(90)
            zeile.addWidget(feld)
        zeile.addStretch(1)
        innen.addLayout(zeile)

        self.schnellwahl = []
        zeile = QHBoxLayout()
        zeile.setSpacing(4)
        for kante in veroeffentlichen.KANTEN:
            knopf = QPushButton(f"{kante}", objectName="kanal")
            knopf.clicked.connect(lambda _a=False, k=kante: self.kante.setValue(k))
            zeile.addWidget(knopf)
            self.schnellwahl.append(knopf)
        zeile.addStretch(1)
        innen.addLayout(zeile)
        hinweis = QLabel(_("Kleinere Bilder werden nicht vergrößert."), objectName="nebentext")
        hinweis.setWordWrap(True)
        innen.addWidget(hinweis)
        return karte

    def _wasserzeichen_karte(self) -> QFrame:
        karte, innen = self._karte(_("Wasserzeichen"))
        gitter = QGridLayout()
        gitter.setColumnStretch(1, 1)
        self.text = QLineEdit(placeholderText=_("z. B. Spieltitel oder Webadresse"))
        gitter.addWidget(QLabel(_("Text")), 0, 0)
        gitter.addWidget(self.text, 0, 1, 1, 2)

        self.logo = QLineEdit(readOnly=True, placeholderText=_("kein Logo"))
        logo_knopf = QPushButton(_("Wählen …"), objectName="kanal")
        logo_knopf.clicked.connect(self._logo_waehlen)
        self.logo_weg = QPushButton("✕", objectName="kanal")
        self.logo_weg.setToolTip(_("Logo entfernen"))
        self.logo_weg.clicked.connect(lambda: self.logo.setText(""))
        logo_zeile = QHBoxLayout()
        logo_zeile.setSpacing(4)
        logo_zeile.addWidget(logo_knopf)
        logo_zeile.addWidget(self.logo_weg)
        gitter.addWidget(QLabel(_("Logo")), 1, 0)
        gitter.addWidget(self.logo, 1, 1)
        gitter.addLayout(logo_zeile, 1, 2)

        self.position = QComboBox()
        for wert, titel in position_titel():
            self.position.addItem(titel, wert)
        gitter.addWidget(QLabel(_("Position")), 2, 0)
        gitter.addWidget(self.position, 2, 1, 1, 2)

        self.textfarbe = QComboBox()
        self.textfarbe.addItem(_("Weiß"), "weiss")
        self.textfarbe.addItem(_("Schwarz"), "schwarz")
        gitter.addWidget(QLabel(_("Textfarbe")), 3, 0)
        gitter.addWidget(self.textfarbe, 3, 1, 1, 2)

        self.wz_groesse, self.wz_groesse_wert = self._regler(5, 80)
        gitter.addWidget(QLabel(_("Größe")), 4, 0)
        gitter.addWidget(self.wz_groesse, 4, 1)
        gitter.addWidget(self.wz_groesse_wert, 4, 2)
        self.deckkraft, self.deckkraft_wert = self._regler(10, 100)
        gitter.addWidget(QLabel(_("Deckkraft")), 5, 0)
        gitter.addWidget(self.deckkraft, 5, 1)
        gitter.addWidget(self.deckkraft_wert, 5, 2)
        innen.addLayout(gitter)

        self.muster = QCheckBox(_("Als Muster über das ganze Bild"))
        self.muster.setToolTip(_("Schwerer zu entfernen als ein Zeichen in der Ecke, "
                                 "stört aber mehr beim Ansehen."))
        innen.addWidget(self.muster)
        hinweis = QLabel(_("Logo am besten als PNG mit durchsichtigem Hintergrund."),
                         objectName="nebentext")
        hinweis.setWordWrap(True)
        innen.addWidget(hinweis)
        return karte

    @staticmethod
    def _regler(minimum: int, maximum: int) -> tuple[QSlider, QLabel]:
        regler = QSlider(Qt.Orientation.Horizontal)
        regler.setRange(minimum, maximum)
        regler.setMinimumWidth(160)
        wert = QLabel(objectName="wert")
        wert.setMinimumWidth(40)
        wert.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return regler, wert

    def _rechte_karte(self, ki_inhalt: bool) -> QFrame:
        karte, innen = self._karte(_("Rechte & Metadaten"))
        gitter = QGridLayout()
        gitter.setColumnStretch(1, 1)
        self.urheber = QLineEdit()
        self.copyright = QLineEdit()
        self.webadresse = QLineEdit(placeholderText="https://…")
        for zeile, (titel, feld) in enumerate(((_("Urheber"), self.urheber),
                                               (_("Copyright"), self.copyright),
                                               (_("Webadresse"), self.webadresse))):
            gitter.addWidget(QLabel(titel), zeile, 0)
            gitter.addWidget(feld, zeile, 1)
        innen.addLayout(gitter)

        self.ki_verbot = QCheckBox(_("KI-Training und Data-Mining untersagen"))
        self.ki_verbot.setToolTip(_(
            "Schreibt einen maschinenlesbaren Nutzungsvorbehalt nach IPTC in die Datei. "
            "Suchmaschinen dürfen das Bild weiterhin finden."))
        self.kameradaten = QCheckBox(_("Alle Kameradaten entfernen"))
        self.kameradaten.setToolTip(_(
            "Kamera, Objektiv, Belichtung und Aufnahmezeit. GPS-Position und "
            "Seriennummern werden in jedem Fall entfernt."))
        innen.addWidget(self.ki_verbot)
        innen.addWidget(self.kameradaten)
        if ki_inhalt:
            hinweis = QLabel(_("Das Bild enthält von der KI erfundene Teile – das wird in "
                               "den Metadaten vermerkt."), objectName="nebentext")
            hinweis.setWordWrap(True)
            innen.addWidget(hinweis)
        return karte

    # ------------------------------------------------------------------
    # Werte
    # ------------------------------------------------------------------

    def _setzen(self, vorlage: Vorlage):
        self.groesse_art.setCurrentIndex(max(0, self.groesse_art.findData(vorlage.groesse_art)))
        self.kante.setValue(vorlage.kante)
        self.prozent.setValue(vorlage.prozent)
        self.text.setText(vorlage.text)
        self.logo.setText(vorlage.logo)
        self.position.setCurrentIndex(max(0, self.position.findData(vorlage.position)))
        self.textfarbe.setCurrentIndex(max(0, self.textfarbe.findData(vorlage.textfarbe)))
        self.wz_groesse.setValue(vorlage.wz_groesse)
        self.deckkraft.setValue(vorlage.deckkraft)
        self.muster.setChecked(vorlage.muster)
        self.urheber.setText(vorlage.urheber)
        self.copyright.setText(vorlage.copyright)
        self.webadresse.setText(vorlage.webadresse)
        self.ki_verbot.setChecked(vorlage.ki_training_verbieten)
        self.kameradaten.setChecked(vorlage.kameradaten_entfernen)
        self._format = vorlage.format

    def vorlage(self) -> Vorlage:
        """Die Einstellungen des Dialogs; ein leeres Copyright folgt dem Urheber."""
        return Vorlage(
            groesse_art=self.groesse_art.currentData(),
            kante=self.kante.value(),
            prozent=self.prozent.value(),
            text=self.text.text(),
            logo=self.logo.text(),
            position=self.position.currentData(),
            wz_groesse=self.wz_groesse.value(),
            deckkraft=self.deckkraft.value(),
            muster=self.muster.isChecked(),
            textfarbe=self.textfarbe.currentData(),
            urheber=self.urheber.text().strip(),
            copyright=self.copyright.text().strip() or copyright_vorschlag(self.urheber.text()),
            webadresse=self.webadresse.text().strip(),
            ki_training_verbieten=self.ki_verbot.isChecked(),
            kameradaten_entfernen=self.kameradaten.isChecked(),
            format=self._format,
        ).begrenzt()

    def _verbinden(self):
        for feld in (self.groesse_art, self.position, self.textfarbe):
            feld.currentIndexChanged.connect(self._aktualisieren)
        for feld in (self.kante, self.prozent, self.wz_groesse, self.deckkraft):
            feld.valueChanged.connect(self._aktualisieren)
        for feld in (self.text, self.logo, self.urheber):
            feld.textChanged.connect(self._aktualisieren)
        self.muster.toggled.connect(self._aktualisieren)

    def _logo_waehlen(self):
        start = os.path.dirname(self.logo.text()) if self.logo.text() else ""
        pfad, _filter = QFileDialog.getOpenFileName(
            self, _("Logo wählen"), start,
            _("Bilder") + " (*.png *.webp *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if pfad:
            self.logo.setText(pfad)

    def _logo(self, pfad: str):
        if self._logo_cache[0] != pfad:
            self._logo_cache = (pfad, veroeffentlichen.logo_laden(pfad))
        return self._logo_cache[1]

    # ------------------------------------------------------------------
    # Anzeige
    # ------------------------------------------------------------------

    def _aktualisieren(self, *_args):
        vorlage = self.vorlage()
        kante = vorlage.groesse_art == "kante"
        self.kante.setVisible(kante)
        self.prozent.setVisible(not kante)
        for knopf in self.schnellwahl:
            knopf.setVisible(kante)
        self.position.setEnabled(not vorlage.muster)
        self.textfarbe.setEnabled(bool(vorlage.text.strip()))
        self.logo_weg.setEnabled(bool(vorlage.logo))
        self.wz_groesse_wert.setText(f"{vorlage.wz_groesse} %")
        self.deckkraft_wert.setText(f"{vorlage.deckkraft} %")
        self.copyright.setPlaceholderText(copyright_vorschlag(self.urheber.text())
                                          or _("z. B. © 2026 Name"))

        hoehe, breite = self._volle_form
        neu_h, neu_b = veroeffentlichen.zielgroesse(hoehe, breite, vorlage)
        if (neu_h, neu_b) == (hoehe, breite):
            self.groesse_info.setText(_("Das Bild bleibt {breite} × {hoehe} Pixel groß.")
                                      .format(breite=breite, hoehe=hoehe))
        else:
            self.groesse_info.setText(_("Ergebnis: {breite} × {hoehe} Pixel (Original "
                                        "{b0} × {h0})").format(breite=neu_b, hoehe=neu_h,
                                                               b0=breite, h0=hoehe))

        logo = self._logo(vorlage.logo)
        if vorlage.logo and logo is None:
            self.logo.setToolTip(_("Die Datei lässt sich nicht als Bild lesen."))
        else:
            self.logo.setToolTip(vorlage.logo)
        zeichen = veroeffentlichen.zeichen_bild(vorlage, logo)
        bild, _alpha = veroeffentlichen.wasserzeichen(self._vorschau, None, vorlage, zeichen)
        self._bild_zeigen(veroeffentlichen.als_8bit(bild))

    def _bild_zeigen(self, rgb: np.ndarray):
        hoehe, breite = rgb.shape[:2]
        rgb = np.ascontiguousarray(rgb)
        qbild = QImage(rgb.data, breite, hoehe, 3 * breite, QImage.Format.Format_RGB888).copy()
        self.bild.setPixmap(QPixmap.fromImage(qbild).scaled(
            *VORSCHAU, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def mit_format(self, endung: str) -> Vorlage:
        """Die Vorlage mit dem zuletzt gewaehlten Dateiformat."""
        return dataclasses.replace(self.vorlage(), format=endung).begrenzt()
