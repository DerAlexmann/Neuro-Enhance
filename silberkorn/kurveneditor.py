"""
Kurveneditor: Gradationskurve mit Histogramm im Hintergrund

Klick ins Feld setzt einen Punkt, Ziehen verschiebt ihn, Doppelklick auf einen
Punkt entfernt ihn. Die Endpunkte lassen sich nur in der Hoehe ziehen - so
werden Schwarz- und Weisspunkt angehoben oder abgesenkt.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from . import farben, kurven

RAND = 7                     # Platz um das Feld, damit Punkte am Rand ganz sichtbar sind
FANGRADIUS = 9               # so nah muss ein Klick an einem Punkt liegen

# Kurve -> Farbrolle im Farbschema
KURVENFARBE = {
    "kurve_hell": "TEXT",
    "kurve_rot": "DANGER",
    "kurve_gruen": "OK",
    "kurve_blau": "ACCENT",
}


class KurvenFlaeche(QWidget):
    geaendert = Signal(str, tuple)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)
        self.punkte = dict.fromkeys(KURVENFARBE, kurven.IDENTITAET)
        self.aktiv = "kurve_hell"
        self.histogramm: np.ndarray | None = None
        self._gezogen: int | None = None

    def heightForWidth(self, breite):                # noqa: N802 - Qt-Name
        return breite

    def hasHeightForWidth(self):                     # noqa: N802 - Qt-Name
        return True

    # ------------------------------------------------------------------
    # Umrechnung Feld <-> Kurve
    # ------------------------------------------------------------------

    def _feld(self) -> QRectF:
        seite = min(self.width(), self.height()) - 2 * RAND
        return QRectF((self.width() - seite) / 2, RAND, seite, seite)

    def _zu_bild(self, x: float, y: float) -> QPointF:
        feld = self._feld()
        return QPointF(feld.left() + x * feld.width(), feld.bottom() - y * feld.height())

    def _zu_kurve(self, punkt: QPointF) -> tuple[float, float]:
        feld = self._feld()
        return ((punkt.x() - feld.left()) / feld.width(),
                (feld.bottom() - punkt.y()) / feld.height())

    def _punkt_bei(self, position: QPointF) -> int | None:
        for index, (x, y) in enumerate(self.punkte[self.aktiv]):
            abstand = self._zu_bild(x, y) - position
            if abstand.x() ** 2 + abstand.y() ** 2 <= FANGRADIUS ** 2:
                return index
        return None

    # ------------------------------------------------------------------
    # Maus
    # ------------------------------------------------------------------

    def _melden(self):
        self.update()
        self.geaendert.emit(self.aktiv, self.punkte[self.aktiv])

    def mousePressEvent(self, ereignis):             # noqa: N802 - Qt-Name
        if ereignis.button() != Qt.MouseButton.LeftButton or not self.isEnabled():
            return
        position = ereignis.position()
        index = self._punkt_bei(position)
        if index is None:
            x, y = self._zu_kurve(position)
            neu, index = kurven.einfuegen(self.punkte[self.aktiv], x, y)
            if index is None:
                return
            self.punkte[self.aktiv] = neu
            self._melden()
        self._gezogen = index

    def mouseMoveEvent(self, ereignis):              # noqa: N802 - Qt-Name
        position = ereignis.position()
        if self._gezogen is None:
            ueber = self._punkt_bei(position) is not None
            self.setCursor(Qt.CursorShape.SizeAllCursor if ueber else Qt.CursorShape.CrossCursor)
            return
        x, y = self._zu_kurve(position)
        self.punkte[self.aktiv] = kurven.verschieben(self.punkte[self.aktiv], self._gezogen, x, y)
        self._melden()

    def mouseReleaseEvent(self, ereignis):           # noqa: N802 - Qt-Name
        self._gezogen = None

    def mouseDoubleClickEvent(self, ereignis):       # noqa: N802 - Qt-Name
        index = self._punkt_bei(ereignis.position())
        if index is not None:
            self.punkte[self.aktiv] = kurven.entfernen(self.punkte[self.aktiv], index)
            self._gezogen = None
            self._melden()

    # ------------------------------------------------------------------
    # Zeichnen
    # ------------------------------------------------------------------

    def paintEvent(self, ereignis):                  # noqa: N802 - Qt-Name
        rollen = farben.THEMES[farben.CURRENT_THEME]
        maler = QPainter(self)
        maler.setRenderHint(QPainter.RenderHint.Antialiasing)
        feld = self._feld()

        maler.fillRect(feld, QColor(rollen["CARD_ALT"]))
        if self.histogramm is not None and self.histogramm.max() > 0:
            # Wurzel statt linear: sonst schluckt eine grosse Flaeche alles andere
            hoehen = np.sqrt(self.histogramm / self.histogramm.max())
            pfad = QPainterPath(QPointF(feld.left(), feld.bottom()))
            for i, h in enumerate(hoehen):
                pfad.lineTo(feld.left() + (i + 0.5) / len(hoehen) * feld.width(),
                            feld.bottom() - h * feld.height() * 0.9)
            pfad.lineTo(feld.right(), feld.bottom())
            pfad.closeSubpath()
            flaeche = QColor(rollen["MUTED"])
            flaeche.setAlpha(55)
            maler.fillPath(pfad, flaeche)

        maler.setPen(QPen(QColor(rollen["BORDER"]), 1))
        for anteil in (0.25, 0.5, 0.75):
            maler.drawLine(QPointF(feld.left() + anteil * feld.width(), feld.top()),
                           QPointF(feld.left() + anteil * feld.width(), feld.bottom()))
            maler.drawLine(QPointF(feld.left(), feld.top() + anteil * feld.height()),
                           QPointF(feld.right(), feld.top() + anteil * feld.height()))
        maler.drawRect(feld)
        maler.setPen(QPen(QColor(rollen["MUTED"]), 1, Qt.PenStyle.DashLine))
        maler.drawLine(feld.bottomLeft(), feld.topRight())

        # Die anderen Kurven blass dahinter, sofern sie veraendert sind
        for name, rolle in KURVENFARBE.items():
            if name != self.aktiv and not kurven.ist_identitaet(self.punkte[name]):
                farbe = QColor(rollen[rolle])
                farbe.setAlpha(90)
                self._kurve_zeichnen(maler, name, farbe, 1.2)

        farbe = QColor(rollen[KURVENFARBE[self.aktiv]])
        self._kurve_zeichnen(maler, self.aktiv, farbe, 2.0)
        maler.setPen(QPen(QColor(rollen["CARD"]), 1.5))
        maler.setBrush(farbe)
        for x, y in self.punkte[self.aktiv]:
            maler.drawEllipse(self._zu_bild(x, y), 5, 5)

    def _kurve_zeichnen(self, maler: QPainter, name: str, farbe: QColor, dicke: float):
        werte = kurven.tabelle(self.punkte[name], 256)
        pfad = QPainterPath(self._zu_bild(0.0, float(werte[0])))
        for i in range(1, len(werte)):
            pfad.lineTo(self._zu_bild(i / (len(werte) - 1), float(werte[i])))
        maler.setPen(QPen(farbe, dicke))
        maler.setBrush(Qt.BrushStyle.NoBrush)
        maler.drawPath(pfad)


class KurvenEditor(QWidget):
    """Kanalwahl und Kurvenfeld. Sendet geaendert(name, punkte)."""

    geaendert = Signal(str, tuple)

    def __init__(self, beschriften, kanal_texte: dict[str, str], zuruecksetzen_text: str):
        super().__init__()
        aufbau = QVBoxLayout(self)
        aufbau.setContentsMargins(0, 0, 0, 0)
        aufbau.setSpacing(8)

        leiste = QHBoxLayout()
        leiste.setSpacing(4)
        self._gruppe = QButtonGroup(self)
        self._knoepfe: dict[str, QPushButton] = {}
        for name in KURVENFARBE:
            knopf = QPushButton(objectName="kanal")
            knopf.setCheckable(True)
            beschriften(knopf.setText, kanal_texte[name])
            knopf.clicked.connect(lambda _an, n=name: self._kanal_waehlen(n))
            self._gruppe.addButton(knopf)
            self._knoepfe[name] = knopf
            leiste.addWidget(knopf)
        self._knoepfe["kurve_hell"].setChecked(True)
        leiste.addStretch(1)
        self.zuruecksetzen_knopf = QPushButton(objectName="kanal")
        beschriften(self.zuruecksetzen_knopf.setText, zuruecksetzen_text)
        self.zuruecksetzen_knopf.clicked.connect(self._aktive_zuruecksetzen)
        leiste.addWidget(self.zuruecksetzen_knopf)
        aufbau.addLayout(leiste)

        self.flaeche = KurvenFlaeche()
        self.flaeche.geaendert.connect(self.geaendert)
        aufbau.addWidget(self.flaeche)

    def _kanal_waehlen(self, name: str):
        self.flaeche.aktiv = name
        self.flaeche.update()

    def _aktive_zuruecksetzen(self):
        name = self.flaeche.aktiv
        self.flaeche.punkte[name] = kurven.IDENTITAET
        self.flaeche.update()
        self.geaendert.emit(name, kurven.IDENTITAET)

    def alle_setzen(self, werte: dict[str, kurven.Punkte]):
        self.flaeche.punkte.update(werte)
        self.flaeche.update()

    def histogramm_zeigen(self, histogramm: np.ndarray | None):
        self.flaeche.histogramm = histogramm
        self.flaeche.update()

    def setEnabled(self, an: bool):                  # noqa: N802 - Qt-Name
        super().setEnabled(an)
        self.flaeche.setEnabled(an)
