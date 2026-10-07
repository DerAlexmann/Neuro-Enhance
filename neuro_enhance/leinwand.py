"""
Leinwand: Bild eingepasst oder vergroessert, Verschieben, Zuschnittrahmen

Eingepasst zeigt die Leinwand die Vorschau des ganzen Bildes. Vergroessert
(100, 200, 400 %) zeigt sie einen Ausschnitt des Bildes in voller Aufloesung;
welcher Bereich gebraucht wird, meldet sie mit `ansicht_geaendert`, und die
Seite liefert ihn mit `zeigen_ausschnitt` nach. 100 % heisst: ein Bildpixel
auf einem Bildschirmpixel.

Im Zuschnittmodus liegt ueber dem eingepassten Bild ein Rahmen mit Griffen an
Ecken und Kanten; ein festes Seitenverhaeltnis wird beim Ziehen eingehalten.

Im Klickmodus meldet die Leinwand Klicks ins Bild (`bild_geklickt`: Bildpunkt und
ob links geklickt wurde) und zeichnet die bisherigen Klicks als Punkte. Ziehen
verschiebt vergroessert weiterhin das Bild - als Klick zaehlt nur, was kaum bewegt
wurde.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

from . import farben

ZOOMSTUFEN = (None, 1.0, 2.0, 4.0)   # None = eingepasst
RAND = 8
GRIFF = 10                           # so nah muss die Maus an Kante oder Ecke sein
KLICK_WEG = 4                        # weiter bewegt ist es Ziehen, kein Klick
MIN_ZUSCHNITT = 0.02


class Leinwand(QFrame):
    datei_abgelegt = Signal(str)
    ansicht_geaendert = Signal()
    zuschnitt_geaendert = Signal(tuple)
    bild_geklickt = Signal(float, float, bool)       # x, y im Bild; True = linke Taste

    def __init__(self):
        super().__init__(objectName="leinwand")
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setMinimumSize(320, 240)
        self._bild: QImage | None = None
        self._puffer = None                   # haelt die Pixel fest, auf die QImage zeigt
        self._ausschnitt_ort = (0, 0)         # Lage des gezeigten Ausschnitts im Vollbild

        self.zoom: float | None = None
        self.voll_form: tuple[int, int] | None = None   # Hoehe, Breite des Ergebnisses
        self.mitte = (0.0, 0.0)              # Bildpunkt (x, y) in der Fenstermitte
        self.zuschnitt: tuple[float, float, float, float] | None = None
        self.seitenverhaeltnis: float | None = None      # Breite / Hoehe in Pixeln
        self._ziehen = None                   # (Art, Startpunkt, Ausgangswert)
        self.klickmodus = False
        self.klickpunkte: list[tuple[float, float, bool]] = []   # im Bild, zum Zeichnen
        self._klick = None                    # (Fensterpunkt, linke Taste) beim Druecken

        aufbau = QVBoxLayout(self)
        aufbau.addStretch(1)
        self.leer = QLabel(objectName="leer")
        self.leer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        aufbau.addWidget(self.leer)
        self.hinweis = QLabel(objectName="nebentext")
        self.hinweis.setAlignment(Qt.AlignmentFlag.AlignCenter)
        aufbau.addWidget(self.hinweis)
        aufbau.addStretch(1)

    # ------------------------------------------------------------------
    # Bild setzen
    # ------------------------------------------------------------------

    def _setzen(self, rgb: np.ndarray | None):
        if rgb is None:
            self._bild = self._puffer = None
        else:
            self._puffer = np.ascontiguousarray(rgb)
            hoehe, breite = self._puffer.shape[:2]
            self._bild = QImage(self._puffer.data, breite, hoehe, breite * 3,
                                QImage.Format.Format_RGB888)
        self.leer.setVisible(self._bild is None)
        self.hinweis.setVisible(self._bild is None)
        self.update()

    def zeigen(self, rgb: np.ndarray | None):
        """Vorschau des ganzen Bildes (eingepasste Ansicht)."""
        self._ausschnitt_ort = (0, 0)
        self._setzen(rgb)

    def zeigen_ausschnitt(self, rgb: np.ndarray, x0: int, y0: int):
        """Ausschnitt in voller Aufloesung, dessen linke obere Ecke bei (x0, y0) liegt."""
        self._ausschnitt_ort = (x0, y0)
        self._setzen(rgb)

    # ------------------------------------------------------------------
    # Abbildung Bild <-> Fenster
    # ------------------------------------------------------------------

    def _dpr(self) -> float:
        return self.devicePixelRatioF()

    def _flaeche(self) -> QRectF:
        return QRectF(self.contentsRect()).adjusted(RAND, RAND, -RAND, -RAND)

    def _massstab(self) -> float:
        """Fensterpunkte je Bildpixel des Vollbildes."""
        if self.voll_form is None:
            return 1.0
        if self.zoom is None:
            flaeche = self._flaeche()
            hoehe, breite = self.voll_form
            return min(flaeche.width() / breite, flaeche.height() / hoehe, 1.0 / self._dpr())
        return self.zoom / self._dpr()

    def _ursprung(self) -> QPointF:
        """Fensterposition des Bildpunkts (0, 0)."""
        s = self._massstab()
        mitte = self._flaeche().center()
        if self.zoom is None:
            hoehe, breite = self.voll_form
            return QPointF(mitte.x() - breite * s / 2, mitte.y() - hoehe * s / 2)
        return QPointF(mitte.x() - self.mitte[0] * s, mitte.y() - self.mitte[1] * s)

    def zu_bild(self, punkt: QPointF) -> tuple[float, float]:
        s, ursprung = self._massstab(), self._ursprung()
        return (punkt.x() - ursprung.x()) / s, (punkt.y() - ursprung.y()) / s

    def zu_fenster(self, x: float, y: float) -> QPointF:
        s, ursprung = self._massstab(), self._ursprung()
        return QPointF(ursprung.x() + x * s, ursprung.y() + y * s)

    def sichtbarer_bereich(self) -> tuple[int, int, int, int] | None:
        """Sichtbarer Bereich des Vollbildes als (x0, y0, breite, hoehe), oder None."""
        if self.zoom is None or self.voll_form is None:
            return None
        hoehe, breite = self.voll_form
        links, oben = self.zu_bild(self._flaeche().topLeft())
        rechts, unten = self.zu_bild(self._flaeche().bottomRight())
        x0, y0 = max(0, int(links)), max(0, int(oben))
        x1, y1 = min(breite, int(np.ceil(rechts)) + 1), min(hoehe, int(np.ceil(unten)) + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1 - x0, y1 - y0

    def _mitte_begrenzen(self):
        if self.voll_form is None:
            return
        hoehe, breite = self.voll_form
        self.mitte = (min(max(self.mitte[0], 0.0), float(breite)),
                      min(max(self.mitte[1], 0.0), float(hoehe)))

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def zoom_setzen(self, zoom: float | None, um: QPointF | None = None):
        """Zoomstufe wechseln; der Bildpunkt unter `um` bleibt, wo er ist."""
        if self.voll_form is None or zoom == self.zoom:
            return
        if um is None:
            um = self._flaeche().center()
        x, y = self.zu_bild(um)
        self.zoom = zoom
        if zoom is not None:
            s = self._massstab()
            mitte = self._flaeche().center()
            self.mitte = (x - (um.x() - mitte.x()) / s, y - (um.y() - mitte.y()) / s)
            self._mitte_begrenzen()
        self._zeiger()
        self.ansicht_geaendert.emit()

    def zoom_schritt(self, richtung: int, um: QPointF | None = None):
        stufe = ZOOMSTUFEN.index(self.zoom) if self.zoom in ZOOMSTUFEN else 0
        neu = min(max(stufe + richtung, 0), len(ZOOMSTUFEN) - 1)
        self.zoom_setzen(ZOOMSTUFEN[neu], um)

    def wheelEvent(self, ereignis):                   # noqa: N802 - Qt-Name
        if self._bild is None or self.zuschnitt is not None:
            return
        richtung = 1 if ereignis.angleDelta().y() > 0 else -1
        self.zoom_schritt(richtung, ereignis.position())

    # ------------------------------------------------------------------
    # Zuschnittrahmen
    # ------------------------------------------------------------------

    def _zuschnitt_fenster(self) -> QRectF:
        hoehe, breite = self.voll_form
        x0, y0, x1, y1 = self.zuschnitt
        return QRectF(self.zu_fenster(x0 * breite, y0 * hoehe),
                      self.zu_fenster(x1 * breite, y1 * hoehe))

    def _griff_bei(self, punkt: QPointF) -> str | None:
        rahmen = self._zuschnitt_fenster()
        links = abs(punkt.x() - rahmen.left()) <= GRIFF
        rechts = abs(punkt.x() - rahmen.right()) <= GRIFF
        oben = abs(punkt.y() - rahmen.top()) <= GRIFF
        unten = abs(punkt.y() - rahmen.bottom()) <= GRIFF
        innen_x = rahmen.left() - GRIFF <= punkt.x() <= rahmen.right() + GRIFF
        innen_y = rahmen.top() - GRIFF <= punkt.y() <= rahmen.bottom() + GRIFF
        art = ("o" if oben and innen_x else "u" if unten and innen_x else "") + \
              ("l" if links and innen_y else "r" if rechts and innen_y else "")
        if art:
            return art
        return "innen" if rahmen.contains(punkt) else None

    def _zuschnitt_ziehen(self, art: str, start, ausgang, punkt: QPointF):
        hoehe, breite = self.voll_form
        x, y = self.zu_bild(punkt)
        dx, dy = (x - start[0]) / breite, (y - start[1]) / hoehe
        x0, y0, x1, y1 = ausgang
        if art == "innen":
            dx = min(max(dx, -x0), 1 - x1)
            dy = min(max(dy, -y0), 1 - y1)
            self.zuschnitt = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
            return
        if "l" in art:
            x0 = min(max(x0 + dx, 0.0), x1 - MIN_ZUSCHNITT)
        if "r" in art:
            x1 = max(min(x1 + dx, 1.0), x0 + MIN_ZUSCHNITT)
        if "o" in art:
            y0 = min(max(y0 + dy, 0.0), y1 - MIN_ZUSCHNITT)
        if "u" in art:
            y1 = max(min(y1 + dy, 1.0), y0 + MIN_ZUSCHNITT)
        if self.seitenverhaeltnis:
            # Hoehe aus der Breite ableiten; die gezogene Seite bestimmt, wo angepasst wird
            soll_h = (x1 - x0) * breite / self.seitenverhaeltnis / hoehe
            if "o" in art:
                y0 = y1 - soll_h
            else:
                y1 = y0 + soll_h
            if y0 < 0 or y1 > 1:
                # passt nicht: von der Hoehe her begrenzen
                y0, y1 = max(y0, 0.0), min(y1, 1.0)
                soll_w = (y1 - y0) * hoehe * self.seitenverhaeltnis / breite
                if "l" in art:
                    x0 = x1 - soll_w
                else:
                    x1 = x0 + soll_w
        self.zuschnitt = (x0, y0, x1, y1)

    def seitenverhaeltnis_setzen(self, verhaeltnis: float | None):
        """Festes Verhaeltnis waehlen und den Rahmen mittig darauf bringen."""
        self.seitenverhaeltnis = verhaeltnis
        if verhaeltnis is None or self.zuschnitt is None or self.voll_form is None:
            return
        self.zuschnitt = rahmen_mit_verhaeltnis(self.zuschnitt, verhaeltnis, self.voll_form)
        self.zuschnitt_geaendert.emit(self.zuschnitt)
        self.update()

    # ------------------------------------------------------------------
    # Maus
    # ------------------------------------------------------------------

    def klickmodus_setzen(self, an: bool):
        self.klickmodus = an
        self.klickpunkte = []
        self._klick = None
        self._zeiger()
        self.update()

    def _zeiger(self):
        if self.klickmodus:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.OpenHandCursor if self.zoom
                           else Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, ereignis):              # noqa: N802 - Qt-Name
        knopf = ereignis.button()
        if self.voll_form is None:
            return
        punkt = ereignis.position()
        if self.klickmodus and self.zuschnitt is None and knopf in (
                Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._klick = (punkt, knopf == Qt.MouseButton.LeftButton)
        if knopf != Qt.MouseButton.LeftButton:
            return
        if self.zuschnitt is not None:
            art = self._griff_bei(punkt)
            if art:
                self._ziehen = (art, self.zu_bild(punkt), self.zuschnitt)
        elif self.zoom is not None:
            self._ziehen = ("schieben", (punkt.x(), punkt.y()), self.mitte)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, ereignis):               # noqa: N802 - Qt-Name
        punkt = ereignis.position()
        if self._ziehen is None:
            if self.zuschnitt is not None and self.voll_form is not None:
                art = self._griff_bei(punkt)
                form = {"ol": Qt.CursorShape.SizeFDiagCursor, "ur": Qt.CursorShape.SizeFDiagCursor,
                        "or": Qt.CursorShape.SizeBDiagCursor, "ul": Qt.CursorShape.SizeBDiagCursor,
                        "l": Qt.CursorShape.SizeHorCursor, "r": Qt.CursorShape.SizeHorCursor,
                        "o": Qt.CursorShape.SizeVerCursor, "u": Qt.CursorShape.SizeVerCursor,
                        "innen": Qt.CursorShape.SizeAllCursor}.get(art,
                                                                   Qt.CursorShape.ArrowCursor)
                self.setCursor(form)
            return
        art, start, ausgang = self._ziehen
        if self._klick is not None and art == "schieben":
            weg = punkt - self._klick[0]
            if abs(weg.x()) + abs(weg.y()) <= KLICK_WEG:
                return                            # noch ein Klick, kein Verschieben
            self._klick = None
        if art == "schieben":
            s = self._massstab()
            self.mitte = (ausgang[0] - (punkt.x() - start[0]) / s,
                          ausgang[1] - (punkt.y() - start[1]) / s)
            self._mitte_begrenzen()
            self.update()
            self.ansicht_geaendert.emit()
        else:
            self._zuschnitt_ziehen(art, start, ausgang, punkt)
            self.update()

    def mouseReleaseEvent(self, ereignis):            # noqa: N802 - Qt-Name
        if self._klick is not None:
            start, links = self._klick
            self._klick = None
            weg = ereignis.position() - start
            if abs(weg.x()) + abs(weg.y()) <= KLICK_WEG:
                x, y = self.zu_bild(start)
                hoehe, breite = self.voll_form
                if 0 <= x < breite and 0 <= y < hoehe:
                    self.bild_geklickt.emit(x, y, links)
        if self._ziehen is None:
            return
        art = self._ziehen[0]
        self._ziehen = None
        if art == "schieben":
            self._zeiger()
        else:
            self.zuschnitt_geaendert.emit(self.zuschnitt)

    def mouseDoubleClickEvent(self, ereignis):        # noqa: N802 - Qt-Name
        if self._bild is None or self.zuschnitt is not None or self.klickmodus:
            return
        self.zoom_setzen(None if self.zoom else 1.0, ereignis.position())

    # ------------------------------------------------------------------
    # Zeichnen
    # ------------------------------------------------------------------

    def paintEvent(self, ereignis):                   # noqa: N802 - Qt-Name
        super().paintEvent(ereignis)
        if self._bild is None or self.voll_form is None:
            return
        maler = QPainter(self)
        maler.setClipRect(self._flaeche())
        maler.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom is None)
        if self.zoom is None:
            hoehe, breite = self.voll_form
            ziel = QRectF(self.zu_fenster(0, 0), self.zu_fenster(breite, hoehe))
        else:
            x0, y0 = self._ausschnitt_ort
            ziel = QRectF(self.zu_fenster(x0, y0),
                          self.zu_fenster(x0 + self._bild.width(), y0 + self._bild.height()))
        maler.drawImage(ziel, self._bild)
        if self.zuschnitt is not None:
            self._rahmen_zeichnen(maler, ziel)
        elif self.klickmodus:
            self._klicks_zeichnen(maler)

    def _klicks_zeichnen(self, maler: QPainter):
        """Bisherige Klicks: gruen dazu, rot weg - mit dunklem Rand auf jedem Grund."""
        maler.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for x, y, dazu in self.klickpunkte:
            mitte = self.zu_fenster(x, y)
            maler.setPen(QPen(QColor(0, 0, 0, 200), 2))
            maler.setBrush(QColor(60, 200, 90) if dazu else QColor(230, 60, 50))
            maler.drawEllipse(mitte, 6, 6)
            maler.setPen(QPen(QColor(255, 255, 255), 2))
            maler.drawLine(mitte + QPointF(-3, 0), mitte + QPointF(3, 0))
            if dazu:
                maler.drawLine(mitte + QPointF(0, -3), mitte + QPointF(0, 3))

    def _rahmen_zeichnen(self, maler: QPainter, bild: QRectF):
        rollen = farben.THEMES[farben.CURRENT_THEME]
        rahmen = self._zuschnitt_fenster()
        abdunkeln = QColor(0, 0, 0, 120)
        maler.fillRect(QRectF(bild.left(), bild.top(), bild.width(), rahmen.top() - bild.top()),
                       abdunkeln)
        maler.fillRect(QRectF(bild.left(), rahmen.bottom(), bild.width(),
                              bild.bottom() - rahmen.bottom()), abdunkeln)
        maler.fillRect(QRectF(bild.left(), rahmen.top(), rahmen.left() - bild.left(),
                              rahmen.height()), abdunkeln)
        maler.fillRect(QRectF(rahmen.right(), rahmen.top(), bild.right() - rahmen.right(),
                              rahmen.height()), abdunkeln)
        drittel = QPen(QColor(255, 255, 255, 110), 1)
        maler.setPen(drittel)
        for anteil in (1 / 3, 2 / 3):
            x = rahmen.left() + anteil * rahmen.width()
            y = rahmen.top() + anteil * rahmen.height()
            maler.drawLine(QPointF(x, rahmen.top()), QPointF(x, rahmen.bottom()))
            maler.drawLine(QPointF(rahmen.left(), y), QPointF(rahmen.right(), y))
        maler.setPen(QPen(QColor(rollen["ON_ACCENT"]), 1.5))
        maler.drawRect(rahmen)
        maler.setPen(QPen(QColor(rollen["ACCENT"]), 4))
        lang = 16
        for ecke, (sx, sy) in ((rahmen.topLeft(), (1, 1)), (rahmen.topRight(), (-1, 1)),
                               (rahmen.bottomLeft(), (1, -1)), (rahmen.bottomRight(), (-1, -1))):
            maler.drawLine(ecke, ecke + QPointF(sx * lang, 0))
            maler.drawLine(ecke, ecke + QPointF(0, sy * lang))

    def resizeEvent(self, ereignis):                  # noqa: N802 - Qt-Name
        super().resizeEvent(ereignis)
        if self.zoom is not None:
            self.ansicht_geaendert.emit()

    # ------------------------------------------------------------------
    # Ziehen und Ablegen von Dateien
    # ------------------------------------------------------------------

    def dragEnterEvent(self, ereignis):               # noqa: N802 - Qt-Name
        if ereignis.mimeData().hasUrls():
            ereignis.acceptProposedAction()

    def dropEvent(self, ereignis):                    # noqa: N802 - Qt-Name
        for url in ereignis.mimeData().urls():
            if url.isLocalFile():
                self.datei_abgelegt.emit(url.toLocalFile())
                break


def rahmen_mit_verhaeltnis(rahmen, verhaeltnis: float, form) -> tuple[float, float, float, float]:
    """Groesster Rahmen mit dem Verhaeltnis (Breite/Hoehe in Pixeln) mittig im alten."""
    hoehe, breite = form
    x0, y0, x1, y1 = rahmen
    w, h = (x1 - x0) * breite, (y1 - y0) * hoehe
    if w / h > verhaeltnis:
        w = h * verhaeltnis
    else:
        h = w / verhaeltnis
    mx, my = (x0 + x1) / 2 * breite, (y0 + y1) / 2 * hoehe
    return ((mx - w / 2) / breite, (my - h / 2) / hoehe,
            (mx + w / 2) / breite, (my + h / 2) / hoehe)
