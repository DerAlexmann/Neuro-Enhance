"""
Mausrad blaettert, statt Regler zu verstellen

Qt gibt das Mausrad an Schieberegler, Auswahllisten und Zahlenfelder unter dem
Zeiger. In der langen Reglerspalte hielt das Blaettern dann an, sobald ein
Regler unter den Zeiger kam, und verstellte ihn. Der Filter leitet das Rad
dieser Elemente an die Bildlaufleiste der umgebenden Flaeche weiter; Regler
reagieren nur noch auf Ziehen, Klicken und Tastatur.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QScrollBar,
)

BEDIENELEMENTE = (QAbstractSlider, QComboBox, QAbstractSpinBox)


def _bildlaufleiste(element, waagrecht: bool) -> QScrollBar | None:
    """Leiste der naechsten umgebenden Bildlaufflaeche, die blaettern kann."""
    eltern = element.parentWidget()
    while eltern is not None:
        if isinstance(eltern, QAbstractScrollArea):
            leiste = eltern.horizontalScrollBar() if waagrecht else eltern.verticalScrollBar()
            if leiste.maximum() > leiste.minimum():
                return leiste
        eltern = eltern.parentWidget()
    return None


class MausradFilter(QObject):
    def eventFilter(self, ziel, ereignis):                # noqa: N802 - Qt-Name
        if (ereignis.type() != QEvent.Type.Wheel or not isinstance(ziel, BEDIENELEMENTE)
                or isinstance(ziel, QScrollBar)):
            return False
        delta = ereignis.angleDelta()
        leiste = _bildlaufleiste(ziel, abs(delta.x()) > abs(delta.y()))
        if leiste is not None:
            QApplication.sendEvent(leiste, ereignis)
        return True                                       # nie an den Regler selbst


def einrichten(app: QApplication) -> MausradFilter:
    filter_ = MausradFilter(app)
    app.installEventFilter(filter_)
    return filter_
