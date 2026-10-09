"""Mausrad blaettert die Reglerspalte, statt Regler zu verstellen; Linkfarbe."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QPalette, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from silberkorn import farben, mausrad  # noqa: E402


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication([])
    filter_ = mausrad.einrichten(app)
    yield app
    app.removeEventFilter(filter_)


def rad(element, nach_unten=True):
    schritt = -120 if nach_unten else 120
    ereignis = QWheelEvent(QPointF(5, 5), QPointF(element.mapToGlobal(QPoint(5, 5))),
                           QPoint(0, 0), QPoint(0, schritt), Qt.MouseButton.NoButton,
                           Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(element, ereignis)


def test_regler_bleibt_flaeche_blaettert(app):
    flaeche = QScrollArea()
    inhalt = QWidget()
    spalte = QVBoxLayout(inhalt)
    regler = QSlider(Qt.Orientation.Horizontal)
    regler.setRange(-100, 100)
    auswahl = QComboBox()
    auswahl.addItems(["a", "b", "c"])
    spalte.addWidget(regler)
    spalte.addWidget(auswahl)
    inhalt.setMinimumHeight(2000)
    flaeche.setWidget(inhalt)
    flaeche.resize(300, 200)
    flaeche.show()
    app.processEvents()
    leiste = flaeche.verticalScrollBar()
    assert leiste.maximum() > 0

    rad(regler)
    assert regler.value() == 0
    assert leiste.value() > 0
    vorher = leiste.value()
    rad(auswahl)
    assert auswahl.currentIndex() == 0
    assert leiste.value() > vorher
    flaeche.close()


def test_regler_ohne_flaeche_bleibt_auch(app):
    regler = QSlider(Qt.Orientation.Horizontal)
    regler.setRange(0, 10)
    rad(regler, nach_unten=False)
    assert regler.value() == 0


def test_linkfarbe_folgt_dem_schema(app):
    for schema in ("light", "dark"):
        farben.apply_theme(schema)
        farben.palette_setzen(app)
        assert app.palette().color(QPalette.ColorRole.Link).name() == \
            farben.THEMES[schema]["LINK"]
    farben.apply_theme(farben.DEFAULT_THEME)
