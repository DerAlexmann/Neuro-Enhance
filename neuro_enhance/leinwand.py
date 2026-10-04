"""
Leinwand: zeigt das Bild eingepasst und nimmt Dateien per Ziehen und Ablegen an

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


class Leinwand(QFrame):
    datei_abgelegt = Signal(str)

    def __init__(self):
        super().__init__(objectName="leinwand")
        self.setAcceptDrops(True)
        self.setMinimumSize(320, 240)
        self._bild: QImage | None = None
        self._puffer = None                   # haelt die Pixel fest, auf die QImage zeigt

        aufbau = QVBoxLayout(self)
        aufbau.addStretch(1)
        self.leer = QLabel(objectName="leer")
        self.leer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        aufbau.addWidget(self.leer)
        self.hinweis = QLabel(objectName="nebentext")
        self.hinweis.setAlignment(Qt.AlignmentFlag.AlignCenter)
        aufbau.addWidget(self.hinweis)
        aufbau.addStretch(1)

    def zeigen(self, rgb: np.ndarray | None):
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

    def paintEvent(self, ereignis):               # noqa: N802 - Qt-Name
        super().paintEvent(ereignis)
        if self._bild is None:
            return
        rahmen = QRectF(self.contentsRect()).adjusted(8, 8, -8, -8)
        # Eingepasst, aber nie groesser als ein Bildpunkt je Bildschirmpunkt -
        # kleine Bilder wuerden sonst unscharf aufgeblasen.
        massstab = min(rahmen.width() / self._bild.width(),
                       rahmen.height() / self._bild.height(),
                       1.0 / self.devicePixelRatioF())
        breite = self._bild.width() * massstab
        hoehe = self._bild.height() * massstab
        ziel = QRectF(rahmen.center().x() - breite / 2, rahmen.center().y() - hoehe / 2,
                      breite, hoehe)
        maler = QPainter(self)
        maler.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        maler.drawImage(ziel, self._bild)

    def dragEnterEvent(self, ereignis):           # noqa: N802 - Qt-Name
        if ereignis.mimeData().hasUrls():
            ereignis.acceptProposedAction()

    def dropEvent(self, ereignis):                # noqa: N802 - Qt-Name
        for url in ereignis.mimeData().urls():
            if url.isLocalFile():
                self.datei_abgelegt.emit(url.toLocalFile())
                break
