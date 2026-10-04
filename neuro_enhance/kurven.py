"""
Gradationskurven: Stuetzpunkte bearbeiten und in eine Nachschlagetabelle wandeln

Eine Kurve ist ein Tupel von Stuetzpunkten (x, y) im Bereich 0..1, nach x
sortiert, mit festen Endpunkten bei x = 0 und x = 1. Zwischen den Punkten wird
monoton kubisch interpoliert (Fritsch-Carlson): Die Kurve laeuft glatt durch
jeden Punkt, schiesst aber nie ueber - eine gewoehnliche kubische Spline wuerde
bei eng liegenden Punkten Wellen schlagen und die Tonwerte umkehren.

Reine Python-/NumPy-Logik ohne Qt und ohne Grafikkarte.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import numpy as np

Punkte = tuple[tuple[float, float], ...]

IDENTITAET: Punkte = ((0.0, 0.0), (1.0, 1.0))
TABELLE = 1024                     # Stuetzstellen der Nachschlagetabelle
MIN_ABSTAND = 0.02                 # kleinster Abstand zweier Punkte auf der x-Achse
MAX_PUNKTE = 16


def _begrenzen(wert: float, unten: float, oben: float) -> float:
    return min(max(wert, unten), oben)


def einfuegen(punkte: Punkte, x: float, y: float) -> tuple[Punkte, int | None]:
    """Fuegt einen Punkt ein; Rueckgabe: neue Kurve und Index des Punktes.

    Liegt x zu nah an einem vorhandenen Punkt oder ist die Kurve voll, bleibt
    sie unveraendert und der Index ist None.
    """
    x, y = _begrenzen(x, 0.0, 1.0), _begrenzen(y, 0.0, 1.0)
    if len(punkte) >= MAX_PUNKTE or any(abs(px - x) < MIN_ABSTAND for px, _ in punkte):
        return punkte, None
    neu = tuple(sorted((*punkte, (x, y))))
    return neu, neu.index((x, y))


def verschieben(punkte: Punkte, index: int, x: float, y: float) -> Punkte:
    """Verschiebt einen Punkt, ohne dass er seine Nachbarn ueberholt.

    Die Endpunkte bleiben auf ihrer x-Position und lassen sich nur in der
    Hoehe bewegen - so kann man Schwarz- und Weisspunkt anheben oder absenken.
    """
    liste = list(punkte)
    if index == 0:
        x = 0.0
    elif index == len(liste) - 1:
        x = 1.0
    else:
        x = _begrenzen(x, liste[index - 1][0] + MIN_ABSTAND, liste[index + 1][0] - MIN_ABSTAND)
    liste[index] = (x, _begrenzen(y, 0.0, 1.0))
    return tuple(liste)


def entfernen(punkte: Punkte, index: int) -> Punkte:
    """Entfernt einen inneren Punkt; die Endpunkte bleiben."""
    if index <= 0 or index >= len(punkte) - 1:
        return punkte
    return punkte[:index] + punkte[index + 1:]


def _steigungen(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tangenten nach Fritsch-Carlson - monoton, ohne Ueberschwingen."""
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.empty_like(y)
    m[0], m[-1] = d[0], d[-1]
    for i in range(1, len(y) - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0.0
        else:
            # gewichtetes harmonisches Mittel der Nachbarsteigungen
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    return m


def tabelle(punkte: Punkte, groesse: int = TABELLE) -> np.ndarray:
    """Kurve als float32-Tabelle mit `groesse` Werten fuer x = 0 .. 1."""
    x = np.array([p[0] for p in punkte], dtype=np.float64)
    y = np.array([p[1] for p in punkte], dtype=np.float64)
    t = np.linspace(0.0, 1.0, groesse)
    if len(punkte) == 2:
        return np.interp(t, x, y).astype(np.float32)
    m = _steigungen(x, y)
    i = np.clip(np.searchsorted(x, t, side="right") - 1, 0, len(x) - 2)
    h = x[i + 1] - x[i]
    s = (t - x[i]) / h
    h00 = (1 + 2 * s) * (1 - s) ** 2
    h10 = s * (1 - s) ** 2
    h01 = s * s * (3 - 2 * s)
    h11 = s * s * (s - 1)
    werte = h00 * y[i] + h10 * h * m[i] + h01 * y[i + 1] + h11 * h * m[i + 1]
    return np.clip(werte, 0.0, 1.0).astype(np.float32)


def ist_identitaet(punkte: Punkte) -> bool:
    return all(abs(px - py) < 1e-6 for px, py in punkte)
