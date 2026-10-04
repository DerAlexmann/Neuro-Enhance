"""Erzeugt das Programmsymbol neuro_enhance.ico.

Gezeichnet wird ein Foto - Rahmen mit Berg und Sonne - in Weiss auf einer
abgerundeten Flaeche im Akzentblau des Farbschemas, oben rechts ein
vierzackiger Funke fuer die Verbesserung. Die .ico enthaelt alle ueblichen
Groessen, vom Reiter der Taskleiste bis zur grossen Kachel im Explorer.

Aufruf:  python icon_erzeugen.py
Benoetigt:  pip install Pillow

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw

ZIEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neuro_enhance.ico")
KANTE = 1024                                  # Vorlage, wird heruntergerechnet
GROESSEN = [16, 24, 32, 48, 64, 128, 256]

BLAU = (47, 125, 225, 255)                    # ACCENT des hellen Schemas
WEISS = (255, 255, 255, 255)


def funke(stift, mitte_x, mitte_y, radius, farbe):
    """Vierzackiger Stern mit schlanker Taille."""
    taille = radius * 0.28
    punkte = [
        (mitte_x, mitte_y - radius), (mitte_x + taille, mitte_y - taille),
        (mitte_x + radius, mitte_y), (mitte_x + taille, mitte_y + taille),
        (mitte_x, mitte_y + radius), (mitte_x - taille, mitte_y + taille),
        (mitte_x - radius, mitte_y), (mitte_x - taille, mitte_y - taille),
    ]
    stift.polygon(punkte, fill=farbe)


def zeichnen() -> Image.Image:
    bild = Image.new("RGBA", (KANTE, KANTE), (0, 0, 0, 0))
    stift = ImageDraw.Draw(bild)

    rand = KANTE // 16
    stift.rounded_rectangle((rand, rand, KANTE - rand, KANTE - rand),
                            radius=KANTE // 5, fill=BLAU)

    # Foto: weisser Rahmen, innen blau, darin Berge und Sonne in Weiss
    links, oben = int(KANTE * 0.21), int(KANTE * 0.30)
    rechts, unten = int(KANTE * 0.73), int(KANTE * 0.76)
    dicke = KANTE // 22
    stift.rounded_rectangle((links, oben, rechts, unten), radius=KANTE // 22, fill=WEISS)
    stift.rounded_rectangle((links + dicke, oben + dicke, rechts - dicke, unten - dicke),
                            radius=KANTE // 60, fill=BLAU)
    boden = unten - dicke
    stift.polygon([(links + dicke, boden), (int(KANTE * 0.38), int(KANTE * 0.50)),
                   (int(KANTE * 0.52), boden)], fill=WEISS)
    stift.polygon([(int(KANTE * 0.42), boden), (int(KANTE * 0.56), int(KANTE * 0.56)),
                   (rechts - dicke, boden)], fill=WEISS)
    r = KANTE // 22
    sx, sy = int(KANTE * 0.60), int(KANTE * 0.43)
    stift.ellipse((sx - r, sy - r, sx + r, sy + r), fill=WEISS)

    # Funke ueber der Ecke; ein blauer Hof trennt ihn sauber vom Rahmen
    fx, fy = int(KANTE * 0.71), int(KANTE * 0.31)
    funke(stift, fx, fy, KANTE * 0.155, BLAU)
    funke(stift, fx, fy, KANTE * 0.12, WEISS)
    return bild


def main() -> None:
    vorlage = zeichnen()
    vorlage.save(ZIEL, sizes=[(n, n) for n in GROESSEN])
    print("geschrieben:", ZIEL)


if __name__ == "__main__":
    main()
