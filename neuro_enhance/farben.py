"""
Farbschemata und Schriften

apply_theme() schreibt die Werte der gewaehlten Palette in die
Modulvariablen - der uebrige Code benutzt einfach BG, CARD, TEXT ... und muss
vom Umschalten nichts wissen. Ein eigenes Schema entsteht durch eine weitere
Palette mit denselben Namen. stylesheet() setzt daraus das Qt-Stylesheet fuer
die ganze Anwendung zusammen; ein Wechsel des Schemas ist damit ein einziger
Aufruf von QApplication.setStyleSheet().

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

THEMES = {
    "light": {
        "BG": "#eef1f5",             # Seitenhintergrund
        "CARD": "#ffffff",           # Karten
        "CARD_ALT": "#fbfcfe",       # Text- und Listenflaechen in Karten
        "BORDER": "#d7dce4",
        "TEXT": "#1b2430",
        "MUTED": "#6c7684",          # Nebentext
        "HEADER": "#1d2330",         # Kopfzeile mit Titel und Umschaltern
        "HEADER_TEXT": "#c2cad8",
        "HEADER_TITLE": "#ffffff",
        "HEADER_GROUP": "#69748c",
        "HEADER_HOVER": "#2b3346",
        "ACCENT": "#2f7de1",
        "ACCENT_DARK": "#1f66c4",
        "ON_ACCENT": "#ffffff",      # Schrift auf farbigen Flaechen
        "OK": "#2e9e5b",
        "OK_DARK": "#25864b",
        "WARN": "#e08b1f",
        "WARN_DARK": "#c4770f",
        "DANGER": "#d64545",
        "DANGER_DARK": "#b83a3a",
        "BTN_BG": "#e3e8f0",         # unauffaelliger Schalter
        "BTN_HOVER": "#d2d9e6",
        "BTN_TEXT": "#1b2430",
        "BTN_DISABLED": "#9aa3b0",
        "FIELD_BG": "#ffffff",       # Eingabefelder
        "TROUGH": "#e3e8f0",         # Rille von Schiebereglern
        "TAB_BG": "#e3e8f0",         # nicht gewaehlter Reiter
        "STATUS_BG": "#e4e8ef",
    },
    "dark": {
        "BG": "#12161d",
        "CARD": "#1a1f28",
        "CARD_ALT": "#151a22",
        "BORDER": "#2c3441",
        "TEXT": "#e6eaf0",
        "MUTED": "#98a2b3",
        "HEADER": "#0e1218",
        "HEADER_TEXT": "#b8c2d0",
        "HEADER_TITLE": "#ffffff",
        "HEADER_GROUP": "#6b7688",
        "HEADER_HOVER": "#212a38",
        "ACCENT": "#4a90e8",
        "ACCENT_DARK": "#3a7ad0",
        "ON_ACCENT": "#ffffff",
        "OK": "#3fb972",
        "OK_DARK": "#349b60",
        "WARN": "#e9a23b",
        "WARN_DARK": "#cc8a26",
        "DANGER": "#e05a5a",
        "DANGER_DARK": "#c44a4a",
        "BTN_BG": "#2a323f",
        "BTN_HOVER": "#353f4f",
        "BTN_TEXT": "#e6eaf0",
        "BTN_DISABLED": "#626c7a",
        "FIELD_BG": "#232b36",
        "TROUGH": "#2a323f",
        "TAB_BG": "#232b36",
        "STATUS_BG": "#0e1218",
    },
}

DEFAULT_THEME = "light"
CURRENT_THEME = DEFAULT_THEME


def apply_theme(name):
    """Farbwerte des gewaehlten Schemas in die Modulvariablen schreiben."""
    global CURRENT_THEME
    CURRENT_THEME = name if name in THEMES else DEFAULT_THEME
    globals().update(THEMES[CURRENT_THEME])


apply_theme(DEFAULT_THEME)      # legt BG, CARD, TEXT ... ueberhaupt erst an

FONT = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_TINY = ("Segoe UI", 8)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_H2 = ("Segoe UI", 12, "bold")
FONT_MONO = ("Consolas", 10)
FONT_MONO_SMALL = ("Consolas", 9)
FONT_ZAHL = ("Consolas", 12, "bold")


def stylesheet() -> str:
    """Qt-Stylesheet aus den Rollen des aktuellen Schemas.

    Die Widgets tragen Objektnamen (setObjectName) statt eigener Farben:
    #kopf, #karte, #kartentitel, #nebentext, #wert ... Damit genuegt beim
    Umschalten das neue Stylesheet, kein Widget muss einzeln umgefaerbt werden.
    """
    t = THEMES[CURRENT_THEME]
    return f"""
    QWidget {{
        font-family: "Segoe UI"; font-size: 10pt; color: {t["TEXT"]};
    }}
    QMainWindow, QWidget#seite {{ background: {t["BG"]}; }}
    QDialog, QMessageBox {{ background: {t["BG"]}; }}
    QMessageBox QLabel {{ color: {t["TEXT"]}; background: transparent; }}
    QScrollArea, QScrollArea > QWidget > QWidget {{ background: {t["BG"]}; border: none; }}

    QFrame#kopf {{ background: {t["HEADER"]}; }}
    QFrame#kopf QLabel {{ color: {t["HEADER_TEXT"]}; background: transparent; }}
    QLabel#titel {{ color: {t["HEADER_TITLE"]}; font-size: 13pt; font-weight: bold; }}
    QLabel#kopfgruppe {{ color: {t["HEADER_GROUP"]}; font-size: 8pt; }}
    QFrame#kopf QCheckBox {{ color: {t["HEADER_TEXT"]}; background: transparent; }}
    QFrame#kopf QComboBox {{
        background: {t["HEADER_HOVER"]}; color: {t["HEADER_TEXT"]};
        border: 1px solid {t["HEADER_GROUP"]}; border-radius: 4px; padding: 2px 8px;
    }}
    QFrame#kopf QComboBox QAbstractItemView {{
        background: {t["HEADER"]}; color: {t["HEADER_TEXT"]};
        selection-background-color: {t["ACCENT"]}; selection-color: {t["ON_ACCENT"]};
    }}

    QFrame#karte {{
        background: {t["CARD"]}; border: 1px solid {t["BORDER"]}; border-radius: 8px;
    }}
    QFrame#karte QLabel {{ background: transparent; }}
    QLabel#kartentitel {{ font-size: 12pt; font-weight: bold; }}
    QLabel#nebentext {{ color: {t["MUTED"]}; }}
    QLabel#wert {{ font-family: "Consolas"; }}
    QLabel#warnung {{ color: {t["WARN"]}; }}
    QLabel#leer {{ color: {t["MUTED"]}; font-size: 12pt; }}
    QFrame#leinwand {{
        background: {t["CARD_ALT"]}; border: 1px dashed {t["BORDER"]}; border-radius: 8px;
    }}

    QTabWidget::pane {{ border: none; background: {t["BG"]}; }}
    QTabBar {{ qproperty-drawBase: 0; background: {t["BG"]}; }}
    QTabBar::tab {{
        background: {t["TAB_BG"]}; color: {t["MUTED"]};
        padding: 7px 16px; margin-right: 2px;
        border-top-left-radius: 6px; border-top-right-radius: 6px;
    }}
    QTabBar::tab:selected {{ background: {t["CARD"]}; color: {t["TEXT"]}; font-weight: bold; }}
    QTabBar::tab:hover:!selected {{ background: {t["BTN_HOVER"]}; }}

    QPushButton {{
        background: {t["BTN_BG"]}; color: {t["BTN_TEXT"]};
        border: none; border-radius: 6px; padding: 7px 16px;
    }}
    QPushButton:hover {{ background: {t["BTN_HOVER"]}; }}
    QPushButton:disabled {{ color: {t["BTN_DISABLED"]}; }}
    QPushButton#hauptschalter {{ background: {t["ACCENT"]}; color: {t["ON_ACCENT"]}; }}
    QPushButton#hauptschalter:hover {{ background: {t["ACCENT_DARK"]}; }}

    QStatusBar {{ background: {t["STATUS_BG"]}; color: {t["MUTED"]}; }}
    QStatusBar::item {{ border: none; }}
    QStatusBar QLabel {{ color: {t["MUTED"]}; background: transparent; padding: 2px 10px; }}
    QStatusBar QLabel#wert {{ font-family: "Consolas"; font-size: 9pt; }}

    QToolTip {{
        background: {t["CARD"]}; color: {t["TEXT"]}; border: 1px solid {t["BORDER"]};
    }}
    """
