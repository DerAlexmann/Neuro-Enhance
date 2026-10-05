"""
Sprachumschaltung

Deutsch ist die Quellsprache: im Code steht der deutsche Text, _("...")
sucht ihn zur Laufzeit in der Sprachtabelle TRANSLATIONS (ganz unten in
dieser Datei). Dort ist auch beschrieben, wie eine weitere Sprache dazukommt.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

SOURCE_LANGUAGE = "de"


class Uebersetzt(str):
    """Uebersetzter Text, der seinen deutschen Schluessel kennt.

    Verhaelt sich ueberall wie ein gewoehnlicher String. Das Hauptfenster
    liest daraus ab, wie sich eine Beschriftung nach einem Sprachwechsel neu
    bilden laesst - samt der Werte, die mit format() eingesetzt wurden.
    """

    def __new__(cls, text, schluessel, werte=None):
        neu = super().__new__(cls, text)
        neu.schluessel = schluessel
        neu.werte = werte or {}
        return neu

    def format(self, *args, **kwargs):
        if args:                             # Positionsargumente nutzt hier niemand
            return str.format(self, *args, **kwargs)
        return Uebersetzt(str.format(self, **kwargs), self.schluessel, kwargs)


class Translator:
    """Uebersetzt einen deutschen Quelltext in die eingestellte Sprache."""

    def __init__(self, language=SOURCE_LANGUAGE):
        self.language = language

    def __call__(self, text):
        if self.language == SOURCE_LANGUAGE:
            return Uebersetzt(text, text)
        return Uebersetzt(TRANSLATIONS.get(self.language, {}).get(text, text), text)

    def available(self):
        """Sprachkuerzel -> Anzeigename, Quellsprache immer zuerst."""
        names = {SOURCE_LANGUAGE: LANGUAGE_NAMES[SOURCE_LANGUAGE]}
        for code in TRANSLATIONS:
            names[code] = LANGUAGE_NAMES.get(code, code)
        return names


_ = Translator()


# --------------------------------------------------------------------------
# SPRACHTABELLE / LANGUAGE TABLE
#
# Quellsprache ist Deutsch - der deutsche Text im Code ist zugleich der
# Schluessel. Eine weitere Sprache kommt in drei Schritten dazu:
#   1. Kuerzel und Anzeigename in LANGUAGE_NAMES eintragen,
#      z. B.  "fr": "Francais"
#   2. In TRANSLATIONS einen Eintrag "fr": { ... } anlegen und die
#      gewuenschten Zeilen uebersetzen.
#   3. Fertig - die Auswahl oben rechts zeigt die Sprache sofort an.
#
# Nicht uebersetzte Zeilen erscheinen automatisch auf Deutsch, eine
# unvollstaendige Tabelle ist also unproblematisch. Platzhalter in
# geschweiften Klammern - {karte}, {stufe}, {treiber} ... - muessen in der
# Uebersetzung unveraendert vorkommen; ihre Reihenfolge im Satz ist frei.
# --------------------------------------------------------------------------

LANGUAGE_NAMES = {
    "de": "Deutsch",
    "en": "English",
}

TRANSLATIONS = {
    "en": {
        # Kopfzeile, Reiter, Statuszeile
        "Version {version}": "Version {version}",
        "Sprache & Darstellung": "Language & appearance",
        "Dunkel": "Dark",
        "Bearbeiten": "Edit",
        "Info & Copyright": "About & copyright",
        "Bereit.": "Ready.",
        "Stufe {stufe}": "Tier {stufe}",

        # Reiter Bearbeiten
        "Noch kein Bild geöffnet.": "No image opened yet.",
        "Bild hierher ziehen oder „Öffnen …“ wählen.":
            "Drag an image here or choose “Open …”.",
        "Öffnen …": "Open …",
        "Speichern unter …": "Save as …",
        "Vorher": "Before",
        "Gedrückt halten, um das unbearbeitete Bild zu sehen.":
            "Hold down to see the unedited image.",
        "Alles zurücksetzen": "Reset all",
        "Doppelklick setzt den Regler zurück.": "Double-click resets the slider.",

        # Reglergruppen und Regler
        "Weißabgleich": "White balance",
        "Licht": "Light",
        "Präsenz": "Presence",
        "Gradationskurve": "Tone curve",
        "Hell": "Lum",
        "R": "R",
        "G": "G",
        "B": "B",
        "Zurücksetzen": "Reset",
        "Klicken setzt einen Punkt, Ziehen verschiebt ihn, Doppelklick entfernt ihn.":
            "Click to add a point, drag to move it, double-click to remove it.",
        "Klarheit": "Clarity",
        "Dunst entfernen": "Dehaze",
        "Rauschminderung": "Noise reduction",
        "Luminanz": "Luminance",
        "Die Vorschau ist verkleinert und zeigt Rauschen schwächer als das "
        "gespeicherte Bild.":
            "The preview is scaled down and shows less noise than the saved image.",
        "Farbbereiche": "Colour ranges",
        "Farbton": "Hue",
        "Rot": "Red",
        "Orange": "Orange",
        "Gelb": "Yellow",
        "Grün": "Green",
        "Aqua": "Aqua",
        "Blau": "Blue",
        "Lila": "Purple",
        "Magenta": "Magenta",
        "LUT": "LUT",
        "Geometrie": "Geometry",
        "KI-Hochskalieren": "AI upscaling",
        "Aus": "Off",
        "Schnell": "Fast",
        "Hohe Qualität": "High quality",
        "Entrauschen": "Denoise",
        "KI-Funktionen brauchen mindestens 4 GB Grafikspeicher.":
            "AI features need at least 4 GB of video memory.",
        "Dieses Modell braucht mindestens Funktionsstufe {stufe}.":
            "This model needs at least feature tier {stufe}.",
        "Modelldateien fehlen im Ordner {ordner}.": "Model files are missing in {ordner}.",
        "Wird beim Speichern angewendet. KI ergänzt Details, die im Original "
        "nicht vorhanden waren.":
            "Applied when saving. AI adds details that were not present in the original.",
        "KI vergrößert das Bild …": "AI is upscaling the image …",
        "Abbrechen": "Cancel",
        "Speichern abgebrochen.": "Saving cancelled.",
        "Die KI-Vergrößerung ist fehlgeschlagen.": "AI upscaling failed.",
        "Objektiv": "Lens",
        "Begradigen": "Straighten",
        "Perspektive senkrecht": "Vertical perspective",
        "Perspektive waagrecht": "Horizontal perspective",
        "Verzeichnung": "Distortion",
        "Vignette": "Vignetting",
        "Farbsaum Rot/Cyan": "Fringe red/cyan",
        "Farbsaum Blau/Gelb": "Fringe blue/yellow",
        "↺ Links": "↺ Left",
        "↻ Rechts": "↻ Right",
        "⇋ Spiegeln": "⇋ Flip",
        "Zuschneiden": "Crop",
        "Rahmen auf dem Bild ziehen; Eingabetaste übernimmt, Esc bricht ab.":
            "Drag the frame on the image; Enter applies, Esc cancels.",
        "Voll": "Full",
        "Zuschnitt zurücksetzen": "Reset crop",
        "Frei": "Free",
        "Original": "Original",
        "Einpassen": "Fit",
        "Ganzes Bild zeigen (Strg+0)": "Show the whole image (Ctrl+0)",
        "Ein Bildpixel je Bildschirmpixel (Strg+1). Mausrad zoomt, Ziehen verschiebt, "
        "Doppelklick wechselt.":
            "One image pixel per screen pixel (Ctrl+1). Mouse wheel zooms, dragging pans, "
            "double-click toggles.",
        "Stärke": "Strength",
        "LUT laden …": "Load LUT …",
        "Entfernen": "Remove",
        "Geladen: {name}": "Loaded: {name}",
        "Keine LUT geladen.": "No LUT loaded.",
        "LUT laden": "Load LUT",
        "Die LUT lässt sich nicht lesen.": "The LUT cannot be read.",
        "Farbe": "Colour",
        "Details": "Detail",
        "Temperatur": "Temperature",
        "Tönung": "Tint",
        "Belichtung": "Exposure",
        "Kontrast": "Contrast",
        "Lichter": "Highlights",
        "Tiefen": "Shadows",
        "Dynamik": "Vibrance",
        "Sättigung": "Saturation",
        "Schärfen": "Sharpening",
        "Radius": "Radius",

        # Öffnen und Speichern
        "Bild öffnen": "Open image",
        "Bilder": "Images",
        "Bild speichern": "Save image",
        "Dieses Dateiformat wird nicht unterstützt.": "This file format is not supported.",
        "Die Datei lässt sich nicht öffnen.": "The file cannot be opened.",
        "Das Bild lässt sich nicht speichern.": "The image cannot be saved.",
        "Für dieses Bild reicht der Grafikspeicher nicht.":
            "There is not enough video memory for this image.",
        "Ungespeicherte Änderungen": "Unsaved changes",
        "Die Änderungen an {name} sind nicht gespeichert. Trotzdem fortfahren?":
            "The changes to {name} have not been saved. Continue anyway?",
        "{name} geöffnet – {breite} × {hoehe} Pixel, {art}":
            "{name} opened – {breite} × {hoehe} pixels, {art}",
        "8 Bit": "8-bit",
        "16 Bit": "16-bit",
        "Alle Bilder": "All images",
        "RAW wird entwickelt …": "Developing RAW …",
        "Gespeichert: {name} ({ms} ms)": "Saved: {name} ({ms} ms)",

        # Reiter Info & Copyright, Statuszeile
        "Grafikkarte": "Graphics card",
        "Nur klassische Filter – für KI-Funktionen sind mindestens "
        "4 GB Grafikspeicher nötig.":
            "Classic filters only – AI features need at least 4 GB of video memory.",
        "Klassische Filter und kleine KI-Modelle, in kleinen Kacheln gerechnet.":
            "Classic filters and small AI models, processed in small tiles.",
        "Klassische Filter und alle Restaurierungsmodelle: Hochskalieren, "
        "Entrauschen, Freistellen und Objektauswahl.":
            "Classic filters and all restoration models: upscaling, denoising, "
            "background removal and object selection.",
        "Zusätzlich größere Modelle, größere Kacheln und generative Füllung.":
            "In addition larger models, larger tiles and generative fill.",
        "Voller Umfang, auch große generative Modelle und mehrere Modelle "
        "gleichzeitig im Speicher.":
            "Full feature set, including large generative models and several models "
            "held in memory at once.",

        # Reiter Info & Copyright
        "Ein quelloffener, GPU-beschleunigter KI-Bild- und Fotoverbesserer "
        "auf Basis neuronaler Netze.":
            "An open-source, GPU-accelerated AI image & photo enhancer powered by "
            "neural networks.",
        "Erstellt mit Unterstützung von Claude AI": "Created with assistance of Claude AI",
        "Veröffentlicht unter der MIT-Lizenz.": "Released under the MIT licence.",
        "Technisches": "Technical details",
        "Grafikspeicher": "Video memory",
        "Compute Capability": "Compute capability",
        "Treiber": "Driver",
        "Funktionsstufe": "Feature tier",
        "Rechengenauigkeit": "Precision",
        "Python": "Python",
        "PySide6 / Qt": "PySide6 / Qt",
        "CuPy / CUDA": "CuPy / CUDA",
        "Einstellungen": "Settings",
        "Marken & Hinweise": "Trademarks & notices",
        "NVIDIA, RTX, GeForce, CUDA und TensorRT sind Marken oder eingetragene Marken "
        "der NVIDIA Corporation in den USA und anderen Ländern. Neuro-Enhance ist ein "
        "unabhängiges Projekt und steht in keiner Verbindung zur NVIDIA Corporation; "
        "es wird von ihr weder unterstützt noch gesponsert.":
            "NVIDIA, RTX, GeForce, CUDA and TensorRT are trademarks or registered "
            "trademarks of NVIDIA Corporation in the U.S. and other countries. "
            "Neuro-Enhance is an independent project and is not affiliated with, "
            "endorsed or sponsored by NVIDIA Corporation.",
        "Qt ist eine Marke der The Qt Company Ltd., Python eine Marke der Python "
        "Software Foundation, Windows eine Marke der Microsoft Corporation. Alle "
        "weiteren Marken gehören ihren jeweiligen Inhabern.":
            "Qt is a trademark of The Qt Company Ltd., Python a trademark of the Python "
            "Software Foundation, Windows a trademark of Microsoft Corporation. All "
            "other trademarks are the property of their respective owners.",
        "KI-Verfahren ergänzen Bilddetails, die im Original nicht vorhanden waren. "
        "Bearbeitete Bilder eignen sich deshalb nicht als Beweis- oder "
        "Dokumentationsmittel.":
            "AI methods add image details that were not present in the original. "
            "Processed images are therefore not suitable as evidence or for "
            "documentation purposes.",
        "Verwendete Fremdkomponenten und ihre Lizenzen stehen in der Datei NOTICE.":
            "Third-party components and their licences are listed in the NOTICE file.",

        # Startmeldungen
        "Keine NVIDIA-Grafikkarte gefunden": "No NVIDIA graphics card found",
        "Neuro-Enhance benötigt eine NVIDIA-RTX-Grafikkarte (ab der RTX-2000-Serie) "
        "mit installiertem NVIDIA-Treiber. Auf diesem Rechner wurde weder eine "
        "NVIDIA-Grafikkarte noch ein NVIDIA-Treiber gefunden.":
            "Neuro-Enhance requires an NVIDIA RTX graphics card (RTX 2000 series or "
            "newer) with the NVIDIA driver installed. Neither an NVIDIA graphics card "
            "nor an NVIDIA driver was found on this computer.",
        "Der NVIDIA-Treiber ist installiert, meldet aber keine Grafikkarte. "
        "Neuro-Enhance benötigt eine NVIDIA-RTX-Grafikkarte (ab der RTX-2000-Serie).":
            "The NVIDIA driver is installed but reports no graphics card. Neuro-Enhance "
            "requires an NVIDIA RTX graphics card (RTX 2000 series or newer).",
        "Keine RTX-Grafikkarte gefunden": "No RTX graphics card found",
        "Gefunden: {karte}. Neuro-Enhance benötigt eine RTX-Grafikkarte mit "
        "Tensor Cores (ab der RTX-2000-Serie). GTX-Karten und ältere Modelle "
        "werden nicht unterstützt.":
            "Found: {karte}. Neuro-Enhance requires an RTX graphics card with Tensor "
            "Cores (RTX 2000 series or newer). GTX cards and older models are not "
            "supported.",
        "Grafiktreiber zu alt": "Graphics driver too old",
        "Gefunden: {karte} mit Treiber {treiber}. Neuro-Enhance benötigt den "
        "NVIDIA-Treiber {mindestens} oder neuer. Bitte den Grafiktreiber "
        "aktualisieren und das Programm danach erneut starten.":
            "Found: {karte} with driver {treiber}. Neuro-Enhance requires NVIDIA driver "
            "{mindestens} or newer. Please update the graphics driver and then start "
            "the program again.",
        "Die vollständigen Systemvoraussetzungen stehen in der README.":
            "The full system requirements are listed in the README.",
        "Treiber herunterladen": "Download driver",
        "Beenden": "Quit",
    },
}
