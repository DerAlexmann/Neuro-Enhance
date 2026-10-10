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
        "Zurücksetzen": "Reset",
        # Anonymisieren
        "Anonymisieren": "Anonymise",
        "Gesichter finden": "Find faces",
        "Legt um jedes erkannte Gesicht eine Fläche. Prüfen und nachbessern – sehr "
        "kleine, verdeckte oder seitliche Gesichter können fehlen.":
            "Places an area around every detected face. Check and correct – very small, "
            "covered or side-on faces may be missed.",
        "Flächen bearbeiten": "Edit areas",
        "Auf freiem Grund ziehen legt eine Fläche an; Flächen verschieben, an den Ecken "
        "die Größe ändern, Entf löscht die gewählte.":
            "Drag on empty ground to add an area; move areas, resize them at the corners, "
            "Del removes the selected one.",
        "Auf freiem Grund ziehen legt eine Fläche an. Flächen verschieben, an den Ecken "
        "die Größe ändern; Entf löscht, Esc beendet.":
            "Drag on empty ground to add an area. Move areas, resize them at the corners; "
            "Del removes, Esc finishes.",
        "Form": "Shape",
        "Ellipse": "Ellipse",
        "Rechteck": "Rectangle",
        "Wirkung": "Effect",
        "Mosaik": "Mosaic",
        "Weichzeichnen": "Blur",
        "Schwarz füllen": "Fill black",
        "Raster": "Grid",
        "Blöcke über die Breite einer Fläche. Weniger ist sicherer: feine Mosaike "
        "lassen sich teilweise zurückrechnen.":
            "Blocks across the width of an area. Fewer is safer: fine mosaics can be "
            "partly reversed.",
        "Fläche löschen": "Remove area",
        "Alle löschen": "Remove all",
        "Ohne GPS und Seriennummern speichern": "Save without GPS and serial numbers",
        "Nimmt beim Speichern GPS-Position, Seriennummern von Kamera und Objektiv, den "
        "Besitzernamen und die Herstellerdaten aus den Metadaten.":
            "Removes the GPS position, the camera and lens serial numbers, the owner name "
            "and the maker notes from the metadata when saving.",
        "Standort und Seriennummern werden nicht mitgespeichert.":
            "Location and serial numbers will not be saved.",
        "{n} Flächen. Das Mosaik wird beim Speichern fest ins Bild gerechnet.":
            "{n} areas. The mosaic is burned into the image when saving.",
        "Gesichter, Kennzeichen oder Hausnummern unkenntlich machen.":
            "Make faces, number plates or house numbers unrecognisable.",
        "Die Gesichtserkennung ist fehlgeschlagen.": "Face detection failed.",
        "Keine Gesichter gefunden – Flächen von Hand aufziehen.":
            "No faces found – drag areas by hand.",
        "{n} Gesichter gefunden – bitte prüfen und nachbessern.":
            "{n} faces found – please check and correct.",
        "Setzt alle Regler, Drehung und Zuschnitt zurück und verwirft eine KI-Erweiterung.":
            "Resets all sliders, rotation and crop and discards an AI extension.",
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
        "Dieses Modell ist noch nicht geladen.": "This model has not been downloaded yet.",
        "Modell herunterladen ({mb} MB)": "Download model ({mb} MB)",
        "Silberkorn lädt {mb} MB von:\n{quelle}\n\n"
        "Das Modell stammt von {herkunft} und wird nach dem Laden gegen seine "
        "Prüfsumme geprüft.\n\n"
        "Ablage: {ordner}\n\nJetzt herunterladen?":
            "Silberkorn downloads {mb} MB from:\n{quelle}\n\n"
            "The model comes from {herkunft} and is checked against its checksum "
            "after downloading.\n\n"
            "Location: {ordner}\n\nDownload now?",
        "KI-Entrauschen": "AI denoise",
        "Für dieses Bild berechnet. Der Regler mischt zwischen Original und "
        "entrauschtem Bild.":
            "Computed for this image. The slider blends between the original and the "
            "denoised image.",
        "Rechnet einmal über das ganze Bild – bei 24 Megapixeln 10 bis 30 "
        "Sekunden. Danach wirkt der Regler sofort.":
            "Runs once over the whole image – 10 to 30 seconds for 24 megapixels. "
            "After that the slider responds instantly.",
        "Entrauschen berechnen": "Compute denoise",
        "KI entrauscht das Bild …": "AI is denoising the image …",
        "KI-Entrauschen abgebrochen.": "AI denoise cancelled.",
        "Das KI-Entrauschen ist fehlgeschlagen.": "AI denoise failed.",
        "KI-Entrauschen fertig ({s} s, über {weg})": "AI denoise done ({s} s, via {weg})",
        "KI-Schärfen": "AI sharpen",
        "Für dieses Bild berechnet. Der Regler bestimmt, wie stark die Schärfung wirkt.":
            "Computed for this image. The slider sets how strongly the sharpening applies.",
        "Gegen leichte Fokus-Unschärfe. Rechnet einmal über das ganze Bild – bei 24 "
        "Megapixeln {zeit}. Erst entrauschen, sonst schärft die KI das Rauschen mit.":
            "For slight focus blur. Runs once over the whole image – {zeit} for 24 "
            "megapixels. Denoise first, otherwise the AI sharpens the noise too.",
        "knapp eine Minute": "just under a minute",
        "etwa drei Minuten": "about three minutes",
        "Schärfen berechnen": "Compute sharpening",
        "Motiv & Hintergrund": "Subject & background",
        "Motiv erkennen": "Detect subject",
        "Maske zeigen": "Show mask",
        "Umkehren": "Invert",
        "Hintergrund durchsichtig speichern": "Save background as transparent",
        "Unschärfe": "Blur",
        "Kante weicher": "Soften edge",
        "Kante verschieben": "Shift edge",
        "Färbt den Hintergrund in der Vorschau rot ein – nur zur Kontrolle, nicht im "
        "gespeicherten Bild.":
            "Tints the background red in the preview – only to check, not in the saved image.",
        "Die Regler dieser Karte wirken auf das Motiv statt auf den Hintergrund.":
            "The sliders of this card affect the subject instead of the background.",
        "Beim Speichern als PNG oder TIFF wird der Hintergrund transparent.":
            "When saving as PNG or TIFF, the background becomes transparent.",
        "Das Erkennen ohne Klick braucht 6 GB Grafikspeicher.":
            "Detection without clicking needs 6 GB of graphics memory.",
        "Motiv erkannt. Die Regler wirken auf den Hintergrund.":
            "Subject detected. The sliders affect the background.",
        "Die KI erkennt das Motiv – bei 24 Megapixeln in wenigen Sekunden – oder wählt "
        "aus, was man anklickt. Danach lässt sich der Hintergrund getrennt bearbeiten oder "
        "durchsichtig speichern.":
            "The AI detects the subject – in a few seconds for 24 megapixels – or selects what "
            "you click. After that the background can be edited separately or saved as "
            "transparent.",
        "Motiv erkennen · {mb} MB": "Detect subject · {mb} MB",
        "Objekt anklicken": "Click object",
        "Objekt anklicken · {mb} MB": "Click object · {mb} MB",
        "Ein Objekt im Bild per Klick auswählen – die Regler dieser Karte wirken dann auf "
        "alles andere.":
            "Select an object in the image by clicking – the sliders of this card then affect "
            "everything else.",
        "Klick zurück": "Undo click",
        "Den letzten Klick zurücknehmen (Strg+Z)": "Undo the last click (Ctrl+Z)",
        "Linksklick ins Bild nimmt einen Bereich dazu, Rechtsklick nimmt einen weg. Strg+Z "
        "nimmt den letzten Klick zurück, Esc beendet.":
            "Left-click in the image adds an area, right-click removes one. Ctrl+Z undoes the "
            "last click, Esc finishes.",
        "Objekt ausgewählt. Die Regler wirken auf alles andere.":
            "Object selected. The sliders affect everything else.",
        "Die Auswahl per Klick ließ sich nicht starten.": "Click selection could not be started.",
        "Die Auswahl ist fehlgeschlagen.": "The selection failed.",
        "Bereit zum Klicken ({s} s)": "Ready for clicking ({s} s)",
        "Objekte entfernen": "Remove objects",
        "Anklicken": "Click",
        "Anklicken · {mb} MB": "Click · {mb} MB",
        "Ein Objekt per Klick markieren – Rechtsklick nimmt einen Bereich wieder weg.":
            "Mark an object by clicking – right-click removes an area again.",
        "Pinsel": "Brush",
        "Kleinigkeiten übermalen, etwa Flecken oder Leitungen – die rechte Maustaste "
        "radiert.":
            "Paint over small things such as spots or wires – the right mouse button erases.",
        "Pinselgröße": "Brush size",
        "Entfernen · {mb} MB": "Remove · {mb} MB",
        "Markierung löschen": "Clear marking",
        "Letzte Entfernung zurücknehmen": "Undo last removal",
        "Zurücknehmen": "Undo",
        "Die blaue Markierung verwerfen, ohne etwas zu entfernen.":
            "Discard the blue marking without removing anything.",
        "Füllt die blau markierte Stelle mit passendem Hintergrund.":
            "Fills the area marked in blue with matching background.",
        "Lädt zuerst das KI-Modell ({mb} MB) aus den Releases von Silberkorn "
        "auf GitHub – nach einer Rückfrage mit Quelle, Größe und Lizenz.":
            "First downloads the AI model ({mb} MB) from Silberkorn's releases on GitHub "
            "– after asking, with source, size and licence.",
        "Berechnet das Ergebnis einmal für das ganze Bild; mit „Stärke“ lässt "
        "es sich danach stufenlos einblenden.":
            "Computes the result once for the whole image; “Strength” then blends it "
            "in smoothly.",
        "Die KI erkennt das Hauptmotiv – die Regler dieser Karte wirken dann auf den "
        "Hintergrund.":
            "The AI detects the main subject – the sliders of this card then affect the "
            "background.",
        "Schätzt die Tiefe des ganzen Bildes – danach wirken Unschärfe, Fokus und "
        "Schärfentiefe.":
            "Estimates the depth of the whole image – then blur, focus and depth of field "
            "take effect.",
        "Linksklick markiert ein Objekt, Rechtsklick nimmt einen Bereich weg. Strg+Z nimmt "
        "den letzten Klick zurück, Esc beendet.":
            "Left-click marks an object, right-click removes an area. Ctrl+Z undoes the last "
            "click, Esc finishes.",
        "Mit der linken Maustaste übermalen, was weg soll; die rechte radiert. Die mittlere "
        "verschiebt das vergrößerte Bild, Esc beendet.":
            "Paint over what should go with the left mouse button; the right one erases. The "
            "middle one moves the zoomed image, Esc finishes.",
        "Blau markiert ist, was verschwindet. „Entfernen“ füllt die Stelle mit passendem "
        "Hintergrund.":
            "What is marked in blue will disappear. “Remove” fills the spot with "
            "matching background.",
        "Ein Objekt anklicken oder übermalen – die KI füllt die Stelle mit dem, was dahinter "
        "liegen könnte.":
            "Click or paint over an object – the AI fills the spot with what could be behind "
            "it.",
        "Bisher entfernt: {n}.": "Removed so far: {n}.",
        "Das Entfernen ist fehlgeschlagen.": "Removing failed.",
        "Entfernt ({s} s)": "Removed ({s} s)",
        "Mit KI erweitern statt beschneiden": "Extend with AI instead of cropping",
        "Statt das Bild auf das Seitenverhältnis zuzuschneiden, erfindet die KI die "
        "fehlenden Ränder dazu. Das Original bleibt unverändert.":
            "Instead of cropping the image to the aspect ratio, the AI invents the missing "
            "borders. The original stays unchanged.",
        "KI-Erweitern braucht mindestens 8 GB Grafikspeicher.":
            "Extending with AI needs at least 8 GB of graphics memory.",
        "Ein Seitenverhältnis wählen – die KI erfindet, was dafür fehlt.":
            "Choose an aspect ratio – the AI invents what is missing for it.",
        "Das Bild hat dieses Seitenverhältnis schon.":
            "The image already has this aspect ratio.",
        "Neue Größe: {breite} × {hoehe} px. Die KI rechnet rund eine halbe Minute.":
            "New size: {breite} × {hoehe} px. The AI takes about half a minute.",
        "Über 25 % je Seite erfindet die KI mehr, als sie sieht – das Ergebnis kann "
        "unstimmig werden.":
            "Beyond 25 % per side the AI invents more than it sees – the result may not "
            "fit together.",
        "Erweitern · {mb} MB": "Extend · {mb} MB",
        "Erweitern": "Extend",
        "Neu erzeugen": "Regenerate",
        "Erfindet die Ränder noch einmal, mit anderem Zufall.":
            "Invents the borders once more, with a different random seed.",
        "Erfindet die fehlenden Ränder für das gewählte Seitenverhältnis. Das Original "
        "bleibt unverändert.":
            "Invents the missing borders for the chosen aspect ratio. The original stays "
            "unchanged.",
        "Erweiterung verworfen.": "Extension discarded.",
        "KI erweitert das Bild …": "AI is extending the image …",
        "KI-Erweitern abgebrochen.": "Extending with AI cancelled.",
        "Das KI-Erweitern ist fehlgeschlagen.": "Extending with AI failed.",
        "KI-Erweitern fertig ({s} s)": "Extended with AI ({s} s)",
        "Tiefe & Bokeh": "Depth & bokeh",
        "Fokus (fern – nah)": "Focus (far – near)",
        "Schärfentiefe": "Depth of field",
        "Fokus ins Bild klicken": "Click focus in image",
        "Ein Klick ins Bild stellt auf diese Entfernung scharf.":
            "A click in the image focuses at that distance.",
        "Tiefenkarte zeigen": "Show depth map",
        "Zeigt in der Vorschau die geschätzte Tiefe: hell ist nah, dunkel fern.":
            "Shows the estimated depth in the preview: bright is near, dark is far.",
        "Motiv scharf halten": "Keep subject sharp",
        "Ist das Motiv erkannt oder angeklickt, bleibt es scharf, gleich wie tief es liegt.":
            "If the subject is detected or clicked, it stays sharp, whatever its depth.",
        "Tiefe geschätzt. Die Unschärfe wächst mit dem Abstand zur Fokusebene, nach vorn "
        "wie nach hinten.":
            "Depth estimated. The blur grows with the distance from the focal plane, towards "
            "the front as well as the back.",
        "Die KI schätzt, wie weit alles im Bild entfernt ist – in Sekundenbruchteilen. "
        "Danach lässt sich der Hintergrund wie mit einem lichtstarken Objektiv "
        "weichzeichnen.":
            "The AI estimates how far away everything in the image is – in a fraction of a "
            "second. After that the background can be blurred as with a fast lens.",
        "Tiefe berechnen": "Estimate depth",
        "Die Tiefe ließ sich nicht schätzen.": "The depth could not be estimated.",
        "Tiefe geschätzt ({s} s)": "Depth estimated ({s} s)",
        "RAW": "RAW",
        "Schnell (GPU)": "Fast (GPU)",
        "Beste Qualität": "Best quality",
        "Schnell: Demosaicing nach Malvar, He und Cutler auf der Grafikkarte, eine 24-MP-RAW "
        "in rund 0,2 s. Beste Qualität: LibRaw mit dem Verfahren DHT auf dem Prozessor, etwa "
        "1,5 s, mit weniger Farbsäumen an feinen Mustern. Gilt beim nächsten Öffnen einer RAW.":
            "Fast: demosaicing after Malvar, He and Cutler on the graphics card, a 24 MP RAW in "
            "about 0.2 s. Best quality: LibRaw with the DHT method on the processor, about "
            "1.5 s, with fewer colour fringes on fine patterns. Applies the next time a RAW "
            "is opened.",
        "Das Motiv ließ sich nicht erkennen.": "The subject could not be detected.",
        "Motiv erkannt ({s} s, über {weg})": "Subject detected ({s} s, via {weg})",
        "Gespeichert: {name} – ohne durchsichtigen Hintergrund, das kann JPEG nicht. Als PNG "
        "oder TIFF speichern.":
            "Saved: {name} – without a transparent background, JPEG cannot do that. Save as "
            "PNG or TIFF.",
        "KI schärft das Bild …": "AI is sharpening the image …",
        "KI-Schärfen abgebrochen.": "AI sharpen cancelled.",
        "Das KI-Schärfen ist fehlgeschlagen.": "AI sharpen failed.",
        "KI-Schärfen fertig ({s} s, über {weg})": "AI sharpen done ({s} s, via {weg})",
        "KI-Modell herunterladen": "Download AI model",
        "TensorRT bereitet das Modell einmalig für diese Grafikkarte vor – das dauert "
        "einige Minuten. Danach rechnet die KI rund doppelt so schnell.":
            "TensorRT is preparing the model for this graphics card once – this takes a few "
            "minutes. After that, the AI runs about twice as fast.",
        "Gespeichert: {name} ({ms} ms, KI über {weg})": "Saved: {name} ({ms} ms, AI via {weg})",
        "nicht installiert (optional, siehe README)": "not installed (optional, see README)",
        "TensorRT ließ sich nicht vorbereiten – die KI rechnet mit CUDA.":
            "TensorRT could not be prepared – the AI computes with CUDA.",
        "Modell wird heruntergeladen …": "Downloading model …",
        "Herunterladen abgebrochen.": "Download cancelled.",
        "Das Modell ließ sich nicht herunterladen.": "The model could not be downloaded.",
        "KI-Modell geladen: {ordner}": "AI model downloaded: {ordner}",
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
        "Einpassen: das ganze Bild zeigen (Strg+0)": "Fit: show the whole image (Ctrl+0)",
        "100 %: ein Bildpixel je Bildschirmpixel (Strg+1). Mausrad zoomt, Ziehen "
        "verschiebt, Doppelklick wechselt.":
            "100 %: one image pixel per screen pixel (Ctrl+1). Mouse wheel zooms, dragging "
            "pans, double-click toggles.",
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
        "CUDA-Architektur": "Compute capability",
        "Treiber": "Driver",
        "Funktionsstufe": "Feature tier",
        "Rechengenauigkeit": "Precision",
        "Python": "Python",
        "PySide6 / Qt": "PySide6 / Qt",
        "CuPy / CUDA": "CuPy / CUDA",
        "Einstellungen": "Settings",
        "Marken & Hinweise": "Trademarks & notices",
        "NVIDIA, RTX, GeForce, CUDA und TensorRT sind Marken oder eingetragene Marken "
        "der NVIDIA Corporation in den USA und anderen Ländern. Silberkorn ist ein "
        "unabhängiges Projekt und steht in keiner Verbindung zur NVIDIA Corporation; "
        "es wird von ihr weder unterstützt noch gesponsert.":
            "NVIDIA, RTX, GeForce, CUDA and TensorRT are trademarks or registered "
            "trademarks of NVIDIA Corporation in the U.S. and other countries. "
            "Silberkorn is an independent project and is not affiliated with, "
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
        "Silberkorn benötigt eine NVIDIA-RTX-Grafikkarte (ab der RTX-2000-Serie) "
        "mit installiertem NVIDIA-Treiber. Auf diesem Rechner wurde weder eine "
        "NVIDIA-Grafikkarte noch ein NVIDIA-Treiber gefunden.":
            "Silberkorn requires an NVIDIA RTX graphics card (RTX 2000 series or "
            "newer) with the NVIDIA driver installed. Neither an NVIDIA graphics card "
            "nor an NVIDIA driver was found on this computer.",
        "Der NVIDIA-Treiber ist installiert, meldet aber keine Grafikkarte. "
        "Silberkorn benötigt eine NVIDIA-RTX-Grafikkarte (ab der RTX-2000-Serie).":
            "The NVIDIA driver is installed but reports no graphics card. Silberkorn "
            "requires an NVIDIA RTX graphics card (RTX 2000 series or newer).",
        "Keine RTX-Grafikkarte gefunden": "No RTX graphics card found",
        "Gefunden: {karte}. Silberkorn benötigt eine RTX-Grafikkarte mit "
        "Tensor Cores (ab der RTX-2000-Serie). GTX-Karten und ältere Modelle "
        "werden nicht unterstützt.":
            "Found: {karte}. Silberkorn requires an RTX graphics card with Tensor "
            "Cores (RTX 2000 series or newer). GTX cards and older models are not "
            "supported.",
        "Grafiktreiber zu alt": "Graphics driver too old",
        "Gefunden: {karte} mit Treiber {treiber}. Silberkorn benötigt den "
        "NVIDIA-Treiber {mindestens} oder neuer. Bitte den Grafiktreiber "
        "aktualisieren und das Programm danach erneut starten.":
            "Found: {karte} with driver {treiber}. Silberkorn requires NVIDIA driver "
            "{mindestens} or newer. Please update the graphics driver and then start "
            "the program again.",
        "Die vollständigen Systemvoraussetzungen stehen in der README.":
            "The full system requirements are listed in the README.",
        "Treiber herunterladen": "Download driver",
        "Beenden": "Quit",
        "NVIDIA-Bibliotheken einrichten": "Set up NVIDIA libraries",
        "Silberkorn rechnet mit CUDA und cuDNN von NVIDIA. Diese Bibliotheken "
        "werden nicht mit dem Programm ausgeliefert, sondern einmalig aus NVIDIAs "
        "offiziellen Paketen vom Python Package Index (pypi.org) geladen – "
        "{laden} GB, ausgepackt rund {platz} GB. Jede Datei wird gegen ihre "
        "Prüfsumme geprüft.":
            "Silberkorn computes with NVIDIA's CUDA and cuDNN. These libraries are not "
            "shipped with the program but downloaded once from NVIDIA's official "
            "packages on the Python Package Index (pypi.org) – {laden} GB, about "
            "{platz} GB unpacked. Every file is verified against its checksum.",
        "Für diese Bibliotheken gelten die Lizenzbedingungen von NVIDIA:":
            "These libraries are subject to NVIDIA's licence terms:",
        "Mit „Zustimmen und herunterladen“ werden sie anerkannt.":
            "Clicking “Agree and download” accepts them.",
        "Ablage: {ordner}": "Location: {ordner}",
        "Zustimmen und herunterladen": "Agree and download",
        "CuPy ließ sich nicht laden": "CuPy could not be loaded",
        "Das Hauptfenster ließ sich nicht öffnen": "The main window could not be opened",
        "Die Einzelheiten stehen unter „Details“ und in {log}.":
            "The details are under “Details” and in {log}.",
        "Die Einzelheiten stehen unter „Details“.": "The details are under “Details”.",
        "NVIDIA-Bibliotheken werden heruntergeladen …": "Downloading NVIDIA libraries …",
        "Die NVIDIA-Bibliotheken ließen sich nicht einrichten. "
        "Beim nächsten Start versucht Silberkorn es erneut.":
            "The NVIDIA libraries could not be set up. Silberkorn will try again at "
            "the next start.",

        # KI-Kolorieren
        "KI-Kolorieren": "AI colourise",
        "Kolorieren berechnen": "Compute colours",
        "Für Schwarzweiß- und Sepiabilder: Die KI schätzt die Farben aus der Helligkeit – "
        "glaubwürdig, aber geraten. Rechnet in unter einer Sekunde.":
            "For black-and-white and sepia images: the AI estimates the colours from the "
            "brightness – plausible, but guessed. Takes less than a second.",
        "Für dieses Bild berechnet. Der Regler mischt zwischen Original und kolorierter "
        "Fassung; Weißabgleich, Sättigung und Farbbereiche wirken danach wie gewohnt.":
            "Computed for this image. The slider blends between the original and the "
            "colourised version; white balance, saturation and colour ranges work as usual "
            "afterwards.",
        "KI koloriert das Bild …": "AI is colourising the image …",
        "KI-Kolorieren abgebrochen.": "AI colourising cancelled.",
        "Das KI-Kolorieren ist fehlgeschlagen.": "AI colourising failed.",
        "KI-Kolorieren fertig ({s} s, über {weg})": "AI colourising done ({s} s, via {weg})",
        # Objektivprofile (lensfun)
        "%d.%m.%Y": "%Y-%m-%d",
        "Objektivprofil anwenden": "Apply lens profile",
        "Gleicht Verzeichnung, Farbsäume und Vignette nach der Objektivdatenbank "
        "lensfun aus. Die Regler darunter wirken zusätzlich.":
            "Corrects distortion, colour fringes and vignetting from the lensfun lens "
            "database. The sliders below act on top of it.",
        "Objektivdatenbank laden": "Load lens database",
        "Nach neuer Datenbank suchen": "Check for a newer database",
        "Fragt bei lensfun nach, ob es eine neuere Objektivdatenbank gibt. Silberkorn "
        "fragt nie von selbst.":
            "Asks lensfun whether a newer lens database exists. Silberkorn never asks on "
            "its own.",
        "Die Objektivdatenbank ist noch nicht geladen (etwa 0,5 MB).":
            "The lens database has not been loaded yet (about 0.5 MB).",
        "Objektivdatenbank vom {datum}.": "Lens database of {datum}.",
        "Die Datei nennt kein Objektiv.": "The file does not name a lens.",
        "Kein Profil für „{objektiv}“ gefunden.": "No profile found for “{objektiv}”.",
        "Farbsäume": "Colour fringes",
        "Profil für: {arten}": "Profile for: {arten}",
        "Kamera-JPEGs sind oft schon in der Kamera korrigiert – deshalb hier nicht "
        "automatisch.":
            "Camera JPEGs are often already corrected in the camera – so not applied "
            "automatically here.",
        "Silberkorn lädt die Objektivdatenbank (etwa 0,5 MB) von:\n{quelle}\n\n"
        "Sie stammt vom Projekt {herkunft}.\n\n"
        "Ablage: {ordner}\n\nJetzt herunterladen?":
            "Silberkorn downloads the lens database (about 0.5 MB) from:\n{quelle}\n\n"
            "It comes from the {herkunft} project.\n\n"
            "Location: {ordner}\n\nDownload now?",
        "Objektivdatenbank wird geladen …": "Loading the lens database …",
        "Die Objektivdatenbank ließ sich nicht laden.": "The lens database could not be loaded.",
        "Objektivdatenbank geladen: {ordner}": "Lens database loaded: {ordner}",
        "Ob es eine neuere Objektivdatenbank gibt, ließ sich nicht feststellen.":
            "Could not determine whether a newer lens database exists.",
        "Objektivdatenbank": "Lens database",
        "Die Objektivdatenbank ist aktuell (Stand {datum}).":
            "The lens database is up to date (as of {datum}).",
        "Es gibt eine neuere Objektivdatenbank vom {datum} (etwa 0,5 MB, von {quelle})."
        "\n\nJetzt laden?":
            "A newer lens database of {datum} is available (about 0.5 MB, from {quelle})."
            "\n\nLoad it now?",
        # Für Veröffentlichung speichern
        "Fürs Netz …": "For the web …",
        "Verkleinert, mit Wasserzeichen und Rechteangaben speichern – etwa für Bilder "
        "im Netz. Das Original bleibt unverändert.":
            "Save downsized, with watermark and rights information – for images on the "
            "web, for instance. The original stays unchanged.",
        "Für Veröffentlichung speichern": "Save for publishing",
        "Für Veröffentlichung gespeichert: {name} – {breite} × {hoehe} Pixel ({ms} ms)":
            "Saved for publishing: {name} – {breite} × {hoehe} pixels ({ms} ms)",
        "Speichern …": "Save …",
        "Größe": "Size",
        "Längste Kante": "Longest edge",
        "Prozent": "Percent",
        "Kleinere Bilder werden nicht vergrößert.": "Smaller images are not enlarged.",
        "Das Bild bleibt {breite} × {hoehe} Pixel groß.":
            "The image stays {breite} × {hoehe} pixels.",
        "Ergebnis: {breite} × {hoehe} Pixel (Original {b0} × {h0})":
            "Result: {breite} × {hoehe} pixels (original {b0} × {h0})",
        "Wasserzeichen": "Watermark",
        "Text": "Text",
        "z. B. Spieltitel oder Webadresse": "e.g. game title or web address",
        "Logo": "Logo",
        "kein Logo": "no logo",
        "Wählen …": "Choose …",
        "Logo entfernen": "Remove logo",
        "Logo wählen": "Choose logo",
        "Die Datei lässt sich nicht als Bild lesen.": "The file cannot be read as an image.",
        "Position": "Position",
        "Unten rechts": "Bottom right",
        "Unten links": "Bottom left",
        "Oben rechts": "Top right",
        "Oben links": "Top left",
        "Mitte": "Centre",
        "Textfarbe": "Text colour",
        "Weiß": "White",
        "Schwarz": "Black",
        "Deckkraft": "Opacity",
        "Als Muster über das ganze Bild": "As a pattern across the whole image",
        "Schwerer zu entfernen als ein Zeichen in der Ecke, stört aber mehr beim Ansehen.":
            "Harder to remove than a mark in the corner, but more distracting to look at.",
        "Logo am besten als PNG mit durchsichtigem Hintergrund.":
            "Ideally a PNG logo with a transparent background.",
        "Rechte & Metadaten": "Rights & metadata",
        "Urheber": "Creator",
        "Copyright": "Copyright",
        "Webadresse": "Web address",
        "z. B. © 2026 Name": "e.g. © 2026 Name",
        "KI-Training und Data-Mining untersagen": "Prohibit AI training and data mining",
        "Schreibt einen maschinenlesbaren Nutzungsvorbehalt nach IPTC in die Datei. "
        "Suchmaschinen dürfen das Bild weiterhin finden.":
            "Writes a machine-readable reservation of rights (IPTC) into the file. Search "
            "engines may still find the image.",
        "Alle Kameradaten entfernen": "Remove all camera data",
        "Kamera, Objektiv, Belichtung und Aufnahmezeit. GPS-Position und "
        "Seriennummern werden in jedem Fall entfernt.":
            "Camera, lens, exposure and capture time. GPS position and serial numbers are "
            "always removed.",
        "Das Bild enthält von der KI erfundene Teile – das wird in den Metadaten vermerkt.":
            "The image contains parts invented by AI – this is noted in the metadata.",
    },
}
