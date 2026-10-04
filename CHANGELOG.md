# Änderungen

Das Format folgt [Keep a Changelog](https://keepachangelog.com/de/1.1.0/),
die Versionsnummern der [semantischen Versionierung](https://semver.org/lang/de/).

## [Unveröffentlicht]

### Hinzugefügt

- RAW-Entwicklung auf der GPU für Bayer-Sensoren: LibRaw liest nur noch das
  Mosaik, Schwarzwert, Weißabgleich, Demosaicing nach Malvar-He-Cutler und
  Farbmatrix rechnet ein CUDA-Kernel in einem Durchlauf. Eine 24-MP-RAW ist
  in rund 0,15 s offen statt in 0,85 s; die Farben stimmen mit LibRaw
  überein. X-Trans, Foveon und bereits entwickelte DNG entwickelt weiterhin
  LibRaw.
- 16 Bit: TIFF und PNG mit 16 Bit je Kanal öffnen und speichern (tifffile,
  imagecodecs). Der Speichern-Dialog bietet PNG und TIFF mit 8 oder 16 Bit an
  und schlägt für Bilder mit mehr als 8 Bit ein 16-Bit-TIFF vor. 16-Bit-PNG
  behält EXIF-Daten; 16-Bit-TIFF trägt das sRGB-Profil, aber noch keine
  EXIF-Daten.
- RAW: Entwicklung mit LibRaw (rawpy), linear in 16 Bit, mit dem
  Weißabgleich der Kamera und ohne automatische Aufhellung.
- Farbprofile: RGB-Matrixprofile (Adobe RGB, ProPhoto, Display P3 …) werden
  selbst gelesen und auf der GPU in lineares sRGB umgerechnet – bei jeder
  Bittiefe und ohne Farben außerhalb von sRGB vorzeitig abzuschneiden.
  CMYK-, Graustufen- und Tabellenprofile rechnet weiterhin LittleCMS.
- Die Vorschau wird auf der GPU in linearem Licht verkleinert.
- Gradationskurve für Helligkeit sowie Rot, Grün und Blau, monoton kubisch
  interpoliert, mit Histogramm der Vorschau im Hintergrund. Die
  Helligkeitskurve wirkt farbtreu, die Kanalkurven verschieben gezielt die
  Farbe.
- Klarheit: lokaler Kontrast in den Mitteltönen mit einem Radius relativ zur
  Bildgröße – Vorschau und Export wirken gleich.
- Dunst entfernen nach dem Dark Channel Prior mit Guided Filter; negative
  Werte fügen Dunst hinzu. Die Schätzung läuft auf einer Kopie mit 512 Pixeln
  Kantenlänge und wird beim Ziehen am Regler wiederverwendet. Mit allen
  Filtern zusammen braucht die Vorschau mit 10 Megapixeln rund 20 ms.
- Erste klassische Filter auf der GPU: Weißabgleich (Temperatur, Tönung),
  Belichtung, Kontrast, Lichter, Tiefen, Dynamik, Sättigung und Schärfen mit
  Radius. Gerechnet wird nicht-destruktiv in linearem Licht; die Kette läuft
  als drei zusammengefasste CUDA-Kernel (Vorschau mit 10 Megapixeln in rund
  12 ms auf einer RTX 4060, Export von 24 Megapixeln in unter 0,2 s).
- Bilder öffnen (JPEG, PNG, TIFF, WebP, BMP, optional HEIC) per Dialog,
  `Strg`+`O` oder Ziehen und Ablegen; speichern als JPEG, PNG, TIFF oder WebP.
  EXIF-Ausrichtung wird angewendet, eingebettete Farbprofile werden nach sRGB
  umgerechnet, EXIF-Daten und Alphakanal bleiben erhalten.
- „Vorher“-Knopf, Zurücksetzen per Doppelklick und „Alles zurücksetzen“;
  Nachfrage vor dem Verwerfen ungespeicherter Änderungen.
- Rechenzeit der GPU in der Statuszeile, CuPy- und NVRTC-Version im Reiter
  „Info & Copyright“.
- Grundgerüst der Anwendung mit PySide6: Kopfzeile mit Sprachwahl (Deutsch,
  Englisch) und dunklem Farbschema, Reiter „Bearbeiten" und „Info & Copyright",
  Statuszeile mit Grafikkarte und Funktionsstufe.
- Startprüfung über NVML, bevor CUDA-Bibliotheken geladen werden. Ohne
  NVIDIA-RTX-Grafikkarte oder mit einem Treiber älter als 570.65 meldet das
  Programm beim Start, was gefunden wurde und was fehlt, und beendet sich.
  GTX-Karten werden auch dann erkannt und abgelehnt, wenn sie zur
  Turing-Generation gehören.
- Funktionsstufen nach Grafikspeicher (–, S, M, L, XL) und Erkennung der
  Rechengenauigkeit (FP16, FP8 ab Ada, FP4 ab Blackwell).
- Programmsymbol, Marken- und Lizenzhinweise (NOTICE).
