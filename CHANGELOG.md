# Änderungen

Das Format folgt [Keep a Changelog](https://keepachangelog.com/de/1.1.0/),
die Versionsnummern der [semantischen Versionierung](https://semver.org/lang/de/).

## [Unveröffentlicht]

### Geändert

- KI-Hochskalieren rechnet in halber Genauigkeit (FP16) und im Speicherformat
  NHWC auf den Tensorkernen – auf einer RTX 4060 doppelt so schnell wie bisher
  (24 MP × 2 mit dem schnellen Modell: 5,7 s statt 11,1 s; 12 MP × 2 mit dem
  großen: 54 s statt 111 s). Die Modelldateien bleiben FP32 und werden beim
  Laden im Speicher gewandelt; neue Abhängigkeit dafür: `onnx`. Die
  Abweichung zu FP32 liegt im Mittel bei 0,0002.

### Behoben

- `python -m neuro_enhance` startete das Programm auch in jedem Hilfsprozess
  erneut; der Start steht jetzt hinter `if __name__ == "__main__"`.
- Ein Bild mit Alphakanal ließ sich nach Zuschnitt oder Drehung nicht
  speichern: Der Alphakanal behielt seine alte Größe. Er bekommt jetzt
  dieselbe Geometrie wie das Bild (ohne Vignette und Farbsäume).
- RAWs der meisten Kameras (getestet mit Nikon NEF und Canon CR2) liefen nicht
  über die GPU, sondern fielen auf LibRaw zurück: rawpy gibt die Farbmatrix
  bei ihnen nur als Nullen heraus. Sie wird jetzt wie in LibRaw aus der
  Kameratabelle (XYZ → Kamera) berechnet. Die Farben stimmen mit LibRaw
  überein; eine 24-MP-RAW ist in rund 0,5 s offen statt in 1,3 s.

### Hinzugefügt

- TensorRT als optionales Zusatzpaket (`requirements-tensorrt.txt`, rund
  1,8 GB von NVIDIA): Ist es installiert, rechnet die KI darüber – auf einer
  RTX 4060 noch einmal rund doppelt so schnell (12 MP × 2 mit dem großen
  Modell: 24 s statt 54 s). Die Engine entsteht beim ersten Einsatz einmalig
  je Modell, Grafikkarte und Kachelgröße in einem eigenen Prozess (ein bis
  zwei Minuten, mit Hinweisfenster) und liegt danach unter
  `modelle/tensorrt`. Scheitert TensorRT, rechnet das Programm mit CUDA
  weiter. „Info & Copyright“ zeigt die TensorRT-Fassung, die Statuszeile nach
  dem Speichern den Rechenweg.

- KI-Modelle per Knopfdruck laden: Die Karte „KI-Hochskalieren“ bietet
  „Modell herunterladen“ an, fragt mit Quelle, Größe und Lizenz nach und lädt
  aus dem Release `modelle-1` dieses Projekts. Jede Datei wird zuerst unter
  `.teil` gespeichert und erst nach bestandener SHA-256-Prüfung umbenannt;
  Abbrechen hinterlässt nichts. Ist der Programmordner schreibgeschützt,
  landen die Modelle unter `%LOCALAPPDATA%\Neuro-Enhance\modelle`.
- KI-Hochskalieren um 2 × oder 4 × beim Speichern mit Real-ESRGAN über ONNX
  Runtime (CUDA): schnelles Modell (realesr-general-x4v3) mit Regler für die
  Entrauschstärke, für den die Gewichte zweier Modelle auf der GPU gemischt
  werden, und großes Modell (RealESRGAN_x4plus) ab Funktionsstufe M.
  Kacheln mit Überlappung, Größe je VRAM-Stufe, bei Speichermangel kleiner;
  Fortschritt mit Abbrechen. Jede Modelldatei wird vor dem Laden gegen ihre
  SHA-256-Prüfsumme geprüft.
- `werkzeuge/modelle_exportieren.py` wandelt die offiziellen Real-ESRGAN-
  Gewichte nach ONNX und prüft das Ergebnis gegen PyTorch.
- 100-%-Ansicht: Zoom auf 100, 200 und 400 % per Mausrad oder `Strg`+`1`,
  Verschieben durch Ziehen, Doppelklick wechselt zur Einpassung. Gerechnet
  wird das ganze Bild in voller Auflösung, gezeigt der sichtbare Ausschnitt –
  pixelgleich mit dem gespeicherten Bild. Verschieben braucht rund 2 ms je
  Ausschnitt, weil das Ergebnis auf der GPU liegen bleibt.
- Geometrie: 90° drehen, spiegeln, begradigen, Perspektive senkrecht und
  waagrecht, Zuschneiden mit Rahmen, Drittellinien und Seitenverhältnissen.
  Eine automatische Vergrößerung verhindert leere Ecken.
- Objektiv: Verzeichnung, Vignette, Farbsäume Rot/Cyan und Blau/Gelb.
- Alle Geometrie- und Objektivkorrekturen sind eine einzige Rückwärts-
  abbildung mit bikubischer Abtastung (Catmull-Rom), auf der GPU ein Kernel.
- Farbbereiche (HSL): Farbton, Sättigung und Luminanz für acht Bereiche, in
  OkLCh mit Kosinus-Übergängen; Grautöne bleiben unberührt. Bedienung über
  eine Karte mit Umschaltern für die drei Eigenschaften.
- LUTs im `.cube`-Format (3D und 1D, DOMAIN_MIN/MAX), tetraedrisch
  interpoliert, mit Stärkeregler; auf der GPU im selben Durchlauf wie Kurven
  und Farbe.
- Rauschminderung: Luminanz mit Non-Local Means (7 × 7 Suchfenster,
  3 × 3 Flecken), Farbe mit einem farbgeführten Fast Guided Filter. Das
  Ergebnis wird für die Vorschau zwischengespeichert; beim Ziehen an den
  Rauschreglern braucht eine Vorschau mit 10 Megapixeln 10 bis 40 ms.
- Mittelwertfilter auf der GPU über Kastenfilter statt Summentabellen:
  Dunst entfernen ist dadurch rund dreimal so schnell.
- RAW-Entwicklung auf der GPU für Bayer-Sensoren: LibRaw liest nur noch das
  Mosaik; Schwarzwert, Weißabgleich, Demosaicing und Farbmatrix rechnet die
  Grafikkarte. Als Verfahren dient RCD (Ratio Corrected Demosaicing) in vier
  CUDA-Durchläufen – an feinen Mustern etwa halb so viele Farbsäume wie
  Malvar-He-Cutler, das als schnellere Alternative im Code bleibt. Eine
  24-MP-RAW ist in rund 0,2 s offen statt in 0,85 s; die Farben stimmen mit
  LibRaw überein. X-Trans, Foveon und bereits entwickelte DNG entwickelt
  weiterhin LibRaw.
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
