# Änderungen

Das Format folgt [Keep a Changelog](https://keepachangelog.com/de/1.1.0/),
die Versionsnummern der [semantischen Versionierung](https://semver.org/lang/de/).

## [Unveröffentlicht]

### Geändert

- Das Programm heißt jetzt **Silberkorn** (bisher Arbeitstitel Neuro-Enhance):
  Paket `silberkorn`, Startdatei `Silberkorn.pyw`, Einstellungen in
  `silberkorn.json`, Modelle bei schreibgeschütztem Programmordner unter
  `%LOCALAPPDATA%\Silberkorn\modelle`.
- RAW: Das Demosaicing auf der GPU arbeitet jetzt nach Malvar, He und Cutler.
  Neu ist die Wahl „RAW: Beste Qualität“ in der Kopfzeile; dann entwickelt
  LibRaw mit dem Verfahren DHT auf dem Prozessor (etwa 1,5 s bei 26 MP).
- Die Nachbauten fremder Netze liegen je Herkunft in eigenen Dateien
  `werkzeuge/netz_*.py` unter der Lizenz ihres Vorbilds; die Lizenztexte der
  Vorbilder (MIT, BSD-3-Clause, Apache-2.0) liegen in `LICENSES/`. README,
  NOTICE und `pyproject.toml` nennen die Ausnahmen von der MIT-Lizenz.
- Gesperrte Hauptknöpfe (etwa „Entfernen“ ohne Markierung) erscheinen grau
  statt in der Akzentfarbe.
- ONNX Runtime bekommt höchstens so viel Grafikspeicher, wie gerade frei ist.
  Ohne Grenze wuchs sein Speicherpool über die Karte hinaus, und Windows
  lagerte aus – ein Netz rechnete dann bis zu zwanzigmal langsamer.
- KI-Kacheln von KI-Entrauschen und KI-Schärfen werden in ihrer Überlappung
  weich ineinander geblendet statt hart aneinandergesetzt; Netze, die über die
  ganze Kachel schauen, liefern sonst sichtbare Nähte.
- Die Marke einer gebauten TensorRT-Engine enthält jetzt die Prüfsumme der
  Modelldatei; ändert sich ein Modell, wird die Engine neu gebaut.
- KI-Hochskalieren rechnet in halber Genauigkeit (FP16) und im Speicherformat
  NHWC auf den Tensorkernen – auf einer RTX 4060 doppelt so schnell wie bisher
  (24 MP × 2 mit dem schnellen Modell: 5,7 s statt 11,1 s; 12 MP × 2 mit dem
  großen: 54 s statt 111 s). Die Modelldateien bleiben FP32 und werden beim
  Laden im Speicher gewandelt; neue Abhängigkeit dafür: `onnx`. Die
  Abweichung zu FP32 liegt im Mittel bei 0,0002.

### Entfernt

- RCD-Demosaicing (`silberkorn/rcd.py`). Es war eine Übertragung von RCD
  (Luis Sanz Rodríguez) in der Fassung von RawTherapee und darktable, die unter
  der GPL-3.0 stehen; als solche durfte es nicht unter der MIT-Lizenz stehen.
  Die frühere Fassung bleibt in der Git-Historie (ab Commit 75bae17) und ist
  dort als GPL-3.0-Bearbeitung zu verstehen.

### Behoben

- Sprachwechsel: Die Auffrischung brach am ersten Text ohne Übersetzung
  („100 %“) ab, fast alles blieb in der alten Sprache. Auswahllisten und Qts
  eigene Knöpfe (Ja/Nein/Abbrechen) wechseln jetzt ebenfalls mit.
- Reglerleiste: Ihr Inhalt ist immer genau so breit wie sichtbar; ein zu
  breites Element schob vorher alle Karten nach rechts aus dem Bild.
- Darstellung: Aufklapplisten, Kästchen, Fortschrittsbalken, Kontextmenüs
  und das Detailfeld von Fehlermeldungen folgen dem Farbschema; Listen
  zeigen ihren längsten Eintrag ganz und flackern beim Aufklappen nicht mehr.
- KI-Knöpfe: kurze Beschriftungen („Anklicken · 147 MB“) statt abgeschnittener
  Texte; der Tooltip erklärt ohne Modell den Download, mit Modell die Aktion.
- `python -m silberkorn` startete das Programm auch in jedem Hilfsprozess
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

- Anonymisieren: Flächen als Rechteck oder Ellipse verpixeln, weichzeichnen oder
  schwarz füllen; „Gesichter finden“ mit YuNet (MIT, liegt bei, Prozessor),
  Flächen auf dem Bild verschieben, in der Größe ändern und aufziehen. Speichern
  auf Wunsch ohne GPS, Seriennummern, Besitzername und Herstellerdaten.
- KI-Erweitern (Outpainting) mit FLUX.2 [klein] 4B und der Outpaint-LoRA von
  fal (beide Apache-2.0): In der Karte „Geometrie“ erweitert „Mit KI erweitern
  statt beschneiden“ das Bild auf das gewählte Seitenverhältnis, statt es
  zuzuschneiden. Das Original kommt mittig auf eine reingrüne Leinwand von rund
  einem Megapixel, die KI füllt das Grün in 4 Schritten; danach wird ihr
  Farbstich am Original herausgenommen und das unveränderte Original mit
  weichem Saum eingesetzt. „Neu erzeugen“ rechnet mit neuem Zufall; über 25 %
  je Seite gibt es einen Hinweis. Die Erweiterung liegt vor allen Reglern und
  lässt sich wieder abwählen. Ab 8 GB Grafikspeicher; das Modell (rund 4,1 GB,
  Release „modelle-8“, auf mehrere Dateien verteilt) wird erst auf Wunsch
  geladen.
- `werkzeuge/outpaint_export.py`: rechnet die LoRA ein, exportiert Transformer
  und VAE nach ONNX, quantisiert den Transformer nach int8 (MatMulNBits,
  Block 128) und prüft jedes Netz sowie den ganzen Ablauf gegen diffusers.
  `werkzeuge/outpaint_referenz.py` ist der lokal geprüfte Stand in PyTorch.
- Zuschnitt: Seitenverhältnisse 4:5 (etwa für Instagram) und 5:4.
- Windows-EXE: `build.cmd` baut mit PyInstaller einen Programmordner und ein
  ZIP (ohne Installation). Die EXE bringt CUDA und cuDNN nicht mit, sondern
  lädt beim ersten Start nach Zustimmung zu NVIDIAs Lizenzen NVIDIAs
  offizielle Pakete von pypi.org (1,6 GB) und prüft sie gegen ihre
  Prüfsummen. Die Lizenztexte aller eingepackten Pakete liegen bei.
- Neue Karte „Tiefe & Bokeh“ mit Depth Anything V2 Small (Apache-2.0): Die
  Tiefe wird einmal geschätzt (0,16 s bei 26 MP, RTX 4060) und mit einem
  geführten Filter an die Kanten des Bildes gelegt. „Unschärfe“ zeichnet nach
  dem Abstand zur Fokusebene weich, „Fokus“ und „Schärfentiefe“ stellen ein,
  ein Klick ins Bild setzt den Fokus, eine Tiefenkarte zeigt die Schätzung.
  Das Bokeh mischt vier unterschiedlich stark weichgezeichnete Fassungen in
  linearem Licht; nur ähnlich Unscharfes trägt zu ihnen bei, damit Scharfes
  nicht ausblutet. Ein erkanntes Motiv bleibt auf Wunsch scharf. Nach dem
  Schätzen wird auf das Motiv bzw. das Nächste im Bild scharf gestellt. Das
  Modell (98 MB) kommt aus dem neuen Release `modelle-7`; es ist in
  `werkzeuge/netz_tiefe.py` ohne xFormers, OpenCV und torchvision nachgebaut.
- Neue Karte „Objekte entfernen“ mit LaMa (Big LaMa, Apache-2.0): Objekte per
  Klick (SAM 2) oder mit dem Pinsel markieren, „Entfernen“ füllt die Stelle.
  Gerechnet wird ein Ausschnitt um die Markierung, höchstens 768 bis 1536 Pixel
  je nach Grafikspeicher, in 0,1 bis 0,4 s (RTX 4060). Die Markierung wächst
  vorher um 5 % ihrer Ausdehnung, damit kein Umriss stehen bleibt. Entfernte
  Objekte liegen als Flicken über dem entrauschten und geschärften Bild,
  stapeln sich und lassen sich einzeln zurücknehmen; ungespeicherte
  Entfernungen zählen als Änderung. Die mittlere Maustaste verschiebt das
  vergrößerte Bild jetzt in jedem Modus. Das Modell (196 MB) kommt aus dem
  neuen Release `modelle-6`; es ist in `werkzeuge/netz_lama.py` mit der
  Fouriertransformation als DFT-Matrizen nachgebaut, der Checkpoint wird mit
  `werkzeuge/checkpoint_lesen.py` gelesen, ohne Code aus dem Pickle
  auszuführen.
- Objekt per Klick auswählen in der Karte „Motiv & Hintergrund“ mit
  Segment Anything 2 (SAM 2.1 small, Apache-2.0): Linksklick nimmt dazu,
  Rechtsklick weg, Strg+Z nimmt den letzten Klick zurück, Esc beendet. Der
  Encoder sieht das Bild einmal (0,1 s), jeder Klick kostet danach rund
  40 ms bei 24 MP (RTX 4060). Die Maske kommt grob aus SAM (256 × 256) und
  wird mit einem geführten Filter an die Kanten des Bildes gelegt. Klicks
  werden durch die Geometrie zurück ins Original gerechnet; bei 100 % lässt
  sich das Bild im Klickmodus weiter verschieben. Ab 4 GB Grafikspeicher.
  Das Modell (Encoder und Decoder, 147 MB) kommt aus dem neuen Release
  `modelle-5`; der Bildteil von SAM 2 ist in `werkzeuge/netz_sam2.py` ohne
  hydra, iopath und torchvision nachgebaut.
- Neue Karte „Motiv & Hintergrund“: BiRefNet (lite, 2K, MIT) erkennt das Motiv
  in rund zwei Sekunden (24 MP, RTX 4060). Belichtung, Kontrast, Sättigung,
  Temperatur und Unschärfe lassen sich für den Hintergrund getrennt einstellen
  (oder mit „Umkehren“ fürs Motiv); dafür rechnet die Kette das Bild ein
  zweites Mal und mischt beide über die Maske, die Unschärfe ohne Lichtsaum
  um das Motiv. Kante weicher und verschieben, Maskenansicht, Hintergrund
  beim Speichern als PNG oder TIFF durchsichtig. Ab 6 GB Grafikspeicher. Das
  Modell (177 MB) kommt aus dem neuen Release `modelle-4`; es ist in
  `werkzeuge/netz_birefnet.py` ohne einops, timm, kornia und torchvision
  nachgebaut.
- KI-Schärfen gegen leichte Fokus-Unschärfe mit Restormer
  (single_image_defocus_deblurring, MIT) in einer eigenen Karte: einmal über
  das ganze Bild gerechnet, danach regelt ein Stärkeregler ohne Wartezeit.
  Geschärft wird das KI-entrauschte Bild, falls vorhanden; gespeichert wird nur
  die Korrektur, und von ihr nur der feine Anteil (Hochpass, Sigma 10 px) –
  Restormer hellt nebenbei auf und verschiebt die Farbe je Kachel ein wenig.
  24 MP brauchen auf einer RTX 4060 rund 3 min, mit TensorRT 45 s. Das Modell
  (101 MB) kommt aus dem neuen Release `modelle-3`. Getestet und verworfen:
  NAFNet (GoPro/REDS) – es erzeugt bei Fotos in voller Auflösung Fehlmuster –
  und das Restormer-Modell gegen Verwacklung, das bei echten Fotos kaum wirkt.
- KI-Entrauschen mit SCUNet (scunet_color_real_psnr, Apache-2.0) in einer
  eigenen Karte: einmal über das ganze Bild gerechnet, danach mischt ein
  Stärkeregler ohne Wartezeit zwischen Original und entrauschtem Bild – in
  Vorschau, 100-%-Ansicht und Export. Gerechnet wird in sRGB wie beim
  Training; Lichter über Weiß bleiben, weil nur die Korrektur des Netzes aufs
  Original gelegt wird. 18 MP brauchen auf einer RTX 4060 rund 20 s, mit
  TensorRT 11 s. Das Modell (71 MB) kommt aus dem neuen Release `modelle-2`;
  `modelle-1` bleibt unverändert.
- `werkzeuge/modelle_exportieren.py` baut SCUNet ohne einops und timm nach
  (rechnet bitgenau wie das Original) und exportiert es für beliebige
  Bildgrößen; mit Argumenten lassen sich einzelne Modelle exportieren.

- TensorRT als optionales Zusatzpaket (`requirements-tensorrt.txt`, rund
  1,8 GB von NVIDIA): Ist es installiert, rechnet die KI darüber – auf einer
  RTX 4060 noch einmal rund doppelt so schnell (12 MP × 2 mit dem großen
  Modell: 24 s statt 54 s). Die Engine entsteht beim ersten Einsatz einmalig
  je Modell, Grafikkarte und Kachelgröße in einem eigenen Prozess (ein bis
  drei Minuten, mit Hinweisfenster) und liegt danach unter
  `modelle/tensorrt`. Scheitert TensorRT, rechnet das Programm mit CUDA
  weiter. „Info & Copyright“ zeigt die TensorRT-Fassung, die Statuszeile nach
  dem Speichern den Rechenweg.

- KI-Modelle per Knopfdruck laden: Die Karte „KI-Hochskalieren“ bietet
  „Modell herunterladen“ an, fragt mit Quelle, Größe und Lizenz nach und lädt
  aus dem Release `modelle-1` dieses Projekts. Jede Datei wird zuerst unter
  `.teil` gespeichert und erst nach bestandener SHA-256-Prüfung umbenannt;
  Abbrechen hinterlässt nichts. Ist der Programmordner schreibgeschützt,
  landen die Modelle unter `%LOCALAPPDATA%\Silberkorn\modelle`.
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
