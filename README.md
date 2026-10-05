# Neuro-Enhance

**Neuro-Enhance - An open-source, GPU-accelerated AI image & photo enhancer powered by neural networks.**<br>
**Neuro-Enhance - ein quelloffener, GPU-beschleunigter KI-Bild- und Fotoverbesserer auf Basis neuronaler Netze.**

[![CI](https://github.com/DerAlexmann/Neuro-Enhance/actions/workflows/ci.yml/badge.svg)](https://github.com/DerAlexmann/Neuro-Enhance/actions/workflows/ci.yml)

Supports hardware acceleration via NVIDIA CUDA / RTX series GPUs.<br>
Unterstützt Hardwarebeschleunigung über NVIDIA CUDA und Grafikkarten der RTX-Serie.

*NVIDIA and RTX are trademarks of NVIDIA Corporation. – NVIDIA und RTX sind Marken der
NVIDIA Corporation.*

*[English version: README.en.md](README.en.md)*

> **Frühe Entwicklungsphase.** Die klassischen Grundregler laufen bereits auf der GPU;
> die KI-Funktionen folgen in den nächsten Versionen.

## Systemvoraussetzungen

Neuro-Enhance rechnet ausschließlich auf der Grafikkarte. **Ohne NVIDIA-RTX-Grafikkarte
startet das Programm nicht**, sondern meldet beim Start, was gefunden wurde und was fehlt.
Einen Notbetrieb auf dem Prozessor gibt es bewusst nicht.

| | Mindestens |
|---|---|
| Betriebssystem | Windows 10 oder 11, 64 Bit |
| Grafikkarte | NVIDIA GeForce RTX 2000-Serie oder neuer, auch RTX-Profikarten (RTX A-Serie, RTX Ada) |
| Grafikspeicher | 4 GB für KI-Funktionen; darunter nur klassische Filter |
| Grafiktreiber | NVIDIA 570.65 oder neuer |
| Python (nur für den Start als Skript) | 3.10 oder neuer |

**Nicht unterstützt** werden GTX-Karten – auch die GTX-16-Serie, die zwar zur selben
Generation wie die RTX 2000 gehört, aber keine Tensor Cores hat – sowie Grafikkarten
anderer Hersteller.

### Funktionsstufen

Was verfügbar ist, hängt vor allem vom Grafikspeicher ab. Neuro-Enhance erkennt die
Stufe beim Start; Funktionen einer höheren Stufe erscheinen ausgegraut mit Hinweis.

| Stufe | Grafikspeicher | Beispiele | Umfang |
|---|---|---|---|
| – | unter 4 GB | RTX 2050 (Laptop) | nur klassische Filter |
| S | 4 GB | RTX 3050 Laptop | klassische Filter, kleine KI-Modelle |
| M | 6–8 GB | RTX 2060, 3060 Ti, 4060, 5060 | alle Restaurierungsmodelle: Hochskalieren, Entrauschen, Freistellen, Objektauswahl |
| L | 12–16 GB | RTX 3060 12 GB, 4070, 5070 Ti | zusätzlich größere Modelle und generative Füllung |
| XL | 24 GB und mehr | RTX 3090, 4090, 5090 | voller Umfang |

Rechengenauigkeit: alle RTX-Karten rechnen in FP16; ab RTX 4000 (Ada) zusätzlich FP8, ab
RTX 5000 (Blackwell) auch FP4. Die Werte stehen auf dem Reiter „Info & Copyright".

## Was es schon kann

- **Grundregler auf der GPU**: Weißabgleich (Temperatur, Tönung), Belichtung,
  Kontrast, Lichter, Tiefen, Dynamik, Sättigung und Schärfen mit einstellbarem
  Radius.
- **Gradationskurve** für die Helligkeit und für Rot, Grün und Blau einzeln, mit
  Histogramm im Hintergrund. Klicken setzt einen Punkt, Ziehen verschiebt ihn,
  Doppelklick entfernt ihn; die Helligkeitskurve verändert die Farben nicht.
- **Klarheit**: mehr oder weniger lokaler Kontrast in den Mitteltönen.
- **Dunst entfernen** nach dem Dark-Channel-Prior-Verfahren – auch umgekehrt, um
  Dunst hinzuzufügen.
- **Farbbereiche (HSL)**: Farbton, Sättigung und Luminanz getrennt für Rot,
  Orange, Gelb, Grün, Aqua, Blau, Lila und Magenta. Gerechnet wird im
  wahrnehmungsgleichmäßigen Farbraum OkLCh, mit weichen Übergängen zwischen den
  Bereichen; Grautöne bleiben unberührt.
- **LUTs**: Looks als `.cube`-Datei (3D und 1D) laden, tetraedrisch
  interpoliert, mit Stärkeregler.
- **Rauschminderung**: Luminanzrauschen mit Non-Local Means, Farbrauschen mit
  einem farbgeführten Filter, der Farbkanten erhält – auch zwischen Farben
  gleicher Helligkeit. Die verkleinerte Vorschau zeigt Rauschen schwächer als
  das gespeicherte Bild – zum Beurteilen gibt es die 100-%-Ansicht.
- **KI-Entrauschen** mit SCUNet, trainiert auf echtes Kamerarauschen und auf
  Treue statt auf erfundene Details: einmal über das ganze Bild gerechnet (ein
  18-MP-Foto braucht auf einer RTX 4060 rund 20 Sekunden, mit TensorRT 11),
  danach mischt ein Stärkeregler sofort zwischen Original und entrauschtem
  Bild – in Vorschau, 100-%-Ansicht und Export gleichermaßen. Lichter über
  Weiß, etwa aus RAWs, bleiben erhalten.
- **Geometrie**: um 90° drehen, spiegeln, begradigen, Perspektive senkrecht
  und waagrecht, Zuschneiden mit Rahmen und festen Seitenverhältnissen. Leere
  Ecken nach dem Begradigen werden automatisch weggeschnitten.
- **Objektiv**: Verzeichnung, Vignette und Farbsäume (chromatische Aberration)
  ausgleichen. Alle Geometrie- und Objektivkorrekturen werden in einem einzigen
  bikubischen Schritt abgetastet.
- **KI-Hochskalieren** um 2 × oder 4 × beim Speichern, mit Real-ESRGAN über ONNX
  Runtime auf der GPU: ein schnelles Modell mit Regler für die Entrauschstärke
  (ab 4 GB Grafikspeicher) und ein großes Modell mit mehr Schärfe (ab
  Funktionsstufe M). Gerechnet wird in halber Genauigkeit (FP16) auf den
  Tensorkernen und in Kacheln, deren Größe sich nach dem Grafikspeicher
  richtet; ein 24-MP-Bild ist mit dem schnellen Modell auf einer RTX 4060 in
  knapp 6 Sekunden auf 96 MP vergrößert. Mit dem optionalen
  [TensorRT](#tensorrt-optional) geht es noch einmal rund doppelt so schnell.
- **100-%-Ansicht**: Zoom auf 100, 200 und 400 % per Mausrad, Verschieben durch
  Ziehen, Doppelklick wechselt zwischen eingepasst und 100 %. Gezeigt wird ein
  Ausschnitt des Bildes in voller Auflösung – genau das, was gespeichert wird.
- **Nicht-destruktiv**: Jede Änderung wird aus dem unveränderten Original neu
  gerechnet. Doppelklick setzt einen Regler zurück, „Vorher“ zeigt das Original,
  solange der Knopf gedrückt ist.
- **Schnell**: Die Vorschau entsteht in Bildschirmauflösung in wenigen
  Millisekunden; die Rechenzeit steht in der Statuszeile. Gespeichert wird in
  voller Auflösung.
- **Farbrichtig**: Gerechnet wird in linearem Licht mit 32-Bit-Gleitkomma.
  RGB-Farbprofile wie Adobe RGB, ProPhoto RGB oder Display P3 werden auf der GPU
  nach sRGB umgerechnet, ohne Umweg über 8 Bit; Farben außerhalb von sRGB
  bleiben bis zur Ausgabe erhalten. Die EXIF-Ausrichtung wird angewendet,
  EXIF-Daten und ein Alphakanal bleiben beim Speichern erhalten.
- **16 Bit und RAW**: TIFF und PNG mit 16 Bit je Kanal öffnen und speichern.
  RAW-Dateien praktisch aller Kameras (CR2, CR3, NEF, ARW, RAF, ORF, RW2, DNG
  und weitere) werden linear und ohne automatische Aufhellung entwickelt – der
  volle Umfang des Sensors bleibt erhalten, die Helligkeit regelt die
  Belichtung. Bei Sensoren mit Bayer-Mosaik, also bei fast allen Kameras,
  läuft das Demosaicing auf der GPU – mit RCD (Ratio Corrected Demosaicing),
  das an feinen Mustern besonders wenige Farbsäume hinterlässt. Eine
  24-Megapixel-RAW ist in rund 0,2 s offen statt in knapp einer Sekunde. Fujis X-Trans und
  andere Sensoren entwickelt weiterhin LibRaw.
- **Formate**: Öffnen von JPEG, PNG, TIFF, WebP, BMP und RAW (HEIC mit dem
  optionalen Paket `pillow-heif`), Speichern als JPEG und WebP mit 8 Bit, PNG
  und TIFF wahlweise mit 8 oder 16 Bit. Bilder mit mehr als 8 Bit schlägt der
  Speichern-Dialog als 16-Bit-TIFF vor.

| Taste | Wirkung |
|---|---|
| `Strg`+`O` | Bild öffnen (oder Datei auf die Fläche ziehen) |
| `Strg`+`S` | Speichern unter … – das Original wird nie stillschweigend überschrieben |
| `Strg`+`0` / `Strg`+`1` | eingepasst / 100 % |
| Mausrad, Ziehen, Doppelklick | zoomen, verschieben, zwischen eingepasst und 100 % wechseln |
| `Eingabe` / `Esc` | Zuschnitt übernehmen / abbrechen |

## Geplant

- **Objektivprofile**: Verzeichnung, Vignette und Farbsäume automatisch aus
  einer Objektivdatenbank (lensfun).
- **X-Trans auf der GPU**: Demosaicing auch für Fujis Sensoren auf der
  Grafikkarte.
- **Weitere KI-Funktionen**: Entrauschen, Entschärfen, Freistellen, Objektauswahl
  per Klick, Objekte entfernen, Tiefenkarte für künstliche Unschärfe.

## Starten

**Als Skript:**

```bash
pip install -r requirements.txt
python Neuro-Enhance.pyw
```

Die Pakete bringen CuPy und die nötigen CUDA-Bibliotheken von NVIDIA mit
(zusammen gut 1 GB); ein eigenes CUDA-Toolkit muss nicht installiert sein. Beim
ersten Einsatz eines Filters übersetzt die Grafikkarte das passende Programm
einmalig; das dauert ein bis drei Sekunden und wird für alle weiteren Starts
gespeichert.

Eine fertige EXE folgt mit dem ersten Release.

### TensorRT (optional)

Mit TensorRT rechnet die KI noch einmal rund doppelt so schnell – auf einer
RTX 4060 vergrößert das große Modell ein 12-MP-Bild in 24 statt 54 Sekunden:

```bash
pip install -r requirements-tensorrt.txt
```

Das sind rund 1,8 GB Download (installiert 2,8 GB) vom Paketindex von NVIDIA.
Neuro-Enhance erkennt TensorRT von selbst; unter „Info & Copyright“ steht die
Fassung. Beim ersten Vergrößern mit einem Modell baut TensorRT einmalig eine für
die Grafikkarte passende Engine – das dauert ein bis drei Minuten, ein Fenster
weist darauf hin. Die Engines liegen danach im Ordner `modelle/tensorrt`. Klappt
TensorRT nicht, rechnet das Programm wie ohne mit CUDA.

TensorRT steht unter einer proprietären Lizenz von NVIDIA, die mit der
Installation gilt. Sie erlaubt nicht, TensorRT mit Neuro-Enhance weiterzugeben –
deshalb gehört es auch zu keiner fertigen EXE, sondern wird immer selbst
installiert.

## Datenschutz

Neuro-Enhance arbeitet vollständig auf dem eigenen Rechner. Es werden keine Bilder
hochgeladen und keine Nutzungsdaten gesendet. Auch die KI rechnet lokal auf der
Grafikkarte. Die einzige Verbindung ins Netz ist das Laden eines KI-Modells – und
das nur, wenn man in der Karte „KI-Hochskalieren“ oder „KI-Entrauschen“
ausdrücklich darauf klickt und die Rückfrage mit Quelle, Größe und Lizenz
bestätigt.

## KI-Modelle

Die Modelle gehören nicht zum Programm. „Modell herunterladen“ lädt sie aus
Releases dieses Projekts: in der Karte „KI-Hochskalieren“ aus
[modelle-1](https://github.com/DerAlexmann/Neuro-Enhance/releases/tag/modelle-1)
das schnelle Modell mit knapp 10 MB und das große mit 64 MB, in der Karte
„KI-Entrauschen“ aus
[modelle-2](https://github.com/DerAlexmann/Neuro-Enhance/releases/tag/modelle-2)
SCUNet mit 71 MB. Jede Datei wird gegen ihre SHA-256-Prüfsumme geprüft, bevor
sie verwendet wird.
Abgelegt werden sie im Ordner `modelle` neben dem Programm oder, wenn der
schreibgeschützt ist, unter `%LOCALAPPDATA%\Neuro-Enhance\modelle`.

Die Modelle sind die offiziellen Gewichte von
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (BSD 3-Clause, Copyright 2021
Xintao Wang) und [SCUNet](https://github.com/cszn/SCUNet) (Apache-2.0, Copyright
2022 Kai Zhang), nach ONNX gewandelt. Wer das selbst nachvollziehen will:

1. Aus den [Releases von Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/releases)
   `realesr-general-x4v3.pth`, `realesr-general-wdn-x4v3.pth` und
   `RealESRGAN_x4plus.pth`, aus dem
   [Release v1.0 von KAIR](https://github.com/cszn/KAIR/releases/tag/v1.0), wo die
   SCUNet-Gewichte liegen, `scunet_color_real_psnr.pth` nach `modelle/quellen/`
   laden.
2. `pip install torch` (nur für diesen Schritt nötig).
3. `python werkzeuge/modelle_exportieren.py` – das legt die ONNX-Modelle in
   `modelle/` ab und prüft sie gegen PyTorch.

## Hinweis zu KI-Ergebnissen

KI-Verfahren wie Hochskalieren oder Entrauschen ergänzen Bilddetails, die im Original
nicht vorhanden waren. Bearbeitete Bilder eignen sich deshalb nicht als Beweis- oder
Dokumentationsmittel.

## Mitmachen

Fehlermeldungen, Vorschläge und Übersetzungen sind willkommen –
[CONTRIBUTING.md](CONTRIBUTING.md) erklärt Aufbau, Stil und Tests.
Sicherheitslücken bitte nicht als Issue, sondern über den Weg in
[SECURITY.md](SECURITY.md).

## Lizenz und Rechtliches

[MIT](LICENSE) – Copyright 2026 Alexander Unverhau.
Erstellt mit Unterstützung von Claude AI.

Verwendete Fremdkomponenten und ihre Lizenzen stehen in [NOTICE](NOTICE). KI-Modelle
stehen unter eigenen Lizenzen, die vor dem Herunterladen angezeigt werden.

**Marken:** NVIDIA, RTX, GeForce, CUDA und TensorRT sind Marken oder eingetragene Marken
der NVIDIA Corporation in den USA und anderen Ländern. Qt ist eine Marke der The Qt
Company Ltd., Python eine Marke der Python Software Foundation, Windows eine Marke der
Microsoft Corporation. Alle weiteren Marken gehören ihren jeweiligen Inhabern.

Neuro-Enhance ist ein unabhängiges Projekt und steht in keiner Verbindung zur NVIDIA
Corporation; es wird von ihr weder unterstützt noch gesponsert.
