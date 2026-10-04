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

## Geplant

- **Weitere klassische Filter**: LUTs, HSL je Farbbereich, Entrauschen,
  Geometrie.
- **X-Trans auf der GPU**: Demosaicing auch für Fujis Sensoren auf der
  Grafikkarte.
- **KI-Funktionen**: Hochskalieren, Entrauschen, Entschärfen, Freistellen,
  Objektauswahl per Klick, Objekte entfernen, Tiefenkarte für künstliche Unschärfe.

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

## Datenschutz

Neuro-Enhance arbeitet vollständig auf dem eigenen Rechner. Es werden keine Bilder
hochgeladen und keine Nutzungsdaten gesendet. In späteren Versionen lädt das Programm
KI-Modelle **nur auf ausdrücklichen Wunsch** von der jeweils angegebenen Quelle herunter.

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
