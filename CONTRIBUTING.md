# Mitmachen

Fehlermeldungen, Verbesserungsvorschläge und Übersetzungen sind willkommen.

*English: this file is in German, but issues and pull requests in English are
just as welcome.*

## Einen Fehler melden

Ein [Issue](../../issues) mit Version, Grafikkarte, Grafikspeicher und
Treiberversion – alles steht auf dem Reiter „Info & Copyright" und lässt sich
dort markieren und kopieren – sowie dem Betriebssystem und dem, was du erwartet
hast. Sicherheitslücken gehören nicht ins Issue, sondern in eine private
Meldung – siehe [SECURITY.md](SECURITY.md).

## Das Projekt im Überblick

| Datei | Inhalt |
|---|---|
| `Silberkorn.pyw` | Startdatei für den Doppelklick |
| `silberkorn/start.py` | Programmstart, Startprüfung, Startmeldungen |
| `silberkorn/gpu_pruefung.py` | Grafikkarte über NVML erkennen und bewerten |
| `silberkorn/hauptfenster.py` | Hauptfenster mit Kopfzeile, Reitern und Statuszeile |
| `silberkorn/bearbeiten_seite.py` | Reiter „Bearbeiten“: Werkzeugleiste, Regler, Öffnen und Speichern |
| `silberkorn/leinwand.py` | Anzeige, Zoom und Verschieben, Zuschnittrahmen, Ziehen und Ablegen |
| `silberkorn/bearbeitung.py` | Sitzung: Original und Vorschau im Grafikspeicher |
| `silberkorn/filter.py` | Filterformeln mit NumPy/CuPy – die Referenz |
| `silberkorn/filter_gpu.py` | dieselben Formeln als zusammengefasste CUDA-Kernel |
| `silberkorn/geometrie.py` | Zuschnitt, Drehen, Begradigen, Perspektive, Objektiv |
| `silberkorn/lut.py` | `.cube`-LUTs lesen, tetraedrisch anwenden |
| `silberkorn/kurven.py` | Gradationskurven: Punkte bearbeiten, Tabellen bilden |
| `silberkorn/kurveneditor.py` | Kurvenfeld mit Histogramm |
| `silberkorn/bilddatei.py` | Laden und Speichern: 8 Bit, 16 Bit, RAW, EXIF |
| `silberkorn/demosaik.py` | RAW-Entwicklung auf der GPU (Bayer-Mosaik), Malvar-He-Cutler |
| `silberkorn/icc.py` | ICC-Matrixprofile lesen und nach linearem sRGB umrechnen |
| `silberkorn/ki.py` | KI-Hochskalieren: Modellkatalog, ONNX Runtime, Kacheln |
| `werkzeuge/modelle_exportieren.py` | Real-ESRGAN-Gewichte nach ONNX wandeln (braucht PyTorch) |
| `silberkorn/cuda.py` | lädt CuPy an einer Stelle |
| `silberkorn/farben.py` | Farbschemata, Schriften, Qt-Stylesheet |
| `silberkorn/uebersetzung.py` | Sprachumschaltung und Sprachtabelle |
| `silberkorn/einstellungen.py` | Einstellungsdatei neben dem Programm |
| `tests/` | Tests für Kartenbewertung, Startmeldungen, Sprachtabelle, Filter und Dateien |
| `icon_erzeugen.py` | Programmsymbol erzeugen |

## Entwickeln

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
python -m ruff check .
```

Die Tests brauchen weder eine Grafikkarte noch Qt; die Filter rechnen dort mit
NumPy. Mit Grafikkarte vergleichen zusätzliche Tests die CUDA-Kernel Pixel für
Pixel mit dieser Referenz. Ist TensorRT installiert, vergleichen weitere Tests
dessen Ergebnis mit CUDA; der erste Lauf baut dafür Engines und dauert einige
Minuten. Die Startmeldungen lassen
sich ohne passende Hardware ansehen:

```bash
python Silberkorn.pyw --simulieren=keine_rtx
```

Ebenso `kein_treiber`, `keine_nvidia`, `treiber_alt` und `wenig_vram`.

## Stil

- **Deutsch** in Kommentaren, Docstrings und Bezeichnern; englische Begriffe
  nur, wo sie die eingeführten sind (`widget`, `layout`, Qt-Methodennamen).
- **Zeilenlänge 100**, vier Leerzeichen, keine Tabs. `ruff` prüft das.
- **Kommentare erklären das Warum**, nicht das Was.
- **Neue Texte** immer durch `_("…")` führen, sonst fehlen sie in der anderen
  Sprache. `tests/test_uebersetzungen.py` achtet darauf.
- **Farben** nie direkt hinschreiben. Widgets bekommen einen Objektnamen, die
  Farbe kommt aus dem Stylesheet in `farben.py`.
- **Beschriftungen** über `beschriften()` setzen, damit sie den Sprachwechsel
  mitmachen.
- **Keine CUDA-Bibliothek vor der Startprüfung laden.** CuPy, ONNX Runtime und
  TensorRT werden erst importiert, wenn `gpu_pruefung` eine passende Karte
  gefunden hat.

## Einen Filter ergänzen

1. Die Formel als Funktion in `filter.py` schreiben – mit `xp_von()`, damit sie
   mit NumPy und CuPy läuft – und in `anwenden()` einhängen.
2. Den Regler in `REGLER` und `Einstellungen` eintragen, den Titel in
   `bearbeiten_seite.regler_titel()`, die Übersetzung in `uebersetzung.py`.
3. Dieselbe Formel in den passenden Kernel in `filter_gpu.py` übernehmen.
   `test_cuda_kernel_rechnen_wie_die_referenz` zeigt, ob beide übereinstimmen.

## KI-Modelle

Neue Modelle nur mit einer Lizenz, die Nutzung und Weitergabe in einem
quelloffenen Projekt erlaubt (etwa Apache 2.0, MIT, BSD). Modelle mit
nicht-kommerziellen Lizenzen (CC BY-NC, S-Lab und Ähnliche) werden nicht
standardmäßig angeboten. Zu jedem Modell gehören Quelle, Lizenz und
SHA-256-Prüfsumme.

## Pull Requests

- Eine Änderung pro Pull Request, mit kurzer Begründung.
- Tests und `ruff` müssen durchlaufen; die CI prüft beides.
- Für sichtbare Änderungen einen Eintrag in `CHANGELOG.md` unter
  „Unveröffentlicht" ergänzen.
- Bildschirmfotos zeigen ausschließlich das Programmfenster – kein Desktop,
  keine anderen Fenster, keine persönlichen Pfade.

## Lizenz

Mit deinem Beitrag stimmst du zu, dass er unter der [MIT-Lizenz](LICENSE)
steht.
