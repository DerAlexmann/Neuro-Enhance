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
| `Neuro-Enhance.pyw` | Startdatei für den Doppelklick |
| `neuro_enhance/start.py` | Programmstart, Startprüfung, Startmeldungen |
| `neuro_enhance/gpu_pruefung.py` | Grafikkarte über NVML erkennen und bewerten |
| `neuro_enhance/hauptfenster.py` | Hauptfenster mit Kopfzeile, Reitern und Statuszeile |
| `neuro_enhance/farben.py` | Farbschemata, Schriften, Qt-Stylesheet |
| `neuro_enhance/uebersetzung.py` | Sprachumschaltung und Sprachtabelle |
| `neuro_enhance/einstellungen.py` | Einstellungsdatei neben dem Programm |
| `tests/` | Tests für Kartenbewertung, Startmeldungen und Sprachtabelle |
| `icon_erzeugen.py` | Programmsymbol erzeugen |

## Entwickeln

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
python -m ruff check .
```

Die Tests brauchen weder eine Grafikkarte noch Qt. Die Startmeldungen lassen
sich ohne passende Hardware ansehen:

```bash
python Neuro-Enhance.pyw --simulieren=keine_rtx
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
