# Änderungen

Das Format folgt [Keep a Changelog](https://keepachangelog.com/de/1.1.0/),
die Versionsnummern der [semantischen Versionierung](https://semver.org/lang/de/).

## [Unveröffentlicht]

### Hinzugefügt

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
