# Sicherheit

*English: please report security issues privately as described below; reports in English
are welcome.*

## Eine Lücke melden

Bitte **kein öffentliches Issue** für Sicherheitslücken. Melde sie über
GitHub: Reiter **Security** → **Report a vulnerability**
([private Sicherheitsmeldung](../../security/advisories/new)). Die Meldung ist
nur für die Projektbetreuung sichtbar.

Hilfreich sind: eine Beschreibung des Problems, die Version des Programms (sie
steht in der Kopfzeile und auf dem Reiter „Info & Copyright"), Grafikkarte und
Treiberversion (ebenfalls auf diesem Reiter), das Betriebssystem und, wenn
möglich, eine Anleitung zum Nachstellen.

Eine Antwort kommt in der Regel innerhalb einer Woche.

## Unterstützte Versionen

Gepflegt wird jeweils die neueste veröffentlichte Version.

## Wie das Programm mit Daten umgeht

- **Bilder** bleiben auf dem Rechner. Sie werden weder hochgeladen noch an
  einen Dienst übergeben; gerechnet wird auf der eigenen Grafikkarte.
- **Netzwerk**: Das Grundgerüst baut keine Verbindungen auf. Spätere Versionen
  laden KI-Modelle nur auf ausdrücklichen Wunsch und nur von der angezeigten
  Quelle; jede Modelldatei wird dabei gegen eine im Programm hinterlegte
  SHA-256-Prüfsumme geprüft.
- **Gespeichert** werden in `neuro-enhance.json` neben dem Programm nur
  Sprache, Farbschema und Fensterlage.
