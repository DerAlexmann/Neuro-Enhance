"""
KI-Hochskalieren (Real-ESRGAN), KI-Entrauschen (SCUNet), KI-Schaerfen
(Restormer), Freistellen (BiRefNet), Auswahl per Klick (SAM 2) und Objekte
entfernen (LaMa) ueber ONNX Runtime

Die Modelle stammen aus den offiziellen Releases von Real-ESRGAN (BSD-3-Clause),
SCUNet (Apache-2.0), Restormer (MIT), BiRefNet (MIT), SAM 2 und LaMa (beide
Apache-2.0) und sind mit
werkzeuge/modelle_exportieren.py nach ONNX gewandelt. Bereit liegen sie als
Dateien eigener Releases dieses Projekts, mit dem jeweiligen Lizenztext
daneben. Heruntergeladen wird nur auf Wunsch des Anwenders; jede Datei wird
gegen ihre SHA-256-Pruefsumme geprueft, bevor sie an ihren Platz kommt und
bevor sie geladen wird.

Gesucht wird im Ordner `modelle` neben dem Programm und - falls der
schreibgeschuetzt ist, etwa unter "Programme" - im Benutzerordner.

Gerechnet wird in Kacheln mit Ueberlappung, damit auch grosse Bilder in den
Grafikspeicher passen. Die Daten bleiben dabei auf der GPU: ONNX Runtime liest
und schreibt ueber IOBinding direkt in CuPy-Speicher. Nur die fertigen Kacheln
wandern in den Arbeitsspeicher - ein vierfach vergroessertes 24-MP-Bild hat
384 Megapixel und passt auf keine Grafikkarte mehr.

Gerechnet wird in halber Genauigkeit (FP16) und im Speicherformat NHWC - so
arbeiten die Tensorkerne aller RTX-Karten am schnellsten, auf einer RTX 4060
rund 2,7-mal schneller als in FP32. Die Modelle liegen als FP32 vor und werden
beim Laden im Speicher gewandelt (etwa eine halbe Sekunde); Ein- und Ausgaenge
bleiben FP32. Die Abweichung zu FP32 liegt im Mittel bei 0,0002 - weit unter
einer Stufe von 8 Bit. Auch Real-ESRGAN selbst rechnet standardmaessig in FP16.
Ausnahme ist Restormer: Es normiert ueber alle Pixel einer Kachel, ONNX Runtime
summiert dabei in FP16 und liefe ueber - es rechnet mit CUDA in FP32, gleich schnell.

Ist das optionale Paket tensorrt-cu12-libs installiert (requirements-tensorrt.txt),
rechnet TensorRT - noch einmal rund doppelt so schnell. TensorRT baut dafuer je
Modell, Grafikkarte und Kachelgroesse einmalig eine Engine (ein bis drei Minuten)
und legt sie im Ordner `tensorrt` neben den Modellen ab. Gebaut wird in einem
eigenen Prozess, denn ONNX Runtime haelt dabei die Python-Sperre (GIL) - im
selben Prozess stuende die Oberflaeche minutenlang still. Scheitert TensorRT,
rechnet das Programm mit CUDA weiter. TensorRT steht unter einer NVIDIA-Lizenz,
die die Weitergabe des ONNX-Parsers nicht erlaubt - es gehoert deshalb nie zum
Programm, sondern wird vom Anwender selbst installiert.

Dieses Modul laedt ONNX Runtime erst beim ersten Gebrauch, also immer nach
der Startpruefung.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np

from . import einstellungen, filter
from .cuda import cupy as cp

QUELLE = "https://github.com/xinntao/Real-ESRGAN"
QUELLE_SCUNET = "https://github.com/cszn/SCUNet"
QUELLE_RESTORMER = "https://github.com/swz30/Restormer"
QUELLE_BIREFNET = "https://github.com/ZhengPeng7/BiRefNet"
QUELLE_SAM2 = "https://github.com/facebookresearch/sam2"
QUELLE_LAMA = "https://github.com/advimman/lama"
MODELL_RELEASE = "https://github.com/DerAlexmann/Neuro-Enhance/releases/download/modelle-1/"
MODELL_RELEASE_2 = "https://github.com/DerAlexmann/Neuro-Enhance/releases/download/modelle-2/"
MODELL_RELEASE_3 = "https://github.com/DerAlexmann/Neuro-Enhance/releases/download/modelle-3/"
MODELL_RELEASE_4 = "https://github.com/DerAlexmann/Neuro-Enhance/releases/download/modelle-4/"
MODELL_RELEASE_5 = "https://github.com/DerAlexmann/Neuro-Enhance/releases/download/modelle-5/"
MODELL_RELEASE_6 = "https://github.com/DerAlexmann/Neuro-Enhance/releases/download/modelle-6/"
BLOCK = 1 << 20

# Datei -> (SHA-256, Groesse in Bytes)
DATEIEN = {
    "realesr-general-x4v3.onnx":
        ("68b895a3b16ba12734c0c5c76e5293dbee13b3d86a6a2120c5bce1ac35507b56", 20751),
    "realesr-general-x4v3.npz":
        ("83000adde069d737a495f409607e4a3a404d7c0085d8267884d67fcb518e72bd", 4879300),
    "realesr-general-wdn-x4v3.npz":
        ("c8aaf96ed78d524e836a990d102e09cde7e045ab2c8aaf8acb0068528e7eeb05", 4879300),
    "realesrgan-x4plus.onnx":
        ("9c887160648173a00ef2e998609bc198f0868a11e6801c9c3abeb1868808926a", 67051953),
    "LICENSE-Real-ESRGAN.txt":
        ("4a699ec4863d96a91fc265948a0c90033f7e8735d515524dcf3444736406e0c2", 1519),
    "scunet-color-real-psnr.onnx":
        ("cd446d36b78ca0c92d1c01052f7e3d331e5e35080cf597404ea9b8c9106bf157", 74754219),
    "LICENSE-SCUNet.txt":
        ("db7c4e3148d7ff287423d0d62716c6519c71cbcabeb0369cf24c4e14a33865cb", 11409),
    "restormer-defocus.onnx":
        ("8d51ab54539cc0383208a036380a2118df6551b693900380d7ed242032b009b9", 105955794),
    "LICENSE-Restormer.txt":
        ("2b93776512924bc095ec5d97a79d76cf1ab401ae641d1ae1f46e94f1c53a2e59", 1090),
    "birefnet-lite-2k.onnx":
        ("c3c8c750ca533f691a12902f28d4712f0331900c32907f2f762352427a415a42", 185559417),
    "LICENSE-BiRefNet.txt":
        ("92a7089e0915fc32bc40067560b398f1e6a7a5958abd7d04eda393629a5acefb", 1066),
    "sam2.1-small-kodierer.onnx":
        ("ec764cb857928d5b80ad740a65fbafb54867f8b446b72f02bbeb414bea91a0a8", 137861210),
    "sam2.1-small-dekodierer.onnx":
        ("aa6140c678916f505133f8a1cd45f7b84b18dcd2ec058249bdcc16ab6fdeb8fc", 16510918),
    "LICENSE-SAM2.txt":
        ("c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4", 11357),
    "big-lama.onnx":
        ("05242ecae18e453d4fc7cf7df015d80c9c1e96f82b7ef7376f2e0579b8444770", 205471670),
    "LICENSE-LaMa.txt":
        ("4ceeeac5a802e86c413c22b16cce8e9a22027b0250c97e6f8ac97c14cf0542c0", 11348),
}
LIZENZDATEI = "LICENSE-Real-ESRGAN.txt"
RAND = 10                    # Ueberlappung je Kachelseite in Eingabepixeln, wie in Real-ESRGAN


@dataclass(frozen=True)
class Modell:
    schluessel: str
    datei: str
    sha256: str
    # Zwei Gewichtssaetze (Datei, SHA-256) zum Mischen - starkes und schwaches
    # Entrauschen; None, wenn die Gewichte in der ONNX-Datei stecken. Mit
    # Mischung enthaelt die ONNX-Datei keine Gewichte: Sie sind Eingaenge des
    # Netzes, denn eingebaute Gewichte laesst ONNX Runtime bei der Ausfuehrung
    # auf der GPU nicht ersetzen.
    mischung: tuple[tuple[str, str], tuple[str, str]] | None
    mindeststufe: str                # kleinste VRAM-Stufe, ab der es angeboten wird
    kacheln: dict                    # Kachelgroesse in Eingabepixeln je Stufe
    lizenz: str = "BSD-3-Clause"
    quelle: str = QUELLE
    herkunft: str = "Real-ESRGAN (BSD 3-Clause, Copyright 2021 Xintao Wang)"
    lizenzdatei: str = LIZENZDATEI
    release: str = MODELL_RELEASE    # woher die App die Dateien laedt
    massstab: int = 4                # Ausgabe ist massstab-mal so gross wie die Eingabe
    rand: int = RAND                 # Ueberlappung je Kachelseite
    vielfaches: int = 1              # Hoehe und Breite einer Kachel muessen Vielfache sein
    # NHWC beschleunigt reine Faltungsnetze; Netze mit Normierung ueber die Kanaele
    # (Restormer) bremst es auf ein Drittel, weil ONNX Runtime dann staendig umsortiert
    nhwc: bool = True
    # Alle Kacheln gleich gross: Randkacheln ruecken ins Bild hinein, statt kleiner zu
    # werden. cuDNN kompiliert manche Faltungen (Restormer in FP16) fuer jede neue
    # Kachelgroesse eigens - rund 26 s je Groesse auf einer RTX 4060.
    feste_kachel: bool = False
    # Mit CUDA in FP16 rechnen? Restormer normiert ueber alle Pixel einer Kachel; ONNX
    # Runtime summiert dabei in FP16 und laeuft ueber. In FP32 ist es gleich schnell.
    cuda_fp16: bool = True
    # Netze mit fester Eingabegroesse (Hoehe, Breite) rechnen das ganze Bild auf
    # einmal, verkleinert, statt in Kacheln
    feste_groesse: tuple[int, int] | None = None
    ausgabe_kanaele: int = 3
    # Weitere ONNX-Dateien (Datei, SHA-256), die zum Modell gehoeren - etwa der
    # Decoder neben dem Encoder
    weitere: tuple[tuple[str, str], ...] = ()


MODELLE = {
    "schnell": Modell(
        "schnell", "realesr-general-x4v3.onnx",
        "68b895a3b16ba12734c0c5c76e5293dbee13b3d86a6a2120c5bce1ac35507b56",
        (("realesr-general-x4v3.npz",
          "83000adde069d737a495f409607e4a3a404d7c0085d8267884d67fcb518e72bd"),
         ("realesr-general-wdn-x4v3.npz",
          "c8aaf96ed78d524e836a990d102e09cde7e045ab2c8aaf8acb0068528e7eeb05")),
        "S", {"S": 384, "M": 768, "L": 1024, "XL": 1536}),
    "qualitaet": Modell(
        "qualitaet", "realesrgan-x4plus.onnx",
        "9c887160648173a00ef2e998609bc198f0868a11e6801c9c3abeb1868808926a",
        None,
        "M", {"M": 256, "L": 512, "XL": 768}),
}
# Entrauschen: SCUNet fuer echtes Kamerarauschen, auf Treue trainiert (PSNR) -
# es erfindet keine Details. Die Swin-Fenster verlangen Vielfache von 64; die
# Ueberlappung ist groesser als beim Hochskalieren, weil SCUNet weiter schaut.
ENTRAUSCH_MODELLE = {
    "scunet": Modell(
        "scunet", "scunet-color-real-psnr.onnx",
        "cd446d36b78ca0c92d1c01052f7e3d331e5e35080cf597404ea9b8c9106bf157",
        None,
        "S", {"S": 256, "M": 512, "L": 768, "XL": 1024},
        lizenz="Apache-2.0", quelle=QUELLE_SCUNET,
        herkunft="SCUNet (Apache-2.0, Copyright 2022 Kai Zhang)",
        lizenzdatei="LICENSE-SCUNet.txt", release=MODELL_RELEASE_2,
        massstab=1, rand=32, vielfaches=64),
}
# Schaerfen: Restormer gegen Fokus-Unschaerfe, trainiert auf echte Fotos (DPDD) und
# auf Treue (L1). Das Modell gegen Verwacklung ist nur auf kuenstliche Bewegung aus
# Videobildern trainiert und hilft bei Fotos kaum - es fehlt deshalb bewusst.
SCHAERF_MODELLE = {
    "restormer": Modell(
        "restormer", "restormer-defocus.onnx",
        "8d51ab54539cc0383208a036380a2118df6551b693900380d7ed242032b009b9",
        None,
        "S", {"S": 256, "M": 384, "L": 448, "XL": 448},
        lizenz="MIT", quelle=QUELLE_RESTORMER,
        herkunft="Restormer (MIT, Copyright 2022 Syed Waqas Zamir)",
        lizenzdatei="LICENSE-Restormer.txt", release=MODELL_RELEASE_3,
        massstab=1, rand=32, vielfaches=8, nhwc=False, feste_kachel=True, cuda_fp16=False),
}
# Freistellen: BiRefNet in der leichten Fassung fuer 2560 x 1440 (Swin-T) - die Maske
# eines 24-MP-Fotos in gut einer Sekunde, mit feinen Kanten und Haaren. Es braucht
# rund 4 bis 6 GB Grafikspeicher, daher erst ab Stufe M. FP16 rechnet ONNX Runtime
# hier zehnmal langsamer (grid_sample), daher FP32.
MASKEN_MODELLE = {
    "birefnet": Modell(
        "birefnet", "birefnet-lite-2k.onnx",
        "c3c8c750ca533f691a12902f28d4712f0331900c32907f2f762352427a415a42",
        None,
        "M", {"M": 2560, "L": 2560, "XL": 2560},
        lizenz="MIT", quelle=QUELLE_BIREFNET,
        herkunft="BiRefNet (MIT, Copyright 2024 ZhengPeng)",
        lizenzdatei="LICENSE-BiRefNet.txt", release=MODELL_RELEASE_4,
        massstab=1, rand=0, vielfaches=32, nhwc=False, cuda_fp16=False,
        feste_groesse=(1440, 2560), ausgabe_kanaele=1),
}
# Auswahl per Klick: SAM 2.1 small. Der Encoder sieht das Bild einmal (1024 x 1024,
# rund 0,1 s), danach braucht jeder Klick nur noch den kleinen Decoder (unter 10 ms).
# In FP32: schnell genug, und die FP16-Wandlung des Encoders laedt ONNX Runtime nicht.
AUSWAHL_MODELLE = {
    "sam2": Modell(
        "sam2", "sam2.1-small-kodierer.onnx",
        "ec764cb857928d5b80ad740a65fbafb54867f8b446b72f02bbeb414bea91a0a8",
        None,
        "S", {"S": 1024, "M": 1024, "L": 1024, "XL": 1024},
        lizenz="Apache-2.0", quelle=QUELLE_SAM2,
        herkunft="SAM 2 (Apache-2.0, Copyright Meta Platforms, Inc. and affiliates)",
        lizenzdatei="LICENSE-SAM2.txt", release=MODELL_RELEASE_5,
        massstab=1, rand=0, vielfaches=32, nhwc=False, cuda_fp16=False,
        feste_groesse=(1024, 1024),
        weitere=(("sam2.1-small-dekodierer.onnx",
                  "aa6140c678916f505133f8a1cd45f7b84b18dcd2ec058249bdcc16ab6fdeb8fc"),)),
}
# Objekte entfernen: Big LaMa malt den markierten Bereich aus seiner Umgebung neu.
# Gerechnet wird ein Ausschnitt um die Markierung, verkleinert auf hoechstens die
# Kantenlaenge der Stufe. In FP32: in FP16 laeuft die Fourier-Faltung ueber.
ENTFERN_MODELLE = {
    "lama": Modell(
        "lama", "big-lama.onnx",
        "05242ecae18e453d4fc7cf7df015d80c9c1e96f82b7ef7376f2e0579b8444770",
        None,
        "S", {"S": 768, "M": 1024, "L": 1280, "XL": 1536},
        lizenz="Apache-2.0", quelle=QUELLE_LAMA,
        herkunft="LaMa (Apache-2.0, Copyright 2021 Samsung Research)",
        lizenzdatei="LICENSE-LaMa.txt", release=MODELL_RELEASE_6,
        massstab=1, rand=0, vielfaches=8, nhwc=False, cuda_fp16=False),
}
ALLE_MODELLE = {**MODELLE, **ENTRAUSCH_MODELLE, **SCHAERF_MODELLE, **MASKEN_MODELLE,
                **AUSWAHL_MODELLE, **ENTFERN_MODELLE}
STUFEN = ("S", "M", "L", "XL")


class KiFehler(Exception):
    """Modell fehlt, ist beschaedigt oder laesst sich nicht ausfuehren."""


def ordner_kandidaten() -> list[str]:
    """Wo Modelle liegen duerfen: neben dem Programm, sonst im Benutzerordner."""
    benutzer = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return [os.path.join(einstellungen.programm_ordner(), "modelle"),
            os.path.join(benutzer, "Neuro-Enhance", "modelle")]


def _beschreibbar(ordner: str) -> bool:
    try:
        os.makedirs(ordner, exist_ok=True)
        probe = os.path.join(ordner, ".schreibprobe")
        with open(probe, "w", encoding="ascii") as datei:
            datei.write("x")
        os.remove(probe)
        return True
    except OSError:
        return False


def modell_ordner() -> str:
    """Ordner fuer neue Modelle: der erste beschreibbare Kandidat."""
    for ordner in ordner_kandidaten():
        if _beschreibbar(ordner):
            return ordner
    return ordner_kandidaten()[-1]


def datei_pfad(name: str) -> str | None:
    for ordner in ordner_kandidaten():
        pfad = os.path.join(ordner, name)
        if os.path.isfile(pfad):
            return pfad
    return None


def dateien(modell: Modell) -> list[str]:
    """Alle Dateien, die ein Modell braucht - samt Lizenztext."""
    namen = [modell.datei] + ([n for n, _s in modell.mischung] if modell.mischung else [])
    return [*namen, *(n for n, _s in modell.weitere), modell.lizenzdatei]


def angeboten(modell: Modell, stufe: str) -> bool:
    return stufe in STUFEN and STUFEN.index(stufe) >= STUFEN.index(modell.mindeststufe)


def kachelgroesse(modell: Modell, stufe: str) -> int:
    if stufe in modell.kacheln:
        return modell.kacheln[stufe]
    return min(modell.kacheln.values())


def _sha256(pfad: str) -> str:
    h = hashlib.sha256()
    with open(pfad, "rb") as datei:
        for block in iter(lambda: datei.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


_GEPRUEFT: dict[str, float] = {}


def datei_pruefen(name: str, sha256: str) -> str:
    """Pfad der Modelldatei, wenn sie da ist und die Pruefsumme stimmt."""
    pfad = datei_pfad(name)
    if pfad is None:
        raise KiFehler(f"Modelldatei fehlt: {name}")
    zeit = os.path.getmtime(pfad)
    if _GEPRUEFT.get(pfad) != zeit:
        if _sha256(pfad) != sha256:
            raise KiFehler(f"Prüfsumme stimmt nicht: {pfad}")
        _GEPRUEFT[pfad] = zeit
    return pfad


def vorhanden(modell: Modell) -> bool:
    return all(datei_pfad(n) is not None for n in dateien(modell))


def fehlende(modell: Modell) -> list[str]:
    return [n for n in dateien(modell) if datei_pfad(n) is None]


def download_groesse(modell: Modell) -> int:
    return sum(DATEIEN[n][1] for n in fehlende(modell))


def herunterladen(modell: Modell, fortschritt=None, quelle: str | None = None) -> str:
    """Fehlende Dateien eines Modells laden und pruefen; Rueckgabe: Zielordner.

    Jede Datei wird zuerst unter `.teil` geschrieben und erst nach bestandener
    Pruefsumme umbenannt - ein abgebrochener oder verfaelschter Download landet
    nie dort, wo das Programm Modelle sucht. fortschritt(geladen, gesamt) wird
    nach jedem Block aufgerufen; gibt es False zurueck, wird abgebrochen.
    """
    quelle = modell.release if quelle is None else quelle
    ziel = modell_ordner()
    os.makedirs(ziel, exist_ok=True)
    namen = fehlende(modell)
    gesamt = sum(DATEIEN[n][1] for n in namen)
    geladen = 0
    for name in namen:
        sha256, groesse = DATEIEN[name]
        teil = os.path.join(ziel, name + ".teil")
        pruef = hashlib.sha256()
        try:
            anfrage = urllib.request.Request(quelle + name,
                                             headers={"User-Agent": "Neuro-Enhance"})
            with urllib.request.urlopen(anfrage, timeout=30) as antwort, \
                    open(teil, "wb") as datei:
                while True:
                    block = antwort.read(BLOCK)
                    if not block:
                        break
                    datei.write(block)
                    pruef.update(block)
                    geladen += len(block)
                    if fortschritt is not None and fortschritt(geladen, gesamt) is False:
                        raise KiAbbruch()
        except KiAbbruch:
            _entfernen(teil)
            raise
        except (urllib.error.URLError, OSError, TimeoutError) as fehler:
            _entfernen(teil)
            raise KiFehler(f"{name}: {fehler}") from fehler
        if pruef.hexdigest() != sha256:
            _entfernen(teil)
            raise KiFehler(f"{name}: Prüfsumme stimmt nicht - die Datei wurde verworfen.")
        os.replace(teil, os.path.join(ziel, name))
    return ziel


def _entfernen(pfad: str):
    try:
        os.remove(pfad)
    except OSError:
        pass


def _halbe_genauigkeit(pfad: str) -> bytes:
    """ONNX-Modell im Speicher nach FP16 wandeln; Ein- und Ausgaenge bleiben FP32.

    Als Gewichts-Eingaenge uebergebene Gewichte (Mischung) bleiben ebenfalls
    FP32 - das Netz wandelt sie selbst, das kostet je Kachel nur Mikrosekunden.
    """
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    try:
        modell = convert_float_to_float16(onnx.load(pfad), keep_io_types=True)
    except Exception as fehler:                  # beschaedigtes oder unerwartetes Modell
        raise KiFehler(f"Modell laesst sich nicht nach FP16 wandeln: {fehler}") from fehler
    return modell.SerializeToString()


TENSORRT_PAKET = "tensorrt-cu12-libs"
_tensorrt_geladen = False
_cache_ordner: str | None = None          # im Bauprozess vom Hauptprozess vorgegeben


def tensorrt_ordner() -> str | None:
    """Ordner mit den TensorRT-Bibliotheken aus dem pip-Paket - oder None."""
    try:
        spec = importlib.util.find_spec("tensorrt_libs")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    ordner = list(spec.submodule_search_locations)[0]
    try:
        namen = os.listdir(ordner)
    except OSError:
        return None
    if not any(n.startswith(("nvinfer_10", "libnvinfer.so.10")) for n in namen):
        return None
    return ordner


def tensorrt_fassung() -> str | None:
    """Fassung des installierten TensorRT - oder None, wenn es fehlt."""
    if tensorrt_ordner() is None:
        return None
    try:
        return importlib.metadata.version(TENSORRT_PAKET)
    except importlib.metadata.PackageNotFoundError:
        return "?"


def _tensorrt_laden() -> bool:
    """TensorRT-Bibliotheken auffindbar machen, bevor ONNX Runtime sie sucht."""
    global _tensorrt_geladen
    if _tensorrt_geladen:
        return True
    ordner = tensorrt_ordner()
    if ordner is None:
        return False
    if sys.platform == "win32":
        # Nur den Suchpfad erweitern - das Paket selbst wuerde beim Import alle
        # Bibliotheken samt der 1,7 GB Builder-Ressourcen laden.
        os.add_dll_directory(ordner)
        os.environ["PATH"] = ordner + os.pathsep + os.environ.get("PATH", "")
    else:
        import tensorrt_libs  # noqa: F401  laedt libnvinfer & Co. global
    _tensorrt_geladen = True
    return True


def tensorrt_cache() -> str:
    return _cache_ordner or os.path.join(modell_ordner(), "tensorrt")


def eingabe_groesse(modell: Modell, kachel: int) -> int:
    """Groesste Kachel samt Ueberlappung, aufgerundet auf das noetige Vielfache."""
    roh = kachel + 2 * modell.rand
    return -(-roh // modell.vielfaches) * modell.vielfaches


def tensorrt_marke(modell: Modell, kachel: int, fp16: bool = True) -> str:
    """Datei, die anzeigt, dass die Engine fuer diese Kombination schon gebaut ist -
    auch fuer genau diese Modelldatei, denn TensorRT baut fuer eine neue neu."""
    return os.path.join(tensorrt_cache(), "{}-{}-{}-{}-sm{}-trt{}.fertig".format(
        modell.schluessel, modell.sha256[:8], "fp16" if fp16 else "fp32",
        eingabe_groesse(modell, kachel),
        cp.cuda.Device().compute_capability, tensorrt_fassung()))


def tensorrt_baut(modell: Modell, kachel: int, fp16: bool = True) -> bool:
    """Muss TensorRT beim Laden dieses Modells erst eine Engine bauen (Minuten)?"""
    return tensorrt_ordner() is not None and not os.path.exists(
        tensorrt_marke(modell, kachel, fp16))


def tensorrt_vorbereiten(modell: Modell, kachel: int, fp16: bool = True):
    """Engine in einem eigenen Prozess bauen; gibt ein Future zurueck.

    Danach laedt Hochskalierer sie in einer halben Sekunde aus dem Zwischenspeicher.
    Scheitert der Bau, wirft future.result() den Fehler.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    ausfuehrer = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
    zukunft = ausfuehrer.submit(_engine_bauen, modell.schluessel, kachel, fp16, tensorrt_cache())
    ausfuehrer.shutdown(wait=False)               # der Auftrag laeuft weiter
    return zukunft


def _engine_bauen(schluessel: str, kachel: int, fp16: bool, cache: str) -> None:
    """Laeuft im Bauprozess."""
    global _cache_ordner
    _cache_ordner = cache
    netz = _Netz(ALLE_MODELLE[schluessel], fp16, kachel, tensorrt=True)
    if not netz.tensorrt:
        raise KiFehler(netz.tensorrt_fehler)


def ausgabe_form(form, faktor: int) -> tuple[int, int]:
    return form[0] * faktor, form[1] * faktor


def _speichergrenze() -> int:
    """Hoechstens so viel Grafikspeicher fuer ONNX Runtime, wie gerade frei ist.

    Ohne Grenze waechst sein Speicherpool ueber den Grafikspeicher hinaus, und
    Windows lagert in den Arbeitsspeicher aus - dann rechnet ein Netz zwanzigmal
    langsamer, statt mit einem Speicherfehler abzubrechen (auf den die Kacheln
    reagieren). Was CuPy in seinem Pool bereithaelt, ist ebenfalls frei.
    """
    pool = cp.get_default_memory_pool()
    frei = cp.cuda.runtime.memGetInfo()[0] + pool.free_bytes()
    return max(1 << 30, frei - (512 << 20))


class _Netz:
    """Ein geladenes Modell mit CUDA (FP16, NHWC) oder TensorRT - teuer anzulegen.

    kachel ist die groesste Kachel in Eingabepixeln. Nur mit ihr kann TensorRT
    rechnen, denn die Engine wird fuer eine feste Hoechstgroesse gebaut;
    tensorrt=None nimmt TensorRT, sobald es installiert ist.
    """

    def __init__(self, modell: Modell, fp16: bool = True, kachel: int | None = None,
                 tensorrt: bool | None = None):
        import onnxruntime as ort
        self._ort = ort
        ort.set_default_logger_severity(3)       # nur Fehler, keine Hinweise auf der Konsole
        # CUDA- und cuDNN-Bibliotheken aus den pip-Paketen von NVIDIA laden
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(cuda=True, cudnn=True, msvc=False)
        self.modell = modell
        self.fp16 = fp16
        self.kachel = kachel
        self._pfad = datei_pruefen(modell.datei, modell.sha256)
        self._gewichte: dict = {}
        if tensorrt is None:
            tensorrt = kachel is not None and tensorrt_ordner() is not None
        self.tensorrt = False
        self.tensorrt_fehler = ""
        self.erster_lauf = False                  # TensorRT hat beim Laden die Engine gebaut
        if tensorrt:
            if kachel is None:
                raise ValueError("TensorRT braucht die Kachelgroesse")
            try:
                self.sitzung = self._tensorrt_sitzung()
                self.tensorrt = True
            except Exception as fehler:           # dann eben mit CUDA
                self.tensorrt_fehler = str(fehler)
                self.erster_lauf = False
        if not self.tensorrt:
            self.sitzung = self._cuda_sitzung()

    @property
    def beschleuniger(self) -> str:
        return "TensorRT" if self.tensorrt else "CUDA"

    def _optionen(self):
        optionen = self._ort.SessionOptions()
        optionen.log_severity_level = 3
        return optionen

    def _cuda_sitzung(self, pfad: str | None = None):
        halb = self.fp16 and self.modell.cuda_fp16
        pfad = pfad or self._pfad
        netz = _halbe_genauigkeit(pfad) if halb else pfad
        anbieter = [("CUDAExecutionProvider", {"device_id": 0,
                                               "cudnn_conv_algo_search": "HEURISTIC",
                                               "prefer_nhwc": "1" if halb and self.modell.nhwc
                                               else "0",
                                               "arena_extend_strategy": "kSameAsRequested",
                                               "gpu_mem_limit": str(_speichergrenze())})]
        try:
            sitzung = self._ort.InferenceSession(netz, self._optionen(), providers=anbieter)
        except Exception as fehler:               # ORT wirft eigene Fehlerklassen
            raise KiFehler(str(fehler)) from fehler
        if "CUDAExecutionProvider" not in sitzung.get_providers():
            raise KiFehler("ONNX Runtime kann die Grafikkarte nicht nutzen (CUDA).")
        return sitzung

    def _tensorrt_sitzung(self):
        """Sitzung ueber TensorRT. FP16 waehlt TensorRT selbst, das Netz bleibt FP32."""
        if not _tensorrt_laden():
            raise KiFehler("TensorRT ist nicht installiert.")
        if "TensorrtExecutionProvider" not in self._ort.get_available_providers():
            raise KiFehler("Diese ONNX-Runtime-Fassung kennt TensorRT nicht.")
        cache = tensorrt_cache()
        os.makedirs(cache, exist_ok=True)
        groesse = eingabe_groesse(self.modell, self.kachel)
        klein = self.modell.vielfaches
        self._marke = tensorrt_marke(self.modell, self.kachel, self.fp16)
        self.erster_lauf = not os.path.exists(self._marke)
        trt = {"device_id": 0,
               "trt_fp16_enable": self.fp16,
               "trt_engine_cache_enable": True,
               "trt_engine_cache_path": cache,
               "trt_timing_cache_enable": True,
               "trt_timing_cache_path": cache,
               # Randkacheln sind kleiner, groesser als die volle Kachel wird keine
               "trt_profile_min_shapes": f"eingabe:1x3x{klein}x{klein}",
               "trt_profile_opt_shapes": f"eingabe:1x3x{groesse}x{groesse}",
               "trt_profile_max_shapes": f"eingabe:1x3x{groesse}x{groesse}"}
        if self.modell.feste_groesse:
            form = "eingabe:1x3x{}x{}".format(*self.modell.feste_groesse)
            for art in ("min", "opt", "max"):
                trt[f"trt_profile_{art}_shapes"] = form
        sitzung = self._ort.InferenceSession(
            self._pfad, self._optionen(),
            providers=[("TensorrtExecutionProvider", trt),
                       ("CUDAExecutionProvider", {"device_id": 0})])
        if "TensorrtExecutionProvider" not in sitzung.get_providers():
            raise KiFehler("ONNX Runtime konnte TensorRT nicht laden.")
        # Die Engine entsteht schon beim Anlegen der Sitzung, nicht erst beim Rechnen
        with open(self._marke, "w", encoding="utf-8"):
            pass
        return sitzung

    def _auf_cuda_wechseln(self, fehler: Exception):
        self.tensorrt_fehler = str(fehler)
        self.tensorrt = False
        self.erster_lauf = False
        self.sitzung = None                       # TensorRT-Speicher zuerst freigeben
        self.sitzung = self._cuda_sitzung()

    def _kachel(self, eingabe):
        """Eine Kachel (1, 3, h, w) float32 auf der GPU -> (1, 3, m*h, m*w)."""
        _n, _k, h, w = eingabe.shape
        m = self.modell.massstab
        ausgabe = cp.empty((1, self.modell.ausgabe_kanaele, m * h, m * w), dtype=cp.float32)
        bindung = self.sitzung.io_binding()
        bindung.bind_input("eingabe", "cuda", 0, np.float32, list(eingabe.shape),
                           eingabe.data.ptr)
        for name, gewicht in self._gewichte.items():
            bindung.bind_input(name, "cuda", 0, np.float32, list(gewicht.shape), gewicht.data.ptr)
        bindung.bind_output("ausgabe", "cuda", 0, np.float32, list(ausgabe.shape),
                            ausgabe.data.ptr)
        cp.cuda.Device().synchronize()             # CuPy hat fertig geschrieben
        self.sitzung.run_with_iobinding(bindung)
        return ausgabe

    def _kacheln(self, kanaele, kachel: int, fortschritt, ablegen, ganz: bool = False):
        """Alle Kacheln rechnen; ablegen(y0, y1, x0, x1, ergebnis) erhaelt den Kern
        jeder Kachel (3, m*h, m*w) ohne Ueberlappung - mit ganz=True die ganze Kachel
        samt Ueberlappung, zum Ueberblenden."""
        _n, _k, hoehe, breite = kanaele.shape
        m, rand, vielfaches = self.modell.massstab, self.modell.rand, self.modell.vielfaches
        zeilen = range(0, hoehe, kachel)
        spalten = range(0, breite, kachel)
        anzahl, nummer = len(zeilen) * len(spalten), 0
        fest = eingabe_groesse(self.modell, kachel) if self.modell.feste_kachel else 0
        for y0 in zeilen:
            for x0 in spalten:
                y1, x1 = min(y0 + kachel, hoehe), min(x0 + kachel, breite)
                ya, xa = max(y0 - rand, 0), max(x0 - rand, 0)
                ye, xe = min(y1 + rand, hoehe), min(x1 + rand, breite)
                if fest:                           # Randkacheln ruecken ins Bild hinein
                    ya, xa = max(min(ya, hoehe - fest), 0), max(min(xa, breite - fest), 0)
                    ye, xe = min(ya + fest, hoehe), min(xa + fest, breite)
                teil = kanaele[:, :, ya:ye, xa:xe]
                fh = -(ye - ya) % vielfaches
                fb = -(xe - xa) % vielfaches
                if fh or fb:                       # am Bildrand auffuellen wie im Original
                    teil = cp.pad(teil, ((0, 0), (0, 0), (0, fh), (0, fb)), mode="edge")
                ergebnis = self._kachel(cp.ascontiguousarray(teil))[0]
                if ganz:
                    ablegen(ya, ye, xa, xe, ergebnis[:, :m * (ye - ya), :m * (xe - xa)])
                else:
                    ablegen(y0, y1, x0, x1, ergebnis[:, m * (y0 - ya):m * (y1 - ya),
                                                     m * (x0 - xa):m * (x1 - xa)])
                nummer += 1
                if fortschritt is not None and fortschritt(nummer, anzahl) is False:
                    raise KiAbbruch()

    def _mit_wiederholung(self, arbeit, kachel: int):
        """arbeit(kachel) ausfuehren; bei Speichermangel mit halber Kachel noch einmal,
        scheitert TensorRT, mit CUDA."""
        if self.tensorrt:
            kachel = min(kachel, self.kachel)    # groesser kann die Engine nicht
        while True:
            try:
                return arbeit(kachel)
            except cp.cuda.memory.OutOfMemoryError:
                pass
            except KiAbbruch:
                raise
            except Exception as fehler:          # ORT meldet Speichermangel als eigenen Fehler
                if "memory" not in str(fehler).lower() and "alloc" not in str(fehler).lower():
                    if not self.tensorrt:
                        raise KiFehler(str(fehler)) from fehler
                    self._auf_cuda_wechseln(fehler)  # und mit CUDA noch einmal
                    continue
            cp.get_default_memory_pool().free_all_blocks()
            if kachel <= 64:
                raise KiFehler("Zu wenig Grafikspeicher - auch mit kleinsten Kacheln.")
            kachel = max(64, kachel // 2 // self.modell.vielfaches * self.modell.vielfaches)


class Hochskalierer(_Netz):
    """Real-ESRGAN - je Modell und Entrauschstaerke einmal anzulegen."""

    def __init__(self, modell: Modell, entrauschen: float = 0.5, fp16: bool = True,
                 kachel: int | None = None, tensorrt: bool | None = None):
        super().__init__(modell, fp16, kachel, tensorrt)
        self.entrauschen = entrauschen
        if modell.mischung:
            self._gewichte = self._mischen(entrauschen)

    def _mischen(self, staerke: float) -> dict:
        """Beide Gewichtssaetze mischen (staerke 1 = volles, 0 = schwaches Entrauschen).

        Wie in Real-ESRGAN werden die Gewichte selbst linear gemischt, nicht die
        Bilder - so bleibt es bei einem Durchlauf. Die Mischung liegt danach im
        Grafikspeicher und wird bei jeder Kachel nur noch verknuepft.
        """
        (stark_name, stark_sha), (schwach_name, schwach_sha) = self.modell.mischung
        stark = np.load(datei_pruefen(stark_name, stark_sha))
        schwach = np.load(datei_pruefen(schwach_name, schwach_sha))
        return {name: cp.ascontiguousarray(cp.asarray(
                    staerke * stark[name] + (1 - staerke) * schwach[name], dtype=cp.float32))
                for name in stark.files}

    def hochskalieren(self, srgb, faktor: int, kachel: int, bits: int = 8,
                      fortschritt=None) -> np.ndarray:
        """sRGB-Bild (H, W, 3) float32 0..1 auf der GPU -> numpy uint8/uint16 (fH, fW, 3).

        faktor 2 rechnet mit 4 und mittelt danach je 2 x 2 Pixel. fortschritt(i, n)
        wird nach jeder Kachel aufgerufen; gibt es False zurueck, wird abgebrochen.
        """
        if faktor not in (2, 4):
            raise ValueError("faktor muss 2 oder 4 sein")
        hoehe, breite = srgb.shape[:2]
        typ, hoechst = (np.uint16, 65535) if bits == 16 else (np.uint8, 255)
        ziel = np.empty((hoehe * faktor, breite * faktor, 3), dtype=typ)
        kanaele = cp.ascontiguousarray(cp.moveaxis(srgb.astype(cp.float32), -1, 0))[None]

        def ablegen(y0, y1, x0, x1, gross):
            if faktor == 2:
                c, h, w = gross.shape
                gross = gross.reshape(c, h // 2, 2, w // 2, 2).mean(axis=(2, 4))
            werte = (cp.clip(gross, 0, 1) * hoechst + 0.5).astype(typ)
            ziel[y0 * faktor:y1 * faktor, x0 * faktor:x1 * faktor] = \
                cp.asnumpy(cp.moveaxis(werte, 0, -1))

        self._mit_wiederholung(lambda k: self._kacheln(kanaele, k, fortschritt, ablegen), kachel)
        return ziel


class _Bildnetz(_Netz):
    """Netz, das ein sRGB-Bild in gleicher Groesse zurueckgibt; alles bleibt auf der GPU."""

    STANDARD = ""

    def __init__(self, modell: Modell | None = None, fp16: bool = True,
                 kachel: int | None = None, tensorrt: bool | None = None):
        super().__init__(modell or ALLE_MODELLE[self.STANDARD], fp16, kachel, tensorrt)

    def rechnen(self, srgb, kachel: int, fortschritt=None):
        """sRGB (H, W, 3) float32 0..1 auf der GPU -> Ergebnis, gleiche Form, auf der GPU.

        Die Kacheln werden in ihrer Ueberlappung weich ineinander geblendet, nicht
        hart aneinandergesetzt: Netze, die ueber die ganze Kachel schauen (SCUNet,
        Restormer), liefern in benachbarten Kacheln leicht verschiedene Helligkeit -
        eine harte Grenze waere als Naht zu sehen.
        """
        hoehe, breite = srgb.shape[:2]
        kanaele = cp.ascontiguousarray(cp.moveaxis(srgb.astype(cp.float32), -1, 0))[None]
        rampe = 2 * self.modell.rand

        def gewicht(anfang, ende, laenge):
            stelle = cp.arange(ende - anfang, dtype=cp.float32)
            w = cp.ones(ende - anfang, dtype=cp.float32)
            if anfang > 0:
                w = cp.minimum(w, (stelle + 0.5) / rampe)
            if ende < laenge:
                w = cp.minimum(w, (ende - anfang - stelle - 0.5) / rampe)
            return w

        def arbeit(k):
            summe = cp.zeros((3, hoehe, breite), dtype=cp.float32)
            gewichte = cp.zeros((hoehe, breite), dtype=cp.float32)

            def ablegen(ya, ye, xa, xe, teil):
                w = gewicht(ya, ye, hoehe)[:, None] * gewicht(xa, xe, breite)[None, :]
                summe[:, ya:ye, xa:xe] += teil * w
                gewichte[ya:ye, xa:xe] += w

            self._kacheln(kanaele, k, fortschritt, ablegen, ganz=True)
            summe /= gewichte
            return summe

        ziel = self._mit_wiederholung(arbeit, kachel)
        return cp.ascontiguousarray(cp.moveaxis(cp.clip(ziel, 0, 1), 0, -1))


class Entrauscher(_Bildnetz):
    """SCUNet - entrauscht ein sRGB-Bild in voller Groesse."""

    STANDARD = "scunet"


class Schaerfer(_Bildnetz):
    """Restormer - nimmt Fokus-Unschaerfe aus einem sRGB-Bild in voller Groesse."""

    STANDARD = "restormer"


class KiAbbruch(Exception):
    """Der Anwender hat eine KI-Berechnung oder einen Download abgebrochen."""


class Freisteller(_Netz):
    """BiRefNet - die Maske des Motivs (1 = Motiv, 0 = Hintergrund) fuer ein ganzes Bild.

    Das Netz kennt nur 2560 x 1440 Pixel im Querformat. Das Bild wird darauf
    verkleinert - Hochformate vorher um 90 Grad gedreht, damit sie nicht gestaucht
    werden - und die Maske danach auf die volle Groesse zurueck vergroessert.
    """

    def __init__(self, modell: Modell | None = None, fp16: bool = True,
                 tensorrt: bool | None = None):
        modell = modell or MASKEN_MODELLE["birefnet"]
        super().__init__(modell, fp16, modell.kacheln[modell.mindeststufe], tensorrt)

    def maske(self, srgb):
        """sRGB (H, W, 3) float32 0..1 auf der GPU -> Maske (H, W) float32 0..1 auf der GPU."""
        hoehe, breite = srgb.shape[:2]
        hochformat = hoehe > breite
        bild = cp.rot90(srgb) if hochformat else srgb
        mh, mb = self.modell.feste_groesse
        faktor = max(1, min(bild.shape[0] // mh, bild.shape[1] // mb))
        klein = filter.vergroessern(filter.verkleinern_box(bild, faktor).astype(cp.float32),
                                    mh, mb)
        eingabe = cp.ascontiguousarray(cp.moveaxis(klein, -1, 0))[None]
        del klein
        try:
            ergebnis = self._kachel(eingabe)
        except cp.cuda.memory.OutOfMemoryError as fehler:
            raise KiFehler("Zu wenig Grafikspeicher fuer die Maske.") from fehler
        except Exception as fehler:              # ORT wirft eigene Fehlerklassen
            if not self.tensorrt:
                raise KiFehler(str(fehler)) from fehler
            self._auf_cuda_wechseln(fehler)      # und mit CUDA noch einmal
            ergebnis = self._kachel(eingabe)
        maske = filter.vergroessern(ergebnis[0, 0], bild.shape[0], bild.shape[1])
        if hochformat:
            maske = cp.rot90(maske, -1)
        return cp.ascontiguousarray(cp.clip(maske, 0, 1))


def _binden(sitzung, eingaben: dict, ausgaben: dict):
    """Netz mit CuPy-Arrays (float32) als Ein- und Ausgaengen rechnen, alles auf der GPU."""
    bindung = sitzung.io_binding()
    for name, wert in eingaben.items():
        bindung.bind_input(name, "cuda", 0, np.float32, list(wert.shape), wert.data.ptr)
    for name, wert in ausgaben.items():
        bindung.bind_output(name, "cuda", 0, np.float32, list(wert.shape), wert.data.ptr)
    cp.cuda.Device().synchronize()                 # CuPy hat fertig geschrieben
    sitzung.run_with_iobinding(bindung)


class Auswaehler(_Netz):
    """SAM 2 - waehlt per Klick ein Objekt aus; die Maske (1 = Objekt) in voller Groesse.

    bild_setzen rechnet den Encoder einmal je Bild, danach liefert roh() zu jeder
    Liste von Klicks in Millisekunden die passende Maske. SAM sieht das Bild
    gestaucht auf 1024 x 1024 Pixel und gibt eine Maske mit 256 x 256 Punkten
    zurueck. Ihre Kante legt maske() mit einem gefuehrten Filter an die Kanten
    des Bildes, bevor sie auf die volle Groesse kommt.
    """

    ARBEIT = 2048                    # laengste Kante, auf der die Kante verfeinert wird
    STABIL = 0.98                    # wie SAM2ImagePredictor: Grenze der Stabilitaet

    def __init__(self, modell: Modell | None = None):
        modell = modell or AUSWAHL_MODELLE["sam2"]
        super().__init__(modell, fp16=False, kachel=None, tensorrt=False)
        (name, sha256), = modell.weitere
        self.dekodierer = self._cuda_sitzung(datei_pruefen(name, sha256))
        self.form: tuple[int, int] | None = None
        self._merkmale = None
        self._fuehrung = None

    def bild_setzen(self, srgb):
        """sRGB (H, W, 3) float32 0..1 auf der GPU einmal durch den Encoder schicken."""
        hoehe, breite = srgb.shape[:2]
        groesse = self.modell.feste_groesse[0]
        faktor = max(1, min(hoehe, breite) // groesse)
        klein = filter.vergroessern(filter.verkleinern_box(srgb, faktor).astype(cp.float32),
                                    groesse, groesse)
        eingabe = cp.ascontiguousarray(cp.moveaxis(klein, -1, 0))[None]
        del klein
        merkmale = {"merkmale": cp.empty((1, 256, 64, 64), dtype=cp.float32),
                    "s0": cp.empty((1, 32, 256, 256), dtype=cp.float32),
                    "s1": cp.empty((1, 64, 128, 128), dtype=cp.float32)}
        try:
            _binden(self.sitzung, {"eingabe": eingabe}, merkmale)
        except Exception as fehler:              # ORT wirft eigene Fehlerklassen
            raise KiFehler(str(fehler)) from fehler
        faktor = max(1, -(-max(hoehe, breite) // self.ARBEIT))
        self._fuehrung = filter.luminanz(filter.verkleinern_box(srgb, faktor)).astype(cp.float32)
        self._merkmale = merkmale
        self.form = (hoehe, breite)

    def roh(self, klicks, vorige=None):
        """Klicks [(x, y, dazu), ...] in Pixeln des Bildes -> Logits (256, 256) der besten
        Maske; vorige: Logits der letzten Maske, die SAM als Hinweis bekommt."""
        hoehe, breite = self.form
        groesse = self.modell.feste_groesse[0]
        punkte = cp.asarray([[[x / breite * groesse - 0.5, y / hoehe * groesse - 0.5]
                              for x, y, _dazu in klicks]], dtype=cp.float32)
        etiketten = cp.asarray([[1.0 if dazu else 0.0 for _x, _y, dazu in klicks]],
                               dtype=cp.float32)
        mit = vorige is not None
        maske = vorige if mit else cp.zeros((256, 256), dtype=cp.float32)
        eingaben = {**self._merkmale, "punkte": punkte, "etiketten": etiketten,
                    "maske": cp.ascontiguousarray(maske[None, None], dtype=cp.float32),
                    "mit_maske": cp.asarray([1.0 if mit else 0.0], dtype=cp.float32)}
        masken = cp.empty((1, 4, 256, 256), dtype=cp.float32)
        guete = cp.empty((1, 4), dtype=cp.float32)
        try:
            _binden(self.dekodierer, eingaben, {"masken": masken, "guete": guete})
        except Exception as fehler:
            raise KiFehler(str(fehler)) from fehler
        return masken[0, self._waehlen(masken[0], guete[0], len(klicks))]

    def _waehlen(self, masken, guete, anzahl: int) -> int:
        """Welche der vier Masken - wie SAM2ImagePredictor.

        Ein einzelner Klick ist mehrdeutig (Knopf, Hemd oder ganze Person?): dann
        die der drei Deutungen mit der hoechsten geschaetzten Guete. Bei mehreren
        Klicks die eigene Maske dafuer, ausser sie ist instabil - aendert sich ihre
        Flaeche schon bei einer kleinen Verschiebung der Schwelle merklich.
        """
        beste = 1 + int(cp.argmax(guete[1:]))
        if anzahl == 1:
            return beste
        innen = int((masken[0] > 0.05).sum())
        aussen = int((masken[0] > -0.05).sum())
        stabil = aussen == 0 or innen / aussen >= self.STABIL
        return 0 if stabil else beste

    def maske(self, logits):
        """Logits (256, 256) -> Maske (H, W) float32 0..1 auf der GPU, Kante am Bild."""
        hoehe, breite = self.form
        fh, fb = self._fuehrung.shape
        # Wie SAM selbst an der Schwelle 0 entscheiden: Unsicheres (oft Haar) halb
        # zu nehmen hiesse, die Regler wirkten dort halb
        hart = (filter.vergroessern(logits, fh, fb) > 0).astype(cp.float32)
        # Ein Punkt der SAM-Maske deckt einige Pixel der Arbeitsgroesse; so weit darf
        # die Kante wandern, um sich an eine Kante im Bild zu legen
        radius = max(2, round(max(fh, fb) / 256))
        fein = filter.gefuehrter_filter(self._fuehrung, hart, radius, 1e-3)
        return cp.ascontiguousarray(cp.clip(filter.vergroessern(fein, hoehe, breite), 0, 1))


class Entferner(_Netz):
    """LaMa - fuellt einen markierten Bereich mit dem, was dahinter liegen koennte.

    Gerechnet wird nur ein Ausschnitt: die Markierung mit einem Rand, der so breit
    ist wie ihre halbe Ausdehnung - genug Umgebung, aus der LaMa die Struktur
    fortsetzt. Ist der Ausschnitt groesser als `kante`, wird er dafuer verkleinert
    und die Fuellung danach wieder vergroessert. Die Markierung waechst vorher um
    einige Pixel, damit auch der Saum des Objekts (Schatten, Kantenlicht)
    verschwindet; eingesetzt wird mit weicher Kante.
    """

    # Die Markierung waechst um diesen Anteil ihrer Ausdehnung, mindestens um einige
    # Pixel der Arbeitsgroesse: Unscharfe Objekte strahlen ueber ihre Kante hinaus, und
    # bleibt ein Rest ihres Umrisses stehen, setzt LaMa ihn fort
    WACHSEN = 0.05
    WACHSEN_MIN = 6

    def __init__(self, modell: Modell | None = None, kante: int | None = None):
        modell = modell or ENTFERN_MODELLE["lama"]
        super().__init__(modell, fp16=False, kachel=None, tensorrt=False)
        self.kante = kante or modell.kacheln[modell.mindeststufe]

    def fuellen(self, srgb, maske):
        """sRGB (H, W, 3) und Markierung (H, W) 0..1 auf der GPU ->
        (y0, y1, x0, x1, Fuellung (h, w, 3) sRGB, Deckkraft (h, w)) oder None ohne
        Markierung. Ausserhalb des Ausschnitts bleibt das Bild, wie es ist."""
        hoehe, breite = srgb.shape[:2]
        markiert = maske > 0.5
        zeilen = cp.flatnonzero(markiert.any(axis=1))
        spalten = cp.flatnonzero(markiert.any(axis=0))
        if zeilen.size == 0:
            return None
        ya, ye = int(zeilen[0]), int(zeilen[-1]) + 1
        xa, xe = int(spalten[0]), int(spalten[-1]) + 1
        rand = max(ye - ya, xe - xa) * 2 // 3 + 48
        y0, y1 = max(0, ya - rand), min(hoehe, ye + rand)
        x0, x1 = max(0, xa - rand), min(breite, xe + rand)
        kante = self.kante
        while True:
            try:
                return (y0, y1, x0, x1, *self._ausschnitt(srgb[y0:y1, x0:x1],
                                                          markiert[y0:y1, x0:x1], kante))
            except Exception as fehler:          # ORT meldet Speichermangel als eigenen Fehler
                text = str(fehler).lower()
                speicher = isinstance(fehler, cp.cuda.memory.OutOfMemoryError) or \
                    "memory" in text or "alloc" in text
                if not speicher:
                    raise KiFehler(str(fehler)) from fehler
                cp.get_default_memory_pool().free_all_blocks()
                if kante <= 256:
                    raise KiFehler("Zu wenig Grafikspeicher zum Entfernen.") from fehler
                kante //= 2

    def _ausschnitt(self, bild, markiert, kante: int):
        from cupyx.scipy import ndimage
        h, w = bild.shape[:2]
        faktor = max(1, -(-max(h, w) // kante))
        # Auf ein Vielfaches des Faktors auffuellen, damit am Rand nichts wegfaellt
        fh, fw = -h % faktor, -w % faktor
        bild = cp.pad(bild, ((0, fh), (0, fw), (0, 0)), mode="edge")
        klein = filter.verkleinern_box(bild, faktor).astype(cp.float32)
        loch = filter.verkleinern_box(cp.pad(markiert, ((0, fh), (0, fw))).astype(cp.float32),
                                      faktor) > 0
        kh, kw = loch.shape
        zeilen, spalten = cp.flatnonzero(loch.any(axis=1)), cp.flatnonzero(loch.any(axis=0))
        ausdehnung = max(int(zeilen[-1] - zeilen[0]), int(spalten[-1] - spalten[0])) + 1
        weite = max(self.WACHSEN_MIN, round(self.WACHSEN * ausdehnung))
        loch = ndimage.maximum_filter(loch.astype(cp.float32), size=2 * weite + 1,
                                      mode="constant")
        v = self.modell.vielfaches
        ph, pw = -kh % v, -kw % v
        eingabe = cp.pad(klein, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
        lochrand = cp.pad(loch, ((0, ph), (0, pw)), mode="symmetric")
        eingabe = cp.ascontiguousarray(cp.moveaxis(eingabe, -1, 0))[None]
        lochrand = cp.ascontiguousarray(lochrand)[None, None]
        ausgabe = cp.empty_like(eingabe)
        _binden(self.sitzung, {"eingabe": eingabe, "maske": lochrand}, {"ausgabe": ausgabe})
        fuellung = cp.moveaxis(ausgabe[0, :, :kh, :kw], 0, -1)
        # Deckkraft: die gewachsene Markierung mit weicher Kante, auf voller Groesse
        weich = filter.gauss(loch, 1.0)
        deckkraft = cp.clip(filter.vergroessern(weich, h + fh, w + fw)[:h, :w], 0, 1)
        deckkraft = cp.maximum(deckkraft, markiert.astype(cp.float32))
        fuellung = filter.vergroessern(cp.ascontiguousarray(fuellung), h + fh, w + fw)[:h, :w]
        return cp.clip(fuellung, 0, 1), cp.ascontiguousarray(deckkraft)
