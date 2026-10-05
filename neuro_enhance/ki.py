"""
KI-Hochskalieren (Real-ESRGAN) und KI-Entrauschen (SCUNet) ueber ONNX Runtime

Die Modelle stammen aus den offiziellen Releases von Real-ESRGAN (BSD-3-Clause)
und SCUNet (Apache-2.0) und sind mit werkzeuge/modelle_exportieren.py nach ONNX
gewandelt. Bereit liegen sie als Dateien eigener Releases dieses Projekts, mit
dem jeweiligen Lizenztext daneben. Heruntergeladen wird nur auf Wunsch
des Anwenders; jede Datei wird gegen ihre SHA-256-Pruefsumme geprueft, bevor
sie an ihren Platz kommt und bevor sie geladen wird.

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

from . import einstellungen
from .cuda import cupy as cp

QUELLE = "https://github.com/xinntao/Real-ESRGAN"
QUELLE_SCUNET = "https://github.com/cszn/SCUNet"
MODELL_RELEASE = "https://github.com/DerAlexmann/Neuro-Enhance/releases/download/modelle-1/"
MODELL_RELEASE_2 = "https://github.com/DerAlexmann/Neuro-Enhance/releases/download/modelle-2/"
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
ALLE_MODELLE = {**MODELLE, **ENTRAUSCH_MODELLE}
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
    return [*namen, modell.lizenzdatei]


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
    """Datei, die anzeigt, dass die Engine fuer diese Kombination schon gebaut ist."""
    return os.path.join(tensorrt_cache(), "{}-{}-{}-sm{}-trt{}.fertig".format(
        modell.schluessel, "fp16" if fp16 else "fp32", eingabe_groesse(modell, kachel),
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

    def _cuda_sitzung(self):
        netz = _halbe_genauigkeit(self._pfad) if self.fp16 else self._pfad
        anbieter = [("CUDAExecutionProvider", {"device_id": 0,
                                               "cudnn_conv_algo_search": "HEURISTIC",
                                               "prefer_nhwc": "1" if self.fp16 else "0"})]
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
        ausgabe = cp.empty((1, 3, m * h, m * w), dtype=cp.float32)
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

    def _kacheln(self, kanaele, kachel: int, fortschritt, ablegen):
        """Alle Kacheln rechnen; ablegen(y0, y1, x0, x1, ergebnis) erhaelt den Kern
        jeder Kachel (3, m*h, m*w) ohne Ueberlappung."""
        _n, _k, hoehe, breite = kanaele.shape
        m, rand, vielfaches = self.modell.massstab, self.modell.rand, self.modell.vielfaches
        zeilen = range(0, hoehe, kachel)
        spalten = range(0, breite, kachel)
        anzahl, nummer = len(zeilen) * len(spalten), 0
        for y0 in zeilen:
            for x0 in spalten:
                y1, x1 = min(y0 + kachel, hoehe), min(x0 + kachel, breite)
                ya, xa = max(y0 - rand, 0), max(x0 - rand, 0)
                ye, xe = min(y1 + rand, hoehe), min(x1 + rand, breite)
                teil = kanaele[:, :, ya:ye, xa:xe]
                fh = -(ye - ya) % vielfaches
                fb = -(xe - xa) % vielfaches
                if fh or fb:                       # am Bildrand auffuellen wie im Original
                    teil = cp.pad(teil, ((0, 0), (0, 0), (0, fh), (0, fb)), mode="edge")
                ergebnis = self._kachel(cp.ascontiguousarray(teil))[0]
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


class Entrauscher(_Netz):
    """SCUNet - entrauscht ein sRGB-Bild in voller Groesse, Ergebnis bleibt auf der GPU."""

    def __init__(self, modell: Modell | None = None, fp16: bool = True,
                 kachel: int | None = None, tensorrt: bool | None = None):
        super().__init__(modell or ENTRAUSCH_MODELLE["scunet"], fp16, kachel, tensorrt)

    def entrauschen(self, srgb, kachel: int, fortschritt=None):
        """sRGB (H, W, 3) float32 0..1 auf der GPU -> entrauscht, gleiche Form, auf der GPU."""
        kanaele = cp.ascontiguousarray(cp.moveaxis(srgb.astype(cp.float32), -1, 0))[None]
        ziel = cp.empty((3,) + srgb.shape[:2], dtype=cp.float32)

        def ablegen(y0, y1, x0, x1, kern):
            ziel[:, y0:y1, x0:x1] = kern

        self._mit_wiederholung(lambda k: self._kacheln(kanaele, k, fortschritt, ablegen), kachel)
        return cp.ascontiguousarray(cp.moveaxis(cp.clip(ziel, 0, 1), 0, -1))


class KiAbbruch(Exception):
    """Der Anwender hat eine KI-Berechnung oder einen Download abgebrochen."""
