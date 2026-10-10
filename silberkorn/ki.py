"""
KI-Hochskalieren (Real-ESRGAN), KI-Entrauschen (SCUNet), KI-Schaerfen
(Restormer), Freistellen (BiRefNet), Auswahl per Klick (SAM 2), Objekte
entfernen (LaMa), Tiefe schaetzen (Depth Anything V2) und Bild erweitern
(FLUX.2 [klein] mit Outpaint-LoRA) ueber ONNX Runtime

Die Modelle stammen aus den offiziellen Releases von Real-ESRGAN (BSD-3-Clause),
SCUNet (Apache-2.0), Restormer (MIT), BiRefNet (MIT), SAM 2, LaMa und Depth
Anything V2 Small (alle drei Apache-2.0) und sind mit
werkzeuge/modelle_exportieren.py nach ONNX gewandelt; FLUX.2 [klein] 4B und
die Outpaint-LoRA von fal (beide Apache-2.0) mit werkzeuge/outpaint_export.py. Bereit liegen sie als
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
import math
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np

from . import einstellungen, filter, nvidia_laufzeit
from .cuda import cupy as cp

QUELLE = "https://github.com/xinntao/Real-ESRGAN"
QUELLE_SCUNET = "https://github.com/cszn/SCUNet"
QUELLE_RESTORMER = "https://github.com/swz30/Restormer"
QUELLE_BIREFNET = "https://github.com/ZhengPeng7/BiRefNet"
QUELLE_SAM2 = "https://github.com/facebookresearch/sam2"
QUELLE_LAMA = "https://github.com/advimman/lama"
QUELLE_TIEFE = "https://github.com/DepthAnything/Depth-Anything-V2"
QUELLE_FLUX2 = "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B"
QUELLE_DDCOLOR = "https://github.com/piddnad/DDColor"
MODELL_RELEASE = "https://github.com/DerAlexmann/Silberkorn/releases/download/modelle-1/"
MODELL_RELEASE_2 = "https://github.com/DerAlexmann/Silberkorn/releases/download/modelle-2/"
MODELL_RELEASE_3 = "https://github.com/DerAlexmann/Silberkorn/releases/download/modelle-3/"
MODELL_RELEASE_4 = "https://github.com/DerAlexmann/Silberkorn/releases/download/modelle-4/"
MODELL_RELEASE_5 = "https://github.com/DerAlexmann/Silberkorn/releases/download/modelle-5/"
MODELL_RELEASE_6 = "https://github.com/DerAlexmann/Silberkorn/releases/download/modelle-6/"
MODELL_RELEASE_7 = "https://github.com/DerAlexmann/Silberkorn/releases/download/modelle-7/"
MODELL_RELEASE_8 = "https://github.com/DerAlexmann/Silberkorn/releases/download/modelle-8/"
MODELL_RELEASE_9 = "https://github.com/DerAlexmann/Silberkorn/releases/download/modelle-9/"
BLOCK = 1 << 20
# Pruefsumme einer Datei, deren Release noch aussteht - werkzeuge/outpaint_export.py
# gibt die echten Summen und Groessen aus. Bis dahin lehnt datei_pruefen() jede
# Datei mit diesem Namen ab.
OFFEN = "0" * 64

# Bild erweitern: FLUX.2 [klein] 4B mit eingerechneter Outpaint-LoRA, int8. GitHub
# nimmt je Release-Datei hoechstens 2 GB an - die Gewichte des Transformers liegen
# deshalb als externe ONNX-Daten auf mehrere Dateien verteilt; ONNX Runtime liest sie
# aus dem Ordner der .onnx-Datei. MatMul (int8-Gewichte) und Attention rechnen in
# FP16, der Rest in FP32 - ganz in FP16 laufen die Aktivierungen ueber (NaN).
OUTPAINT_TRANSFORMER = "flux2-klein-outpaint.onnx"
OUTPAINT_GEWICHTE = ("flux2-klein-outpaint-1.bin", "flux2-klein-outpaint-2.bin",
                     "flux2-klein-outpaint-3.bin")
OUTPAINT_KODIERER = "flux2-vae-kodierer.onnx"
OUTPAINT_DEKODIERER = "flux2-vae-dekodierer.onnx"
OUTPAINT_DATEN = "flux2-klein-outpaint-daten.npz"     # Prompt-Einbettung, VAE-Normierung
OUTPAINT_NOTICE = "NOTICE-FLUX2-klein.txt"

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
    "depth-anything-v2-small.onnx":
        ("2b7c680369d243ed48c240d98f6263e1ca1a03e8c1035445b8614c0e4a909f6a", 102597798),
    "LICENSE-DepthAnythingV2.txt":
        ("c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4", 11357),
    OUTPAINT_TRANSFORMER:
        ("7b142a619b1fe979afea9cebcca58b7f190ca0ed432870cb6b7a8b9ad975c052", 2090922),
    OUTPAINT_GEWICHTE[0]:
        ("05175916238ef91c1d5d6934c5a54841240e756c30dcbc0b11d4eea254a6df32", 1893335040),
    OUTPAINT_GEWICHTE[1]:
        ("df5979950f12e816cc5764aebf1533e4de09364b8939eff7c401b41415de8990", 1840250880),
    OUTPAINT_GEWICHTE[2]:
        ("cd45115e7f23e4473afadc3a7c7f693f7b7385c31fbd961d1b95a7912c19b9cc", 206794752),
    OUTPAINT_KODIERER:
        ("a90651cbe02e4f761988cbf2f7ba3cacdd60d1d158dc67d301594c14778c6bce", 68942587),
    OUTPAINT_DEKODIERER:
        ("cd844387a12a49598479d3938ace3c0bdea4e199fc0aec477037eddd3e89150a", 99345574),
    OUTPAINT_DATEN:
        ("85aceb5808cd6ccd44d58815a08757db3fdeb2b4f06773fa97c1285860c08cba", 15731444),
    "LICENSE-FLUX2-klein.txt":
        ("c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4", 11357),
    OUTPAINT_NOTICE:
        ("dc4d5e17f058373b302df5cc7496972d2955779caaaf6e5b299cdf6fbdeb40ab", 1610),
    "ddcolor-tiny.onnx":
        ("f0b915eced94a8b15991172bd7dc38603f3dc8978cd0383d312be1313de17618", 220472850),
    "LICENSE-DDColor.txt":
        ("43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1", 11356),
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
    # Weitere Dateien (Datei, SHA-256), die zum Modell gehoeren - etwa der Decoder
    # neben dem Encoder
    weitere: tuple[tuple[str, str], ...] = ()
    # Wie viel Grafikspeicher (GiB, wie die Karte ihn meldet) das Modell mindestens
    # braucht, falls mehr als die Mindeststufe verspricht
    mindest_vram: float = 0.0


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
# Tiefe: Depth Anything V2 Small - nur dieses Modell der Familie steht unter
# Apache-2.0. Es sieht das Bild auf 1050 x 700 verkleinert (0,13 s); die Tiefe
# legt ein gefuehrter Filter an die Kanten des Bildes.
TIEFEN_MODELLE = {
    "tiefe": Modell(
        "tiefe", "depth-anything-v2-small.onnx",
        "2b7c680369d243ed48c240d98f6263e1ca1a03e8c1035445b8614c0e4a909f6a",
        None,
        "S", {"S": 1050, "M": 1050, "L": 1050, "XL": 1050},
        lizenz="Apache-2.0", quelle=QUELLE_TIEFE,
        herkunft="Depth Anything V2 Small (Apache-2.0, Lihe Yang u. a.)",
        lizenzdatei="LICENSE-DepthAnythingV2.txt", release=MODELL_RELEASE_7,
        massstab=1, rand=0, vielfaches=14, nhwc=False, cuda_fp16=False,
        feste_groesse=(700, 1050), ausgabe_kanaele=1),
}
# Bild erweitern: FLUX.2 [klein] 4B, destilliert auf 4 Schritte, mit der Outpaint-LoRA
# von fal (Staerke 1,1 fest eingerechnet) und in int8 (MatMulNBits). Es rechnet auf
# rund einem Megapixel; der Transformer allein belegt gut 4 GB, mit den
# Zwischenergebnissen braucht es eine Karte mit 8 GB (Spitze in PyTorch 6,2 GB).
ERWEITER_MODELLE = {
    "outpaint": Modell(
        "outpaint", OUTPAINT_TRANSFORMER, DATEIEN[OUTPAINT_TRANSFORMER][0],
        None,
        "M", {"M": 1024, "L": 1024, "XL": 1024},
        lizenz="Apache-2.0", quelle=QUELLE_FLUX2,
        herkunft="FLUX.2 [klein] 4B (Apache-2.0, Black Forest Labs) mit der "
                 "Outpaint-LoRA von fal (Apache-2.0)",
        lizenzdatei="LICENSE-FLUX2-klein.txt", release=MODELL_RELEASE_8,
        massstab=1, rand=0, vielfaches=16, nhwc=False, cuda_fp16=False,
        weitere=tuple((name, DATEIEN[name][0]) for name in (
            *OUTPAINT_GEWICHTE, OUTPAINT_KODIERER, OUTPAINT_DEKODIERER, OUTPAINT_DATEN,
            OUTPAINT_NOTICE)),
        mindest_vram=7.0),          # 8-GB-Karten melden 7,6 bis 8 GiB
}
# Kolorieren: DDColor in der kleinen Fassung (ConvNeXt-T). Es sieht das Graubild auf
# 512 x 512 und schaetzt die Farbanteile a und b (Lab); die Helligkeit bleibt die des
# Bildes in voller Aufloesung. Rund 60 ms auf einer RTX 4060, in FP32.
FARB_MODELLE = {
    "ddcolor": Modell(
        "ddcolor", "ddcolor-tiny.onnx",
        "f0b915eced94a8b15991172bd7dc38603f3dc8978cd0383d312be1313de17618",
        None,
        "S", {"S": 512, "M": 512, "L": 512, "XL": 512},
        lizenz="Apache-2.0", quelle=QUELLE_DDCOLOR,
        herkunft="DDColor (Apache-2.0, Xiaoyang Kang u. a.)",
        lizenzdatei="LICENSE-DDColor.txt", release=MODELL_RELEASE_9,
        massstab=1, rand=0, vielfaches=32, nhwc=False, cuda_fp16=False,
        feste_groesse=(512, 512), ausgabe_kanaele=2),
}
ALLE_MODELLE = {**MODELLE, **ENTRAUSCH_MODELLE, **SCHAERF_MODELLE, **MASKEN_MODELLE,
                **AUSWAHL_MODELLE, **ENTFERN_MODELLE, **TIEFEN_MODELLE, **ERWEITER_MODELLE,
                **FARB_MODELLE}
STUFEN = ("S", "M", "L", "XL")


class KiFehler(Exception):
    """Modell fehlt, ist beschaedigt oder laesst sich nicht ausfuehren."""


def ordner_kandidaten() -> list[str]:
    """Wo Modelle liegen duerfen: neben dem Programm, sonst im Benutzerordner."""
    benutzer = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return [os.path.join(einstellungen.programm_ordner(), "modelle"),
            os.path.join(benutzer, "Silberkorn", "modelle")]


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


def genug_vram(modell: Modell, vram_bytes: int) -> bool:
    """Reicht der Grafikspeicher der Karte, auch ueber die Stufe hinaus?"""
    return vram_bytes >= modell.mindest_vram * 2**30


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
                                             headers={"User-Agent": "Silberkorn"})
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
            # (in der EXE aus dem beim ersten Start geladenen Ordner)
            ort.preload_dlls(cuda=True, cudnn=True, msvc=False,
                             directory=nvidia_laufzeit.dll_ordner())
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

    def _speichergrenze(self) -> int:
        return _speichergrenze()

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
                                               "gpu_mem_limit": str(self._speichergrenze())})]
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


def _binden(sitzung, eingaben: dict, ausgaben: dict, schrumpfen: bool = False):
    """Netz mit CuPy-Arrays (float32) als Ein- und Ausgaengen rechnen, alles auf der GPU.
    schrumpfen: danach gibt ONNX Runtime seinen Zwischenspeicher wieder frei, statt ihn
    fuer den naechsten Lauf zu behalten - fuer Netze, die sich den Speicher teilen."""
    bindung = sitzung.io_binding()
    for name, wert in eingaben.items():
        bindung.bind_input(name, "cuda", 0, np.float32, list(wert.shape), wert.data.ptr)
    for name, wert in ausgaben.items():
        bindung.bind_output(name, "cuda", 0, np.float32, list(wert.shape), wert.data.ptr)
    cp.cuda.Device().synchronize()                 # CuPy hat fertig geschrieben
    optionen = None
    if schrumpfen:
        import onnxruntime as ort
        optionen = ort.RunOptions()
        optionen.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
    sitzung.run_with_iobinding(bindung, optionen)


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


class Tiefenschaetzer(_Netz):
    """Depth Anything V2 - die relative Tiefe eines ganzen Bildes, 1 = am naechsten.

    Wie beim Freistellen wird das Bild auf die feste Groesse des Netzes gebracht,
    Hochformate vorher gedreht. Die Tiefe kommt auf hoechstens ARBEIT Pixel
    Kantenlaenge zurueck, an die Kanten des Bildes gelegt, und wird auf 0..1
    gestreckt (1. bis 99. Perzentil) - mehr Aufloesung braucht eine Unschaerfe nach
    Tiefe nicht.
    """

    ARBEIT = 2048

    def __init__(self, modell: Modell | None = None):
        modell = modell or TIEFEN_MODELLE["tiefe"]
        super().__init__(modell, fp16=False, kachel=None, tensorrt=False)

    def tiefe(self, srgb):
        """sRGB (H, W, 3) float32 0..1 auf der GPU -> Tiefe (h, w) 0..1 auf der GPU,
        in der Lage des Bildes, h und w hoechstens ARBEIT."""
        hoehe, breite = srgb.shape[:2]
        hochformat = hoehe > breite
        bild = cp.rot90(srgb) if hochformat else srgb
        mh, mb = self.modell.feste_groesse
        faktor = max(1, min(bild.shape[0] // mh, bild.shape[1] // mb))
        klein = filter.vergroessern(filter.verkleinern_box(bild, faktor).astype(cp.float32),
                                    mh, mb)
        eingabe = cp.ascontiguousarray(cp.moveaxis(klein, -1, 0))[None]
        del klein
        ausgabe = cp.empty((1, 1, mh, mb), dtype=cp.float32)
        try:
            _binden(self.sitzung, {"eingabe": eingabe}, {"ausgabe": ausgabe})
        except Exception as fehler:              # ORT wirft eigene Fehlerklassen
            raise KiFehler(str(fehler)) from fehler
        roh = cp.rot90(ausgabe[0, 0], -1) if hochformat else ausgabe[0, 0]
        unten, oben = (float(v) for v in cp.percentile(roh, cp.asarray([1.0, 99.0])))
        roh = cp.clip((roh - unten) / max(oben - unten, 1e-6), 0, 1)
        # Fuehrung: das Bild selbst, verkleinert auf die Arbeitsgroesse
        faktor = max(1, -(-max(hoehe, breite) // self.ARBEIT))
        fuehrung = filter.luminanz(filter.verkleinern_box(srgb, faktor)).astype(cp.float32)
        fh, fb = fuehrung.shape
        grob = filter.vergroessern(cp.ascontiguousarray(roh, dtype=cp.float32), fh, fb)
        radius = max(2, round(max(fh, fb) / max(mh, mb)))
        fein = filter.gefuehrter_filter(fuehrung, grob, radius, 1e-4)
        return cp.ascontiguousarray(cp.clip(fein, 0, 1))


class Kolorierer(_Netz):
    """DDColor - Farbe fuer ein Schwarzweiss- oder Sepiabild.

    Das Netz sieht nur die Helligkeit: das Bild als Graubild in sRGB, auf
    512 x 512 gebracht (wie in der Pipeline von DDColor, ohne Drehen). Zurueck
    kommen die Farbanteile a und b im Lab-Raum; eingesetzt werden sie spaeter mit
    der Helligkeit des Bildes in voller Aufloesung (farbe_einsetzen).
    """

    def __init__(self, modell: Modell | None = None, kachel: int | None = None, **_art):
        modell = modell or FARB_MODELLE["ddcolor"]
        super().__init__(modell, fp16=False, kachel=None, tensorrt=False)

    def farben(self, linear):
        """Lineares Bild (H, W, 3) auf der GPU -> Farbanteile a, b (512, 512, 2)."""
        mh, mb = self.modell.feste_groesse
        hoehe, breite = linear.shape[:2]
        faktor = max(1, min(hoehe // mh, breite // mb))
        hell = filter.luminanz(cp.clip(filter.verkleinern_box(linear, faktor), 0, None))
        grau = filter.linear_zu_srgb(cp.clip(hell, 0, 1)).astype(cp.float32)
        grau = filter.vergroessern(cp.ascontiguousarray(grau), mh, mb)
        eingabe = cp.ascontiguousarray(cp.broadcast_to(grau, (1, 3, mh, mb)))
        ausgabe = cp.empty((1, 2, mh, mb), dtype=cp.float32)
        try:
            _binden(self.sitzung, {"eingabe": eingabe}, {"ausgabe": ausgabe})
        except Exception as fehler:              # ORT wirft eigene Fehlerklassen
            raise KiFehler(str(fehler)) from fehler
        return cp.ascontiguousarray(cp.moveaxis(ausgabe[0], 0, -1))


# Lineares sRGB -> XYZ (D65) und zurueck; Lab mit dem Weisspunkt D65 - wie OpenCV, mit
# dem DDColor trainiert ist
_LAB_PRAEAMBEL = r"""
__device__ __forceinline__ float lab_f(float t) {
    return t > 0.008856f ? cbrtf(t) : 7.787f * t + 16.0f / 116.0f;
}
__device__ __forceinline__ float lab_finv(float f) {
    return f > 0.206893f ? f * f * f : (f - 16.0f / 116.0f) / 7.787f;
}
"""

_FARBE_EINSETZEN = None


def farbe_einsetzen(linear, ab, staerke: float):
    """Farbanteile a, b (H, W, 2) mit der Helligkeit des linearen Bildes (H, W, 3) zu
    einem farbigen Bild verbinden und mit staerke (0..1) ueber das Bild legen.

    Die Helligkeit Y bleibt genau erhalten - wie bei DDColor, das L des Originals
    behaelt; ein Sepiaton faellt bei voller Staerke weg."""
    global _FARBE_EINSETZEN
    if _FARBE_EINSETZEN is None:
        _FARBE_EINSETZEN = cp.ElementwiseKernel(
            "raw float32 bild, raw float32 ab, float32 staerke",
            "raw float32 ziel",
            r"""
            float r = bild[3 * i], g = bild[3 * i + 1], b = bild[3 * i + 2];
            float y = fmaxf(0.212671f * r + 0.715160f * g + 0.072169f * b, 0.0f);
            float fy = lab_f(y);
            float x = 0.950456f * lab_finv(fy + ab[2 * i] / 500.0f);
            float z = 1.088754f * lab_finv(fy - ab[2 * i + 1] / 200.0f);
            float fr = 3.240479f * x - 1.537150f * y - 0.498535f * z;
            float fg = -0.969256f * x + 1.875992f * y + 0.041556f * z;
            float fb = 0.055648f * x - 0.204043f * y + 1.057311f * z;
            ziel[3 * i] = r + staerke * (fmaxf(fr, 0.0f) - r);
            ziel[3 * i + 1] = g + staerke * (fmaxf(fg, 0.0f) - g);
            ziel[3 * i + 2] = b + staerke * (fmaxf(fb, 0.0f) - b);
            """,
            "silberkorn_farbe_einsetzen", preamble=_LAB_PRAEAMBEL)
    hoehe, breite = linear.shape[:2]
    ziel = cp.empty((hoehe, breite, 3), dtype=cp.float32)
    _FARBE_EINSETZEN(cp.ascontiguousarray(linear, dtype=cp.float32),
                     cp.ascontiguousarray(ab, dtype=cp.float32), cp.float32(staerke), ziel,
                     size=hoehe * breite)
    return ziel


# --------------------------------------------------------------------------
# Bild erweitern (Outpainting): FLUX.2 [klein] mit Outpaint-LoRA
# --------------------------------------------------------------------------
#
# Das Original kommt mittig auf eine Leinwand im Zielformat, der Rest wird
# reingruen; die LoRA hat gelernt, gruene Flaechen passend zum Bild zu fuellen.
# Gerechnet wird auf rund einem Megapixel (Kanten durch 16 teilbar), mit festem
# Prompt, dessen Einbettung beiliegt - der Text-Encoder (Qwen3, 8 GB) entfaellt.
# Die Ablaeufe folgen Flux2KleinPipeline aus diffusers 0.41: Latents in 2 x 2
# zusammenfassen (patchify) und normieren, das Leinwandbild als Referenz-Tokens
# neben die verrauschten Tokens legen (Zeitkoordinate 10 statt 0), vier
# Euler-Schritte des Flow Matching mit verschobenem Zeitplan (compute_empirical_mu).
#
# Das Modell zeichnet das ganze Bild neu, auch den Bereich des Originals - mit
# demselben leichten Farbstich wie die neuen Raender. farbe_angleichen() misst
# den Stich am Original und nimmt ihn ueberall heraus; eingesetzt wird danach das
# unveraenderte Original in voller Aufloesung, mit weichem Saum nur an den
# erweiterten Seiten.
#
# Die Rechenhilfen nehmen NumPy- und CuPy-Arrays; die Tests laufen so auch ohne
# Grafikkarte.

OUTPAINT_FLAECHE = 1024 * 1024     # Arbeitsgroesse in Pixeln, hoechstens
# Mit weniger als 12 GB: Gewichte (4,2 GB) und Zwischenwerte von 1 MP (3,5 GB) passen
# nicht zusammen in 8 GB. RTX 4060 (8 GB): 0,6 MP, 4 Schritte je ~3,5 s
OUTPAINT_FLAECHE_KNAPP = 640 * 1024
OUTPAINT_VRAM_VOLL = 11.5 * 2**30  # ab hier die volle Arbeitsgroesse (12-GB-Karten)
OUTPAINT_FLAECHE_MIN = 256 * 1024  # kleiner wird nicht versucht - zu wenig Details
OUTPAINT_RESERVE = 768 << 20       # Grafikspeicher fuer Windows und die Anzeige


class _ErweiternSpeicher(KiFehler):
    """Zu wenig Grafikspeicher fuer die gewaehlte Arbeitsgroesse."""
OUTPAINT_VIELFACHES = 16           # VAE (8) mal patchify (2)
OUTPAINT_GRUEN = (0.0, 1.0, 0.0)   # was die LoRA fuellt (#00FF00)
OUTPAINT_SCHRITTE = 4              # destilliertes Modell, ohne CFG (guidance 1.0)
OUTPAINT_REFERENZ_T = 10           # Zeitkoordinate des Referenzbildes (diffusers: scale)
OUTPAINT_SAUM = 0.015              # weicher Saum: Anteil der kurzen Kante des Originals
OUTPAINT_GROSSER_RAND = 0.25       # ab hier erfindet die KI mehr, als sie sieht


def _ndimage(xp):
    if xp is np:
        from scipy import ndimage
    else:
        from cupyx.scipy import ndimage
    return ndimage


def leinwand_planen(breite: int, hoehe: int, verhaeltnis: float, anteil: float = 1.0,
                    raster: int = 1) -> tuple[int, int, int, int]:
    """Leinwand im Seitenverhaeltnis verhaeltnis (Breite/Hoehe) um ein Original.

    Rueckgabe (W, H, x, y) in Pixeln des Originals: Groesse der Leinwand und Lage
    des Originals darin, mittig. anteil < 1 erweitert zusaetzlich rundum. x und y
    liegen auf einem Vielfachen von raster (fuer die verkleinerte Vorschau).
    """
    bw, bh = breite / anteil, hoehe / anteil
    if bw / bh < verhaeltnis:
        bw = bh * verhaeltnis
    else:
        bh = bw / verhaeltnis
    gross_b, gross_h = max(round(bw), breite), max(round(bh), hoehe)
    x = (gross_b - breite) // 2 // raster * raster
    y = (gross_h - hoehe) // 2 // raster * raster
    return gross_b, gross_h, x, y


def rand_anteile(breite: int, hoehe: int, gross_b: int, gross_h: int, x: int,
                 y: int) -> tuple[float, float, float, float]:
    """Wie viel je Seite dazukommt (links, oben, rechts, unten), als Anteil der
    Kante des Originals in dieser Richtung."""
    return (x / breite, y / hoehe, (gross_b - breite - x) / breite,
            (gross_h - hoehe - y) / hoehe)


def grosser_rand(breite: int, hoehe: int, gross_b: int, gross_h: int, x: int, y: int) -> bool:
    return max(rand_anteile(breite, hoehe, gross_b, gross_h, x, y)) > OUTPAINT_GROSSER_RAND


def arbeitsgroesse(gross_b: int, gross_h: int, flaeche: int = OUTPAINT_FLAECHE,
                   vielfaches: int = OUTPAINT_VIELFACHES) -> tuple[int, int]:
    """(Breite, Hoehe) fuers Modell: Seitenverhaeltnis der Leinwand, rund `flaeche`
    Pixel, Kanten Vielfache von `vielfaches`. Nie mehr als `flaeche` - sonst
    verkleinerte diffusers das Referenzbild selbst noch einmal."""
    s = (flaeche / (gross_b * gross_h)) ** 0.5
    ab = max(vielfaches, round(gross_b * s / vielfaches) * vielfaches)
    ah = max(vielfaches, round(gross_h * s / vielfaches) * vielfaches)
    while ab * ah > flaeche:
        # Die Kante kuerzen, die beim Runden am meisten zugelegt hat
        if ab / (gross_b * s) >= ah / (gross_h * s):
            ab -= vielfaches
        else:
            ah -= vielfaches
    return ab, ah


def _skalieren(bild, hoehe: int, breite: int):
    """Bild (h, w, k) auf (hoehe, breite): erst ganzzahlig mitteln, dann bilinear."""
    faktor = max(1, min(bild.shape[0] // max(hoehe, 1), bild.shape[1] // max(breite, 1)))
    xp = filter.xp_von(bild)
    klein = filter.verkleinern_box(bild, faktor).astype(xp.float32)
    return filter.vergroessern(klein, hoehe, breite)


def eingabe_bauen(srgb, gross_b: int, gross_h: int, x: int, y: int, ab: int, ah: int):
    """Leinwand fuers Modell: (ah, ab, 3) reingruen, das verkleinerte Original darauf.

    Rueckgabe: Leinwand und (ox, oy, ob, oh), die Lage des Originals darin.
    """
    xp = filter.xp_von(srgb)
    hoehe, breite = srgb.shape[:2]
    sx, sy = ab / gross_b, ah / gross_h
    ox, oy = round(x * sx), round(y * sy)
    ob, oh = min(round(breite * sx), ab - ox), min(round(hoehe * sy), ah - oy)
    leinwand = xp.empty((ah, ab, 3), dtype=xp.float32)
    leinwand[...] = xp.asarray(OUTPAINT_GRUEN, dtype=xp.float32)
    leinwand[oy:oy + oh, ox:ox + ob] = xp.clip(_skalieren(srgb, oh, ob), 0, 1)
    return leinwand, (ox, oy, ob, oh)


def patchify(latent):
    """(1, C, H, W) -> (1, 4C, H/2, W/2): je 2 x 2 Punkte zu einem (wie diffusers)."""
    b, c, h, w = latent.shape
    teile = latent.reshape(b, c, h // 2, 2, w // 2, 2).transpose(0, 1, 3, 5, 2, 4)
    return teile.reshape(b, c * 4, h // 2, w // 2)


def unpatchify(latent):
    b, c, h, w = latent.shape
    teile = latent.reshape(b, c // 4, 2, 2, h, w).transpose(0, 1, 4, 2, 5, 3)
    return teile.reshape(b, c // 4, h * 2, w * 2)


def packen(latent):
    """(1, C, h, w) -> Tokens (1, h*w, C), zeilenweise."""
    b, c, h, w = latent.shape
    return latent.reshape(b, c, h * w).transpose(0, 2, 1)


def entpacken(tokens, hoehe: int, breite: int):
    b, n, c = tokens.shape
    return tokens.transpose(0, 2, 1).reshape(b, c, hoehe, breite)


def positionen(hoehe: int, breite: int, t: int = 0, xp=np):
    """Positions-IDs (h*w, 4) der Bild-Tokens: (T, Zeile, Spalte, 0), zeilenweise."""
    zeilen, spalten = xp.meshgrid(xp.arange(hoehe, dtype=xp.float32),
                                  xp.arange(breite, dtype=xp.float32), indexing="ij")
    ids = xp.zeros((hoehe * breite, 4), dtype=xp.float32)
    ids[:, 0] = t
    ids[:, 1] = zeilen.ravel()
    ids[:, 2] = spalten.ravel()
    return ids


def text_positionen(laenge: int, xp=np):
    """Positions-IDs (L, 4) der Text-Tokens: (0, 0, 0, Stelle)."""
    ids = xp.zeros((laenge, 4), dtype=xp.float32)
    ids[:, 3] = xp.arange(laenge, dtype=xp.float32)
    return ids


def empirischer_mu(bild_tokens: int, schritte: int) -> float:
    """Verschiebung des Zeitplans nach Bildgroesse - compute_empirical_mu aus diffusers."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if bild_tokens > 4300:
        return float(a2 * bild_tokens + b2)
    m_200 = a2 * bild_tokens + b2
    m_10 = a1 * bild_tokens + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * schritte + b)


@dataclass(frozen=True)
class Zeitplan:
    """Die Einstellungen des FlowMatchEulerDiscreteScheduler, die hier zaehlen."""
    dynamisch: bool = True         # use_dynamic_shifting
    exponentiell: bool = True      # time_shift_type == "exponential"
    verschiebung: float = 3.0      # shift, nur ohne dynamische Verschiebung
    ende: float = 0.0              # shift_terminal, 0 = keins


def sigmas_berechnen(schritte: int, mu: float, plan: Zeitplan) -> np.ndarray:
    """Rauschanteile je Schritt, mit 0 am Ende (schritte + 1 Werte, float32).

    Wie Flux2KleinPipeline: linspace(1, 1/n, n), verschoben um mu, und
    FlowMatchEulerDiscreteScheduler.set_timesteps.
    """
    s = np.linspace(1.0, 1 / schritte, schritte).astype(np.float32)
    if plan.dynamisch:
        if plan.exponentiell:
            s = math.exp(mu) / (math.exp(mu) + (1 / s - 1))
        else:
            s = mu / (mu + (1 / s - 1))
    else:
        s = plan.verschiebung * s / (1 + (plan.verschiebung - 1) * s)
    if plan.ende:
        rest = 1 - s
        s = 1 - rest / (rest[-1] / (1 - plan.ende))
    return np.append(s.astype(np.float32), np.float32(0))


def euler_schritt(latents, geschwindigkeit, sigma: float, sigma_danach: float):
    """Ein Schritt des Flow Matching: x <- x + (sigma' - sigma) * v."""
    return latents + np.float32(sigma_danach - sigma) * geschwindigkeit


def _weich(bild, sigma: float):
    nd = _ndimage(filter.xp_von(bild))
    if bild.ndim == 2:
        return nd.gaussian_filter(bild, sigma)
    xp = filter.xp_von(bild)
    return xp.stack([nd.gaussian_filter(bild[..., k], sigma) for k in range(bild.shape[2])],
                    axis=-1)


def farbe_angleichen(roh, orig, ox: int, oy: int):
    """Den Farbstich des Modells herausnehmen.

    roh: Ergebnis des Modells (ah, ab, 3) sRGB 0..1; orig: das Original in der
    Groesse, in der es auf der Leinwand lag (oh, ob, 3), an der Stelle (ox, oy).
    Je Kanal eine lineare Abbildung Modell -> Original, gemessen im Bereich des
    Originals; danach die weichgezeichnete Restabweichung vom naechsten Punkt des
    Originals nach aussen fortgesetzt - so laufen auch oertliche Unterschiede
    am Rand des Originals sanft in die neuen Raender aus.
    """
    xp = filter.xp_von(roh)
    ah, ab = roh.shape[:2]
    oh, ob = orig.shape[:2]
    m = roh.astype(xp.float32, copy=True)
    o = orig.astype(xp.float32)
    for k in range(3):
        x = m[oy:oy + oh, ox:ox + ob, k].ravel()
        y = o[..., k].ravel()
        mx, my = float(x.mean()), float(y.mean())
        streuung = float(((x - mx) ** 2).mean())
        a = float(((x - mx) * (y - my)).mean()) / streuung if streuung > 1e-8 else 1.0
        m[..., k] = m[..., k] * a + (my - a * mx)
    sigma = max(ab, ah) / 40
    abweichung = xp.zeros_like(m)
    gewicht = xp.zeros((ah, ab), dtype=xp.float32)
    abweichung[oy:oy + oh, ox:ox + ob] = o - m[oy:oy + oh, ox:ox + ob]
    gewicht[oy:oy + oh, ox:ox + ob] = 1
    abweichung = _weich(abweichung, sigma) / xp.maximum(_weich(gewicht, sigma), 1e-3)[..., None]
    _abstand, naechste = _ndimage(xp).distance_transform_edt(gewicht < 0.5,
                                                             return_indices=True)
    fortgesetzt = abweichung[naechste[0], naechste[1]]
    return xp.clip(m + _weich(fortgesetzt, sigma), 0, 1).astype(xp.float32)


def saum_deckkraft(gross_b: int, gross_h: int, x: int, y: int, breite: int, hoehe: int,
                   saum: float, xp=np):
    """Deckkraft des Originals (hoehe, breite) beim Einsetzen in die Leinwand.

    Faellt zu den erweiterten Seiten hin ueber `saum` Pixel auf 0 ab; Seiten, an
    denen das Original schon die Leinwand beruehrt, bleiben hart (1). Das ist der
    euklidische Abstand zur Leinwand ausserhalb des Originals - bei einem Rechteck
    der kleinste Abstand zu einer erweiterten Kante.
    """
    weit = xp.float32(1e9)
    zeilen = xp.arange(hoehe, dtype=xp.float32)
    spalten = xp.arange(breite, dtype=xp.float32)
    oben = zeilen + 1 if y > 0 else xp.full_like(zeilen, weit)
    unten = hoehe - zeilen if y + hoehe < gross_h else xp.full_like(zeilen, weit)
    links = spalten + 1 if x > 0 else xp.full_like(spalten, weit)
    rechts = breite - spalten if x + breite < gross_b else xp.full_like(spalten, weit)
    abstand = xp.minimum(xp.minimum(oben, unten)[:, None], xp.minimum(links, rechts)[None, :])
    return xp.clip(abstand / max(saum, 1e-6), 0, 1).astype(xp.float32)


def saum_breite(breite: int, hoehe: int) -> float:
    return max(4.0, round(min(breite, hoehe) * OUTPAINT_SAUM))


def einsetzen(gross, orig, x: int, y: int, deckkraft):
    """Das Original (h, w, k) mit Deckkraft (h, w) an (x, y) in die Leinwand setzen."""
    hoehe, breite = orig.shape[:2]
    ergebnis = gross.copy()
    teil = ergebnis[y:y + hoehe, x:x + breite]
    teil += deckkraft[..., None] * (orig - teil)
    return ergebnis


@dataclass(frozen=True)
class OutpaintDaten:
    """Was dem Transformer neben den Gewichten beiliegt (flux2-klein-outpaint-daten.npz)."""
    prompt: object                 # Einbettung des festen Prompts (1, 512, 7680)
    mittel: object                 # Normierung der Latents (bn des VAE), (1, 128, 1, 1)
    streuung: object
    zeitplan: Zeitplan


def outpaint_daten_laden(pfad: str, xp=np) -> OutpaintDaten:
    with np.load(pfad) as daten:
        def feld(name, form=None):
            wert = xp.asarray(daten[name], dtype=xp.float32)
            return xp.ascontiguousarray(wert if form is None else wert.reshape(form))
        return OutpaintDaten(
            feld("prompt"), feld("bn_mittel", (1, -1, 1, 1)), feld("bn_std", (1, -1, 1, 1)),
            Zeitplan(bool(daten["dynamisch"]), bool(daten["exponentiell"]),
                     float(daten["verschiebung"]), float(daten["ende"])))


def outpaint_rauschen(seed: int, tokens: int, kanaele: int, xp=np):
    """Startrauschen (1, tokens, kanaele) - gleich fuer gleichen Seed."""
    return xp.random.RandomState(seed).standard_normal((1, tokens, kanaele)).astype(xp.float32)


def outpaint_rechnen(eingabe, rauschen, daten: OutpaintDaten, kodieren, transformieren,
                     dekodieren, weiter=None):
    """Der Ablauf von Flux2KleinPipeline.__call__ fuer ein Referenzbild, ohne CFG.

    eingabe: Leinwand (ah, ab, 3) sRGB 0..1; rauschen: (1, n, 128) mit n = ah*ab/256.
    Die Netze sind Funktionen: kodieren(bild (1, 3, ah, ab)) -> Latent (1, 32, ah/8,
    ab/8), transformieren(eingaben: dict) -> Geschwindigkeit (1, 2n, 128),
    dekodieren(latent) -> Bild (1, 3, ah, ab) sRGB 0..1. weiter() nach jedem Netzlauf.
    NumPy oder CuPy - je nachdem, worauf eingabe liegt.
    """
    xp = filter.xp_von(eingabe)
    weiter = weiter or (lambda: None)
    ah, ab = eingabe.shape[:2]
    h, w = ah // OUTPAINT_VIELFACHES, ab // OUTPAINT_VIELFACHES
    n = h * w
    bild = xp.ascontiguousarray(xp.moveaxis(eingabe.astype(xp.float32), -1, 0))[None]
    roh = kodieren(bild)
    del bild
    weiter()
    referenz = xp.ascontiguousarray(packen((patchify(roh) - daten.mittel) / daten.streuung))
    ids = xp.ascontiguousarray(xp.concatenate([positionen(h, w, 0, xp),
                                               positionen(h, w, OUTPAINT_REFERENZ_T, xp)]))
    text_ids = text_positionen(daten.prompt.shape[1], xp)
    sigmas = sigmas_berechnen(OUTPAINT_SCHRITTE, empirischer_mu(n, OUTPAINT_SCHRITTE),
                              daten.zeitplan)
    latents = rauschen
    for i in range(OUTPAINT_SCHRITTE):
        eingang = xp.ascontiguousarray(xp.concatenate([latents, referenz], axis=1))
        geschwindigkeit = transformieren({
            "latents": eingang, "prompt": daten.prompt,
            "zeit": xp.asarray([sigmas[i]], dtype=xp.float32),
            "bild_ids": ids, "text_ids": text_ids})
        # Nur die eigenen Tokens zaehlen, nicht die des Referenzbildes
        latents = euler_schritt(latents, geschwindigkeit[:, :n], sigmas[i], sigmas[i + 1])
        del eingang, geschwindigkeit
        weiter()
    latent = unpatchify(entpacken(latents, h, w) * daten.streuung + daten.mittel)
    ergebnis = dekodieren(xp.ascontiguousarray(latent, dtype=xp.float32))
    weiter()
    return xp.ascontiguousarray(xp.clip(xp.moveaxis(ergebnis[0], 0, -1), 0, 1))


class Erweiterer(_Netz):
    """FLUX.2 [klein] mit Outpaint-LoRA - erfindet die Raender um ein Bild.

    Drei Netze: Transformer (int8, auf mehrere Dateien verteilt), VAE-Encoder und
    VAE-Decoder (FP16, Ein- und Ausgaenge FP32 in sRGB 0..1). Dazu liegen die
    Prompt-Einbettung, die Normierung der Latents (bn des VAE) und die
    Einstellungen des Zeitplans bei.
    """

    def __init__(self, modell: Modell | None = None):
        modell = modell or ERWEITER_MODELLE["outpaint"]
        pfade = {name: datei_pruefen(name, sha256) for name, sha256 in modell.weitere}
        haupt = datei_pfad(modell.datei)
        for name in OUTPAINT_GEWICHTE:
            # ONNX Runtime sucht die externen Gewichte neben der .onnx-Datei
            if haupt is None or os.path.dirname(pfade[name]) != os.path.dirname(haupt):
                raise KiFehler(f"{name} muss im selben Ordner liegen wie {modell.datei}.")
        super().__init__(modell, fp16=False, kachel=None, tensorrt=False)
        self.kodierer = self._cuda_sitzung(pfade[OUTPAINT_KODIERER])
        self.dekodierer = self._cuda_sitzung(pfade[OUTPAINT_DEKODIERER])
        self.daten = outpaint_daten_laden(pfade[OUTPAINT_DATEN], cp)
        gesamt = cp.cuda.runtime.memGetInfo()[1]
        self.flaeche = OUTPAINT_FLAECHE if gesamt >= OUTPAINT_VRAM_VOLL \
            else OUTPAINT_FLAECHE_KNAPP

    def _speichergrenze(self) -> int:
        """Grenze je Sitzung: der ganze Grafikspeicher bis auf eine Reserve - nicht nur,
        was beim Anlegen frei ist. Drei Sitzungen teilen sich den Speicher und geben
        ihn nach jedem Lauf zurueck; mit der engeren Grenze scheiterte der Transformer
        am zweiten Bild anderer Form, obwohl genug frei war (Arena von ONNX Runtime)."""
        return max(1 << 30, cp.cuda.runtime.memGetInfo()[1] - OUTPAINT_RESERVE)

    def _rechnen(self, sitzung, eingaben: dict, ausgabe_name: str, form):
        ausgabe = cp.empty(form, dtype=cp.float32)
        try:
            # Drei Netze teilen sich den Speicher: jedes gibt seinen Zwischenspeicher
            # zurueck, sonst haelt der Transformer ihn, und der Decoder muss auslagern
            _binden(sitzung, eingaben, {ausgabe_name: ausgabe}, schrumpfen=True)
        except cp.cuda.memory.OutOfMemoryError as fehler:
            raise _ErweiternSpeicher("Zu wenig Grafikspeicher zum Erweitern.") from fehler
        except Exception as fehler:              # ORT wirft eigene Fehlerklassen
            text = str(fehler)
            if "memory" in text.lower() or "alloc" in text.lower():
                raise _ErweiternSpeicher("Zu wenig Grafikspeicher zum Erweitern.") from fehler
            raise KiFehler(text) from fehler
        return ausgabe

    def erzeugen(self, eingabe, seed: int, fortschritt=None):
        """Leinwand (ah, ab, 3) sRGB 0..1 auf der GPU, Gruen = zu fuellen ->
        Ergebnis des Modells, gleiche Form. fortschritt(i, n) nach jedem Netzlauf;
        gibt es False zurueck, wird abgebrochen."""
        ah, ab = eingabe.shape[:2]
        n = (ah // OUTPAINT_VIELFACHES) * (ab // OUTPAINT_VIELFACHES)
        kanaele = self.daten.mittel.shape[1]
        # Was CuPy fuer grosse Bilder vorhaelt, braucht jetzt ONNX Runtime
        cp.get_default_memory_pool().free_all_blocks()
        anzahl, nummer = OUTPAINT_SCHRITTE + 2, 0

        def weiter():
            nonlocal nummer
            nummer += 1
            if fortschritt is not None and fortschritt(nummer, anzahl) is False:
                raise KiAbbruch()

        return outpaint_rechnen(
            eingabe, outpaint_rauschen(seed, n, kanaele, cp), self.daten,
            lambda bild: self._rechnen(self.kodierer, {"bild": bild}, "latent",
                                       (1, kanaele // 4, ah // 8, ab // 8)),
            lambda eingaben: self._rechnen(self.sitzung, eingaben, "geschwindigkeit",
                                           (1, 2 * n, kanaele)),
            lambda latent: self._rechnen(self.dekodierer, {"latent": latent}, "bild",
                                         (1, 3, ah, ab)),
            weiter)

    def raender(self, srgb, gross_b: int, gross_h: int, x: int, y: int, seed: int,
                fortschritt=None):
        """sRGB-Original (h, w, 3) 0..1 auf der GPU -> Leinwand (H, W, 3) in voller
        Groesse, wie sie das Modell sieht: farblich ans Original angeglichen, aber
        noch ohne eingesetztes Original (das setzt einsetzen() mit Saum ein)."""
        while True:
            ab, ah = arbeitsgroesse(gross_b, gross_h, self.flaeche)
            eingabe, (ox, oy, ob, oh) = eingabe_bauen(srgb, gross_b, gross_h, x, y, ab, ah)
            try:
                roh = self.erzeugen(eingabe, seed, fortschritt)
                break
            except _ErweiternSpeicher:
                # Wie viel neben den Gewichten frei bleibt, haengt von Bild und Fragmentierung
                # ab: kleiner versuchen und es fuer die naechsten Bilder so lassen
                del eingabe
                cp.get_default_memory_pool().free_all_blocks()
                if self.flaeche <= OUTPAINT_FLAECHE_MIN:
                    raise
                self.flaeche = max(OUTPAINT_FLAECHE_MIN, int(self.flaeche * 0.75))
        angeglichen = farbe_angleichen(roh, eingabe[oy:oy + oh, ox:ox + ob], ox, oy)
        del roh, eingabe
        return filter.vergroessern(angeglichen, gross_h, gross_b)

    def erweitern(self, srgb, verhaeltnis: float, seed: int, fortschritt=None):
        """sRGB-Original (h, w, 3) -> erweitertes Bild im Seitenverhaeltnis, auf der GPU."""
        hoehe, breite = srgb.shape[:2]
        gross_b, gross_h, x, y = leinwand_planen(breite, hoehe, verhaeltnis)
        gross = self.raender(srgb, gross_b, gross_h, x, y, seed, fortschritt)
        deckkraft = saum_deckkraft(gross_b, gross_h, x, y, breite, hoehe,
                                   saum_breite(breite, hoehe), cp)
        return einsetzen(gross, srgb.astype(cp.float32), x, y, deckkraft)
