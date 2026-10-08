"""
EXE bauen: PyInstaller, Lizenzen beilegen, pruefen, als ZIP packen

Aufruf ueber build.cmd, das die saubere Umgebung .venv-build anlegt. Ergebnis:

    dist/Silberkorn/                      der Programmordner (onedir)
    dist/Silberkorn-<Version>-win64.zip   derselbe Ordner zum Verteilen

Neben die EXE kommen LICENSE, NOTICE, LICENSES/ und der Ordner lizenzen/ mit
den Lizenztexten aller Pakete, die PyInstaller tatsaechlich eingepackt hat,
dazu DRITTANBIETER.txt mit Paket, Fassung und Lizenz. Der Bau bricht ab, wenn
NVIDIA-Bibliotheken oder libheif im Ordner landen.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import re
import shutil
import sys

WURZEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(WURZEL, "dist")
BUILD = os.path.join(WURZEL, "build")
ORDNER = os.path.join(DIST, "Silberkorn")

# Duerfen nie in die EXE (siehe NOTICE und silberkorn/nvidia_laufzeit.py): die
# NVIDIA-DLLs selbst - CuPys gleichnamige Anbindungen (cublas.pyd usw., MIT)
# sind erlaubt - und alles mit libheif
VERBOTEN_DLL = re.compile(r"^(cudart|cublas|cudnn|cufft|curand|cusolver|cusparse|nvrtc|nvjitlink|"
                          r"nvinfer|nvonnxparser|nvblas)", re.I)
VERBOTEN = re.compile(r"^_?(lib)?heif", re.I)

# Lizenzen der Bibliotheken in den eingepackten imagecodecs-Modulen
IMAGECODECS_LIZENZEN = ["LICENSE-zlib", "LICENSE-libdeflate", "LICENSE-libpng",
                        "LICENSE-libjpeg-turbo", "LICENSE-zstd", "LICENSE-liblzma",
                        "LICENSE-libwebp"]
# Pakete ohne Lizenzdatei im dist-info-Ordner: wo die Lizenz sonst steht -
# im Kopf eines Moduls, als Datei im Paket oder (flatbuffers nennt nur
# "Apache-2.0") als Standardtext aus LICENSES/
LIZENZ_ERSATZ = {
    "nvidia-ml-py": ("kopf", "pynvml.py"),
    "onnxruntime-gpu": ("datei", "onnxruntime/LICENSE"),
    "flatbuffers": ("text", "Apache-2.0.txt"),
}
LIZENZDATEI = re.compile(r"(licen[cs]e|copying|notice|authors)", re.I)


def version() -> str:
    with open(os.path.join(WURZEL, "silberkorn", "__init__.py"), encoding="utf-8") as datei:
        return re.search(r'^VERSION = "([^"]+)"', datei.read(), re.M).group(1)


def umgebung_pruefen():
    """Nur in einer Umgebung ohne NVIDIA-Pakete bauen - sonst koennte ein Hook sie einsammeln."""
    if sys.prefix == sys.base_prefix:
        sys.exit("Bitte ueber build.cmd bauen (saubere Umgebung .venv-build).")
    fremd = sorted(d.metadata["Name"] for d in md.distributions()
                   if re.match(r"(nvidia-(?!ml-py)|tensorrt|pillow-heif|onnxruntime$|torch)",
                               d.metadata["Name"] or "", re.I))
    if fremd:
        sys.exit(f"Diese Pakete gehoeren nicht in die Bauumgebung: {', '.join(fremd)}")


def pyinstaller():
    import PyInstaller.__main__
    PyInstaller.__main__.run([os.path.join(WURZEL, "silberkorn.spec"), "--noconfirm", "--clean",
                              "--distpath", DIST, "--workpath", BUILD])


def verbotene_dateien() -> list[str]:
    funde = []
    for wurzel, _ordner, dateien in os.walk(ORDNER):
        for name in dateien:
            endung = os.path.splitext(name)[1].lower()
            if ((endung == ".dll" and VERBOTEN_DLL.match(name))
                    or (endung in (".dll", ".pyd") and VERBOTEN.match(name))):
                funde.append(os.path.relpath(os.path.join(wurzel, name), ORDNER))
    return funde


# Importiert in einem eigenen Prozess alles, was das Programm laedt, und meldet
# die Module mit ihrem echten Namen - Aliase wie os.path fallen so heraus.
_IMPORTPROBE = """
import json, sys
import silberkorn.hauptfenster, silberkorn.einrichten
import cupyx.scipy.ndimage, imagecodecs, rawpy, tifffile, onnx, onnxruntime
import onnxruntime.transformers.float16
for codec in ("png", "zlib", "deflate", "jpeg8", "zstd", "lzma", "webp", "lzw"):
    getattr(imagecodecs, codec + "_decode")
echt = [n for n, m in list(sys.modules.items()) if getattr(m, "__name__", n) == n]
print(json.dumps(sorted(echt)))
"""


# Module, die erst zur Laufzeit entstehen und nie als Datei vorliegen: das
# Programm selbst (steht als Skript in der EXE), Cythons Hilfsmodule, die in
# shiboken eingebettete Signatur-Unterstuetzung von PySide6 und Aliase
LAUFZEIT_MODULE = ("silberkorn", "__main__", "_cython_", "_frozen_importlib", "cython_runtime",
                   "shibokensupport", "signature_bootstrap", "PySide6.support",
                   "collections.abc", "cupy.fft.config")


def module_pruefen() -> list[str]:
    """Module, die das Programm laedt, die aber nicht in der EXE stecken.

    PyInstaller sieht Importe aus kompilierten Modulen (Cython, pybind) nicht -
    so fehlte etwa graphlib, das CuPys Kern importiert. Die Probe laeuft ohne
    Grafikkarte: Importieren genuegt, gerechnet wird nicht.
    """
    import ast
    import subprocess
    import zipfile
    ergebnis = subprocess.run([sys.executable, "-c", _IMPORTPROBE], cwd=WURZEL, check=True,
                              capture_output=True, text=True)
    geladen = json.loads(ergebnis.stdout.strip().splitlines()[-1])
    with open(os.path.join(BUILD, "silberkorn", "PYZ-00.toc"), encoding="utf-8") as datei:
        vorhanden = {e[0] for e in ast.literal_eval(datei.read())[1]}
    with zipfile.ZipFile(os.path.join(BUILD, "silberkorn", "base_library.zip")) as archiv:
        vorhanden |= {n[:-4].replace("/", ".").removesuffix(".__init__")
                      for n in archiv.namelist() if n.endswith(".pyc")}
    intern = os.path.join(ORDNER, "_internal")
    erweiterungen = set()
    for wurzel, _ordner, dateien in os.walk(intern):
        for name in dateien:
            if name.endswith(".pyd"):
                pfad = os.path.relpath(os.path.join(wurzel, name), intern)
                erweiterungen.add(pfad.replace("\\", ".").split(".cp3")[0].removesuffix(".pyd"))
    # Was PyInstaller fuer den Start direkt in die EXE legt (struct u. a.)
    with open(os.path.join(BUILD, "silberkorn", "PKG-00.toc"), encoding="utf-8") as datei:
        vorhanden |= {e[0] for e in ast.literal_eval(datei.read())[2] if e[2] == "PYMODULE"}
    vorhanden |= erweiterungen | set(sys.builtin_module_names)
    return [m for m in geladen
            if m not in vorhanden
            and not m.startswith(LAUFZEIT_MODULE)
            and m.rsplit(".", 1)[0] not in erweiterungen]       # Untermodule von pybind


def lizenz_von(dist: md.Distribution) -> str:
    meta = dist.metadata
    if meta.get("License-Expression"):
        return meta["License-Expression"]
    lizenz = (meta.get("License") or "").strip()
    if lizenz and "\n" not in lizenz and len(lizenz) < 80:
        return lizenz
    klassen = [k.split("::")[-1].strip() for k in meta.get_all("Classifier") or []
               if k.startswith("License ::")]
    return ", ".join(klassen) or "siehe lizenzen/"


def ersatz_lizenz(dist: md.Distribution, ziel: str) -> int:
    """Lizenz nach LIZENZ_ERSATZ als lizenzen/<Paket>/LICENSE.txt ablegen."""
    art, wo = LIZENZ_ERSATZ[dist.metadata["Name"]]
    if art == "text":
        quelle = os.path.join(WURZEL, "LICENSES", wo)
    else:
        quelle = next(f for f in dist.files or [] if f.as_posix() == wo).locate()
    with open(quelle, encoding="utf-8") as datei:
        text = datei.read()
    if art == "kopf":                          # nur der Kommentarblock am Anfang
        zeilen = []
        for zeile in text.splitlines():
            if not zeile.startswith("#"):
                break
            zeilen.append(zeile.lstrip("#").rstrip())
        text = "\n".join(z for z in zeilen if z.strip()).strip() + "\n"
        if "copyright" not in text.lower():
            return 0
    unter = os.path.join(ziel, dist.metadata["Name"], "LICENSE.txt")
    os.makedirs(os.path.dirname(unter), exist_ok=True)
    with open(unter, "w", encoding="utf-8") as datei:
        datei.write(text)
    return 1


def lizenzen_beilegen() -> list[tuple[str, str, str]]:
    """Lizenztexte aller eingepackten Pakete nach lizenzen/<Paket>/ kopieren."""
    with open(os.path.join(BUILD, "silberkorn", "module.json"), encoding="utf-8") as datei:
        module = json.load(datei)
    zuordnung = md.packages_distributions()
    namen = sorted({d for m in module for d in zuordnung.get(m, [])}, key=str.lower)
    ziel = os.path.join(ORDNER, "lizenzen")
    liste = []
    for name in namen:
        dist = md.distribution(name)
        if dist.metadata["Name"].lower() in ("pyinstaller", "pyinstaller-hooks-contrib"):
            continue                           # nur der Bootloader - Lizenz s. u.
        kopiert = 0
        for datei in dist.files or []:
            teile = datei.parts
            if (teile[0].endswith(".dist-info")
                    and (LIZENZDATEI.search(teile[-1]) or "licenses" in teile[1:-1])):
                unter = os.path.join(ziel, dist.metadata["Name"], *teile[1:])
                os.makedirs(os.path.dirname(unter), exist_ok=True)
                shutil.copyfile(datei.locate(), unter)
                kopiert += 1
        if not kopiert and dist.metadata["Name"] in LIZENZ_ERSATZ:
            kopiert = ersatz_lizenz(dist, ziel)
        if not kopiert:
            sys.exit(f"Keine Lizenzdatei gefunden: {name}")
        liste.append((dist.metadata["Name"], dist.version, lizenz_von(dist)))

    # Was nicht in den dist-info-Ordnern steht
    import imagecodecs
    import onnxruntime
    zusatz = [(os.path.join(os.path.dirname(onnxruntime.__file__), "ThirdPartyNotices.txt"),
               os.path.join(ziel, "onnxruntime-gpu", "ThirdPartyNotices.txt"))]
    zusatz += [(os.path.join(os.path.dirname(imagecodecs.__file__), "licenses", n),
                os.path.join(ziel, "imagecodecs", "codecs", n)) for n in IMAGECODECS_LIZENZEN]
    zusatz += [(os.path.join(sys.base_prefix, "LICENSE.txt"),
                os.path.join(ziel, "Python", "LICENSE.txt"))]
    for quelle, unter in zusatz:
        os.makedirs(os.path.dirname(unter), exist_ok=True)
        shutil.copyfile(quelle, unter)
    for name in ("LGPL-3.0.txt", "GPL-3.0.txt"):
        unter = os.path.join(ziel, md.distribution("PySide6-Essentials").metadata["Name"], name)
        shutil.copyfile(os.path.join(WURZEL, "LICENSES", name), unter)
    return liste


def drittanbieter_schreiben(liste: list[tuple[str, str, str]]):
    import PyInstaller
    from PySide6 import __version__ as pyside
    from PySide6.QtCore import qVersion
    qt = qVersion()
    reihe = "\n".join(f"  {n:<28} {v:<14} {lz}" for n, v, lz in liste)
    text = f"""Silberkorn {version()} - Drittanbieter-Komponenten / third-party components
=============================================================================

Silberkorn selbst steht unter der MIT-Lizenz (LICENSE); Hinweise zu fremden
Bestandteilen stehen in NOTICE. Dieser Ordner enthaelt ausserdem die folgenden
Python-Pakete. Ihre Lizenztexte liegen im Ordner lizenzen/.

This folder also contains the following Python packages. Their licence texts
are in the folder lizenzen/.

  Python {sys.version.split()[0]:<21} PSF License (lizenzen/Python/LICENSE.txt)
{reihe}

Qt und PySide6 / Qt and PySide6
-------------------------------
Qt {qt} und PySide6 {pyside} sind unveraendert enthalten und stehen hier unter
der GNU Lesser General Public License v3 (lizenzen/PySide6_Essentials/).
Die Qt-Bibliotheken liegen als eigene Dateien im Ordner _internal/PySide6 und
lassen sich durch eigene, kompatible Fassungen ersetzen. Den Quelltext gibt es
bei The Qt Company:
  https://download.qt.io/official_releases/qt/{qt.rsplit(".", 1)[0]}/{qt}/
  https://download.qt.io/official_releases/QtForPython/pyside6/
Die von Qt selbst verwendeten Fremdbibliotheken nennt
  https://doc.qt.io/qt-6/licenses-used-in-qt.html

Qt {qt} and PySide6 {pyside} are included unmodified under the GNU Lesser
General Public License v3. The Qt libraries are separate files in
_internal/PySide6 and can be replaced with compatible versions of your own.
The source code is available from The Qt Company at the addresses above.

NVIDIA
------
CUDA und cuDNN sind NICHT enthalten. Silberkorn laedt sie beim ersten Start
nach Zustimmung zu NVIDIAs Lizenzbedingungen aus NVIDIAs offiziellen Paketen
vom Python Package Index in den Ordner nvidia/.

CUDA and cuDNN are NOT included. Silberkorn downloads them at first start,
after you accept NVIDIA's licence terms, from NVIDIA's official packages on
the Python Package Index into the folder nvidia/.

PyInstaller
-----------
Gebaut mit PyInstaller {PyInstaller.__version__}; der Startcode (Bootloader)
steht unter der GPL-2.0 mit einer Ausnahme, die die Weitergabe ohne
Einschraenkung erlaubt (lizenzen/PyInstaller/).
Built with PyInstaller; its bootloader is GPL-2.0 with an exception that
allows distribution without restriction.
"""
    with open(os.path.join(ORDNER, "DRITTANBIETER.txt"), "w", encoding="utf-8") as datei:
        datei.write(text)
    pyi = md.distribution("pyinstaller")
    for datei in pyi.files or []:
        if datei.parts[0].endswith(".dist-info") and LIZENZDATEI.search(datei.parts[-1]):
            unter = os.path.join(ORDNER, "lizenzen", "PyInstaller", datei.parts[-1])
            os.makedirs(os.path.dirname(unter), exist_ok=True)
            shutil.copyfile(datei.locate(), unter)


def projekt_dateien():
    for name in ("LICENSE", "NOTICE", "README.md", "README.en.md", "CHANGELOG.md"):
        shutil.copyfile(os.path.join(WURZEL, name), os.path.join(ORDNER, name))
    shutil.copytree(os.path.join(WURZEL, "LICENSES"), os.path.join(ORDNER, "LICENSES"),
                    dirs_exist_ok=True)


def packen() -> str:
    basis = os.path.join(DIST, f"Silberkorn-{version()}-win64")
    zip_pfad = shutil.make_archive(basis, "zip", DIST, "Silberkorn")
    h = hashlib.sha256()
    with open(zip_pfad, "rb") as datei:
        for block in iter(lambda: datei.read(1 << 20), b""):
            h.update(block)
    with open(zip_pfad + ".sha256", "w", encoding="ascii") as datei:
        datei.write(f"{h.hexdigest()}  {os.path.basename(zip_pfad)}\n")
    return zip_pfad


def groesse(pfad: str) -> int:
    return sum(os.path.getsize(os.path.join(w, n)) for w, _o, d in os.walk(pfad) for n in d)


def main():
    umgebung_pruefen()
    pyinstaller()
    funde = verbotene_dateien()
    if funde:
        sys.exit("Diese Dateien duerfen nicht in die EXE:\n  " + "\n  ".join(funde))
    fehlend = module_pruefen()
    if fehlend:
        sys.exit("Diese Module laedt das Programm, sie fehlen aber in der EXE "
                 "(in silberkorn.spec unter VERSTECKT eintragen):\n  " + "\n  ".join(fehlend))
    liste = lizenzen_beilegen()
    drittanbieter_schreiben(liste)
    projekt_dateien()
    zip_pfad = packen()
    print(f"\nOrdner: {ORDNER} ({groesse(ORDNER) / 2**20:.0f} MB)")
    print(f"ZIP:    {zip_pfad} ({os.path.getsize(zip_pfad) / 2**20:.0f} MB)")
    with open(zip_pfad + ".sha256", encoding="ascii") as datei:
        print(f"SHA-256 {datei.read().split()[0]}")


if __name__ == "__main__":
    main()
