"""
KI-Gewichte nach ONNX wandeln - Entwicklerwerkzeug, nicht Teil der App

Laedt die offiziellen Gewichte in die nachgebauten Original-Netzarchitekturen
und exportiert sie als ONNX. Die Nachbauten liegen je Herkunft in eigenen
Dateien netz_*.py, jede unter der Lizenz ihres Vorbilds; die Schluessel der
Gewichte stimmen mit den Originalen ueberein, und load_state_dict(strict=True)
prueft das. Diese Datei selbst enthaelt keinen fremden Code.

Erzeugt in modelle/:
  realesr-general-x4v3.onnx        schnelles 4x-Modell (SRVGGNetCompact) - ohne
                                   Gewichte: sie sind Eingaenge des Netzes
  realesr-general-x4v3.npz         Gewichte beider Zwillinge unter den Namen dieser
  realesr-general-wdn-x4v3.npz     Eingaenge: starkes und schwaches Entrauschen.
                                   Die App mischt sie auf der GPU fuer den
                                   Entrauschregler und reicht sie bei jeder Kachel
                                   mit durch. (Eingebaute Gewichte laesst ONNX
                                   Runtime bei der GPU-Ausfuehrung nicht ersetzen.)
  realesrgan-x4plus.onnx           grosses 4x-Modell (RRDBNet)
  scunet-color-real-psnr.onnx      Entrauschen (SCUNet, Apache-2.0) - Hoehe und
                                   Breite muessen Vielfache von 64 sein
  restormer-defocus.onnx           Schaerfen gegen Fokus-Unschaerfe (Restormer,
                                   MIT) - Vielfache von 8
  birefnet-lite-2k.onnx            Motiv freistellen (BiRefNet, MIT) - feste
                                   Eingabe 2560 x 1440 wie im Training; das Netz
                                   ist in werkzeuge/netz_birefnet.py nachgebaut
  ddcolor-tiny.onnx                Kolorieren (DDColor tiny, Apache-2.0) - feste
                                   Eingabe 512 x 512, Graubild -> Farbanteile a, b
und gibt Groesse und SHA-256 jeder Datei aus.

Aufruf (braucht PyTorch, nur zum Entwickeln):
    python werkzeuge/modelle_exportieren.py              alle Modelle
    python werkzeuge/modelle_exportieren.py scunet       nur ausgewaehlte
                                                         (realesrgan, scunet, restormer,
                                                         birefnet)

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from netz_realesrgan import SRVGGNetCompact  # noqa: E402
from netz_restormer import Restormer  # noqa: E402
from netz_rrdbnet import RRDBNet  # noqa: E402
from netz_scunet import SCUNet  # noqa: E402
from torch import nn  # noqa: E402

PROJEKT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUELLEN = os.path.join(PROJEKT, "modelle", "quellen")
ZIEL = os.path.join(PROJEKT, "modelle")


def gewichte(datei: str) -> dict:
    daten = torch.load(os.path.join(QUELLEN, datei), map_location="cpu", weights_only=True)
    for schluessel in ("params_ema", "params", "model"):
        if schluessel in daten:
            return daten[schluessel]
    return daten


def exportieren(netz: nn.Module, name: str, gewichte_als_eingang: bool = False,
                beispiel_groesse: int = 64) -> str:
    """ONNX-Export; mit gewichte_als_eingang werden die Parameter zu Eingaengen des Netzes."""
    netz.eval()
    pfad = os.path.join(ZIEL, name)
    beispiel = torch.rand(1, 3, beispiel_groesse, beispiel_groesse)
    torch.onnx.export(
        netz, (beispiel,), pfad, input_names=["eingabe"], output_names=["ausgabe"],
        dynamic_axes={"eingabe": {2: "hoehe", 3: "breite"}, "ausgabe": {2: "ah", 3: "ab"}},
        opset_version=17, dynamo=False, do_constant_folding=False,
        export_params=not gewichte_als_eingang, keep_initializers_as_inputs=False)
    return pfad


def pruefsumme(pfad: str) -> str:
    h = hashlib.sha256()
    with open(pfad, "rb") as datei:
        for block in iter(lambda: datei.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def realesrgan() -> list[str]:
    ergebnisse = []

    kompakt = SRVGGNetCompact()
    kompakt.load_state_dict(gewichte("realesr-general-x4v3.pth"), strict=True)
    ergebnisse.append(exportieren(kompakt, "realesr-general-x4v3.onnx",
                                  gewichte_als_eingang=True))

    # Beide Zwillinge als Gewichtssaetze: gleiche Architektur, gleiche Namen
    for name in ("realesr-general-x4v3", "realesr-general-wdn-x4v3"):
        satz = gewichte(f"{name}.pth")
        SRVGGNetCompact().load_state_dict(satz, strict=True)
        npz = os.path.join(ZIEL, f"{name}.npz")
        np.savez(npz, **{k: v.numpy().astype(np.float32) for k, v in satz.items()})
        ergebnisse.append(npz)

    gross = RRDBNet()
    gross.load_state_dict(gewichte("RealESRGAN_x4plus.pth"), strict=True)
    ergebnisse.append(exportieren(gross, "realesrgan-x4plus.onnx"))

    # Pruefen: ONNX rechnet wie PyTorch
    import onnxruntime as ort
    probe = torch.rand(1, 3, 48, 40)
    for netz, pfad, satz in ((kompakt, ergebnisse[0], ergebnisse[1]),
                             (gross, ergebnisse[3], None)):
        with torch.no_grad():
            soll = netz(probe).numpy()
        sitzung = ort.InferenceSession(pfad, providers=["CPUExecutionProvider"])
        eingaben = {"eingabe": probe.numpy()}
        if satz:
            gespeichert = np.load(satz)
            namen = {e.name for e in sitzung.get_inputs()} - {"eingabe"}
            assert namen == set(gespeichert.files), "Eingaenge und Gewichte passen nicht"
            eingaben.update({n: gespeichert[n] for n in namen})
        ist = sitzung.run(None, eingaben)[0]
        print(f"{os.path.basename(pfad)}: ONNX gegen PyTorch, groesste Abweichung "
              f"{np.abs(ist - soll).max():.2e}")
    return ergebnisse


def scunet() -> list[str]:
    netz = SCUNet()
    netz.load_state_dict(gewichte("scunet_color_real_psnr.pth"), strict=True)
    pfad = exportieren(netz, "scunet-color-real-psnr.onnx", beispiel_groesse=128)

    # Pruefen in einer anderen Groesse als beim Export - die Fenstermaske muss mitwachsen
    import onnxruntime as ort
    probe = torch.rand(1, 3, 192, 256)
    with torch.no_grad():
        soll = netz(probe).numpy()
    sitzung = ort.InferenceSession(pfad, providers=["CPUExecutionProvider"])
    ist = sitzung.run(None, {"eingabe": probe.numpy()})[0]
    print(f"{os.path.basename(pfad)}: ONNX gegen PyTorch, groesste Abweichung "
          f"{np.abs(ist - soll).max():.2e}")
    return [pfad]


def restormer() -> list[str]:
    netz = Restormer()
    netz.load_state_dict(gewichte("single_image_defocus_deblurring.pth"), strict=True)
    pfad = exportieren(netz, "restormer-defocus.onnx", beispiel_groesse=64)

    import onnxruntime as ort
    probe = torch.rand(1, 3, 96, 136)
    with torch.no_grad():
        soll = netz(probe).numpy()
    sitzung = ort.InferenceSession(pfad, providers=["CPUExecutionProvider"])
    ist = sitzung.run(None, {"eingabe": probe.numpy()})[0]
    print(f"{os.path.basename(pfad)}: ONNX gegen PyTorch, groesste Abweichung "
          f"{np.abs(ist - soll).max():.2e}")
    return [pfad]


def birefnet() -> list[str]:
    from netz_birefnet import BiRefNet
    daten = gewichte("BiRefNet_lite-general-2K-epoch_232.pth")
    netz = BiRefNet("swin_v1_t")
    netz.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in daten.items()},
                         strict=True)
    netz.eval()
    pfad = os.path.join(ZIEL, "birefnet-lite-2k.onnx")
    # Feste Groesse wie im Training (Breite 2560, Hoehe 1440): Fenstermasken und
    # Positionsindizes werden so zu Konstanten
    torch.onnx.export(netz, (torch.rand(1, 3, 1440, 2560),), pfad, input_names=["eingabe"],
                      output_names=["ausgabe"], opset_version=17, dynamo=False,
                      do_constant_folding=True)

    import onnxruntime as ort
    probe = torch.rand(1, 3, 1440, 2560)
    with torch.no_grad():
        soll = netz(probe).numpy()
    sitzung = ort.InferenceSession(pfad, providers=["CPUExecutionProvider"])
    ist = sitzung.run(None, {"eingabe": probe.numpy()})[0]
    print(f"{os.path.basename(pfad)}: ONNX gegen PyTorch, groesste Abweichung "
          f"{np.abs(ist - soll).max():.2e}")
    return [pfad]


def sam2() -> list[str]:
    from netz_sam2 import Dekodierer, Kodierer, Sam2
    daten = gewichte("sam2.1_hiera_small.pt")
    sam = Sam2()
    sam.load_state_dict(Sam2.gewichte(daten), strict=True)
    sam.eval()
    kodierer, dekodierer = Kodierer(sam).eval(), Dekodierer(sam).eval()
    pfad_k = os.path.join(ZIEL, "sam2.1-small-kodierer.onnx")
    pfad_d = os.path.join(ZIEL, "sam2.1-small-dekodierer.onnx")
    bild = torch.rand(1, 3, 1024, 1024)
    torch.onnx.export(kodierer, (bild,), pfad_k, input_names=["eingabe"],
                      output_names=["merkmale", "s0", "s1"], opset_version=17, dynamo=False,
                      do_constant_folding=True)
    with torch.no_grad():
        merkmale, s0, s1 = kodierer(bild)
    punkte, etiketten = torch.tensor([[[300.0, 420.0], [700.0, 600.0]]]), torch.tensor([[1.0, 0.0]])
    maske, mit = torch.randn(1, 1, 256, 256), torch.tensor([1.0])
    eingaben = (merkmale, s0, s1, punkte, etiketten, maske, mit)
    namen = ["merkmale", "s0", "s1", "punkte", "etiketten", "maske", "mit_maske"]
    # Die Zahl der Klicks ist frei
    torch.onnx.export(dekodierer, eingaben, pfad_d, input_names=namen,
                      output_names=["masken", "guete"], opset_version=17, dynamo=False,
                      do_constant_folding=True,
                      dynamic_axes={"punkte": {1: "klicks"}, "etiketten": {1: "klicks"}})

    import onnxruntime as ort
    k = ort.InferenceSession(pfad_k, providers=["CPUExecutionProvider"])
    ist = k.run(None, {"eingabe": bild.numpy()})
    soll = (merkmale, s0, s1)
    abweichung = max(np.abs(i - s.numpy()).max() for i, s in zip(ist, soll, strict=True))
    print(f"{os.path.basename(pfad_k)}: ONNX gegen PyTorch, groesste Abweichung "
          f"{abweichung:.2e}")
    d = ort.InferenceSession(pfad_d, providers=["CPUExecutionProvider"])
    for n in (1, 3):                         # andere Klickzahl als beim Export
        probe = list(eingaben)
        probe[3], probe[4] = torch.rand(1, n, 2) * 1024, torch.ones(1, n)
        with torch.no_grad():
            soll = dekodierer(*probe)
        ist = d.run(None, {name: t.numpy() for name, t in zip(namen, probe, strict=True)})
        abweichung = max(np.abs(i - s.numpy()).max() for i, s in zip(ist, soll, strict=True))
        print(f"{os.path.basename(pfad_d)} ({n} Klicks): ONNX gegen PyTorch, groesste "
              f"Abweichung {abweichung:.2e}")
    return [pfad_k, pfad_d]


def lama() -> list[str]:
    from checkpoint_lesen import gewichte as lightning_gewichte
    from netz_lama import Lama
    daten = lightning_gewichte(os.path.join(QUELLEN, "big-lama", "models", "best.ckpt"))
    netz = Lama()
    netz.generator.load_state_dict(
        {k[len("generator."):]: v for k, v in daten.items() if k.startswith("generator.")},
        strict=True)
    netz.eval()
    pfad = os.path.join(ZIEL, "big-lama.onnx")
    bild, maske = torch.rand(1, 3, 64, 96), torch.zeros(1, 1, 64, 96)
    maske[..., 16:40, 24:56] = 1
    achsen = {2: "hoehe", 3: "breite"}
    torch.onnx.export(netz, (bild, maske), pfad, input_names=["eingabe", "maske"],
                      output_names=["ausgabe"], opset_version=17, dynamo=False,
                      do_constant_folding=True,
                      dynamic_axes={"eingabe": achsen, "maske": achsen, "ausgabe": achsen})

    import onnxruntime as ort
    sitzung = ort.InferenceSession(pfad, providers=["CPUExecutionProvider"])
    for h, w in ((128, 200), (256, 256)):                # andere Groessen als beim Export
        probe, loch = torch.rand(1, 3, h, w), torch.zeros(1, 1, h, w)
        loch[..., h // 4:h // 2, w // 3:w // 2] = 1
        with torch.no_grad():
            soll = netz(probe, loch).numpy()
        ist = sitzung.run(None, {"eingabe": probe.numpy(), "maske": loch.numpy()})[0]
        print(f"{os.path.basename(pfad)} ({h} x {w}): ONNX gegen PyTorch, groesste "
              f"Abweichung {np.abs(ist - soll).max():.2e}")
    return [pfad]


def tiefe() -> list[str]:
    from netz_tiefe import BREITE, HOEHE, DepthAnythingV2
    netz = DepthAnythingV2()
    netz.load_state_dict(gewichte("depth_anything_v2_vits.pth"), strict=True)
    netz.eval()
    with torch.no_grad():
        frei = netz(torch.rand(1, 3, HOEHE, BREITE) * 0 + 0.5).numpy()
    netz.festlegen()
    pfad = os.path.join(ZIEL, "depth-anything-v2-small.onnx")
    # Feste Groesse: die interpolierte Positionseinbettung wird zur Konstante
    torch.onnx.export(netz, (torch.rand(1, 3, HOEHE, BREITE),), pfad, input_names=["eingabe"],
                      output_names=["ausgabe"], opset_version=17, dynamo=False,
                      do_constant_folding=True)

    import onnxruntime as ort
    probe = torch.rand(1, 3, HOEHE, BREITE)
    with torch.no_grad():
        soll = netz(probe).numpy()
    sitzung = ort.InferenceSession(pfad, providers=["CPUExecutionProvider"])
    ist = sitzung.run(None, {"eingabe": probe.numpy()})[0]
    with torch.no_grad():
        fest = netz(torch.rand(1, 3, HOEHE, BREITE) * 0 + 0.5).numpy()
    print(f"{os.path.basename(pfad)}: vorab festgelegte Positionen gegen interpolierte "
          f"{np.abs(fest - frei).max():.2e}; ONNX gegen PyTorch, groesste Abweichung "
          f"{np.abs(ist - soll).max():.2e} (Werte bis {np.abs(soll).max():.1f})")
    return [pfad]


def ddcolor() -> list[str]:
    from netz_ddcolor import GROESSE, DDColor
    netz = DDColor()
    netz.load_state_dict(DDColor.gewichte_einrechnen(gewichte("ddcolor_paper_tiny.bin")),
                         strict=True)
    netz.eval()
    pfad = os.path.join(ZIEL, "ddcolor-tiny.onnx")
    # Feste Groesse wie in der Pipeline von DDColor: die Positionskodierung wird Konstante
    torch.onnx.export(netz, (torch.rand(1, 3, GROESSE, GROESSE),), pfad, input_names=["eingabe"],
                      output_names=["ausgabe"], opset_version=17, dynamo=False,
                      do_constant_folding=True)

    import onnxruntime as ort
    probe = torch.rand(1, 3, GROESSE, GROESSE)
    with torch.no_grad():
        soll = netz(probe).numpy()
    sitzung = ort.InferenceSession(pfad, providers=["CPUExecutionProvider"])
    ist = sitzung.run(None, {"eingabe": probe.numpy()})[0]
    print(f"{os.path.basename(pfad)}: ONNX gegen PyTorch, groesste Abweichung "
          f"{np.abs(ist - soll).max():.2e} (Werte bis {np.abs(soll).max():.1f})")
    return [pfad]


EXPORTE = {"realesrgan": realesrgan, "scunet": scunet, "restormer": restormer,
           "birefnet": birefnet, "sam2": sam2, "lama": lama, "tiefe": tiefe, "ddcolor": ddcolor}


def main():
    auswahl = sys.argv[1:] or list(EXPORTE)
    ergebnisse = []
    for name in auswahl:
        ergebnisse += EXPORTE[name]()
    for pfad in ergebnisse:
        print(f"{os.path.basename(pfad):34s} {os.path.getsize(pfad) / 2**20:6.1f} MB  "
              f"{pruefsumme(pfad)}")


if __name__ == "__main__":
    main()
