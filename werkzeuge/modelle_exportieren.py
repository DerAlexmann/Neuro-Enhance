"""
Real-ESRGAN-Gewichte nach ONNX wandeln - Entwicklerwerkzeug, nicht Teil der App

Laedt die offiziellen Gewichte aus dem Real-ESRGAN-Release (BSD-3-Clause) in
die Original-Netzarchitekturen und exportiert sie als ONNX. Die Architekturen
sind hier nachgebaut, damit weder BasicSR noch das Real-ESRGAN-Paket noetig
sind; die Schluessel der Gewichte stimmen mit den Originalen ueberein, und
load_state_dict(strict=True) prueft das.

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
und gibt Groesse und SHA-256 jeder Datei aus.

Aufruf (braucht PyTorch, nur zum Entwickeln):
    python werkzeuge/modelle_exportieren.py

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

PROJEKT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUELLEN = os.path.join(PROJEKT, "modelle", "quellen")
ZIEL = os.path.join(PROJEKT, "modelle")


class SRVGGNetCompact(nn.Module):
    """Kompaktes Netz aus Real-ESRGAN (realesr-general-x4v3)."""

    def __init__(self, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList([nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_feat)])
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_feat))
        self.body.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for schicht in self.body:
            out = schicht(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """ESRGAN-Generator aus Real-ESRGAN (RealESRGAN_x4plus)."""

    def __init__(self, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def gewichte(datei: str) -> dict:
    daten = torch.load(os.path.join(QUELLEN, datei), map_location="cpu", weights_only=True)
    for schluessel in ("params_ema", "params"):
        if schluessel in daten:
            return daten[schluessel]
    return daten


def exportieren(netz: nn.Module, name: str, gewichte_als_eingang: bool = False) -> str:
    """ONNX-Export; mit gewichte_als_eingang werden die Parameter zu Eingaengen des Netzes."""
    netz.eval()
    pfad = os.path.join(ZIEL, name)
    beispiel = torch.rand(1, 3, 64, 64)
    torch.onnx.export(
        netz, (beispiel,), pfad, input_names=["eingabe"], output_names=["ausgabe"],
        dynamic_axes={"eingabe": {2: "hoehe", 3: "breite"}, "ausgabe": {2: "h4", 3: "b4"}},
        opset_version=17, dynamo=False, do_constant_folding=False,
        export_params=not gewichte_als_eingang, keep_initializers_as_inputs=False)
    return pfad


def pruefsumme(pfad: str) -> str:
    h = hashlib.sha256()
    with open(pfad, "rb") as datei:
        for block in iter(lambda: datei.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
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
    for pfad in ergebnisse:
        print(f"{os.path.basename(pfad):34s} {os.path.getsize(pfad) / 2**20:6.1f} MB  "
              f"{pruefsumme(pfad)}")


if __name__ == "__main__":
    main()
