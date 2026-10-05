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
  scunet-color-real-psnr.onnx      Entrauschen (SCUNet, Apache-2.0) - Hoehe und
                                   Breite muessen Vielfache von 64 sein
und gibt Groesse und SHA-256 jeder Datei aus.

Aufruf (braucht PyTorch, nur zum Entwickeln):
    python werkzeuge/modelle_exportieren.py              alle Modelle
    python werkzeuge/modelle_exportieren.py scunet       nur ausgewaehlte
                                                         (realesrgan, scunet)

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


# ----------------------------------------------------------------------
# SCUNet - Kai Zhang u. a., "Practical Blind Image Denoising via Swin-Conv-UNet
# and Data Synthesis", https://github.com/cszn/SCUNet (Apache-2.0). Nachgebaut
# ohne einops und timm; die Maske der verschobenen Fenster entsteht hier aus
# Koordinaten statt aus festen Groessen, damit das ONNX-Modell jede Bildgroesse
# annimmt. Hoehe und Breite muessen Vielfache von 64 sein - das Auffuellen, das
# im Original im Netz steckt, uebernimmt die App beim Kacheln.


class WMSA(nn.Module):
    """Selbstaufmerksamkeit in (verschobenen) Fenstern wie im Swin Transformer."""

    def __init__(self, dim, head_dim, window_size, verschoben):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.n_heads = dim // head_dim
        self.p = window_size
        self.verschoben = verschoben
        self.embedding_layer = nn.Linear(dim, 3 * dim, bias=True)
        self.relative_position_params = nn.Parameter(
            torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size - 1))
        self.linear = nn.Linear(dim, dim)
        koord = torch.tensor([[i, j] for i in range(window_size) for j in range(window_size)])
        bezug = koord[:, None, :] - koord[None, :, :] + window_size - 1
        self.register_buffer("bezug_y", bezug[:, :, 0].contiguous(), persistent=False)
        self.register_buffer("bezug_x", bezug[:, :, 1].contiguous(), persistent=False)

    def forward(self, x):                                     # x: b h w c
        p, s = self.p, self.p // 2
        if self.verschoben:
            x = torch.roll(x, shifts=(-s, -s), dims=(1, 2))
        b, hoehe, breite, c = x.shape
        fh, fb = hoehe // p, breite // p
        x = x.reshape(b, fh, p, fb, p, c).permute(0, 1, 3, 2, 4, 5).reshape(b, fh * fb, p * p, c)
        qkv = self.embedding_layer(x)
        qkv = qkv.reshape(b, fh * fb, p * p, 3 * self.n_heads, self.head_dim)
        q, k, v = qkv.permute(3, 0, 1, 2, 4).chunk(3, dim=0)  # je: kopf b fenster pixel c
        sim = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        bias = self.relative_position_params[:, self.bezug_y, self.bezug_x]
        sim = sim + bias[:, None, None]
        if self.verschoben:
            # Nach dem Rollen grenzen im letzten Fenster jeder Zeile und Spalte
            # Pixel aneinander, die im Bild nicht benachbart sind - sie duerfen
            # einander nicht sehen.
            zeilen = (torch.arange(hoehe, device=x.device) >= hoehe - s).long()
            spalten = (torch.arange(breite, device=x.device) >= breite - s).long()
            bereich = zeilen[:, None] * 2 + spalten[None, :]
            bereich = bereich.reshape(fh, p, fb, p).permute(0, 2, 1, 3).reshape(fh * fb, p * p)
            maske = bereich[:, :, None] != bereich[:, None, :]
            sim = sim.masked_fill(maske, float("-inf"))
        aus = torch.matmul(torch.softmax(sim, dim=-1), v)    # kopf b fenster pixel c
        aus = aus.permute(1, 2, 3, 0, 4).reshape(b, fh * fb, p * p, self.n_heads * self.head_dim)
        aus = self.linear(aus)
        aus = aus.reshape(b, fh, fb, p, p, c).permute(0, 1, 3, 2, 4, 5).reshape(b, hoehe, breite, c)
        if self.verschoben:
            aus = torch.roll(aus, shifts=(s, s), dims=(1, 2))
        return aus


class SwinBlock(nn.Module):
    def __init__(self, dim, head_dim, window_size, verschoben):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.msa = WMSA(dim, head_dim, window_size, verschoben)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        x = x + self.msa(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ConvTransBlock(nn.Module):
    """Halb Faltung, halb Swin-Block - der Kern von SCUNet."""

    def __init__(self, conv_dim, trans_dim, head_dim, window_size, verschoben):
        super().__init__()
        self.conv_dim, self.trans_dim = conv_dim, trans_dim
        self.trans_block = SwinBlock(trans_dim, head_dim, window_size, verschoben)
        self.conv1_1 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)
        self.conv_block = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False), nn.ReLU(True),
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False))

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = self.trans_block(trans_x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.conv1_2(torch.cat((conv_x, trans_x), dim=1))


class SCUNet(nn.Module):
    """Swin-Conv-UNet zum Entrauschen (scunet_color_real_psnr: config 4 x 7, dim 64)."""

    def __init__(self, config=(4, 4, 4, 4, 4, 4, 4), dim=64):
        super().__init__()
        kopf, fenster = 32, 8

        def stufe(anzahl, d):
            # abwechselnd normale und verschobene Fenster, wie im Original
            return [ConvTransBlock(d // 2, d // 2, kopf, fenster, i % 2 == 1)
                    for i in range(anzahl)]

        self.m_head = nn.Sequential(nn.Conv2d(3, dim, 3, 1, 1, bias=False))
        self.m_down1 = nn.Sequential(*stufe(config[0], dim),
                                     nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False))
        self.m_down2 = nn.Sequential(*stufe(config[1], 2 * dim),
                                     nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False))
        self.m_down3 = nn.Sequential(*stufe(config[2], 4 * dim),
                                     nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False))
        self.m_body = nn.Sequential(*stufe(config[3], 8 * dim))
        self.m_up3 = nn.Sequential(nn.ConvTranspose2d(8 * dim, 4 * dim, 2, 2, 0, bias=False),
                                   *stufe(config[4], 4 * dim))
        self.m_up2 = nn.Sequential(nn.ConvTranspose2d(4 * dim, 2 * dim, 2, 2, 0, bias=False),
                                   *stufe(config[5], 2 * dim))
        self.m_up1 = nn.Sequential(nn.ConvTranspose2d(2 * dim, dim, 2, 2, 0, bias=False),
                                   *stufe(config[6], dim))
        self.m_tail = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1, bias=False))

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        return self.m_tail(x + x1)


def gewichte(datei: str) -> dict:
    daten = torch.load(os.path.join(QUELLEN, datei), map_location="cpu", weights_only=True)
    for schluessel in ("params_ema", "params"):
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


EXPORTE = {"realesrgan": realesrgan, "scunet": scunet}


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
