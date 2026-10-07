"""
KI-Gewichte nach ONNX wandeln - Entwicklerwerkzeug, nicht Teil der App

Laedt die offiziellen Gewichte aus den Releases von Real-ESRGAN (BSD-3-Clause),
SCUNet (Apache-2.0) und Restormer (MIT) in die Original-Netzarchitekturen und
exportiert sie als ONNX. Die Architekturen sind hier nachgebaut, damit weder
BasicSR noch einops oder timm noetig sind; die Schluessel der Gewichte stimmen
mit den Originalen ueberein, und load_state_dict(strict=True) prueft das.

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


class RestormerNorm(nn.Module):
    """LayerNorm ueber die Kanaele je Pixel (WithBias) - ohne Umsortieren nach (B, HW, C)."""

    def __init__(self, kanaele):
        super().__init__()
        self.body = nn.Module()
        self.body.weight = nn.Parameter(torch.ones(kanaele))
        self.body.bias = nn.Parameter(torch.zeros(kanaele))

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / torch.sqrt(var + 1e-5)
        return y * self.body.weight.view(1, -1, 1, 1) + self.body.bias.view(1, -1, 1, 1)


class GDFN(nn.Module):
    def __init__(self, dim, faktor=2.66):
        super().__init__()
        versteckt = int(dim * faktor)
        self.project_in = nn.Conv2d(dim, 2 * versteckt, 1, bias=False)
        self.dwconv = nn.Conv2d(2 * versteckt, 2 * versteckt, 3, 1, 1, groups=2 * versteckt,
                                bias=False)
        self.project_out = nn.Conv2d(versteckt, dim, 1, bias=False)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class MDTA(nn.Module):
    """Aufmerksamkeit ueber die Kanaele statt ueber die Pixel (transponiert)."""

    def __init__(self, dim, koepfe):
        super().__init__()
        self.koepfe = koepfe
        self.temperature = nn.Parameter(torch.ones(koepfe, 1, 1))
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(3 * dim, 3 * dim, 3, 1, 1, groups=3 * dim, bias=False)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=False)

    @staticmethod
    def normieren(x, h, w):
        """Wie F.normalize ueber alle h * w Pixel, aber FP16-fest: Die Summe der
        Quadrate - und schon die Pixelzahl selbst - uebersteigt bei grossen Kacheln
        das FP16-Maximum (65504). Erst auf den groessten Betrag teilen, dann den
        Mittelwert der Quadrate nehmen; die Wurzel der Pixelzahl kommt getrennt
        aus Hoehe und Breite."""
        groesster = x.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        y = x / groesster
        norm = torch.sqrt(y.pow(2).mean(-1, keepdim=True)) * h ** 0.5 * w ** 0.5
        return y / norm.clamp_min(1e-12)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q, k, v = (t.reshape(b, self.koepfe, c // self.koepfe, h * w) for t in (q, k, v))
        q = self.normieren(q, h, w)
        k = self.normieren(k, h, w)
        gewicht = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        return self.project_out((gewicht @ v).reshape(b, c, h, w))


class RestormerBlock(nn.Module):
    def __init__(self, dim, koepfe):
        super().__init__()
        self.norm1 = RestormerNorm(dim)
        self.attn = MDTA(dim, koepfe)
        self.norm2 = RestormerNorm(dim)
        self.ffn = GDFN(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class Restormer(nn.Module):
    """Restormer (dim 48, Bloecke 4-6-6-8, 4 Verfeinerungsbloecke) zum Entwackeln und
    gegen Fokus-Unschaerfe. Hoehe und Breite muessen Vielfache von 8 sein."""

    def __init__(self, dim=48, bloecke=(4, 6, 6, 8), verfeinerung=4, koepfe=(1, 2, 4, 8)):
        super().__init__()

        def stufe(anzahl, d, k):
            return nn.Sequential(*[RestormerBlock(d, k) for _ in range(anzahl)])

        def runter(d):
            return nn.Sequential(nn.Conv2d(d, d // 2, 3, 1, 1, bias=False), nn.PixelUnshuffle(2))

        def hoch(d):
            return nn.Sequential(nn.Conv2d(d, 2 * d, 3, 1, 1, bias=False), nn.PixelShuffle(2))

        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, 3, 1, 1, bias=False)
        self.encoder_level1 = stufe(bloecke[0], dim, koepfe[0])
        self.down1_2 = nn.Module()
        self.down1_2.body = runter(dim)
        self.encoder_level2 = stufe(bloecke[1], 2 * dim, koepfe[1])
        self.down2_3 = nn.Module()
        self.down2_3.body = runter(2 * dim)
        self.encoder_level3 = stufe(bloecke[2], 4 * dim, koepfe[2])
        self.down3_4 = nn.Module()
        self.down3_4.body = runter(4 * dim)
        self.latent = stufe(bloecke[3], 8 * dim, koepfe[3])
        self.up4_3 = nn.Module()
        self.up4_3.body = hoch(8 * dim)
        self.reduce_chan_level3 = nn.Conv2d(8 * dim, 4 * dim, 1, bias=False)
        self.decoder_level3 = stufe(bloecke[2], 4 * dim, koepfe[2])
        self.up3_2 = nn.Module()
        self.up3_2.body = hoch(4 * dim)
        self.reduce_chan_level2 = nn.Conv2d(4 * dim, 2 * dim, 1, bias=False)
        self.decoder_level2 = stufe(bloecke[1], 2 * dim, koepfe[1])
        self.up2_1 = nn.Module()
        self.up2_1.body = hoch(2 * dim)
        self.decoder_level1 = stufe(bloecke[0], 2 * dim, koepfe[0])
        self.refinement = stufe(verfeinerung, 2 * dim, koepfe[0])
        self.output = nn.Conv2d(2 * dim, 3, 3, 1, 1, bias=False)

    def forward(self, eingang):
        e1 = self.encoder_level1(self.patch_embed.proj(eingang))
        e2 = self.encoder_level2(self.down1_2.body(e1))
        e3 = self.encoder_level3(self.down2_3.body(e2))
        x = self.latent(self.down3_4.body(e3))
        x = self.decoder_level3(self.reduce_chan_level3(torch.cat([self.up4_3.body(x), e3], 1)))
        x = self.decoder_level2(self.reduce_chan_level2(torch.cat([self.up3_2.body(x), e2], 1)))
        x = self.decoder_level1(torch.cat([self.up2_1.body(x), e1], 1))
        return self.output(self.refinement(x)) + eingang


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


EXPORTE = {"realesrgan": realesrgan, "scunet": scunet, "restormer": restormer,
           "birefnet": birefnet, "sam2": sam2, "lama": lama, "tiefe": tiefe}


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
