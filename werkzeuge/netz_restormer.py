"""
Restormer nachgebaut - fuer den ONNX-Export, nicht Teil der App

Restormer (Syed Waqas Zamir u. a., https://github.com/swz30/Restormer) in der
Fassung fuer Fokus-Unschaerfe. Die Namen der Module entsprechen dem Original,
damit load_state_dict(strict=True) die offiziellen Gewichte laedt.

Geaendert gegenueber dem Original, alles rechnet dasselbe: ohne einops; die
LayerNorm rechnet direkt ueber die Kanaele statt nach (B, HW, C) umzusortieren;
die Normierung von Abfragen und Schluesseln ist FP16-fest umgestellt (siehe
MDTA.normieren).

Als Bearbeitung des Codes von Restormer steht diese Datei unter der MIT-Lizenz
von Restormer (Text in LICENSES/MIT-Restormer.txt).

Licensed under MIT License
Copyright (c) 2022 Syed Waqas Zamir and contributors (Restormer)
Copyright 2026 Alexander Unverhau (Aenderungen)
Created with assistance of Claude AI
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


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
