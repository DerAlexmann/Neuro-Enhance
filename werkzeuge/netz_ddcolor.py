"""
DDColor nachgebaut - fuer den ONNX-Export, nicht Teil der App

Nachbau von DDColor (Kang u. a., ICCV 2023) in der kleinen Fassung
(ddcolor_paper_tiny: ConvNeXt-T als Encoder, Pixel- und Farbdecoder) zum
Kolorieren von Schwarzweissbildern. Das Netz bekommt ein Graubild in sRGB
(512 x 512) und liefert die Farbanteile a und b im Lab-Raum (CIE, D65).
Die Namen der Module entsprechen dem Original, damit load_state_dict(strict=True)
die offiziellen Gewichte laedt.

Nach dem Code von DDColor (https://github.com/piddnad/DDColor, Apache License 2.0,
Copyright 2023 Xiaoyang Kang u. a.), der seinerseits enthaelt:
- ConvNeXt (https://github.com/facebookresearch/ConvNeXt, MIT License,
  Copyright (c) Meta Platforms, Inc. and affiliates),
- Aufmerksamkeits- und FFN-Schichten aus Mask2Former
  (https://github.com/facebookresearch/Mask2Former, MIT License,
  Copyright (c) 2022 Meta, Inc.),
- die Positionskodierung aus DETR (https://github.com/facebookresearch/detr,
  Apache License 2.0, Copyright (c) Facebook, Inc. and its affiliates),
- U-Net-Bausteine aus DeOldify (https://github.com/jantic/DeOldify, MIT License,
  Copyright (c) 2018 Jason Antic) nach fastai (Apache License 2.0).

Geaendert gegenueber dem Original, alles rechnet dasselbe:
- Feste Eingabe 512 x 512; die Zwischenergebnisse des Encoders werden direkt
  weitergereicht statt ueber Forward-Hooks.
- Die Spektralnormierung ist beim Laden in die Gewichte eingerechnet
  (gewichte_einrechnen): Im Auswertemodus teilt PyTorch das Gewicht nur durch
  sigma = u . (W v) - das geschieht hier einmal vorab.
- Ohne Initialisierung, Dropout, Drop-Path und den Klassifikationskopf.

Als Bearbeitung des Codes von DDColor steht diese Datei - anders als der Rest
des Projekts - unter der Apache License 2.0 (Text in LICENSES/Apache-2.0.txt).
ConvNeXt, die Schichten aus Mask2Former und die Bausteine aus DeOldify stehen
zusaetzlich unter deren MIT-Lizenzen (Texte in LICENSES/MIT-ConvNeXt.txt,
LICENSES/MIT-Mask2Former.txt und LICENSES/MIT-DeOldify.txt).

Licensed under Apache License 2.0
Copyright 2023 Xiaoyang Kang u. a. (DDColor)
Copyright (c) Meta Platforms, Inc. and affiliates (ConvNeXt, DETR)
Copyright (c) 2022 Meta, Inc. (Mask2Former)
Copyright (c) 2018 Jason Antic (DeOldify)
Copyright 2026 Alexander Unverhau (Aenderungen)
Created with assistance of Claude AI
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

GROESSE = 512
MITTEL = (0.485, 0.456, 0.406)
STREUUNG = (0.229, 0.224, 0.225)
TIEFEN, BREITEN = (3, 3, 9, 3), (96, 192, 384, 768)      # ConvNeXt-T
NF = 512
ANFRAGEN = 100
EBENEN = 3
SCHICHTEN = 9
VERSTECKT = 256


# ----------------------------------------------------------------------
# ConvNeXt-T


class LayerNorm(nn.Module):
    """LayerNorm ueber die Kanaele, fuer (N, C, H, W) oder (N, H, W, C)."""

    def __init__(self, dim, eps=1e-6, kanaele_zuerst=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.kanaele_zuerst = kanaele_zuerst
        self.dim = (dim,)

    def forward(self, x):
        if not self.kanaele_zuerst:
            return F.layer_norm(x, self.dim, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = self.dwconv(x).permute(0, 2, 3, 1)
        y = self.gamma * self.pwconv2(self.act(self.pwconv1(self.norm(y))))
        return x + y.permute(0, 3, 1, 2)


class ConvNeXt(nn.Module):
    """Gibt die vier normierten Stufen zurueck (norm0 bis norm3), wie die Hooks im Original."""

    def __init__(self):
        super().__init__()
        self.downsample_layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3, BREITEN[0], kernel_size=4, stride=4),
            LayerNorm(BREITEN[0], kanaele_zuerst=True))])
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(BREITEN[i], kanaele_zuerst=True),
                nn.Conv2d(BREITEN[i], BREITEN[i + 1], kernel_size=2, stride=2)))
        self.stages = nn.ModuleList(nn.Sequential(*[Block(BREITEN[i]) for _ in range(TIEFEN[i])])
                                    for i in range(4))
        for i in range(4):
            self.add_module(f"norm{i}", LayerNorm(BREITEN[i], kanaele_zuerst=True))
        self.norm = nn.LayerNorm(BREITEN[-1], eps=1e-6)       # nur fuer die Gewichte

    def forward(self, x):
        stufen = []
        for i in range(4):
            x = self.stages[i](self.downsample_layers[i](x))
            stufen.append(getattr(self, f"norm{i}")(x))
        return stufen


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.arch = ConvNeXt()

    def forward(self, x):
        return self.arch(x)


# ----------------------------------------------------------------------
# Pixeldecoder (U-Net-Bausteine nach DeOldify)


def _faltung(ein, aus, ks=3, bias=True, relu=False, bn=False):
    """Faltung (Spektralnormierung eingerechnet), dahinter ReLU und BatchNorm - die
    Positionen im Sequential wie im Original."""
    schichten = [nn.Conv2d(ein, aus, ks, padding=(ks - 1) // 2, bias=bias)]
    if relu:
        schichten.append(nn.ReLU(True))
    if bn:
        schichten.append(nn.BatchNorm2d(aus))
    return nn.Sequential(*schichten)


class CustomPixelShuffle_ICNR(nn.Module):  # noqa: N801  (Name wie im Original)
    def __init__(self, ein, aus, faktor=2, bn=False):
        super().__init__()
        self.conv = _faltung(ein, aus * faktor ** 2, ks=1, bias=not bn, bn=bn)
        self.shuf = nn.PixelShuffle(faktor)
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.blur(self.pad(self.shuf(self.relu(self.conv(x)))))


class UnetBlockWide(nn.Module):
    def __init__(self, hoch_ein, quer_ein, aus):
        super().__init__()
        self.shuf = CustomPixelShuffle_ICNR(hoch_ein, aus, bn=True)
        self.bn = nn.BatchNorm2d(quer_ein)
        self.conv = _faltung(aus + quer_ein, aus, bias=False, relu=True, bn=True)
        self.relu = nn.ReLU()

    def forward(self, hoch, quer):
        return self.conv(self.relu(torch.cat([self.shuf(hoch), self.bn(quer)], dim=1)))


# ----------------------------------------------------------------------
# Farbdecoder (Schichten nach Mask2Former, Positionskodierung nach DETR)


def positionen(hoehe, breite, merkmale=VERSTECKT // 2, temperatur=10000):
    """Sinus-Positionskodierung (C, H, W), normiert auf 2 pi - wie PositionEmbeddingSine."""
    eps, skala = 1e-6, 2 * math.pi
    y = torch.arange(1, hoehe + 1, dtype=torch.float32)[:, None].expand(hoehe, breite)
    x = torch.arange(1, breite + 1, dtype=torch.float32)[None, :].expand(hoehe, breite)
    y = y / (hoehe + eps) * skala
    x = x / (breite + eps) * skala
    dim_t = torch.arange(merkmale, dtype=torch.float32)
    dim_t = temperatur ** (2 * (dim_t // 2) / merkmale)
    px, py = x[..., None] / dim_t, y[..., None] / dim_t
    px = torch.stack((px[..., 0::2].sin(), px[..., 1::2].cos()), dim=3).flatten(2)
    py = torch.stack((py[..., 0::2].sin(), py[..., 1::2].cos()), dim=3).flatten(2)
    return torch.cat((py, px), dim=2).permute(2, 0, 1)


class SelfAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(VERSTECKT, 8)
        self.norm = nn.LayerNorm(VERSTECKT)

    def forward(self, ziel, position):
        q = k = ziel + position
        return self.norm(ziel + self.self_attn(q, k, value=ziel)[0])


class CrossAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(VERSTECKT, 8)
        self.norm = nn.LayerNorm(VERSTECKT)

    def forward(self, ziel, speicher, position, ziel_position):
        neu = self.multihead_attn(query=ziel + ziel_position, key=speicher + position,
                                  value=speicher)[0]
        return self.norm(ziel + neu)


class FFNLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(VERSTECKT, 2048)
        self.linear2 = nn.Linear(2048, VERSTECKT)
        self.norm = nn.LayerNorm(VERSTECKT)

    def forward(self, x):
        return self.norm(x + self.linear2(F.relu(self.linear1(x))))


class MLP(nn.Module):
    def __init__(self, ein, versteckt, aus, schichten):
        super().__init__()
        h = [versteckt] * (schichten - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([ein, *h], [*h, aus], strict=True))

    def forward(self, x):
        for i, schicht in enumerate(self.layers):
            x = F.relu(schicht(x)) if i < len(self.layers) - 1 else schicht(x)
        return x


class MultiScaleColorDecoder(nn.Module):
    def __init__(self, ein_kanaele):
        super().__init__()
        self.query_feat = nn.Embedding(ANFRAGEN, VERSTECKT)
        self.query_embed = nn.Embedding(ANFRAGEN, VERSTECKT)
        self.level_embed = nn.Embedding(EBENEN, VERSTECKT)
        self.input_proj = nn.ModuleList(nn.Conv2d(k, VERSTECKT, 1) for k in ein_kanaele)
        self.transformer_self_attention_layers = nn.ModuleList(
            SelfAttentionLayer() for _ in range(SCHICHTEN))
        self.transformer_cross_attention_layers = nn.ModuleList(
            CrossAttentionLayer() for _ in range(SCHICHTEN))
        self.transformer_ffn_layers = nn.ModuleList(FFNLayer() for _ in range(SCHICHTEN))
        self.decoder_norm = nn.LayerNorm(VERSTECKT)
        self.color_embed = MLP(VERSTECKT, VERSTECKT, VERSTECKT, 3)

    def forward(self, stufen, bild_merkmale):
        quellen, orte = [], []
        for i, merkmal in enumerate(stufen):
            _n, _k, h, w = merkmal.shape
            orte.append(positionen(h, w).flatten(1).permute(1, 0)[:, None, :])
            ebene = self.level_embed.weight[i][None, :, None]
            quelle = self.input_proj[i](merkmal).flatten(2) + ebene
            quellen.append(quelle.permute(2, 0, 1))
        anfrage = self.query_embed.weight[:, None, :]
        x = self.query_feat.weight[:, None, :]
        for i in range(SCHICHTEN):
            ebene = i % EBENEN
            x = self.transformer_cross_attention_layers[i](x, quellen[ebene], orte[ebene], anfrage)
            x = self.transformer_self_attention_layers[i](x, anfrage)
            x = self.transformer_ffn_layers[i](x)
        farben = self.color_embed(self.decoder_norm(x).transpose(0, 1))
        return torch.einsum("bqc,bchw->bqhw", farben, bild_merkmale)


class DuelDecoder(nn.Module):  # Name wie im Original
    def __init__(self):
        super().__init__()
        kanaele = BREITEN[::-1]                              # 768, 384, 192, 96
        self.layers = nn.Sequential(
            UnetBlockWide(kanaele[0], kanaele[1], NF),
            UnetBlockWide(NF, kanaele[2], NF),
            UnetBlockWide(NF, kanaele[3], NF // 2))
        self.last_shuf = CustomPixelShuffle_ICNR(NF // 2, NF // 2, faktor=4)
        self.color_decoder = MultiScaleColorDecoder([NF, NF, NF // 2])

    def forward(self, stufen):
        out0 = self.layers[0](stufen[3], stufen[2])
        out1 = self.layers[1](out0, stufen[1])
        out2 = self.layers[2](out1, stufen[0])
        return self.color_decoder([out0, out1, out2], self.last_shuf(out2))


class DDColor(nn.Module):
    """Graubild in sRGB (1, 3, 512, 512), 0..1 -> Farbanteile a, b (1, 2, 512, 512)."""

    def __init__(self):
        super().__init__()
        self.encoder = ImageEncoder()
        self.decoder = DuelDecoder()
        self.refine_net = nn.Sequential(_faltung(ANFRAGEN + 3, 2, ks=1))
        self.register_buffer("mean", torch.tensor(MITTEL).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STREUUNG).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        merkmale = self.decoder(self.encoder(x))
        return self.refine_net(torch.cat([merkmale, x], dim=1))

    @staticmethod
    def gewichte_einrechnen(daten: dict) -> dict:
        """Spektralnormierung einrechnen: weight = weight_orig / (u . (W v))."""
        neu = {}
        for name, wert in daten.items():
            if name.endswith(("weight_u", "weight_v")):
                continue
            if name.endswith("weight_orig"):
                stamm = name[:-len("_orig")]
                u, v = daten[stamm + "_u"], daten[stamm + "_v"]
                sigma = torch.dot(u, wert.reshape(wert.shape[0], -1) @ v)
                neu[stamm] = wert / sigma
            else:
                neu[name] = wert
        return neu
