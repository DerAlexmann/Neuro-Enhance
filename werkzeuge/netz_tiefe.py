"""
Depth Anything V2 nachgebaut - fuer den ONNX-Export, nicht Teil der App

Nachbau von Depth Anything V2 Small (DINOv2 ViT-S/14 mit DPT-Kopf) zum
Schaetzen der Tiefe eines Fotos. Die Namen der Module entsprechen dem
Original, damit load_state_dict(strict=True) die offiziellen Gewichte
unveraendert laedt.

Nach dem Code von Depth Anything V2 (https://github.com/DepthAnything/Depth-Anything-V2,
Apache License 2.0, Lihe Yang u. a.), der wiederum DINOv2 (Copyright (c) Meta
Platforms, Inc. and affiliates, Apache License 2.0) und den DPT-Kopf aus MiDaS/DPT
(Intel ISL) enthaelt.
Geaendert gegenueber dem Original, alles rechnet dasselbe:
- Feste Eingabe von 700 x 1050 Pixeln (3:2, 50 x 75 Felder); die
  Positionseinbettung wird dafuer vorab wie im Original interpoliert und steht
  als Konstante im Netz. Hochformate dreht die App vorher.
- Das Netz erwartet sRGB in 0..1 und normiert selbst (ImageNet-Mittelwerte).
- Ohne xFormers, Maskentoken und Trainingszweige (Drop-Path, verschachtelte
  Tensoren); Aufmerksamkeit wie Attention.forward im Original.

Als Bearbeitung des Codes von Depth Anything V2 steht diese Datei - anders als
der Rest des Projekts - unter der Apache License 2.0 (Text in
LICENSES/Apache-2.0.txt). Der DPT-Kopf (ResidualConvUnit, FeatureFusionBlock)
stammt aus MiDaS und steht zusaetzlich unter dessen MIT-Lizenz (Text in
LICENSES/MIT-MiDaS.txt).

Licensed under Apache License 2.0
Copyright (c) Meta Platforms, Inc. and affiliates (DINOv2)
Copyright 2024 Depth Anything V2 (Lihe Yang u. a.)
Copyright (c) 2019 Intel ISL (Intel Intelligent Systems Lab) (MiDaS)
Copyright 2026 Alexander Unverhau (Aenderungen)
Created with assistance of Claude AI
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

HOEHE, BREITE = 700, 1050
FELD = 14
MITTEL = (0.485, 0.456, 0.406)
STREUUNG = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------
# DINOv2 ViT-S/14


class PatchEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, FELD, FELD)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, koepfe):
        super().__init__()
        self.koepfe = koepfe
        self.scale = (dim // koepfe) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.koepfe, c // self.koepfe).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        x = (q @ k.transpose(-2, -1)).softmax(dim=-1) @ v
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self, dim, verdeckt):
        super().__init__()
        self.fc1 = nn.Linear(dim, verdeckt)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(verdeckt, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, koepfe):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, koepfe)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, 4 * dim)
        self.ls2 = LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoVisionTransformer(nn.Module):
    def __init__(self, dim=384, tiefe=12, koepfe=6, bildgroesse=518):
        super().__init__()
        felder = (bildgroesse // FELD) ** 2
        self.patch_embed = PatchEmbed(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, felder + 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))     # nur im Training
        self.blocks = nn.ModuleList(Block(dim, koepfe) for _ in range(tiefe))
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fest = None                     # vorab interpolierte Positionen fuer den Export

    def _positionen(self, h, w):
        """interpolate_pos_encoding des Originals, samt seinem Versatz von 0,1."""
        pos = self.pos_embed.float()
        n = pos.shape[1] - 1
        if (h // FELD) * (w // FELD) == n and h == w:      # Trainingsgroesse: unveraendert
            return pos
        seite = math.sqrt(n)
        h0, w0 = h // FELD + 0.1, w // FELD + 0.1
        felder = F.interpolate(
            pos[:, 1:].reshape(1, int(seite), int(seite), -1).permute(0, 3, 1, 2),
            scale_factor=(h0 / seite, w0 / seite), mode="bicubic", antialias=False)
        felder = felder.permute(0, 2, 3, 1).reshape(1, -1, pos.shape[-1])
        return torch.cat((pos[:, :1], felder), dim=1)

    def stufen(self, x, nummern):
        """Normierte Feld-Tokens nach den Bloecken `nummern` (ohne Klassentoken)."""
        h, w = x.shape[-2:]
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + (self.fest if self.fest is not None else self._positionen(h, w))
        aus = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in nummern:
                aus.append(self.norm(x)[:, 1:])
        return aus


# ----------------------------------------------------------------------
# DPT-Kopf


class ResidualConvUnit(nn.Module):
    def __init__(self, kanaele):
        super().__init__()
        self.conv1 = nn.Conv2d(kanaele, kanaele, 3, 1, 1)
        self.conv2 = nn.Conv2d(kanaele, kanaele, 3, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(F.relu(x)))) + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, kanaele):
        super().__init__()
        self.out_conv = nn.Conv2d(kanaele, kanaele, 1)
        self.resConfUnit1 = ResidualConvUnit(kanaele)
        self.resConfUnit2 = ResidualConvUnit(kanaele)

    def forward(self, x, neben=None, groesse=None):
        if neben is not None:
            x = x + self.resConfUnit1(neben)
        x = self.resConfUnit2(x)
        if groesse is None:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        else:
            x = F.interpolate(x, size=groesse, mode="bilinear", align_corners=True)
        return self.out_conv(x)


class _Behaelter(nn.Module):
    """Nur fuer die Namen (depth_head.scratch.*)."""


class DPTHead(nn.Module):
    def __init__(self, ein=384, merkmale=64, kanaele=(48, 96, 192, 384)):
        super().__init__()
        self.projects = nn.ModuleList(nn.Conv2d(ein, k, 1) for k in kanaele)
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(kanaele[0], kanaele[0], 4, 4),
            nn.ConvTranspose2d(kanaele[1], kanaele[1], 2, 2),
            nn.Identity(),
            nn.Conv2d(kanaele[3], kanaele[3], 3, 2, 1)])
        s = self.scratch = _Behaelter()
        for i, k in enumerate(kanaele, 1):
            setattr(s, f"layer{i}_rn", nn.Conv2d(k, merkmale, 3, 1, 1, bias=False))
            setattr(s, f"refinenet{i}", FeatureFusionBlock(merkmale))
        s.output_conv1 = nn.Conv2d(merkmale, merkmale // 2, 3, 1, 1)
        s.output_conv2 = nn.Sequential(nn.Conv2d(merkmale // 2, 32, 3, 1, 1), nn.ReLU(True),
                                       nn.Conv2d(32, 1, 1), nn.ReLU(True), nn.Identity())

    def forward(self, stufen, fh, fw):
        schichten = []
        for i, x in enumerate(stufen):
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], fh, fw)
            schichten.append(self.resize_layers[i](self.projects[i](x)))
        s = self.scratch
        l1, l2, l3, l4 = (getattr(s, f"layer{i}_rn")(x) for i, x in enumerate(schichten, 1))
        p4 = s.refinenet4(l4, groesse=l3.shape[2:])
        p3 = s.refinenet3(p4, l3, groesse=l2.shape[2:])
        p2 = s.refinenet2(p3, l2, groesse=l1.shape[2:])
        p1 = s.refinenet1(p2, l1)
        x = s.output_conv1(p1)
        x = F.interpolate(x, (fh * FELD, fw * FELD), mode="bilinear", align_corners=True)
        return s.output_conv2(x)


class DepthAnythingV2(nn.Module):
    """sRGB 0..1 (1, 3, 700, 1050) -> relative inverse Tiefe (1, 1, 700, 1050):
    groesser heisst naeher."""

    STUFEN = (2, 5, 8, 11)

    def __init__(self):
        super().__init__()
        self.pretrained = DinoVisionTransformer()
        self.depth_head = DPTHead()
        self.register_buffer("mittel", torch.tensor(MITTEL).reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer("streuung", torch.tensor(STREUUNG).reshape(1, 3, 1, 1),
                             persistent=False)

    def festlegen(self, hoehe=HOEHE, breite=BREITE):
        """Positionen fuer eine feste Groesse vorab interpolieren - der ONNX-Export kennt
        die bikubische Interpolation mit Faktoren nicht; das Ergebnis ist dasselbe."""
        with torch.no_grad():
            self.pretrained.fest = self.pretrained._positionen(hoehe, breite)

    def forward(self, bild):
        x = (bild - self.mittel) / self.streuung
        fh, fw = x.shape[-2] // FELD, x.shape[-1] // FELD
        tiefe = self.depth_head(self.pretrained.stufen(x, self.STUFEN), fh, fw)
        return F.relu(tiefe)
