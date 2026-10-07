"""
SCUNet nachgebaut - fuer den ONNX-Export, nicht Teil der App

Swin-Conv-UNet zum Entrauschen (Kai Zhang u. a., "Practical Blind Image Denoising
via Swin-Conv-UNet and Data Synthesis", https://github.com/cszn/SCUNet). Die
Namen der Module entsprechen dem Original, damit load_state_dict(strict=True) die
offiziellen Gewichte laedt. Die Fenster-Aufmerksamkeit folgt dem Swin Transformer
(https://github.com/microsoft/Swin-Transformer, MIT, Copyright (c) Microsoft
Corporation; Text in LICENSES/MIT-Swin-Transformer.txt).

Geaendert gegenueber dem Original, alles rechnet dasselbe: ohne einops und timm;
die Maske der verschobenen Fenster entsteht aus Koordinaten statt aus festen
Groessen, damit das ONNX-Modell jede Bildgroesse annimmt. Hoehe und Breite muessen
Vielfache von 64 sein - das Auffuellen, das im Original im Netz steckt,
uebernimmt die App beim Kacheln.

Als Bearbeitung des Codes von SCUNet steht diese Datei - anders als der Rest des
Projekts - unter der Apache License 2.0 (Text in LICENSES/Apache-2.0.txt).

Licensed under Apache License 2.0
Copyright 2022 Kai Zhang u. a. (SCUNet)
Copyright 2026 Alexander Unverhau (Aenderungen)
Created with assistance of Claude AI
"""

from __future__ import annotations

import torch
from torch import nn


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
