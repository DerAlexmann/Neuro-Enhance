"""
SRVGGNetCompact aus Real-ESRGAN nachgebaut - fuer den ONNX-Export, nicht Teil der App

Das kompakte Netz von realesr-general-x4v3 (und seinem Zwilling mit schwachem
Entrauschen). Nach realesrgan/archs/srvgg_arch.py aus Real-ESRGAN
(https://github.com/xinntao/Real-ESRGAN); die Namen der Module entsprechen dem
Original, damit load_state_dict(strict=True) die offiziellen Gewichte laedt.
Geaendert: nur die Teile, die die Inferenz braucht; feste Aktivierung PReLU.

Als Bearbeitung des Codes von Real-ESRGAN steht diese Datei unter der
BSD-3-Clause-Lizenz von Real-ESRGAN (Text in LICENSES/BSD-3-Clause-Real-ESRGAN.txt).

Copyright (c) 2021, Xintao Wang (Real-ESRGAN)
Copyright 2026 Alexander Unverhau (Aenderungen)
Created with assistance of Claude AI
"""

from __future__ import annotations

from torch import nn
from torch.nn import functional as F  # noqa: N812


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
