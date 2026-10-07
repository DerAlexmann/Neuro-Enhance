"""
LaMa nachgebaut - fuer den ONNX-Export, nicht Teil der App

Nachbau des Generators von LaMa ("Big LaMa", FFC-ResNet) zum Fuellen von
Bildbereichen: entfernt man ein Objekt, malt LaMa den Hintergrund dahinter.
Die Namen der Module entsprechen dem Original, damit load_state_dict(strict=True)
die offiziellen Gewichte unveraendert laedt.

Nach dem Code von LaMa (https://github.com/advimman/lama,
saicinpainting/training/modules/ffc.py), Copyright 2021 Samsung Research,
Apache License 2.0; das wiederum nach der Fast Fourier Convolution
(https://github.com/pkumivision/FFC).
Geaendert gegenueber dem Original, alles rechnet dasselbe:
- Die Fouriertransformation (torch.fft.rfftn/irfftn) ist als Multiplikation mit
  DFT-Matrizen ausgeschrieben. Die Matrizen entstehen im Netz aus der Bildgroesse;
  die Winkel werden ganzzahlig modulo der Laenge gebildet, damit sie auch in
  float32 genau bleiben. So braucht weder ONNX Runtime noch TensorRT einen
  FFT-Operator.
- Nur die Zweige, die Big LaMa nutzt: ohne LFU, ohne Gates, ohne
  Squeeze-Excitation und ohne raeumliche Transformation.
- Eingang sind Bild (sRGB 0..1) und Maske getrennt; das Netz loescht den
  Bereich unter der Maske selbst und haengt die Maske an, wie
  DefaultInpaintingTrainingModule es tut. Ausgang ist das gemalte Bild; das
  Einsetzen in das Original uebernimmt die App.

Als Bearbeitung des Codes von LaMa steht diese Datei - anders als der Rest des
Projekts - unter der Apache License 2.0 (Text in LICENSES/Apache-2.0.txt).

Licensed under Apache License 2.0
Copyright 2021 Samsung Research (LaMa)
Copyright 2026 Alexander Unverhau (Aenderungen)
Created with assistance of Claude AI
"""

from __future__ import annotations

import math

import torch
from torch import nn

GLOBAL = 0.75                # Anteil der Kanaele im spektralen (globalen) Zweig


def _winkel(n, k, laenge):
    """2 pi (n k mod laenge) / laenge - ganzzahlig reduziert, dann erst float."""
    return (n[:, None] * k[None, :]) % laenge * (2 * math.pi / laenge)


class FourierUnit(nn.Module):
    """1x1-Faltung im Frequenzraum: rfft2 -> Faltung -> irfft2 (orthonormiert)."""

    def __init__(self, kanaele):
        super().__init__()
        self.conv_layer = nn.Conv2d(kanaele * 2, kanaele * 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(kanaele * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape
        wf = w // 2 + 1
        nh, nw, kw = torch.arange(h), torch.arange(w), torch.arange(wf)
        phi = _winkel(nh, nh, h)                    # (h, h), symmetrisch
        ch, sh = phi.cos(), phi.sin()
        theta = _winkel(nw, kw, w)                  # (w, wf)
        cw, sw = theta.cos(), theta.sin()
        skala = 1.0 / torch.sqrt((h * w) * torch.ones(()))

        # Hin: reell entlang der Breite, komplex entlang der Hoehe
        re, im = x @ cw, -(x @ sw)
        yr = (ch @ re + sh @ im) * skala
        yi = (ch @ im - sh @ re) * skala
        f = torch.stack((yr, yi), dim=2).reshape(b, 2 * c, h, wf)
        f = self.relu(self.bn(self.conv_layer(f)))
        f = f.reshape(b, c, 2, h, wf)
        zr, zi = f[:, :, 0], f[:, :, 1]

        # Zurueck: komplex entlang der Hoehe, reell (c2r) entlang der Breite
        xr = ch @ zr - sh @ zi
        xi = ch @ zi + sh @ zr
        # Gleichanteil und Nyquist zaehlen einfach, alle anderen doppelt
        gewicht = 2.0 - (kw == 0).float() - (2 * kw == w).float()
        cinv = cw.t() * gewicht[:, None]            # (wf, w)
        sinv = sw.t() * gewicht[:, None]
        return (xr @ cinv - xi @ sinv) * skala


class SpectralTransform(nn.Module):
    def __init__(self, ein, aus):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(ein, aus // 2, 1, bias=False),
                                   nn.BatchNorm2d(aus // 2), nn.ReLU(inplace=True))
        self.fu = FourierUnit(aus // 2)
        self.conv2 = nn.Conv2d(aus // 2, aus, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x + self.fu(x))


def _faltung(ein, aus, kern, schritt=1, rand=0):
    return nn.Conv2d(ein, aus, kern, schritt, rand, bias=False, padding_mode="reflect")


class _Behaelter(nn.Module):
    """Nur fuer die Namen: haelt die Faltungen unter .ffc wie im Original."""


class FfcLokal(nn.Module):
    """FFC_BN_ACT ohne globalen Zweig (ratio 0 -> 0): Faltung, BN, ReLU."""

    def __init__(self, ein, aus, kern, schritt=1, rand=0):
        super().__init__()
        self.ffc = _Behaelter()
        self.ffc.convl2l = _faltung(ein, aus, kern, schritt, rand)
        self.bn_l = nn.BatchNorm2d(aus)
        self.act_l = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act_l(self.bn_l(self.ffc.convl2l(x)))


class FfcTeilen(nn.Module):
    """Letzter Abwaertsschritt (ratio 0 -> 0.75): teilt in lokalen und globalen Zweig."""

    def __init__(self, ein, aus):
        super().__init__()
        g = int(aus * GLOBAL)
        self.ffc = _Behaelter()
        self.ffc.convl2l = _faltung(ein, aus - g, 3, 2, 1)
        self.ffc.convl2g = _faltung(ein, g, 3, 2, 1)
        self.bn_l = nn.BatchNorm2d(aus - g)
        self.bn_g = nn.BatchNorm2d(g)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return (self.act(self.bn_l(self.ffc.convl2l(x))),
                self.act(self.bn_g(self.ffc.convl2g(x))))


class FFC_BN_ACT(nn.Module):  # noqa: N801 - Name wie im Original
    """Volle FFC (ratio 0.75 -> 0.75) mit BN und ReLU je Zweig."""

    def __init__(self, dim):
        super().__init__()
        g = int(dim * GLOBAL)
        lok = dim - g
        self.ffc = _Behaelter()
        self.ffc.convl2l = _faltung(lok, lok, 3, 1, 1)
        self.ffc.convl2g = _faltung(lok, g, 3, 1, 1)
        self.ffc.convg2l = _faltung(g, lok, 3, 1, 1)
        self.ffc.convg2g = SpectralTransform(g, g)
        self.bn_l = nn.BatchNorm2d(lok)
        self.bn_g = nn.BatchNorm2d(g)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x_l, x_g = x
        f = self.ffc
        aus_l = f.convl2l(x_l) + f.convg2l(x_g)
        aus_g = f.convl2g(x_l) + f.convg2g(x_g)
        return self.act(self.bn_l(aus_l)), self.act(self.bn_g(aus_g))


class FFCResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim)
        self.conv2 = FFC_BN_ACT(dim)

    def forward(self, x):
        x_l, x_g = x
        y_l, y_g = self.conv2(self.conv1((x_l, x_g)))
        return x_l + y_l, x_g + y_g


class Zusammen(nn.Module):
    def forward(self, x):
        return torch.cat(x, dim=1)


class Generator(nn.Module):
    """FFCResNetGenerator mit den Werten von Big LaMa (ngf 64, 3 Abwaertsschritte,
    18 Bloecke, Sigmoid am Ausgang); model.N wie im Original."""

    def __init__(self, ngf=64, abwaerts=3, bloecke=18):
        super().__init__()
        teile = [nn.ReflectionPad2d(3), FfcLokal(4, ngf, 7)]
        for i in range(abwaerts):
            ein, aus = ngf * 2 ** i, ngf * 2 ** (i + 1)
            teile.append(FfcTeilen(ein, aus) if i == abwaerts - 1
                         else FfcLokal(ein, aus, 3, 2, 1))
        dim = ngf * 2 ** abwaerts
        teile += [FFCResnetBlock(dim) for _ in range(bloecke)]
        teile.append(Zusammen())
        for i in range(abwaerts):
            ein = ngf * 2 ** (abwaerts - i)
            teile += [nn.ConvTranspose2d(ein, ein // 2, 3, 2, 1, output_padding=1),
                      nn.BatchNorm2d(ein // 2), nn.ReLU(True)]
        teile += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, 7), nn.Sigmoid()]
        self.model = nn.Sequential(*teile)

    def forward(self, x):
        return self.model(x)


class Lama(nn.Module):
    """Bild (1, 3, H, W) sRGB 0..1 und Maske (1, 1, H, W), 1 = fuellen -> gemaltes Bild.

    H und W muessen Vielfache von 8 sein.
    """

    def __init__(self):
        super().__init__()
        self.generator = Generator()

    def forward(self, bild, maske):
        return self.generator(torch.cat([bild * (1 - maske), maske], dim=1))
