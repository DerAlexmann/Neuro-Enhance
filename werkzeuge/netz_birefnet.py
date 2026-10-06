"""
BiRefNet nachgebaut - fuer den ONNX-Export, nicht Teil der App

Nachbau der Inferenz von BiRefNet (https://github.com/ZhengPeng7/BiRefNet, MIT)
mit Swin-Transformer-Rueckgrat, ohne einops, timm, kornia und torchvision. Die
Namen der Module entsprechen dem Original, damit load_state_dict(strict=True)
die offiziellen Gewichte unveraendert laedt.

Zwei Abweichungen vom Original, beide rechnen dasselbe:
- Die verformbare Faltung (torchvision.ops.deform_conv2d) ist mit
  grid_sample nachgebildet - je Kernpunkt eine bilineare Abtastung und eine
  1x1-Faltung, aufsummiert. ONNX Runtime und TensorRT kennen grid_sample.
- Die Eingabegroesse ist fest (die des Trainings); Masken und Positionsindizes
  der Swin-Fenster entstehen deshalb schon beim Export als Konstanten.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

# ----------------------------------------------------------------------
# Swin-Transformer (v1)


def fenster_teilen(x, f):
    b, h, w, c = x.shape
    x = x.view(b, h // f, f, w // f, f, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, f * f, c)


def fenster_zusammen(fenster, f, h, w):
    c = fenster.shape[-1]
    x = fenster.view(-1, h // f, w // f, f, f, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, h, w, c)


class Mlp(nn.Module):
    def __init__(self, dim, versteckt):
        super().__init__()
        self.fc1 = nn.Linear(dim, versteckt)
        self.fc2 = nn.Linear(versteckt, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class WindowAttention(nn.Module):
    def __init__(self, dim, fenster, koepfe):
        super().__init__()
        self.koepfe = koepfe
        self.skala = (dim // koepfe) ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * fenster - 1) ** 2, koepfe))
        koord = torch.stack(torch.meshgrid(torch.arange(fenster), torch.arange(fenster),
                                           indexing="ij")).flatten(1)
        rel = (koord[:, :, None] - koord[:, None, :]).permute(1, 2, 0) + (fenster - 1)
        self.register_buffer("relative_position_index",
                             rel[..., 0] * (2 * fenster - 1) + rel[..., 1])
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, maske=None):
        b, n, c = (int(g) for g in x.shape)
        q, k, v = self.qkv(x).reshape(b, n, 3, self.koepfe, c // self.koepfe) \
            .permute(2, 0, 3, 1, 4).unbind(0)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)] \
            .view(n, n, -1).permute(2, 0, 1)[None]
        gewicht = (q * self.skala) @ k.transpose(-2, -1) + bias
        if maske is not None:                     # (Fenster, n, n), je Bild gleich
            nw = int(maske.shape[0])
            gewicht = (gewicht.view(b // nw, nw, self.koepfe, n, n) + maske[None, :, None]) \
                .view(b, self.koepfe, n, n)
        return self.proj((gewicht.softmax(-1) @ v).transpose(1, 2).reshape(b, n, c))


class SwinBlock(nn.Module):
    def __init__(self, dim, koepfe, fenster, verschiebung):
        super().__init__()
        self.fenster, self.verschiebung = fenster, verschiebung
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, fenster, koepfe)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, 4 * dim)

    def forward(self, x, h, w, maske):
        b, _l, c = (int(g) for g in x.shape)
        f, s = self.fenster, self.verschiebung
        y = self.norm1(x).view(b, h, w, c)
        pr, pu = (f - w % f) % f, (f - h % f) % f
        y = F.pad(y, (0, 0, 0, pr, 0, pu))
        hp, wp = h + pu, w + pr
        if s:
            y = torch.roll(y, shifts=(-s, -s), dims=(1, 2))
        y = fenster_zusammen(self.attn(fenster_teilen(y, f), maske if s else None), f, hp, wp)
        if s:
            y = torch.roll(y, shifts=(s, s), dims=(1, 2))
        y = y[:, :h, :w, :].reshape(b, h * w, c)
        x = x + y
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, h, w):
        b, _l, c = (int(g) for g in x.shape)
        x = x.view(b, h, w, c)
        if h % 2 or w % 2:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.reduction(self.norm(x.reshape(b, -1, 4 * c)))


def fenster_maske(h, w, f, s):
    """Maske der verschobenen Fenster wie im Original (0 oder -inf)."""
    hp, wp = -(-h // f) * f, -(-w // f) * f
    bild = torch.zeros((1, hp, wp, 1))
    zaehler = 0
    for ys in (slice(0, -f), slice(-f, -s), slice(-s, None)):
        for xs in (slice(0, -f), slice(-f, -s), slice(-s, None)):
            bild[:, ys, xs, :] = zaehler
            zaehler += 1
    m = fenster_teilen(bild, f).squeeze(-1)
    m = m[:, None, :] - m[:, :, None]
    return m.masked_fill(m != 0, float("-inf")).masked_fill(m == 0, 0.0)


class BasicLayer(nn.Module):
    def __init__(self, dim, tiefe, koepfe, fenster, runter):
        super().__init__()
        self.fenster = fenster
        self.blocks = nn.ModuleList([SwinBlock(dim, koepfe, fenster, (i % 2) * (fenster // 2))
                                     for i in range(tiefe)])
        self.downsample = PatchMerging(dim) if runter else None

    def forward(self, x, h, w):
        maske = fenster_maske(h, w, self.fenster, self.fenster // 2).to(x.dtype)
        for block in self.blocks:
            x = block(x, h, w, maske)
        if self.downsample is None:
            return x, x, h, w
        return x, self.downsample(x, h, w), (h + 1) // 2, (w + 1) // 2


class PatchEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, 4, 4)
        self.norm = nn.LayerNorm(dim)


class SwinTransformer(nn.Module):
    def __init__(self, dim, tiefen, koepfe, fenster):
        super().__init__()
        self.patch_embed = PatchEmbed(dim)
        self.layers = nn.ModuleList([
            BasicLayer(dim * 2 ** i, tiefen[i], koepfe[i], fenster, i < 3) for i in range(4)])
        self.merkmale = [dim * 2 ** i for i in range(4)]
        for i in range(4):
            self.add_module(f"norm{i}", nn.LayerNorm(self.merkmale[i]))

    def forward(self, x):
        x = self.patch_embed.proj(x)              # Groesse ist ein Vielfaches von 4
        h, w = int(x.shape[2]), int(x.shape[3])    # feste Groesse: beim Export Konstanten
        x = self.patch_embed.norm(x.flatten(2).transpose(1, 2))
        aus = []
        for i, schicht in enumerate(self.layers):
            y, x, h2, w2 = schicht(x, h, w)
            y = getattr(self, f"norm{i}")(y)
            aus.append(y.view(-1, h, w, self.merkmale[i]).permute(0, 3, 1, 2))
            h, w = h2, w2
        return aus


SWIN = {
    "swin_v1_t": {"dim": 96, "tiefen": (2, 2, 6, 2), "koepfe": (3, 6, 12, 24), "fenster": 7},
    "swin_v1_l": {"dim": 192, "tiefen": (2, 2, 18, 2), "koepfe": (6, 12, 24, 48), "fenster": 12},
}

# ----------------------------------------------------------------------
# Decoder


def verformbare_faltung(x, versatz, maske, gewicht, rand):
    """torchvision.ops.deform_conv2d (Schritt 1, Dilatation 1, eine Versatzgruppe)
    ueber grid_sample: Je Kernpunkt wird das Bild an den verschobenen Stellen
    bilinear abgetastet (ausserhalb 0), mit der Maske gewichtet und mit dem
    passenden Teil der Gewichte gefaltet."""
    _b, _c, h, w = x.shape
    kh, kw = gewicht.shape[2:]
    gy, gx = torch.meshgrid(torch.arange(h, dtype=x.dtype), torch.arange(w, dtype=x.dtype),
                            indexing="ij")
    summe = None
    for i in range(kh):
        for j in range(kw):
            k = i * kw + j
            y = gy + (i - rand) + versatz[:, 2 * k]
            xx = gx + (j - rand) + versatz[:, 2 * k + 1]
            gitter = torch.stack((2 * xx / (w - 1) - 1, 2 * y / (h - 1) - 1), dim=-1)
            probe = F.grid_sample(x, gitter, mode="bilinear", padding_mode="zeros",
                                  align_corners=True)
            gewichtung = maske[:, k:k + 1]
            if summe is not None:
                # Formal vom bisherigen Zwischenstand abhaengig (+ 0 * ein Wert): ONNX
                # Runtime rechnet sonst alle 49 Teile eines 7x7-Kerns, bevor es sie
                # addiert, und haelt sie alle zugleich im Grafikspeicher.
                gewichtung = gewichtung + 0 * summe[:, :1, :1, :1]
            teil = F.conv2d(probe * gewichtung, gewicht[:, :, i:i + 1, j:j + 1])
            summe = teil if summe is None else summe + teil
    return summe


class DeformableConv2d(nn.Module):
    def __init__(self, ein, aus, kern):
        super().__init__()
        self.rand = kern // 2
        self.offset_conv = nn.Conv2d(ein, 2 * kern * kern, kern, 1, self.rand)
        self.modulator_conv = nn.Conv2d(ein, kern * kern, kern, 1, self.rand)
        self.regular_conv = nn.Conv2d(ein, aus, kern, 1, self.rand, bias=False)

    def forward(self, x):
        versatz = self.offset_conv(x)
        maske = 2.0 * torch.sigmoid(self.modulator_conv(x))
        return verformbare_faltung(x, versatz, maske, self.regular_conv.weight, self.rand)


class AsppTeil(nn.Module):
    def __init__(self, ein, aus, kern):
        super().__init__()
        self.atrous_conv = DeformableConv2d(ein, aus, kern)
        self.bn = nn.BatchNorm2d(aus)

    def forward(self, x):
        return F.relu(self.bn(self.atrous_conv(x)))


class ASPPDeformable(nn.Module):
    def __init__(self, ein, aus=None, groessen=(1, 3, 7)):
        super().__init__()
        aus = aus or ein
        mitte = 256
        self.aspp1 = AsppTeil(ein, mitte, 1)
        self.aspp_deforms = nn.ModuleList([AsppTeil(ein, mitte, g) for g in groessen])
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(ein, mitte, 1, bias=False),
                                             nn.BatchNorm2d(mitte), nn.ReLU())
        self.conv1 = nn.Conv2d(mitte * (2 + len(groessen)), aus, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(aus)

    def forward(self, x):
        teile = [self.aspp1(x)] + [a(x) for a in self.aspp_deforms]
        teile.append(self.global_avg_pool(x).expand(-1, -1, *x.shape[2:]))
        return F.relu(self.bn1(self.conv1(torch.cat(teile, 1))))


class BasicDecBlk(nn.Module):
    def __init__(self, ein, aus, mitte=64):
        super().__init__()
        self.conv_in = nn.Conv2d(ein, mitte, 3, 1, 1)
        self.bn_in = nn.BatchNorm2d(mitte)
        self.dec_att = ASPPDeformable(mitte)
        self.conv_out = nn.Conv2d(mitte, aus, 3, 1, 1)
        self.bn_out = nn.BatchNorm2d(aus)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        return self.bn_out(self.conv_out(self.dec_att(x)))


class SimpleConvs(nn.Module):
    def __init__(self, ein, aus, mitte=64):
        super().__init__()
        self.conv1 = nn.Conv2d(ein, mitte, 3, 1, 1)
        self.conv_out = nn.Conv2d(mitte, aus, 3, 1, 1)

    def forward(self, x):
        return self.conv_out(self.conv1(x))


class Seitlich(nn.Module):
    def __init__(self, ein, aus):
        super().__init__()
        self.conv = nn.Conv2d(ein, aus, 1)

    def forward(self, x):
        return self.conv(x)


def in_flicken(bild, h, w):
    """'b c (hg h) (wg w) -> b (c hg wg) h w'"""
    b, c, hh, ww = bild.shape
    hg, wg = hh // h, ww // w
    return bild.view(b, c, hg, h, wg, w).permute(0, 1, 2, 4, 3, 5).reshape(b, c * hg * wg, h, w)


def hoch(x, groesse):
    return F.interpolate(x, size=groesse, mode="bilinear", align_corners=True)


class Decoder(nn.Module):
    def __init__(self, kanaele):
        super().__init__()
        ein = [2 ** i * 3 for i in (10, 8, 6, 4, 0)]
        ipt = [kanaele[i] // 8 for i in range(4)]
        self.ipt_blk5 = SimpleConvs(ein[0], ipt[0])
        self.ipt_blk4 = SimpleConvs(ein[1], ipt[0])
        self.ipt_blk3 = SimpleConvs(ein[2], ipt[1])
        self.ipt_blk2 = SimpleConvs(ein[3], ipt[2])
        self.ipt_blk1 = SimpleConvs(ein[4], ipt[3])
        aus = list(kanaele[1:]) + [kanaele[-1] // 2]
        dec_ein = [kanaele[i] + ipt[max(0, i - 1)] for i in range(4)]
        self.decoder_block4 = BasicDecBlk(dec_ein[0], aus[0])
        self.decoder_block3 = BasicDecBlk(dec_ein[1], aus[1])
        self.decoder_block2 = BasicDecBlk(dec_ein[2], aus[2])
        self.decoder_block1 = BasicDecBlk(dec_ein[3], aus[3])
        self.conv_out1 = nn.Sequential(nn.Conv2d(aus[3] + ipt[3], 1, 1))
        self.lateral_block4 = Seitlich(kanaele[1], aus[0])
        self.lateral_block3 = Seitlich(kanaele[2], aus[1])
        self.lateral_block2 = Seitlich(kanaele[3], aus[2])
        n = 16
        for i, c in zip((4, 3, 2), aus[:3], strict=True):
            # Gradienten-Aufmerksamkeit; die Vorhersage-Koepfe nutzt nur das Training,
            # sie sind trotzdem da, damit die Gewichte vollstaendig passen
            setattr(self, f"gdt_convs_{i}",
                    nn.Sequential(nn.Conv2d(c, n, 3, 1, 1), nn.BatchNorm2d(n), nn.ReLU()))
            setattr(self, f"gdt_convs_pred_{i}", nn.Sequential(nn.Conv2d(n, 1, 1)))
            setattr(self, f"gdt_convs_attn_{i}", nn.Sequential(nn.Conv2d(n, 1, 1)))
            setattr(self, f"conv_ms_spvn_{i}", nn.Conv2d(c, 1, 1))

    def stufe(self, p, i):
        merkmal = getattr(self, f"gdt_convs_{i}")(p)
        return p * torch.sigmoid(getattr(self, f"gdt_convs_attn_{i}")(merkmal))

    def forward(self, x, x1, x2, x3, x4):
        def bildteil(blk, ziel):
            h, w = ziel.shape[2:]
            return blk(hoch(in_flicken(x, h, w), (h, w)))

        p4 = self.stufe(self.decoder_block4(torch.cat((x4, bildteil(self.ipt_blk5, x4)), 1)), 4)
        p3 = hoch(p4, x3.shape[2:]) + self.lateral_block4(x3)
        p3 = self.stufe(self.decoder_block3(torch.cat((p3, bildteil(self.ipt_blk4, p3)), 1)), 3)
        p2 = hoch(p3, x2.shape[2:]) + self.lateral_block3(x2)
        p2 = self.stufe(self.decoder_block2(torch.cat((p2, bildteil(self.ipt_blk3, p2)), 1)), 2)
        p1 = hoch(p2, x1.shape[2:]) + self.lateral_block2(x1)
        p1 = self.decoder_block1(torch.cat((p1, bildteil(self.ipt_blk2, p1)), 1))
        return self.ausgang(x, p1)

    def ausgang(self, x, p1):
        """conv_out1(cat(hoch(p1), ipt_blk1(x))) - umgestellt, weil sonst 120 Kanaele in
        voller Aufloesung entstuenden (bei 2560 x 1440 mehrere GB):
        - die 1x1-Faltung vertauscht mit dem bilinearen Hochskalieren (beides linear):
          erst auf einen Kanal falten, dann hochskalieren;
        - ipt_blk1 ist Faltung auf Faltung ohne Aktivierung dazwischen: seine zweite
          Faltung und der passende Teil von conv_out1 werden zu einer Faltung 64 -> 1.
        Rechnet dasselbe bis auf Rundung."""
        aus = self.conv_out1[0]
        n = p1.shape[1]
        w_p, w_i = aus.weight[:, :n], aus.weight[:, n:]
        letzte = self.ipt_blk1.conv_out
        gewicht = torch.einsum("om,mcij->ocij", w_i[:, :, 0, 0], letzte.weight)
        bias = w_i[:, :, 0, 0] @ letzte.bias + aus.bias
        grob = hoch(F.conv2d(p1, w_p), x.shape[2:])
        fein = F.conv2d(self.ipt_blk1.conv1(x), gewicht, bias, padding=1)
        return grob + fein


class BiRefNet(nn.Module):
    """Eingabe: RGB 0..1 (B, 3, H, W), H und W Vielfache von 32. Ausgabe: Maske 0..1
    (B, 1, H, W) - die ImageNet-Normierung des Originals steckt im Netz."""

    def __init__(self, rueckgrat="swin_v1_t"):
        super().__init__()
        self.bb = SwinTransformer(**SWIN[rueckgrat])
        kanaele = [2 * c for c in self.bb.merkmale[::-1]]        # mul_scl_ipt = cat
        kontext = kanaele[1:][::-1]
        self.squeeze_module = nn.Sequential(BasicDecBlk(kanaele[0] + sum(kontext), kanaele[0]))
        self.decoder = Decoder(kanaele)
        self.register_buffer("mittel", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("streuung", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
                             persistent=False)

    def forward(self, bild):
        x = (bild - self.mittel) / self.streuung
        h, w = x.shape[2:]
        voll = self.bb(x)
        halb = self.bb(hoch(x, (h // 2, w // 2)))
        x1, x2, x3, x4 = (torch.cat([a, hoch(b, a.shape[2:])], 1) for a, b in zip(voll, halb,
                                                                                    strict=True))
        x4 = torch.cat([hoch(x1, x4.shape[2:]), hoch(x2, x4.shape[2:]), hoch(x3, x4.shape[2:]),
                        x4], 1)
        x4 = self.squeeze_module(x4)
        return torch.sigmoid(self.decoder(x, x1, x2, x3, x4))
