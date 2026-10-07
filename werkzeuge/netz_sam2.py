"""
SAM 2 nachgebaut - fuer den ONNX-Export, nicht Teil der App

Nachbau des Bildteils von Segment Anything 2 (SAM 2.1, Hiera small) fuer die
Auswahl per Klick: Bild-Encoder (Hiera + FPN-Hals) und Masken-Decoder mit
Prompt-Encoder. Video, Gedaechtnis und Objektverfolgung fehlen. Ohne hydra,
iopath und torchvision; die Namen der Module entsprechen dem Original, damit
load_state_dict(strict=True) die offiziellen Gewichte unveraendert laedt.

Nach dem Code von SAM 2 (https://github.com/facebookresearch/sam2),
Copyright (c) Meta Platforms, Inc. and affiliates, Apache License 2.0.
Geaendert gegenueber dem Original, alles rechnet dasselbe:
- Der Encoder hat eine feste Eingabe von 1024 x 1024, erwartet sRGB in 0..1
  und normiert selbst (ImageNet-Mittelwerte wie SAM2Transforms). Er gibt die
  Merkmale schon so aus, wie SAM2ImagePredictor.set_image sie ablegt: mit
  no_mem_embed und den beiden hochaufgeloesten Stufen durch conv_s0/conv_s1.
- Der Decoder nimmt die Klickpunkte in Pixeln des 1024er-Bildes, haengt den
  Fuellpunkt selbst an und gibt alle vier Masken (Logits, 256 x 256) und ihre
  geschaetzte Guete aus; die Auswahl unter ihnen trifft die App. Eine
  vorherige Maske geht ueber mit_maske (0 oder 1) ein statt ueber None.
- Die Aufmerksamkeit ist ausgeschrieben statt scaled_dot_product_attention.
- Die Positionskodierungen des FPN-Halses und die Kopf-Teile fuer Video
  (Objektzeiger, Objekt-Score) werden nicht gerechnet.

Als Bearbeitung des Codes von SAM 2 steht diese Datei - anders als der Rest
des Projekts - unter der Apache License 2.0 (Text in LICENSES/Apache-2.0.txt).

Licensed under Apache License 2.0
Copyright (c) Meta Platforms, Inc. and affiliates (SAM 2)
Copyright 2026 Alexander Unverhau (Aenderungen)
Created with assistance of Claude AI
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

BILD = 1024                                  # Eingabegroesse des Encoders
MITTEL = (0.485, 0.456, 0.406)
STREUUNG = (0.229, 0.224, 0.225)


def aufmerksamkeit(q, k, v):
    """Skalarprodukt-Aufmerksamkeit ueber (B, Koepfe, N, d)."""
    gewicht = (q * (1.0 / math.sqrt(q.shape[-1]))) @ k.transpose(-2, -1)
    return gewicht.softmax(dim=-1) @ v


class MLP(nn.Module):
    def __init__(self, ein, verdeckt, aus, schichten, aktivierung=nn.ReLU, sigmoid=False):
        super().__init__()
        h = [verdeckt] * (schichten - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([ein] + h, h + [aus], strict=True))
        self.act = aktivierung()
        self.sigmoid = sigmoid

    def forward(self, x):
        for i, schicht in enumerate(self.layers):
            x = schicht(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return x.sigmoid() if self.sigmoid else x


class LayerNorm2d(nn.Module):
    def __init__(self, kanaele, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(kanaele))
        self.bias = nn.Parameter(torch.zeros(kanaele))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


# ----------------------------------------------------------------------
# Hiera (Rueckgrat des Encoders)


def fenster_teilen(x, f):
    """(B, H, W, C) -> (B * Fenster, f, f, C), unten und rechts mit Nullen aufgefuellt."""
    b, h, w, c = x.shape
    ph, pw = (f - h % f) % f, (f - w % f) % f
    if ph or pw:
        x = F.pad(x, (0, 0, 0, pw, 0, ph))
    hp, wp = h + ph, w + pw
    x = x.view(b, hp // f, f, wp // f, f, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, f, f, c), hp, wp


def fenster_zusammen(fenster, f, hp, wp, h, w):
    b = fenster.shape[0] // (hp * wp // f // f)
    x = fenster.reshape(b, hp // f, wp // f, f, f, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, -1)
    return x[:, :h, :w, :] if hp > h or wp > w else x


def buendeln(x, pool):
    """Max-Pooling ueber (B, H, W, C)."""
    return pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class MultiScaleAttention(nn.Module):
    def __init__(self, dim, dim_aus, koepfe, pool=None):
        super().__init__()
        self.koepfe = koepfe
        self.pool = pool
        self.qkv = nn.Linear(dim, dim_aus * 3)
        self.proj = nn.Linear(dim_aus, dim_aus)

    def forward(self, x):
        b, h, w, _ = x.shape
        qkv = self.qkv(x).reshape(b, h * w, 3, self.koepfe, -1)
        q, k, v = torch.unbind(qkv, 2)
        if self.pool is not None:            # Abfragen buendeln beim Stufenwechsel
            q = buendeln(q.reshape(b, h, w, -1), self.pool)
            h, w = int(q.shape[1]), int(q.shape[2])
            q = q.reshape(b, h * w, self.koepfe, -1)
        x = aufmerksamkeit(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        x = x.transpose(1, 2).reshape(b, h, w, -1)
        return self.proj(x)


class MultiScaleBlock(nn.Module):
    def __init__(self, dim, dim_aus, koepfe, fenster, q_schritt):
        super().__init__()
        self.dim, self.dim_aus, self.fenster = dim, dim_aus, fenster
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pool = nn.MaxPool2d(q_schritt, q_schritt) if q_schritt else None
        self.attn = MultiScaleAttention(dim, dim_aus, koepfe, self.pool)
        self.norm2 = nn.LayerNorm(dim_aus, eps=1e-6)
        self.mlp = MLP(dim_aus, 4 * dim_aus, dim_aus, 2, aktivierung=nn.GELU)
        if dim != dim_aus:
            self.proj = nn.Linear(dim, dim_aus)

    def forward(self, x):
        kurz = x
        x = self.norm1(x)
        if self.dim != self.dim_aus:
            kurz = self.proj(x)
            if self.pool is not None:
                kurz = buendeln(kurz, self.pool)
        f = self.fenster
        if f:
            h, w = int(x.shape[1]), int(x.shape[2])
            x, hp, wp = fenster_teilen(x, f)
        x = self.attn(x)
        if f:
            if self.pool is not None:        # Fenster und Bild sind halb so gross
                f //= 2
                h, w = int(kurz.shape[1]), int(kurz.shape[2])
                hp, wp = h + (f - h % f) % f, w + (f - w % f) % f
            x = fenster_zusammen(x, f, hp, wp, h, w)
        x = kurz + x
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, 7, 4, 3)

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)


class Hiera(nn.Module):
    """Hiera-Rueckgrat; Vorgaben von SAM 2.1 small."""

    def __init__(self, dim=96, koepfe=1, stufen=(1, 2, 11, 2), global_bloecke=(7, 10, 13),
                 fenster=(8, 4, 14, 7), hintergrund=(7, 7)):
        super().__init__()
        self.stufen_enden = [sum(stufen[:i]) - 1 for i in range(1, len(stufen) + 1)]
        pool_bloecke = [e + 1 for e in self.stufen_enden[:-1]]
        self.patch_embed = PatchEmbed(dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, *hintergrund))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, dim, fenster[0], fenster[0]))
        self.blocks = nn.ModuleList()
        stufe = 1
        for i in range(sum(stufen)):
            dim_aus = dim
            # Die Fenstergroesse hinkt einen Block nach: der erste Block einer Stufe
            # hat noch das Fenster der vorigen
            f = 0 if i in global_bloecke else fenster[stufe - 1]
            if i - 1 in self.stufen_enden:
                dim_aus, koepfe, stufe = dim * 2, koepfe * 2, stufe + 1
            self.blocks.append(MultiScaleBlock(dim, dim_aus, koepfe, f,
                                               (2, 2) if i in pool_bloecke else None))
            dim = dim_aus

    def forward(self, x):
        x = self.patch_embed(x)
        h, w = int(x.shape[1]), int(x.shape[2])
        pos = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        f = self.pos_embed_window.shape[-1]
        pos = pos + self.pos_embed_window.tile(1, 1, h // f, w // f)
        x = x + pos.permute(0, 2, 3, 1)
        stufen = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.stufen_enden:
                stufen.append(x.permute(0, 3, 1, 2))
        return stufen


class FpnNeck(nn.Module):
    """FPN-Hals; nur die Faltungen, ohne Positionskodierung (die braucht nur das Video)."""

    def __init__(self, d_model=256, kanaele=(768, 384, 192, 96)):
        super().__init__()
        self.convs = nn.ModuleList()
        for dim in kanaele:
            stufe = nn.Sequential()
            stufe.add_module("conv", nn.Conv2d(dim, d_model, 1))
            self.convs.append(stufe)

    def forward(self, xs):
        # Von grob nach fein; nur Stufe 2 bekommt die groebste von oben dazu
        # (fpn_top_down_levels 2 und 3), die groebste selbst faellt weg (scalp 1)
        oben = self.convs[0](xs[3])
        s2 = self.convs[1](xs[2]) + F.interpolate(oben, scale_factor=2.0, mode="nearest")
        return self.convs[3](xs[0]), self.convs[2](xs[1]), s2


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = Hiera()
        self.neck = FpnNeck()

    def forward(self, x):
        return self.neck(self.trunk(x))


# ----------------------------------------------------------------------
# Prompt-Encoder und Masken-Decoder


class PositionEmbeddingRandom(nn.Module):
    def __init__(self, merkmale=128):
        super().__init__()
        self.register_buffer("positional_encoding_gaussian_matrix", torch.randn(2, merkmale))

    def kodieren(self, xy):
        """Punkte in 0..1 (..., 2) -> (..., 2 * merkmale)."""
        xy = (2 * xy - 1) @ self.positional_encoding_gaussian_matrix
        xy = 2 * math.pi * xy
        return torch.cat([xy.sin(), xy.cos()], dim=-1)

    def gitter(self, h, w):
        ys = (torch.arange(h, dtype=torch.float32) + 0.5) / h
        xs = (torch.arange(w, dtype=torch.float32) + 0.5) / w
        xy = torch.stack([xs[None, :].expand(h, w), ys[:, None].expand(h, w)], dim=-1)
        return self.kodieren(xy).permute(2, 0, 1)[None]


class PromptEncoder(nn.Module):
    def __init__(self, dim=256, merkmal_groesse=64, masken_kanaele=16):
        super().__init__()
        self.merkmal_groesse = merkmal_groesse
        self.pe_layer = PositionEmbeddingRandom(dim // 2)
        self.point_embeddings = nn.ModuleList(nn.Embedding(1, dim) for _ in range(4))
        self.not_a_point_embed = nn.Embedding(1, dim)
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, masken_kanaele // 4, 2, 2), LayerNorm2d(masken_kanaele // 4), nn.GELU(),
            nn.Conv2d(masken_kanaele // 4, masken_kanaele, 2, 2), LayerNorm2d(masken_kanaele),
            nn.GELU(), nn.Conv2d(masken_kanaele, dim, 1))
        self.no_mask_embed = nn.Embedding(1, dim)

    def punkte(self, punkte, etiketten):
        """Klickpunkte (B, N, 2) in Pixeln des 1024er-Bildes, Etiketten 1 = dazu, 0 = weg."""
        b = punkte.shape[0]
        punkte = torch.cat([punkte + 0.5, torch.zeros(b, 1, 2)], dim=1)   # Fuellpunkt
        etiketten = torch.cat([etiketten, -torch.ones(b, 1)], dim=1)[..., None]
        einbettung = self.pe_layer.kodieren(punkte / BILD)
        keiner = (etiketten == -1).float()
        einbettung = einbettung * (1 - keiner) + keiner * self.not_a_point_embed.weight
        einbettung = einbettung + (etiketten == 0).float() * self.point_embeddings[0].weight
        return einbettung + (etiketten == 1).float() * self.point_embeddings[1].weight

    def dicht(self, maske, mit_maske):
        """Vorige Maske (B, 1, 256, 256) als dichte Einbettung - oder keine."""
        mit = mit_maske.reshape(-1, 1, 1, 1)
        keine = self.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return mit * self.mask_downscaling(maske) + (1 - mit) * keine


class Attention(nn.Module):
    def __init__(self, dim, koepfe, verkleinern=1):
        super().__init__()
        innen = dim // verkleinern
        self.koepfe = koepfe
        self.q_proj = nn.Linear(dim, innen)
        self.k_proj = nn.Linear(dim, innen)
        self.v_proj = nn.Linear(dim, innen)
        self.out_proj = nn.Linear(innen, dim)

    def _koepfe(self, x):
        b, n, c = x.shape
        return x.reshape(b, n, self.koepfe, c // self.koepfe).transpose(1, 2)

    def forward(self, q, k, v):
        q, k, v = (self._koepfe(p(t)) for p, t in ((self.q_proj, q), (self.k_proj, k),
                                                   (self.v_proj, v)))
        x = aufmerksamkeit(q, k, v)
        b, kp, n, c = x.shape
        return self.out_proj(x.transpose(1, 2).reshape(b, n, kp * c))


class TwoWayAttentionBlock(nn.Module):
    def __init__(self, dim, koepfe, mlp_dim, ohne_pe_zuerst):
        super().__init__()
        self.self_attn = Attention(dim, koepfe)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn_token_to_image = Attention(dim, koepfe, 2)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dim, 2)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.cross_attn_image_to_token = Attention(dim, koepfe, 2)
        self.ohne_pe_zuerst = ohne_pe_zuerst

    def forward(self, abfragen, schluessel, abfrage_pe, schluessel_pe):
        if self.ohne_pe_zuerst:
            abfragen = self.self_attn(abfragen, abfragen, abfragen)
        else:
            q = abfragen + abfrage_pe
            abfragen = abfragen + self.self_attn(q, q, abfragen)
        abfragen = self.norm1(abfragen)
        q, k = abfragen + abfrage_pe, schluessel + schluessel_pe
        abfragen = self.norm2(abfragen + self.cross_attn_token_to_image(q, k, schluessel))
        abfragen = self.norm3(abfragen + self.mlp(abfragen))
        q, k = abfragen + abfrage_pe, schluessel + schluessel_pe
        schluessel = self.norm4(schluessel + self.cross_attn_image_to_token(k, q, abfragen))
        return abfragen, schluessel


class TwoWayTransformer(nn.Module):
    def __init__(self, tiefe=2, dim=256, koepfe=8, mlp_dim=2048):
        super().__init__()
        self.layers = nn.ModuleList(TwoWayAttentionBlock(dim, koepfe, mlp_dim, i == 0)
                                    for i in range(tiefe))
        self.final_attn_token_to_image = Attention(dim, koepfe, 2)
        self.norm_final_attn = nn.LayerNorm(dim)

    def forward(self, bild, bild_pe, punkte):
        bild = bild.flatten(2).permute(0, 2, 1)
        bild_pe = bild_pe.flatten(2).permute(0, 2, 1)
        abfragen, schluessel = punkte, bild
        for schicht in self.layers:
            abfragen, schluessel = schicht(abfragen, schluessel, punkte, bild_pe)
        q, k = abfragen + punkte, schluessel + bild_pe
        abfragen = abfragen + self.final_attn_token_to_image(q, k, schluessel)
        return self.norm_final_attn(abfragen), schluessel


class MaskDecoder(nn.Module):
    def __init__(self, dim=256, masken=3):
        super().__init__()
        self.transformer = TwoWayTransformer(dim=dim)
        self.iou_token = nn.Embedding(1, dim)
        self.anzahl = masken + 1
        self.mask_tokens = nn.Embedding(self.anzahl, dim)
        self.obj_score_token = nn.Embedding(1, dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 4, 2, 2), LayerNorm2d(dim // 4), nn.GELU(),
            nn.ConvTranspose2d(dim // 4, dim // 8, 2, 2), nn.GELU())
        self.conv_s0 = nn.Conv2d(dim, dim // 8, 1)
        self.conv_s1 = nn.Conv2d(dim, dim // 4, 1)
        self.output_hypernetworks_mlps = nn.ModuleList(
            MLP(dim, dim, dim // 8, 3) for _ in range(self.anzahl))
        self.iou_prediction_head = MLP(dim, 256, self.anzahl, 3, sigmoid=True)
        self.pred_obj_score_head = MLP(dim, dim, 1, 3)     # nur Video, wird nicht gerechnet

    def forward(self, bild, bild_pe, punkte, dicht, s0, s1):
        b = punkte.shape[0]
        ausgabe = torch.cat([self.obj_score_token.weight, self.iou_token.weight,
                             self.mask_tokens.weight], dim=0)
        tokens = torch.cat([ausgabe[None].expand(b, -1, -1), punkte], dim=1)
        quelle = bild + dicht
        _, c, h, w = quelle.shape
        hs, quelle = self.transformer(quelle, bild_pe, tokens)
        iou_token = hs[:, 1, :]
        masken_tokens = hs[:, 2:2 + self.anzahl, :]
        quelle = quelle.transpose(1, 2).reshape(b, c, h, w)
        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        hoch = act1(ln1(dc1(quelle) + s1))
        hoch = act2(dc2(hoch) + s0)
        hyper = torch.stack([mlp(masken_tokens[:, i, :])
                             for i, mlp in enumerate(self.output_hypernetworks_mlps)], dim=1)
        b, c, h, w = hoch.shape
        masken = (hyper @ hoch.reshape(b, c, h * w)).reshape(b, -1, h, w)
        return masken, self.iou_prediction_head(iou_token)


class Sam2(nn.Module):
    """Die Teile von SAM 2.1, die ein einzelnes Bild braucht - Namen wie im Original."""

    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.sam_prompt_encoder = PromptEncoder()
        self.sam_mask_decoder = MaskDecoder()
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, 256))
        self.register_buffer("mittel", torch.tensor(MITTEL).reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer("streuung", torch.tensor(STREUUNG).reshape(1, 3, 1, 1),
                             persistent=False)

    @staticmethod
    def gewichte(daten: dict) -> dict:
        """Aus dem Checkpoint nur die Schluessel dieses Nachbaus."""
        teile = ("image_encoder.", "sam_prompt_encoder.", "sam_mask_decoder.", "no_mem_embed")
        return {k: v for k, v in daten.items() if k.startswith(teile)}


class Kodierer(nn.Module):
    """sRGB 0..1 (1, 3, 1024, 1024) -> Bildmerkmale (1, 256, 64, 64), s0 (1, 32, 256, 256),
    s1 (1, 64, 128, 128)."""

    def __init__(self, sam: Sam2):
        super().__init__()
        self.sam = sam

    def forward(self, bild):
        sam = self.sam
        s0, s1, s2 = sam.image_encoder((bild - sam.mittel) / sam.streuung)
        merkmale = s2 + sam.no_mem_embed.reshape(1, -1, 1, 1)
        return (merkmale, sam.sam_mask_decoder.conv_s0(s0),
                sam.sam_mask_decoder.conv_s1(s1))


class Dekodierer(nn.Module):
    """Merkmale + Klicks (+ vorige Maske) -> vier Masken-Logits (1, 4, 256, 256) und Guete (1, 4).

    Maske 0 ist die fuer mehrere Klicks, 1..3 die drei Deutungen eines einzelnen Klicks.
    """

    def __init__(self, sam: Sam2):
        super().__init__()
        self.sam = sam

    def forward(self, merkmale, s0, s1, punkte, etiketten, maske, mit_maske):
        prompt = self.sam.sam_prompt_encoder
        bild_pe = prompt.pe_layer.gitter(prompt.merkmal_groesse, prompt.merkmal_groesse)
        return self.sam.sam_mask_decoder(merkmale, bild_pe, prompt.punkte(punkte, etiketten),
                                         prompt.dicht(maske, mit_maske), s0, s1)
