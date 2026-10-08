"""Referenz fuer KI-Outpainting: FLUX.2 klein 4B (destilliert) + fal-Outpaint-LoRA.

Nur fuer Entwickler, laeuft mit PyTorch/diffusers in einer eigenen venv und gehoert
nicht zur App. Es ist der lokal gepruefte Stand (RTX 4060, 8 GB, ~24 s je Bild),
an dem sich ONNX-Export und CuPy-Pipeline in Silberkorn messen lassen.

Modelle (alle Apache-2.0):
  black-forest-labs/FLUX.2-klein-4B       Transformer (destilliert, is_distilled)
  black-forest-labs/FLUX.2-klein-base-4B  Text-Encoder, Tokenizer, VAE (identisch)
  fal/flux-2-klein-4B-outpaint-lora       flux-outpaint-lora.safetensors

Verfahren:
  1. Leinwand im Zielformat, Original mittig, Rest reingruen (#00FF00); ~1 MP,
     Kanten durch 16 teilbar.
  2. Fester Prompt "Fill the green spaces according to the image"; seine
     Einbettung (1, 512, 7680) wird einmal berechnet und gespeichert.
  3. LoRA mit Staerke 1.1 fest einrechnen, Transformer in FP8 speichern und in
     BF16 rechnen (passt in 8 GB, Spitze 6.2 GB); 4 Schritte, guidance 1.0.
  4. Farbstich korrigieren: Das Modell zeichnet das ganze Bild neu, auch den
     Bereich des Originals, mit demselben Gruenstich wie die neuen Raender. Je
     Kanal eine lineare Abbildung Modell->Original im Originalbereich, danach die
     weichgezeichnete Restabweichung vom naechsten Randpunkt nach aussen fortsetzen.
  5. Hochskalieren auf volle Groesse, Original unveraendert einsetzen, weicher Saum
     (1,5 % der kurzen Kante) nur an erweiterten Seiten.

Aufruf:
  python outpaint_referenz.py <modelle> <bild> <ausgabe.jpg> <breite:hoehe> [anteil] [seed]
  <modelle> enthaelt klein-base-4b/, klein-4b-destilliert/transformer/, lora/.
  anteil < 1 verkleinert das Original zusaetzlich (rundum erweitern).
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

PROMPT = "Fill the green spaces according to the image"
GRUEN = (0, 255, 0)
FLAECHE = 1024 * 1024
SCHRITTE, CFG, LORA_STAERKE = 4, 1.0, 1.1


def leinwand(groesse, verhaeltnis: float, anteil: float = 1.0):
    """(Breite, Hoehe) des Originals -> (W, H, x, y) der Leinwand in Originalpixeln."""
    w, h = groesse
    bw, bh = w / anteil, h / anteil
    if bw / bh < verhaeltnis:
        bw = bh * verhaeltnis
    else:
        bh = bw / verhaeltnis
    W, H = round(bw), round(bh)
    return W, H, (W - w) // 2, (H - h) // 2


def eingabe_bauen(orig: Image.Image, W: int, H: int, x: int, y: int) -> Image.Image:
    s = (FLAECHE / (W * H)) ** 0.5
    aw, ah = round(W * s / 16) * 16, round(H * s / 16) * 16
    sx, sy = aw / W, ah / H
    klein = orig.resize((round(orig.width * sx), round(orig.height * sy)), Image.LANCZOS)
    eingabe = Image.new("RGB", (aw, ah), GRUEN)
    eingabe.paste(klein, (round(x * sx), round(y * sy)))
    return eingabe


def _weich(a, sigma):
    return ndimage.gaussian_filter(a, (sigma, sigma, 0) if a.ndim == 3 else sigma)


def farbe_angleichen(roh: Image.Image, orig: Image.Image, W, H, x, y) -> Image.Image:
    aw, ah = roh.size
    sx, sy = aw / W, ah / H
    ox, oy = round(x * sx), round(y * sy)
    ow, oh = round(orig.width * sx), round(orig.height * sy)
    m = np.asarray(roh, np.float32) / 255
    o = np.asarray(orig.resize((ow, oh), Image.LANCZOS), np.float32) / 255
    innen = m[oy:oy + oh, ox:ox + ow]
    for k in range(3):
        a, b = np.polyfit(innen[..., k].ravel(), o[..., k].ravel(), 1)
        m[..., k] = m[..., k] * a + b
    sigma = max(aw, ah) / 40
    diff = np.zeros_like(m)
    gewicht = np.zeros(m.shape[:2], np.float32)
    diff[oy:oy + oh, ox:ox + ow] = o - m[oy:oy + oh, ox:ox + ow]
    gewicht[oy:oy + oh, ox:ox + ow] = 1
    diff = _weich(diff, sigma) / np.maximum(_weich(gewicht, sigma), 1e-3)[..., None]
    _, (iy, ix) = ndimage.distance_transform_edt(gewicht < 0.5, return_indices=True)
    m = np.clip(m + _weich(diff[iy, ix], sigma), 0, 1)
    return Image.fromarray((m * 255 + 0.5).astype(np.uint8))


def einsetzen(korr: Image.Image, orig: Image.Image, W, H, x, y) -> Image.Image:
    gross = np.asarray(korr.resize((W, H), Image.LANCZOS), np.float32)
    saum = max(4, round(min(orig.size) * 0.015))
    innen = np.zeros((H, W), bool)
    innen[y:y + orig.height, x:x + orig.width] = True
    # Bildkanten der Leinwand zaehlen nicht als Naht
    pad = np.pad(innen, saum + 1, mode="edge")
    d = ndimage.distance_transform_edt(pad)[saum + 1:-saum - 1, saum + 1:-saum - 1]
    deck = np.clip(d / saum, 0, 1)[y:y + orig.height, x:x + orig.width, None]
    teil = gross[y:y + orig.height, x:x + orig.width]
    gross[y:y + orig.height, x:x + orig.width] = (
        np.asarray(orig, np.float32) * deck + teil * (1 - deck))
    return Image.fromarray(np.clip(gross + 0.5, 0, 255).astype(np.uint8))


def pipeline_laden(modelle: str):
    import torch
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel

    basis = os.path.join(modelle, "klein-base-4b")
    einbettung = os.path.join(modelle, "prompt_einbettung.pt")
    if not os.path.exists(einbettung):
        pipe = Flux2KleinPipeline.from_pretrained(basis, transformer=None, vae=None,
                                                  torch_dtype=torch.bfloat16)
        with torch.no_grad():
            positiv, _ = pipe.encode_prompt(PROMPT, device="cpu")
        torch.save({"positiv": positiv}, einbettung)
        del pipe
    positiv = torch.load(einbettung)["positiv"]
    transformer = Flux2Transformer2DModel.from_pretrained(
        os.path.join(modelle, "klein-4b-destilliert", "transformer"), torch_dtype=torch.bfloat16)
    pipe = Flux2KleinPipeline.from_pretrained(basis, text_encoder=None, tokenizer=None,
                                              transformer=transformer,
                                              torch_dtype=torch.bfloat16)
    pipe.register_to_config(is_distilled=True)
    pipe.load_lora_weights(os.path.join(modelle, "lora", "flux-outpaint-lora.safetensors"),
                           adapter_name="outpaint")
    pipe.fuse_lora(lora_scale=LORA_STAERKE)
    pipe.unload_lora_weights()
    pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn,
                                              compute_dtype=torch.bfloat16)
    pipe.to("cuda")
    return pipe, positiv.to("cuda")


def main(argv: list[str]) -> None:
    import torch

    modelle, bild, ausgabe, format_ = argv[:4]
    anteil = float(argv[4]) if len(argv) > 4 else 1.0
    seed = int(argv[5]) if len(argv) > 5 else 1
    b, h = (int(z) for z in format_.split(":"))
    orig = ImageOps.exif_transpose(Image.open(bild)).convert("RGB")
    W, H, x, y = leinwand(orig.size, b / h, anteil)
    eingabe = eingabe_bauen(orig, W, H, x, y)
    pipe, positiv = pipeline_laden(modelle)
    roh = pipe(image=eingabe, prompt_embeds=positiv, width=eingabe.width,
               height=eingabe.height, num_inference_steps=SCHRITTE, guidance_scale=CFG,
               generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    korr = farbe_angleichen(roh, orig, W, H, x, y)
    einsetzen(korr, orig, W, H, x, y).save(ausgabe, quality=92)


if __name__ == "__main__":
    main(sys.argv[1:])
