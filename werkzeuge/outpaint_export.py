"""
KI-Erweitern nach ONNX wandeln - Entwicklerwerkzeug, nicht Teil der App

FLUX.2 [klein] 4B (destilliert) mit der Outpaint-LoRA von fal, so wie
werkzeuge/outpaint_referenz.py sie in PyTorch rechnet: LoRA mit Staerke 1,1 fest
in die Gewichte eingerechnet, fester Prompt, 4 Schritte ohne CFG.

Erzeugt in modelle/ (oder --ziel):
  flux2-klein-outpaint.onnx        Transformer. Die Gewichte der MatMul in int8
                                   (MatMulNBits, Block 128, symmetrisch), der Rest
                                   FP16 - nur die Winkel der Positions- und
                                   Zeiteinbettung (Sin/Cos und was sie speist) bleiben
                                   FP32, in FP16 waeren sie um Zehntel verschoben.
                                   Ein- und Ausgaenge FP32.
  flux2-klein-outpaint-1.bin ...   seine Gewichte als externe ONNX-Daten, auf Dateien
                                   unter 2 GB verteilt (Grenze von GitHub je Datei)
  flux2-vae-kodierer.onnx          VAE-Encoder: sRGB 0..1 -> Mittelwert der Latents
  flux2-vae-dekodierer.onnx        VAE-Decoder: Latents -> sRGB 0..1; beide FP16
  flux2-klein-outpaint-daten.npz   Einbettung des Prompts (1, 512, 7680), Normierung
                                   der Latents (bn des VAE) und Zeitplan des Schedulers
  LICENSE-FLUX2-klein.txt          Apache-2.0 (aus LICENSES/)
  NOTICE-FLUX2-klein.txt           Herkunft und Aenderungen (aus LICENSES/)
und gibt Groesse und SHA-256 jeder Datei aus - fuer DATEIEN in silberkorn/ki.py.

Geprueft wird (abschaltbar mit --ohne-pruefung):
  - jedes Netz gegen PyTorch, als Kosinus-Aehnlichkeit der Ausgaben,
  - der ganze Ablauf, wie ihn silberkorn.ki.outpaint_rechnen() rechnet, gegen
    Flux2KleinPipeline - mit demselben Startrauschen auf einer Probeleinwand.

Aufruf (braucht PyTorch, diffusers 0.41, peft, transformers, onnx und
onnxruntime-gpu; der Transformer wird in FP32 auf dem Prozessor exportiert, das
braucht rund 48 GB Arbeitsspeicher):
    python werkzeuge/outpaint_export.py <modelle> [--ziel ORDNER] [--fp32] [--vae-fp32]
                                        [--ohne-pruefung]
  <modelle> wie bei outpaint_referenz.py: klein-base-4b/, klein-4b-destilliert/transformer/,
  lora/flux-outpaint-lora.safetensors; liegt dort prompt_einbettung.pt (von
  outpaint_referenz.py), wird sie uebernommen, sonst mit dem Text-Encoder berechnet.
  --fp32      Transformer ausser den int8-Gewichten in FP32 statt FP16 (falls FP16
              ueberlaeuft - der Kosinus der Pruefung zeigt es)
  --vae-fp32  VAE in FP32 statt FP16

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import types

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outpaint_referenz import LORA_STAERKE, PROMPT  # noqa: E402

PROJEKT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJEKT)
from silberkorn import ki  # noqa: E402

ZIEL = os.path.join(PROJEKT, "modelle")
LIZENZEN = os.path.join(PROJEKT, "LICENSES")
GRENZE = 1_900_000_000          # Bytes je Gewichtsdatei, unter den 2 GiB von GitHub
AUSRICHTUNG = 4096              # Versatz der Gewichte in der Datei (mmap)
OPSET = 17


# --------------------------------------------------------------------------
# Laden
# --------------------------------------------------------------------------

def pipeline_laden(modelle: str):
    """Pipeline in FP32 auf dem Prozessor, LoRA eingerechnet; dazu die Prompt-Einbettung."""
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel

    basis = os.path.join(modelle, "klein-base-4b")
    einbettung = os.path.join(modelle, "prompt_einbettung.pt")
    if os.path.exists(einbettung):
        prompt = torch.load(einbettung)["positiv"]
    else:
        pipe = Flux2KleinPipeline.from_pretrained(basis, transformer=None, vae=None,
                                                  torch_dtype=torch.bfloat16)
        with torch.no_grad():
            prompt, _ids = pipe.encode_prompt(PROMPT, device="cpu")
        torch.save({"positiv": prompt}, einbettung)
        del pipe
    transformer = Flux2Transformer2DModel.from_pretrained(
        os.path.join(modelle, "klein-4b-destilliert", "transformer"), torch_dtype=torch.float32)
    pipe = Flux2KleinPipeline.from_pretrained(basis, text_encoder=None, tokenizer=None,
                                              transformer=transformer,
                                              torch_dtype=torch.float32)
    pipe.register_to_config(is_distilled=True)
    pipe.load_lora_weights(os.path.join(modelle, "lora", "flux-outpaint-lora.safetensors"),
                           adapter_name="outpaint")
    pipe.fuse_lora(lora_scale=LORA_STAERKE)
    pipe.unload_lora_weights()
    pipe.transformer.eval()
    pipe.vae.eval()
    return pipe, prompt.float()


# --------------------------------------------------------------------------
# Huellen: die Netze mit den Ein- und Ausgaengen, die silberkorn/ki.py erwartet
# --------------------------------------------------------------------------

class TransformerHuelle(nn.Module):
    """latents (1, N, 128), prompt (1, 512, 7680), zeit (1,) = sigma, bild_ids (N, 4),
    text_ids (512, 4) -> geschwindigkeit (1, N, 128). N umfasst eigene und
    Referenz-Tokens; die Pipeline nimmt davon nur die eigenen."""

    def __init__(self, transformer):
        super().__init__()
        self.t = transformer

    def forward(self, latents, prompt, zeit, bild_ids, text_ids):
        return self.t(hidden_states=latents, encoder_hidden_states=prompt, timestep=zeit,
                      img_ids=bild_ids, txt_ids=text_ids, guidance=None,
                      return_dict=False)[0]


class KodiererHuelle(nn.Module):
    """sRGB (1, 3, H, W) 0..1 -> Latent (1, 32, H/8, W/8): der Mittelwert der Verteilung,
    wie retrieve_latents(sample_mode="argmax")."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, bild):
        momente = self.vae._encode(bild * 2 - 1)
        return momente[:, :self.vae.config.latent_channels]


class DekodiererHuelle(nn.Module):
    """Latent (1, 32, h, w) -> sRGB (1, 3, 8h, 8w) 0..1, wie postprocess der Pipeline."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        bild = self.vae.decode(latent, return_dict=False)[0]
        return (bild / 2 + 0.5).clamp(0, 1)


def rope_in_fp32(transformer):
    """Positionseinbettung in FP32 statt FP64 - ONNX Runtime rechnet Sin/Cos auf der
    GPU nicht in FP64. Die Positionen sind ganze Zahlen unter 200, der Fehler bleibt
    unter 1e-5."""
    from diffusers.models.embeddings import get_1d_rotary_pos_embed

    def forward(self, ids):
        cos_teile, sin_teile = [], []
        pos = ids.float()
        for i, dim in enumerate(self.axes_dim):
            cos, sin = get_1d_rotary_pos_embed(dim, pos[..., i], theta=self.theta,
                                               repeat_interleave_real=True, use_real=True,
                                               freqs_dtype=torch.float32)
            cos_teile.append(cos)
            sin_teile.append(sin)
        return torch.cat(cos_teile, dim=-1), torch.cat(sin_teile, dim=-1)

    transformer.pos_embed.forward = types.MethodType(forward, transformer.pos_embed)


def rms_ausschreiben(netz: nn.Module) -> list[nn.Module]:
    """torch.nn.RMSNorm als einzelne Rechenschritte: aten::rms_norm kennt der Export
    erst ab Opset 23, und das kann ONNX Runtime 1.26 auf der GPU nicht. Gerechnet
    wird in FP32, wie in RMSNorm selbst. Rueckgabe: die geaenderten Module."""
    def forward(self, x):
        eps = self.eps if self.eps is not None else torch.finfo(torch.float32).eps
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
        if self.weight is not None:
            y = y * self.weight
        return y.to(x.dtype)

    geaendert = [m for m in netz.modules() if isinstance(m, nn.RMSNorm)]
    for modul in geaendert:
        modul.forward = types.MethodType(forward, modul)
    return geaendert


def drehung_mit_matmul(x, freqs_cis, use_real=True, use_real_unbind_dim=-1, sequence_dim=2):
    """apply_rotary_emb aus diffusers fuer Flux (Paare benachbarter Kanaele), ohne
    reshape: TorchScript schriebe dort die Zahl der Tokens beim Export als feste Zahl
    ins Netz. Das Paar (a, b) wird zu (-b, a) - als Produkt mit einer festen Matrix
    aus 0 und +-1, in FP16 und int8 exakt."""
    assert use_real and use_real_unbind_dim == -1
    cos, sin = freqs_cis
    if sequence_dim == 1:
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    else:
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    d = x.shape[-1]
    tausch = torch.zeros(d, d)
    gerade = torch.arange(0, d, 2)
    tausch[gerade + 1, gerade] = -1
    tausch[gerade, gerade + 1] = 1
    gedreht = torch.matmul(x.float(), tausch)
    return (x.float() * cos + gedreht * sin).to(x.dtype)


class Exportfreundlich:
    """Fuer die Dauer des Exports: RoPE in FP32, RMSNorm ausgeschrieben, Drehung ohne
    reshape. Danach rechnet der Transformer wieder genau wie in diffusers."""

    def __init__(self, transformer):
        self.transformer = transformer

    def __enter__(self):
        import diffusers.models.transformers.transformer_flux2 as flux2
        self._modul, self._drehung = flux2, flux2.apply_rotary_emb
        flux2.apply_rotary_emb = drehung_mit_matmul
        rope_in_fp32(self.transformer)
        self._normen = rms_ausschreiben(self.transformer)
        return self

    def __exit__(self, *_fehler):
        self._modul.apply_rotary_emb = self._drehung
        del self.transformer.pos_embed.forward
        for modul in self._normen:
            del modul.forward


def zeitplan(pipe) -> ki.Zeitplan:
    """Die Einstellungen des Schedulers, die silberkorn.ki.sigmas_berechnen nachrechnet."""
    k = pipe.scheduler.config
    for name in ("use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas",
                 "invert_sigmas", "stochastic_sampling", "use_flow_sigmas"):
        assert not k.get(name), f"{name} rechnet silberkorn/ki.py nicht nach"
    assert k.get("time_shift_type", "exponential") in ("exponential", "linear")
    assert k.get("num_train_timesteps", 1000) == 1000
    return ki.Zeitplan(bool(k.use_dynamic_shifting),
                       k.get("time_shift_type", "exponential") == "exponential",
                       float(k.get("shift", 1.0)), float(k.get("shift_terminal") or 0.0))


def daten_speichern(pipe, prompt, pfad: str):
    vae = pipe.vae
    plan = zeitplan(pipe)
    mittel = vae.bn.running_mean.detach().float().numpy()
    streuung = torch.sqrt(vae.bn.running_var + vae.config.batch_norm_eps).detach().float()
    np.savez(pfad, prompt=prompt.detach().float().numpy(), bn_mittel=mittel,
             bn_std=streuung.numpy(), dynamisch=plan.dynamisch,
             exponentiell=plan.exponentiell, verschiebung=plan.verschiebung, ende=plan.ende)
    return pfad


# --------------------------------------------------------------------------
# ONNX: Export, FP16, int8, verteilt speichern
# --------------------------------------------------------------------------

def nach_onnx(huelle: nn.Module, beispiel: tuple, eingaenge: list[str], ausgaenge: list[str],
              achsen: dict) -> onnx.ModelProto:
    """Export ueber TorchScript; Modelle ueber 2 GB schreibt PyTorch mit externen Daten,
    deshalb in einen Zwischenordner, und laedt sie von dort samt Gewichten."""
    with tempfile.TemporaryDirectory() as ordner:
        pfad = os.path.join(ordner, "netz.onnx")
        with torch.no_grad():
            torch.onnx.export(huelle, beispiel, pfad, input_names=eingaenge,
                              output_names=ausgaenge, dynamic_axes=achsen,
                              opset_version=OPSET, dynamo=False, do_constant_folding=True)
        return onnx.load(pfad, load_external_data=True)


# Knoten der RMSNorm von Query und Key: Quadratsumme in FP32 (wie in PyTorch)
NORM_IN_FP32 = ("/norm_q/", "/norm_k/", "/norm_added_q/", "/norm_added_k/")


def fp32_knoten(modell: onnx.ModelProto) -> list[str]:
    """Knoten, die beim Wandeln nach FP16 in FP32 bleiben: Sin, Cos und alles, was
    ihnen zuarbeitet - ihre Winkel reichen bis ~1000 (Zeit), in FP16 waeren sie auf
    eine halbe Einheit genau, Sinus und Kosinus also wertlos -, dazu die RMSNorm."""
    erzeuger = {ausgang: knoten for knoten in modell.graph.node for ausgang in knoten.output}
    gesperrt = {k.name for k in modell.graph.node if any(n in k.name for n in NORM_IN_FP32)}
    offen = [k for k in modell.graph.node if k.op_type in ("Sin", "Cos")]
    fertig: set[str] = set()
    while offen:
        knoten = offen.pop()
        if knoten.name in fertig:
            continue
        fertig.add(knoten.name)
        gesperrt.add(knoten.name)
        offen.extend(erzeuger[e] for e in knoten.input if e in erzeuger)
    return sorted(gesperrt)


def vereinfachen(modell: onnx.ModelProto):
    """Vor der FP16-Wandlung: Constant-Knoten werden Initialisierer - nur so erkennt
    die Wandlung Eingaenge, die FP32 bleiben muessen (etwa die Massstaebe von
    Resize) -, und Identity auf Initialisierern faellt weg. Mit Identity verweist
    der Export auf gleiche Gewichte (etwa lauter Einsen der RMSNorm); die Wandlung
    uebersieht diesen Umweg und gaebe FP32-Knoten FP16-Gewichte."""
    namen = {i.name for i in modell.graph.initializer}
    ersetzen, bleiben = {}, []
    for knoten in modell.graph.node:
        werte = [a for a in knoten.attribute if a.name == "value"]
        if knoten.op_type == "Constant" and not knoten.domain and werte:
            tensor = onnx.TensorProto()
            tensor.CopyFrom(werte[0].t)
            tensor.name = knoten.output[0]
            modell.graph.initializer.append(tensor)
            namen.add(tensor.name)
        elif knoten.op_type == "Identity" and knoten.input[0] in namen:
            ersetzen[knoten.output[0]] = knoten.input[0]
        else:
            bleiben.append(knoten)
    ausgaenge = {a.name for a in modell.graph.output}
    for knoten in bleiben:
        for i, name in enumerate(knoten.input):
            knoten.input[i] = ersetzen.get(name, name)
    bleiben += [onnx.helper.make_node("Identity", [ersetzen[a]], [a])
                for a in ausgaenge if a in ersetzen]
    del modell.graph.node[:]
    modell.graph.node.extend(bleiben)


def nach_fp16(modell: onnx.ModelProto) -> onnx.ModelProto:
    from onnxruntime.transformers.float16 import convert_float_to_float16
    vereinfachen(modell)
    for i, knoten in enumerate(modell.graph.node):
        if not knoten.name:
            knoten.name = f"knoten_{i}"           # node_block_list braucht Namen
    # Ohne Formfolgerung: sie scheitert an Modellen ueber 2 GB
    return convert_float_to_float16(modell, keep_io_types=True, disable_shape_infer=True,
                                    node_block_list=fp32_knoten(modell))


def nach_int8(modell: onnx.ModelProto) -> onnx.ModelProto:
    """Gewichte aller MatMul mit konstanter Matrix nach int8, je 128 Werte ein Massstab."""
    import logging

    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(
        logging.WARNING)
    quant = MatMulNBitsQuantizer(modell, bits=8, block_size=128, is_symmetric=True)
    quant.process()
    return quant.model.model


def _alle_tensoren(modell: onnx.ModelProto):
    from onnx.external_data_helper import _get_all_tensors
    return _get_all_tensors(modell)


def verteilt_speichern(modell: onnx.ModelProto, ordner: str, name: str,
                       grenze: int | None = None) -> list[str]:
    """Modell speichern, die Gewichte der Reihe nach auf Dateien name-1.bin,
    name-2.bin ... verteilt, jede hoechstens `grenze` Bytes. Kleine Tensoren bleiben
    in der .onnx-Datei. Rueckgabe: alle geschriebenen Pfade, die .onnx-Datei zuerst."""
    from onnx.external_data_helper import set_external_data
    grenze = grenze or GRENZE
    stamm = os.path.splitext(name)[0]
    teile: list[str] = []
    datei, belegt = None, 0
    try:
        for tensor in _alle_tensoren(modell):
            if not tensor.HasField("raw_data"):
                tensor.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor),
                                                        tensor.name))
            roh = tensor.raw_data
            if len(roh) < 1024:
                continue
            versatz = -(-belegt // AUSRICHTUNG) * AUSRICHTUNG
            if datei is None or versatz + len(roh) > grenze:
                if datei is not None:
                    datei.close()
                teile.append(f"{stamm}-{len(teile) + 1}.bin")
                datei = open(os.path.join(ordner, teile[-1]), "wb")
                versatz = 0
            datei.seek(versatz)
            datei.write(roh)
            belegt = versatz + len(roh)
            set_external_data(tensor, location=teile[-1], offset=versatz, length=len(roh))
            tensor.data_location = onnx.TensorProto.EXTERNAL
            tensor.ClearField("raw_data")
    finally:
        if datei is not None:
            datei.close()
    pfad = os.path.join(ordner, name)
    onnx.save_model(modell, pfad)
    return [pfad, *(os.path.join(ordner, t) for t in teile)]


def einzeln_speichern(modell: onnx.ModelProto, ordner: str, name: str) -> list[str]:
    pfad = os.path.join(ordner, name)
    onnx.save_model(modell, pfad)
    return [pfad]


# --------------------------------------------------------------------------
# Die drei Netze
# --------------------------------------------------------------------------

def beispiel_transformer(prompt, hoehe: int = 16, breite: int = 16):
    """Eingaben fuer eine Leinwand von 16*hoehe x 16*breite Pixeln, als Torch-Tensoren."""
    n = hoehe * breite
    ids = np.concatenate([ki.positionen(hoehe, breite, 0),
                          ki.positionen(hoehe, breite, ki.OUTPAINT_REFERENZ_T)])
    generator = torch.Generator().manual_seed(0)
    return (torch.randn(1, 2 * n, 128, generator=generator), prompt,
            torch.tensor([0.75]), torch.from_numpy(ids),
            torch.from_numpy(ki.text_positionen(prompt.shape[1])))


def transformer_exportieren(pipe, prompt, ziel: str, fp32: bool = False) -> list[str]:
    huelle = TransformerHuelle(pipe.transformer).eval()
    with Exportfreundlich(pipe.transformer):
        modell = nach_onnx(huelle, beispiel_transformer(prompt),
                           ["latents", "prompt", "zeit", "bild_ids", "text_ids"],
                           ["geschwindigkeit"],
                           {"latents": {1: "tokens"}, "bild_ids": {0: "tokens"},
                            "geschwindigkeit": {1: "tokens"}})
    if not fp32:
        modell = nach_fp16(modell)
    modell = nach_int8(modell)
    return verteilt_speichern(modell, ziel, ki.OUTPAINT_TRANSFORMER)


def vae_exportieren(pipe, ziel: str, fp32: bool = False) -> list[str]:
    kodierer = nach_onnx(KodiererHuelle(pipe.vae).eval(), (torch.rand(1, 3, 128, 192),),
                         ["bild"], ["latent"],
                         {"bild": {2: "hoehe", 3: "breite"}, "latent": {2: "lh", 3: "lb"}})
    latent = torch.randn(1, pipe.vae.config.latent_channels, 16, 24)
    dekodierer = nach_onnx(DekodiererHuelle(pipe.vae).eval(), (latent,), ["latent"], ["bild"],
                           {"latent": {2: "lh", 3: "lb"}, "bild": {2: "hoehe", 3: "breite"}})
    if not fp32:
        kodierer, dekodierer = nach_fp16(kodierer), nach_fp16(dekodierer)
    return [*einzeln_speichern(kodierer, ziel, ki.OUTPAINT_KODIERER),
            *einzeln_speichern(dekodierer, ziel, ki.OUTPAINT_DEKODIERER)]


# --------------------------------------------------------------------------
# Pruefen
# --------------------------------------------------------------------------

def kosinus(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def sitzung(pfad: str):
    import onnxruntime as ort
    anbieter = [a for a in ("CUDAExecutionProvider", "CPUExecutionProvider")
                if a in ort.get_available_providers()]
    return ort.InferenceSession(pfad, providers=anbieter)


def _melden(name: str, soll, ist):
    soll, ist = np.asarray(soll), np.asarray(ist)
    print(f"{name}: Kosinus {kosinus(soll, ist):.6f}, groesste Abweichung "
          f"{np.abs(soll - ist).max():.3e} (Werte bis {np.abs(soll).max():.2f})")


def netze_pruefen(pipe, prompt, ziel: str):
    """Jedes Netz gegen PyTorch, in anderen Groessen als beim Export."""
    kodierer = sitzung(os.path.join(ziel, ki.OUTPAINT_KODIERER))
    bild = torch.rand(1, 3, 256, 384)
    with torch.no_grad():
        soll = KodiererHuelle(pipe.vae)(bild).numpy()
    _melden("VAE-Encoder", soll, kodierer.run(None, {"bild": bild.numpy()})[0])

    dekodierer = sitzung(os.path.join(ziel, ki.OUTPAINT_DEKODIERER))
    latent = torch.from_numpy(soll)
    with torch.no_grad():
        soll = DekodiererHuelle(pipe.vae)(latent).numpy()
    _melden("VAE-Decoder", soll, dekodierer.run(None, {"latent": latent.numpy()})[0])

    transformer = sitzung(os.path.join(ziel, ki.OUTPAINT_TRANSFORMER))
    namen = ["latents", "prompt", "zeit", "bild_ids", "text_ids"]
    for hoehe, breite in ((16, 24), (32, 32)):
        eingaben = beispiel_transformer(prompt, hoehe, breite)
        with torch.no_grad():
            soll = TransformerHuelle(pipe.transformer)(*eingaben).numpy()
        ist = transformer.run(None, {n: t.numpy() for n, t in zip(namen, eingaben, strict=True)})
        _melden(f"Transformer ({16 * breite} x {16 * hoehe})", soll, ist[0])


def probeleinwand(breite: int = 512, hoehe: int = 384) -> np.ndarray:
    """Synthetische Leinwand (uint8): Verlauf mit Kreis und Streifen, links und rechts
    gruen - der Fall "Querformat breiter machen"."""
    y, x = np.mgrid[0:hoehe, 0:breite].astype(np.float32)
    bild = np.stack([x / breite, y / hoehe, 0.5 + 0.3 * np.sin(x / 23 + y / 31)], axis=-1)
    kreis = (x - breite / 2) ** 2 + (y - hoehe / 2) ** 2 < (hoehe / 4) ** 2
    bild[kreis] = (0.9, 0.6, 0.2)
    bild[(y.astype(int) // 24) % 4 == 0] *= 0.6
    rand = breite // 6 // 16 * 16
    bild[:, :rand] = ki.OUTPAINT_GRUEN
    bild[:, breite - rand:] = ki.OUTPAINT_GRUEN
    return (np.clip(bild, 0, 1) * 255 + 0.5).astype(np.uint8)


def ablauf_pruefen(pipe, prompt, ziel: str, seed: int = 1):
    """silberkorn.ki.outpaint_rechnen mit den ONNX-Netzen gegen Flux2KleinPipeline -
    dieselbe Leinwand, dasselbe Startrauschen."""
    from PIL import Image
    leinwand = probeleinwand()
    hoehe, breite = leinwand.shape[:2]
    h, w = hoehe // ki.OUTPAINT_VIELFACHES, breite // ki.OUTPAINT_VIELFACHES
    daten = ki.outpaint_daten_laden(os.path.join(ziel, ki.OUTPAINT_DATEN))
    rauschen = ki.outpaint_rauschen(seed, h * w, daten.mittel.shape[1])
    with torch.no_grad():
        soll = pipe(image=Image.fromarray(leinwand), prompt_embeds=prompt, width=breite,
                    height=hoehe, num_inference_steps=ki.OUTPAINT_SCHRITTE, guidance_scale=1.0,
                    latents=torch.from_numpy(np.ascontiguousarray(ki.entpacken(rauschen, h, w))),
                    output_type="np").images[0]
    kodierer = sitzung(os.path.join(ziel, ki.OUTPAINT_KODIERER))
    transformer = sitzung(os.path.join(ziel, ki.OUTPAINT_TRANSFORMER))
    dekodierer = sitzung(os.path.join(ziel, ki.OUTPAINT_DEKODIERER))
    ist = ki.outpaint_rechnen(
        leinwand.astype(np.float32) / 255, rauschen, daten,
        lambda bild: kodierer.run(None, {"bild": bild})[0],
        lambda eingaben: transformer.run(None, eingaben)[0],
        lambda latent: dekodierer.run(None, {"latent": latent})[0])
    _melden("Ganzer Ablauf (Bild 0..1)", soll, ist)
    abstand = float(np.abs(np.asarray(soll) - ist).mean() * 255)
    print(f"  mittlere Abweichung {abstand:.2f} von 255 Stufen")
    Image.fromarray((np.clip(ist, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
        os.path.join(ziel, "outpaint-probe-onnx.png"))
    Image.fromarray((np.clip(soll, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
        os.path.join(ziel, "outpaint-probe-pytorch.png"))


# --------------------------------------------------------------------------

def pruefsumme(pfad: str) -> str:
    h = hashlib.sha256()
    with open(pfad, "rb") as datei:
        for block in iter(lambda: datei.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None):
    teiler = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    teiler.add_argument("modelle")
    teiler.add_argument("--ziel", default=ZIEL)
    teiler.add_argument("--fp32", action="store_true")
    teiler.add_argument("--vae-fp32", action="store_true")
    teiler.add_argument("--ohne-pruefung", action="store_true")
    args = teiler.parse_args(argv)
    os.makedirs(args.ziel, exist_ok=True)

    pipe, prompt = pipeline_laden(args.modelle)
    ergebnisse = [daten_speichern(pipe, prompt, os.path.join(args.ziel, ki.OUTPAINT_DATEN))]
    ergebnisse += vae_exportieren(pipe, args.ziel, args.vae_fp32)
    ergebnisse += transformer_exportieren(pipe, prompt, args.ziel, args.fp32)
    for quelle, name in (("Apache-2.0.txt", "LICENSE-FLUX2-klein.txt"),
                         ("NOTICE-FLUX2-klein.txt", ki.OUTPAINT_NOTICE)):
        ergebnisse.append(shutil.copyfile(os.path.join(LIZENZEN, quelle),
                                          os.path.join(args.ziel, name)))
    teile = [os.path.basename(p) for p in ergebnisse if p.endswith(".bin")]
    if tuple(teile) != ki.OUTPAINT_GEWICHTE:
        print(f"ACHTUNG: Gewichtsdateien {teile} - OUTPAINT_GEWICHTE in silberkorn/ki.py "
              "anpassen")
    if not args.ohne_pruefung:
        netze_pruefen(pipe, prompt, args.ziel)
        ablauf_pruefen(pipe, prompt, args.ziel)
    for pfad in ergebnisse:
        print(f"{os.path.basename(pfad):34s} {os.path.getsize(pfad):>13,d} B  "
              f"{pruefsumme(pfad)}")


if __name__ == "__main__":
    main()
