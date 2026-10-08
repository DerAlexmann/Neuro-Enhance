"""
KI-Erweitern nach ONNX wandeln - Entwicklerwerkzeug, nicht Teil der App

FLUX.2 [klein] 4B (destilliert) mit der Outpaint-LoRA von fal, so wie
werkzeuge/outpaint_referenz.py sie in PyTorch rechnet: LoRA mit Staerke 1,1 fest
in die Gewichte eingerechnet, fester Prompt, 4 Schritte ohne CFG.

Erzeugt in modelle/ (oder --ziel):
  flux2-klein-outpaint.onnx        Transformer. Die Gewichte der MatMul in int8
                                   (MatMulNBits, Block 128, symmetrisch), der Rest
                                   FP32: in FP16 laufen die Aktivierungen ueber, der
                                   Transformer gibt dann nur NaN aus (geprueft mit
                                   onnxruntime-gpu 1.26). Ein- und Ausgaenge FP32.
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
onnxruntime-gpu; der Transformer wird in FP32 auf dem Prozessor exportiert. Damit
das PyTorch-Modell und das ONNX-Modell nie gleichzeitig im Speicher liegen, rechnet
ein eigener Prozess den PyTorch-Teil samt Sollwerten der Pruefung und legt alles in
<ziel>/_export-zwischenstand/ ab; der Hauptprozess wandelt und prueft danach ohne
PyTorch-Modell):
    python werkzeuge/outpaint_export.py <modelle> [--ziel ORDNER] [--fp16] [--vae-fp32]
                                        [--ohne-pruefung]
  <modelle> wie bei outpaint_referenz.py: klein-base-4b/, klein-4b-destilliert/transformer/,
  lora/flux-outpaint-lora.safetensors; liegt dort prompt_einbettung.pt (von
  outpaint_referenz.py), wird sie uebernommen, sonst mit dem Text-Encoder berechnet.
  --fp16      Transformer ausser den int8-Gewichten in FP16 statt FP32 - nur zum
              Ausprobieren; Sin/Cos der Positions- und Zeiteinbettung und die
              RMSNorm bleiben dabei FP32
  --vae-fp32  VAE in FP32 statt FP16
Spitze im Arbeitsspeicher rund 30 GB. Besteht die Pruefung nicht, bleibt der
Zwischenstand liegen, und ein weiterer Lauf beginnt bei der Wandlung.

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

def onnx_schreiben(huelle: nn.Module, beispiel: tuple, eingaenge: list[str],
                   ausgaenge: list[str], achsen: dict, pfad: str) -> None:
    """Export ueber TorchScript; Modelle ueber 2 GB schreibt PyTorch mit externen Daten."""
    with torch.no_grad():
        torch.onnx.export(huelle, beispiel, pfad, input_names=eingaenge,
                          output_names=ausgaenge, dynamic_axes=achsen,
                          opset_version=OPSET, dynamo=False, do_constant_folding=True)


def nach_onnx(huelle: nn.Module, beispiel: tuple, eingaenge: list[str], ausgaenge: list[str],
              achsen: dict) -> onnx.ModelProto:
    """Export in einen Zwischenordner und von dort samt Gewichten laden."""
    with tempfile.TemporaryDirectory() as ordner:
        pfad = os.path.join(ordner, "netz.onnx")
        onnx_schreiben(huelle, beispiel, eingaenge, ausgaenge, achsen, pfad)
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


def attention_fusionieren(modell: onnx.ModelProto) -> int:
    """Die vom TorchScript-Export zerlegte scaled_dot_product_attention durch
    com.microsoft.MultiHeadAttention ersetzen. Zerlegt legt ONNX Runtime die ganze
    Matrix (Koepfe x Tokens x Tokens) an - bei 1 MP ueber 7 GB; fusioniert rechnet
    sie blockweise. Muster je Block, Q, K, V jeweils (B, S, H, D):
        Transpose(Q, 0213) * c -> MatMul <- Transpose(K, 0231) * c
        -> Softmax -> MatMul <- Transpose(V, 0213) -> Transpose(0213)
    c ist die Wurzel der Skala 1/sqrt(D), wie sie MultiHeadAttention ohnehin nimmt.
    Rueckgabe: Zahl der ersetzten Attention. Nur der Graph aendert sich, die Gewichte
    bleiben (geht auch an einem schon gespeicherten Modell ohne seine Gewichte)."""
    from onnx import helper
    knoten = list(modell.graph.node)
    erzeuger = {a: k for k in knoten for a in k.output}
    nutzer: dict[str, list] = {}
    for k in knoten:
        for e in k.input:
            nutzer.setdefault(e, []).append(k)

    def perm(k):
        return [a for a in k.attribute if a.name == "perm"][0].ints

    def vor_transpose(name, erwartet):
        """Mul(Transpose(x, erwartet), c) -> x, oder Transpose(x, erwartet) -> x."""
        k = erzeuger[name]
        if k.op_type == "Mul":
            k = next(erzeuger[e] for e in k.input
                     if e in erzeuger and erzeuger[e].op_type == "Transpose")
        assert k.op_type == "Transpose" and list(perm(k)) == erwartet, k.name
        return k.input[0]

    neu, ersetzt = [], 0
    kopf_form = helper.make_tensor("mha_form_kopf", onnx.TensorProto.INT64, [3], [0, 0, -1])
    for softmax in [k for k in knoten if k.op_type == "Softmax"]:
        qk = erzeuger[softmax.input[0]]
        av = nutzer[softmax.output[0]][0]
        zurueck = nutzer[av.output[0]][0]
        assert qk.op_type == "MatMul" and av.op_type == "MatMul", softmax.name
        assert zurueck.op_type == "Transpose" and list(perm(zurueck)) == [0, 2, 1, 3]
        q = vor_transpose(qk.input[0], [0, 2, 1, 3])
        k = vor_transpose(qk.input[1], [0, 2, 3, 1])
        v = vor_transpose(av.input[1], [0, 2, 1, 3])
        stamm = softmax.name.rsplit("/", 1)[0] + "/mha"
        # Form der Ausgabe (B, S, H, D) wie Q
        form = stamm + "_form"
        neu += [helper.make_node("Shape", [q], [form], name=form)]
        flach = []
        for name, x in (("q", q), ("k", k), ("v", v)):
            flach.append(f"{stamm}_{name}")
            neu.append(helper.make_node("Reshape", [x, "mha_form_kopf"], [flach[-1]],
                                        name=flach[-1]))
        aus = stamm + "_aus"
        neu.append(helper.make_node("MultiHeadAttention", flach, [aus], name=stamm,
                                    domain="com.microsoft", num_heads=OUTPAINT_KOEPFE))
        neu.append(helper.make_node("Reshape", [aus, form], [zurueck.output[0]],
                                    name=stamm + "_zurueck"))
        for alt in (qk, softmax, av, zurueck):
            alt.output[0] = alt.output[0] + "_unbenutzt"
        ersetzt += 1
    if not ersetzt:
        return 0
    modell.graph.initializer.append(kopf_form)
    modell.graph.node.extend(neu)
    _unbenutzte_entfernen(modell)
    _topologisch_ordnen(modell)
    return ersetzt


OUTPAINT_KOEPFE = 24        # FLUX.2 klein 4B: num_attention_heads
HALB_GENAU = ("MatMulNBits", "MultiHeadAttention")


def rechenkerne_fp16(modell: onnx.ModelProto) -> int:
    """MatMulNBits und MultiHeadAttention in FP16 rechnen lassen - Cast davor und
    danach, ihre Massstaebe als FP16. Nur sie nutzen dann die Tensorkerne; alles
    andere (Restverbindungen, Normierung, Gating, GELU) bleibt FP32, denn ganz in
    FP16 laufen die Aktivierungen ueber. Auf einer RTX 4060 halbiert das die Zeit
    je Schritt; Kosinus zur FP32-Fassung 0,999999. Rueckgabe: Zahl der Knoten."""
    from onnx import TensorProto, helper
    inits = {i.name: i for i in modell.graph.initializer}
    neu, gewandelt, zahl = [], {}, 0
    for k in modell.graph.node:
        if k.op_type not in HALB_GENAU:
            neu.append(k)
            continue
        zahl += 1
        for j, e in enumerate(k.input):
            if not e:
                continue
            init = inits.get(e)
            if init is not None:
                # gepackte int8-Gewichte (Eingang 1) und Nullpunkte bleiben, wie sie sind
                if init.data_type == TensorProto.FLOAT:
                    if e not in gewandelt:
                        halb = numpy_helper.from_array(
                            numpy_helper.to_array(init).astype(np.float16), e + "_fp16")
                        modell.graph.initializer.append(halb)
                        gewandelt[e] = halb.name
                    k.input[j] = gewandelt[e]
                continue
            cast = f"{k.name}_ein{j}_fp16"
            neu.append(helper.make_node("Cast", [e], [cast], name=cast, to=TensorProto.FLOAT16))
            k.input[j] = cast
        neu.append(k)
        for j, ausgang in enumerate(k.output):
            halb = f"{k.name}_aus{j}_fp16"
            k.output[j] = halb
            neu.append(helper.make_node("Cast", [halb], [ausgang], name=ausgang + "_fp32",
                                        to=TensorProto.FLOAT))
    del modell.graph.node[:]
    modell.graph.node.extend(neu)
    _unbenutzte_entfernen(modell)
    return zahl


def _unbenutzte_entfernen(modell: onnx.ModelProto) -> None:
    """Knoten und Initialisierer, von denen keine Ausgabe des Graphen abhaengt."""
    erzeuger = {a: k for k in modell.graph.node for a in k.output}
    gebraucht, offen = set(), [a.name for a in modell.graph.output]
    while offen:
        name = offen.pop()
        k = erzeuger.get(name)
        if k is None or id(k) in gebraucht:
            continue
        gebraucht.add(id(k))
        offen.extend(e for e in k.input if e)
    bleiben = [k for k in modell.graph.node if id(k) in gebraucht]
    eingaenge = {e for k in bleiben for e in k.input}
    inits = [i for i in modell.graph.initializer if i.name in eingaenge]
    del modell.graph.node[:]
    modell.graph.node.extend(bleiben)
    del modell.graph.initializer[:]
    modell.graph.initializer.extend(inits)


def _topologisch_ordnen(modell: onnx.ModelProto) -> None:
    vorhanden = {e.name for e in modell.graph.input} | {
        i.name for i in modell.graph.initializer} | {""}
    offen, sortiert = list(modell.graph.node), []
    while offen:
        rest = []
        for k in offen:
            if all(e in vorhanden for e in k.input):
                sortiert.append(k)
                vorhanden.update(k.output)
            else:
                rest.append(k)
        assert len(rest) < len(offen), "Graph hat einen Zyklus"
        offen = rest
    del modell.graph.node[:]
    modell.graph.node.extend(sortiert)


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


def transformer_roh(pipe, prompt, pfad: str) -> None:
    """Der Transformer als FP32-ONNX auf die Platte, ohne ihn wieder zu laden: Das
    PyTorch-Modell (16 GB) und das geladene ONNX-Modell gleichzeitig im Speicher
    braeuchten zusammen mit den Kopien der Wandlung rund 48 GB."""
    huelle = TransformerHuelle(pipe.transformer).eval()
    with Exportfreundlich(pipe.transformer):
        onnx_schreiben(huelle, beispiel_transformer(prompt),
                       ["latents", "prompt", "zeit", "bild_ids", "text_ids"],
                       ["geschwindigkeit"],
                       {"latents": {1: "tokens"}, "bild_ids": {0: "tokens"},
                        "geschwindigkeit": {1: "tokens"}}, pfad)


def transformer_wandeln(roh: str, ziel: str, fp32: bool = False) -> list[str]:
    """Rohes FP32-ONNX -> FP16 (ausser fp32) -> int8-Gewichte -> fusionierte Attention,
    verteilt gespeichert."""
    modell = onnx.load(roh, load_external_data=True)
    if not fp32:
        modell = nach_fp16(modell)
    modell = nach_int8(modell)
    print(f"Attention fusioniert: {attention_fusionieren(modell)}")
    if fp32:
        print(f"In FP16 gerechnet: {rechenkerne_fp16(modell)} Knoten")
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


MINDEST_KOSINUS = 0.99
KOSINUS: list[float] = []      # alle Ergebnisse der Pruefung dieses Laufs


def _melden(name: str, soll, ist):
    soll, ist = np.asarray(soll), np.asarray(ist)
    KOSINUS.append(kosinus(soll, ist))
    print(f"{name}: Kosinus {KOSINUS[-1]:.6f}, groesste Abweichung "
          f"{np.abs(soll - ist).max():.3e} (Werte bis {np.abs(soll).max():.2f})")


TRANSFORMER_EINGAENGE = ["latents", "prompt", "zeit", "bild_ids", "text_ids"]
TRANSFORMER_PROBEN = ((16, 24), (32, 32))


def sollwerte_berechnen(pipe, prompt, pfad: str):
    """Ein- und Ausgaben von PyTorch fuer alle Pruefungen, als .npz - so laeuft die
    Pruefung der ONNX-Netze spaeter ohne das PyTorch-Modell im Speicher."""
    werte = {}
    bild = torch.rand(1, 3, 256, 384, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        latent = KodiererHuelle(pipe.vae)(bild)
        werte["kodierer_ein"], werte["kodierer_aus"] = bild.numpy(), latent.numpy()
        werte["dekodierer_aus"] = DekodiererHuelle(pipe.vae)(latent).numpy()
        for hoehe, breite in TRANSFORMER_PROBEN:
            eingaben = beispiel_transformer(prompt, hoehe, breite)
            werte[f"transformer_{hoehe}x{breite}"] = \
                TransformerHuelle(pipe.transformer)(*eingaben).numpy()
    werte["ablauf"] = ablauf_soll(pipe, prompt)
    np.savez(pfad, **werte)


def netze_pruefen(soll: dict, prompt, ziel: str):
    """Jedes Netz gegen die Sollwerte von PyTorch, in anderen Groessen als beim Export."""
    kodierer = sitzung(os.path.join(ziel, ki.OUTPAINT_KODIERER))
    _melden("VAE-Encoder", soll["kodierer_aus"],
            kodierer.run(None, {"bild": soll["kodierer_ein"]})[0])
    dekodierer = sitzung(os.path.join(ziel, ki.OUTPAINT_DEKODIERER))
    _melden("VAE-Decoder", soll["dekodierer_aus"],
            dekodierer.run(None, {"latent": soll["kodierer_aus"]})[0])
    del kodierer, dekodierer

    transformer = sitzung(os.path.join(ziel, ki.OUTPAINT_TRANSFORMER))
    for hoehe, breite in TRANSFORMER_PROBEN:
        eingaben = beispiel_transformer(prompt, hoehe, breite)
        ist = transformer.run(None, {n: t.numpy() for n, t in
                                     zip(TRANSFORMER_EINGAENGE, eingaben, strict=True)})
        _melden(f"Transformer ({16 * breite} x {16 * hoehe})",
                soll[f"transformer_{hoehe}x{breite}"], ist[0])


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


def _probe(seed: int):
    """Probeleinwand und Startrauschen - in beiden Prozessen gleich."""
    leinwand = probeleinwand()
    hoehe, breite = leinwand.shape[:2]
    h, w = hoehe // ki.OUTPAINT_VIELFACHES, breite // ki.OUTPAINT_VIELFACHES
    # Kanaele der Latents nach patchify: 4 * latent_channels des VAE (32)
    rauschen = ki.outpaint_rauschen(seed, h * w, 128)
    return leinwand, rauschen, h, w


def ablauf_soll(pipe, prompt, seed: int = 1) -> np.ndarray:
    """Flux2KleinPipeline auf der Probeleinwand mit festem Startrauschen."""
    from PIL import Image
    leinwand, rauschen, h, w = _probe(seed)
    hoehe, breite = leinwand.shape[:2]
    with torch.no_grad():
        return pipe(image=Image.fromarray(leinwand), prompt_embeds=prompt, width=breite,
                    height=hoehe, num_inference_steps=ki.OUTPAINT_SCHRITTE, guidance_scale=1.0,
                    latents=torch.from_numpy(np.ascontiguousarray(ki.entpacken(rauschen, h, w))),
                    output_type="np").images[0]


def ablauf_pruefen(soll: np.ndarray, ziel: str, seed: int = 1):
    """silberkorn.ki.outpaint_rechnen mit den ONNX-Netzen gegen Flux2KleinPipeline -
    dieselbe Leinwand, dasselbe Startrauschen."""
    from PIL import Image
    leinwand, rauschen, _h, _w = _probe(seed)
    daten = ki.outpaint_daten_laden(os.path.join(ziel, ki.OUTPAINT_DATEN))
    assert daten.mittel.shape[1] == rauschen.shape[-1], "Kanalzahl der Latents"
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
    teiler.add_argument("--fp16", action="store_true")
    teiler.add_argument("--vae-fp32", action="store_true")
    teiler.add_argument("--ohne-pruefung", action="store_true")
    # intern: nur der PyTorch-Teil, als eigener Prozess (siehe unten)
    teiler.add_argument("--nur-pytorch", action="store_true", help=argparse.SUPPRESS)
    args = teiler.parse_args(argv)
    os.makedirs(args.ziel, exist_ok=True)
    zwischen = os.path.join(args.ziel, "_export-zwischenstand")
    roh = os.path.join(zwischen, "transformer-fp32.onnx")
    soll_pfad = os.path.join(zwischen, "sollwerte.npz")

    if args.nur_pytorch:
        os.makedirs(zwischen, exist_ok=True)
        pipe, prompt = pipeline_laden(args.modelle)
        daten_speichern(pipe, prompt, os.path.join(args.ziel, ki.OUTPAINT_DATEN))
        vae_exportieren(pipe, args.ziel, args.vae_fp32)
        transformer_roh(pipe, prompt, roh)
        if not args.ohne_pruefung:
            sollwerte_berechnen(pipe, prompt, soll_pfad)
        return

    # Zwei Prozesse, damit PyTorch-Modell (16 GB in FP32) und ONNX-Modell samt Kopien
    # der Wandlung nie gleichzeitig im Arbeitsspeicher liegen
    import subprocess
    befehl = [sys.executable, os.path.abspath(__file__), args.modelle, "--ziel", args.ziel,
              "--nur-pytorch"]
    befehl += [f"--{n}" for n in ("vae-fp32", "ohne-pruefung")
               if getattr(args, n.replace("-", "_"))]
    if os.path.exists(roh) and (args.ohne_pruefung or os.path.exists(soll_pfad)):
        # Ein frueherer Lauf scheiterte erst nach dem PyTorch-Teil
        print(f"Zwischenstand aus {zwischen} wird weiterverwendet")
    else:
        subprocess.run(befehl, check=True)

    ergebnisse = [os.path.join(args.ziel, n)
                  for n in (ki.OUTPAINT_DATEN, ki.OUTPAINT_KODIERER, ki.OUTPAINT_DEKODIERER)]
    ergebnisse += transformer_wandeln(roh, args.ziel, fp32=not args.fp16)
    for quelle, name in (("Apache-2.0.txt", "LICENSE-FLUX2-klein.txt"),
                         ("NOTICE-FLUX2-klein.txt", ki.OUTPAINT_NOTICE)):
        ergebnisse.append(shutil.copyfile(os.path.join(LIZENZEN, quelle),
                                          os.path.join(args.ziel, name)))
    teile = [os.path.basename(p) for p in ergebnisse if p.endswith(".bin")]
    if tuple(teile) != ki.OUTPAINT_GEWICHTE:
        print(f"ACHTUNG: Gewichtsdateien {teile} - OUTPAINT_GEWICHTE in silberkorn/ki.py "
              "anpassen")
    if not args.ohne_pruefung:
        with np.load(soll_pfad) as soll:
            soll = dict(soll)
        prompt = torch.from_numpy(ki.outpaint_daten_laden(
            os.path.join(args.ziel, ki.OUTPAINT_DATEN)).prompt)
        netze_pruefen(soll, prompt, args.ziel)
        ablauf_pruefen(soll["ablauf"], args.ziel)
    if all(k >= MINDEST_KOSINUS for k in KOSINUS):     # NaN faellt hier durch
        shutil.rmtree(zwischen)
    else:
        print(f"ACHTUNG: Pruefung nicht bestanden (Kosinus unter {MINDEST_KOSINUS} oder NaN) "
              f"- Zwischenstand bleibt fuer einen weiteren Lauf in {zwischen}")
    for pfad in ergebnisse:
        print(f"{os.path.basename(pfad):34s} {os.path.getsize(pfad):>13,d} B  "
              f"{pruefsumme(pfad)}")


if __name__ == "__main__":
    main()
