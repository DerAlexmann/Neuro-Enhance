"""
Checkpoints von PyTorch Lightning lesen, ohne fremden Code auszufuehren - fuer den
ONNX-Export, nicht Teil der App

Ein Lightning-Checkpoint (etwa best.ckpt von LaMa) ist ein Pickle, das neben den
Gewichten auch Objekte von pytorch_lightning, omegaconf und Co. enthaelt.
torch.load(weights_only=True) lehnt sie ab, weights_only=False wuerde beliebigen
Code ausfuehren und braeuchte all diese Pakete. Dieser Leser laesst nur das zu,
was Tensoren und Woerterbuecher brauchen; alles andere wird zu einem leeren
Platzhalter, der seine Argumente schluckt.

Licensed under MIT License
Copyright 2026 Alexander Unverhau
Created with assistance of Claude AI
"""

from __future__ import annotations

import collections
import pickle
import types

import torch

ERLAUBT = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("torch._utils", "_rebuild_parameter"): torch._utils._rebuild_parameter,
    ("torch", "FloatStorage"): torch.FloatStorage,
    ("torch", "LongStorage"): torch.LongStorage,
    ("torch", "IntStorage"): torch.IntStorage,
    ("torch", "HalfStorage"): torch.HalfStorage,
    ("torch", "ByteStorage"): torch.ByteStorage,
    ("torch", "BoolStorage"): torch.BoolStorage,
}


class Platzhalter(dict):
    """Steht fuer jede fremde Klasse oder Funktion: nimmt alles an, tut nichts.

    Ein dict, weil Pickle manchen Objekten Eintraege oder Listenelemente zuweist.
    """

    def __init__(self, *_args, **_kwargs):
        super().__init__()

    def append(self, _wert):
        pass

    def extend(self, _werte):
        pass

    def __call__(self, *_args, **_kwargs):
        return Platzhalter()

    def __setstate__(self, _zustand):
        pass


class _Leser(pickle.Unpickler):
    def find_class(self, modul, name):
        return ERLAUBT.get((modul, name), Platzhalter)


_modul = types.ModuleType("vorsichtig")
_modul.Unpickler = _Leser
_modul.load = pickle.load
_modul.__name__ = "pickle"


def gewichte(pfad: str) -> dict:
    """state_dict eines Lightning-Checkpoints - nur Tensoren."""
    daten = torch.load(pfad, map_location="cpu", weights_only=False, pickle_module=_modul)
    zustand = daten["state_dict"] if "state_dict" in daten else daten
    return {k: v for k, v in zustand.items() if isinstance(v, torch.Tensor)}
