# Neuro-Enhance

**Neuro-Enhance - An open-source, GPU-accelerated AI image & photo enhancer powered by neural networks.**<br>
**Neuro-Enhance - ein quelloffener, GPU-beschleunigter KI-Bild- und Fotoverbesserer auf Basis neuronaler Netze.**

[![CI](https://github.com/DerAlexmann/Neuro-Enhance/actions/workflows/ci.yml/badge.svg)](https://github.com/DerAlexmann/Neuro-Enhance/actions/workflows/ci.yml)

Supports hardware acceleration via NVIDIA CUDA / RTX series GPUs.

*NVIDIA and RTX are trademarks of NVIDIA Corporation.*

*[Deutsche Fassung: README.md](README.md)*

> **Early development.** Version 0.1.0 is the groundwork: the graphics card check at
> start-up, detection of the feature tier and the user interface. The editing and AI
> features will follow in the next versions.

## System requirements

Neuro-Enhance computes on the graphics card only. **Without an NVIDIA RTX graphics card
the program does not start**; instead it reports what was found and what is missing.
There is deliberately no fallback mode on the CPU.

| | Minimum |
|---|---|
| Operating system | Windows 10 or 11, 64-bit |
| Graphics card | NVIDIA GeForce RTX 2000 series or newer, including professional RTX cards (RTX A series, RTX Ada) |
| Video memory | 4 GB for AI features; below that classic filters only |
| Graphics driver | NVIDIA 570.65 or newer |
| Python (only when run as a script) | 3.10 or newer |

**Not supported** are GTX cards – including the GTX 16 series, which belongs to the same
generation as the RTX 2000 but has no Tensor Cores – and graphics cards from other
vendors.

### Feature tiers

What is available depends mainly on the video memory. Neuro-Enhance detects the tier at
start-up; features of a higher tier are shown greyed out with a note.

| Tier | Video memory | Examples | Scope |
|---|---|---|---|
| – | below 4 GB | RTX 2050 (laptop) | classic filters only |
| S | 4 GB | RTX 3050 Laptop | classic filters, small AI models |
| M | 6–8 GB | RTX 2060, 3060 Ti, 4060, 5060 | all restoration models: upscaling, denoising, background removal, object selection |
| L | 12–16 GB | RTX 3060 12 GB, 4070, 5070 Ti | in addition larger models and generative fill |
| XL | 24 GB and more | RTX 3090, 4090, 5090 | full feature set |

Precision: all RTX cards compute in FP16; from the RTX 4000 (Ada) also in FP8, from the
RTX 5000 (Blackwell) also in FP4. The values are shown on the "About & copyright" tab.

## Planned features

- **Classic filters on the GPU**: exposure, curves, colour, LUTs, sharpening, denoising,
  dehaze, geometry, RAW development.
- **AI features**: upscaling, denoising, deblurring, background removal, click-to-select
  objects, object removal, depth maps for synthetic depth of field.
- **Non-destructive**: every step stays adjustable and can be switched off.

## Running it

**As a script:**

```bash
pip install -r requirements.txt
python Neuro-Enhance.pyw
```

A ready-made executable will follow with the first release.

## Privacy

Neuro-Enhance works entirely on your own computer. No images are uploaded and no usage
data is sent. In later versions the program downloads AI models **only on explicit
request**, from the source stated for each model.

## A note on AI results

AI methods such as upscaling or denoising add image details that were not present in the
original. Processed images are therefore not suitable as evidence or for documentation
purposes.

## Contributing

Bug reports, suggestions and translations are welcome – [CONTRIBUTING.md](CONTRIBUTING.md)
explains structure, style and tests (in German; issues and pull requests in English are
just as welcome). Please report security issues not as an issue but as described in
[SECURITY.md](SECURITY.md).

## Licence and legal notices

[MIT](LICENSE) – Copyright 2026 Alexander Unverhau.
Created with assistance of Claude AI.

Third-party components and their licences are listed in [NOTICE](NOTICE). AI models are
subject to their own licences, which are shown before downloading.

**Trademarks:** NVIDIA, RTX, GeForce, CUDA and TensorRT are trademarks or registered
trademarks of NVIDIA Corporation in the U.S. and other countries. Qt is a trademark of The
Qt Company Ltd., Python a trademark of the Python Software Foundation, Windows a trademark
of Microsoft Corporation. All other trademarks are the property of their respective
owners.

Neuro-Enhance is an independent project and is not affiliated with, endorsed or sponsored
by NVIDIA Corporation.
